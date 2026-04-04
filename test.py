"""
Test / Evaluate a trained TRPO policy on the Velocity-Conditioned Ant.

This script loads a saved checkpoint, fires up the MuJoCo GUI so you can
watch the ant walk, and prints live stats to the terminal.

Usage
-----
  # Run the best saved model with the MuJoCo 3-D viewer:
  python test.py --checkpoint checkpoints/best_model.pt

  # Use a specific velocity command instead of random ones:
  python test.py --checkpoint checkpoints/best_model.pt --cmd_vx 1.5 --cmd_vy 0.0

  # Run 5 episodes, then quit automatically:
  python test.py --checkpoint checkpoints/best_model.pt --episodes 5

  # Use stochastic (sampled) actions instead of the deterministic mean:
  python test.py --checkpoint checkpoints/best_model.pt --stochastic
"""

# ─────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────
import argparse
import sys

import numpy as np
import torch

# These are the modules we wrote during training.
# "environment" contains our custom Ant wrapper that adds velocity
# commands to the observation and computes our natural-gait reward.
from enviorment_wrapper import VelocityAntEnv

# "models" contains the neural network architectures:
#   - PolicyNetwork:  takes an observation, outputs a probability
#                     distribution over actions  (a Gaussian).
#   - ValueNetwork:   not needed at test time, but we import it
#                     so we could inspect it if we wanted.
from models import PolicyNetwork


# ─────────────────────────────────────────────────────────────────────
# Helper: load a trained checkpoint from disk
# ─────────────────────────────────────────────────────────────────────

def load_trained_policy(checkpoint_path, device="cpu"):
    """
    Read a .pt checkpoint file and reconstruct the policy network.

    A checkpoint is a Python dictionary saved by torch.save() that
    contains:
      - 'policy_state'  : the learned weights of the policy network
      - 'obs_dim'       : size of the observation vector (31 for us)
      - 'act_dim'       : size of the action vector (8 joints)
      - 'hidden'        : width of hidden layers (e.g. 256)
      - 'iteration'     : training iteration when this was saved
      - 'best_reward'   : best mean episode return seen so far
      - 'args'          : all the command-line arguments from training

    Returns
    -------
    policy : PolicyNetwork   – ready to use, weights loaded, in eval mode
    meta   : dict            – everything else in the checkpoint
    """
    # Load the file.  weights_only=False because we also stored plain
    # Python dicts / ints / floats alongside the tensor weights.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Rebuild the network with the same architecture that was used
    # during training.  If the sizes don't match, load_state_dict will
    # raise an error — that's a safety check for free.
    obs_dim = ckpt["obs_dim"]
    act_dim = ckpt["act_dim"]
    hidden = ckpt["hidden"]

    policy = PolicyNetwork(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(hidden, hidden),
    ).to(device)

    # Copy the learned weights into the network.
    policy.load_state_dict(ckpt["policy_state"])

    # Switch to evaluation mode.  For our network this only affects
    # batch-norm / dropout (we don't use either), but it's good practice.
    policy.eval()

    return policy, ckpt


# ─────────────────────────────────────────────────────────────────────
# Helper: pick an action from the policy
# ─────────────────────────────────────────────────────────────────────

def select_action(policy, obs_numpy, stochastic, device="cpu"):
    """
    Given a numpy observation, ask the policy network for an action.

    Two modes:
      deterministic (default):  use the mean of the Gaussian.
          This is the "best guess" action and gives smooth, repeatable
          behaviour.  Ideal for evaluation / demo.

      stochastic (--stochastic flag):  sample from the Gaussian.
          This is what we do during training so the agent explores.
          At test time it adds visible jitter but is useful to gauge
          how uncertain the policy is.

    Returns
    -------
    action : np.ndarray of shape (act_dim,)
    """
    # Convert numpy → torch tensor, add a batch dimension [1, obs_dim]
    obs_tensor = torch.as_tensor(
        obs_numpy, dtype=torch.float32, device=device
    ).unsqueeze(0)

    # We don't need gradients at test time — save memory and time.
    with torch.no_grad():
        # policy.forward() returns a torch Normal distribution.
        #   dist.loc   = mean  (μ)
        #   dist.scale = std   (σ)
        dist = policy(obs_tensor)

        if stochastic:
            action = dist.sample()           # random draw from N(μ, σ)
        else:
            action = dist.loc                # just the mean μ

    # Remove batch dim and move back to numpy.
    return action.squeeze(0).cpu().numpy()

# ─────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────

def run_evaluation(args):
    device = "cpu"  # rendering is CPU-bound; GPU adds no benefit here

    # ── 1. Load the trained policy from the checkpoint file ───────────
    print(f"Loading checkpoint: {args.checkpoint}")
    policy, ckpt = load_trained_policy(args.checkpoint, device=device)

    train_iter = ckpt.get("iteration", "?")
    train_reward = ckpt.get("best_reward", "?")
    print(f"  Training iteration : {train_iter}")
    print(f"  Best train return  : {train_reward}")
    print(f"  Network hidden     : {ckpt['hidden']}")
    print()

    # ── 2. Create the environment with the MuJoCo 3-D viewer ─────────
    #    render_mode="human" opens a window where you can see the ant.
    env = VelocityAntEnv(render_mode="human")

    # If the user specified a fixed velocity command, we'll override the
    # random sampling that normally happens inside the environment.
    use_fixed_cmd = any(
        x is not None for x in [args.cmd_vx, args.cmd_vy, args.cmd_vz, args.cmd_yaw]
    )
    if use_fixed_cmd:
        fixed_cmd = np.array([
            args.cmd_vx  if args.cmd_vx  is not None else 0.0,
            args.cmd_vy  if args.cmd_vy  is not None else 0.0,
            args.cmd_vz  if args.cmd_vz  is not None else 0.0,
            args.cmd_yaw if args.cmd_yaw is not None else 0.0,
        ], dtype=np.float32)
        print(f"Using FIXED velocity command: "
              f"Vx={fixed_cmd[0]:.2f}  Vy={fixed_cmd[1]:.2f}  "
              f"Vz={fixed_cmd[2]:.2f}  yaw={fixed_cmd[3]:.2f}")
    else:
        print("Velocity commands will be sampled randomly each episode.")

    mode_str = "STOCHASTIC" if args.stochastic else "DETERMINISTIC"
    print(f"Action mode: {mode_str}")
    print(f"Episodes to run: {'∞ (Ctrl-C to stop)' if args.episodes == 0 else args.episodes}")
    print()

    # ── 3. Run episodes ──────────────────────────────────────────────
    all_returns = []
    all_vel_errors = []
    episode_num = 0

    try:
        while True:
            # Check if we've run enough episodes.
            if args.episodes > 0 and episode_num >= args.episodes:
                break

            episode_num += 1

            # Reset the environment — this gives us the first observation
            # and randomly samples a new velocity command (unless we
            # override it below).
            obs, info = env.reset()

            # Override the random velocity command if the user specified one.
            if use_fixed_cmd:
                env._cmd_vel = fixed_cmd.copy()
                # Re-augment observation with the fixed command.
                obs = np.concatenate([obs[:27], fixed_cmd]).astype(np.float32)

            done = False
            ep_return = 0.0       # total reward this episode
            ep_steps = 0          # timesteps this episode
            ep_vel_errors = []    # per-step velocity tracking error

            print(f"── Episode {episode_num} ──")
            cmd = env._cmd_vel
            print(f"  Command: Vx={cmd[0]:+.2f}  Vy={cmd[1]:+.2f}  "
                  f"Vz={cmd[2]:+.2f}  yaw={cmd[3]:+.2f}")

            while not done:
                # Ask the policy for an action.
                action = select_action(policy, obs, args.stochastic, device)

                # Clip to the environment's valid action range.
                # The Ant has 8 torque-controlled joints, each in [-1, 1].
                action = np.clip(action, env.action_space.low, env.action_space.high)

                # Step the simulation forward by one timestep.
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_return += reward
                ep_steps += 1

                if "velocity_error_xy" in info:
                    ep_vel_errors.append(info["velocity_error_xy"])

                # Print a progress dot every 100 steps so you know it's alive
                if ep_steps % 100 == 0:
                    avg_err = np.mean(ep_vel_errors[-100:]) if ep_vel_errors else 0
                    print(f"    step {ep_steps:4d}  |  "
                          f"reward so far: {ep_return:8.1f}  |  "
                          f"vel_err(last 100): {avg_err:.3f}")

            # ── Episode summary ───────────────────────────────────────
            mean_vel_err = np.mean(ep_vel_errors) if ep_vel_errors else 0.0
            reason = "fell" if terminated else "time limit"
            print(f"  Result: {reason} after {ep_steps} steps")
            print(f"  Total return   : {ep_return:.2f}")
            print(f"  Mean vel error : {mean_vel_err:.4f}")
            print()

            all_returns.append(ep_return)
            all_vel_errors.append(mean_vel_err)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        env.close()

    # ── 4. Summary across all episodes ────────────────────────────────
    if all_returns:
        print("\n" + "=" * 50)
        print("  EVALUATION SUMMARY")
        print("=" * 50)
        print(f"  Episodes run        : {len(all_returns)}")
        print(f"  Mean return         : {np.mean(all_returns):.2f}  "
              f"± {np.std(all_returns):.2f}")
        print(f"  Min / Max return    : {np.min(all_returns):.2f}  "
              f"/ {np.max(all_returns):.2f}")
        print(f"  Mean velocity error : {np.mean(all_vel_errors):.4f}")
        print("=" * 50)


# ─────────────────────────────────────────────────────────────────────
# Command-line interface
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained TRPO Ant policy with the MuJoCo viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test.py --checkpoint checkpoints/best_model.pt
  python test.py --checkpoint checkpoints/best_model.pt --cmd_vx 1.0 --cmd_vy 0.0
  python test.py --checkpoint checkpoints/best_model.pt --episodes 10
  python test.py --checkpoint checkpoints/best_model.pt --stochastic
""",
    )

    # required
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a .pt checkpoint file saved during training")

    # behaviour
    p.add_argument("--episodes", type=int, default=0,
                   help="Number of episodes to run (0 = infinite, Ctrl-C to stop)")
    p.add_argument("--stochastic", action="store_true",
                   help="Sample actions from the policy distribution "
                        "(default: use deterministic mean)")

    # optional fixed velocity command
    p.add_argument("--cmd_vx", type=float, default=None,
                   help="Fixed forward velocity command (m/s)")
    p.add_argument("--cmd_vy", type=float, default=None,
                   help="Fixed lateral velocity command (m/s)")
    p.add_argument("--cmd_vz", type=float, default=None,
                   help="Fixed vertical velocity command (m/s, usually 0)")
    p.add_argument("--cmd_yaw", type=float, default=None,
                   help="Fixed yaw-rate command (rad/s)")

    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())