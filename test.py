"""
Test / Evaluate a trained TRPO policy on the Velocity-Conditioned Ant.

This script loads a saved checkpoint, fires up the MuJoCo GUI so you can
watch the ant walk, and prints live stats to the terminal.

"""

# ---------------------------------------------------------------------
# Imports
from types import SimpleNamespace

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
from trpo import RunningMeanStd


# -----------------------------------------------------------------------
# Helper: load a trained checkpoint from disk

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

    obs_rms = None
    if "obs_rms" in ckpt:
        obs_rms = RunningMeanStd(shape=(obs_dim,))
        obs_rms.mean = ckpt["obs_rms"]["mean"]
        obs_rms.var = ckpt["obs_rms"]["var"]
        obs_rms.count = ckpt["obs_rms"]["count"]

    return policy, obs_rms, ckpt


# -----------------------------------------------------------------------------
# Helper: pick an action from the policy

def select_action(policy, obs_numpy, stochastic, device="cpu", obs_rms=None):
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
    if obs_rms is not None:
        obs_numpy = obs_rms.normalize(obs_numpy).astype(np.float32)

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
    policy, obs_rms, ckpt = load_trained_policy(args.checkpoint, device=device)

    train_iter = ckpt.get("iteration", "?")
    train_reward = ckpt.get("best_reward", "?")
    saved_args = ckpt.get("args", {})
    print(f"  Training iteration : {train_iter}")
    print(f"  Best train return  : {train_reward}")
    print(f"  Network hidden     : {ckpt['hidden']}")
    print()

    # Resolve DR settings.
    if args.domain_rand is None:
        use_dr = any(saved_args.get(k, False) for k in
                                                ("randomize_mass",
                                                 "randomize_friction",
                                                 "randomize_action_delay",
                                                 "randomize_obs_delay"))
    else:
        use_dr = args.domain_rand

    
    dr_kwargs = {}
    if use_dr:
        dr_kwargs = {
            "randomize_mass": saved_args.get("randomize_mass", False),
            "mass_scale_range": (
                saved_args.get("mass_min_scale", 0.9),
                saved_args.get("mass_max_scale", 1.1),
            ),
            "randomize_friction": saved_args.get("randomize_friction", False),
            "friction_scale_range": (
                saved_args.get("friction_min_scale", 0.7),
                saved_args.get("friction_max_scale", 1.3),
            ),
            "randomize_action_delay": saved_args.get("randomize_action_delay", False),
            "action_delay_range": (
                saved_args.get("action_delay_min", 0),
                saved_args.get("action_delay_max", 2),
            ),
            "randomize_obs_delay": saved_args.get("randomize_obs_delay", False),
            "obs_delay_range": (
                saved_args.get("obs_delay_min", 0),
                saved_args.get("obs_delay_max", 0),
            ),
        }

    # ── 2. Create the environment with the MuJoCo 3-D viewer ─────────
    #    render_mode="human" opens a window where you can see the ant.
    env = VelocityAntEnv(render_mode="human", **dr_kwargs)

    dr_flags = [k for k in ("randomize_mass", "randomize_friction", "randomize_action_delay", "randomize_obs_delay")
                if dr_kwargs.get(k, False)]
    
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

            done = False
            ep_return = 0.0       # total reward this episode
            ep_steps = 0          # timesteps this episode
            ep_vel_errors = []    # per-step velocity tracking error

            print(f"── Episode {episode_num} ──")

            while not done:
                # Ask the policy for an action.
                action = select_action(policy, obs, args.stochastic, device, obs_rms=obs_rms)

                # Clip to the environment's valid action range.
                # The Ant has 8 torque-controlled joints, each in [-1, 1].
                action = np.clip(action, env.action_space.low, env.action_space.high)

                # Step the simulation forward by one timestep.
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                ep_return += reward
                ep_steps += 1

                if "velocity_error" in info:
                    ep_vel_errors.append(info["velocity_error"])

                # Print a progress dot every 100 steps so you know it's alive
                if ep_steps % 100 == 0:
                    avg_err = np.mean(ep_vel_errors[-100:]) if ep_vel_errors else 0
                    fw_spd = info.get("forward_speed", 0.0)
                    print(f"    step {ep_steps:4d}  |  "
                          f"reward so far: {ep_return:8.1f}  |  "
                          f"fwd: {fw_spd:5.2f} m/s  |  "
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
CONFIG = SimpleNamespace(
    checkpoint = "checkpoints/best_model.pt",
    episodes = 0,
    stochastic = False,
    domain_rand = None,
)

if __name__ == "__main__":
    run_evaluation(CONFIG)