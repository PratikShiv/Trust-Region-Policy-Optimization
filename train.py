"""
Training script for TRPO on the velocity conditioned Ant Env

Usage:
------
    python train.py                                     # Train with defaults
    python train.py --wandb                             # train + log to Weights and Biases
    python train.py --resume checkpoints/ckpt_100.pt    # Resume from checkpoint
    python train.py --render                            # train with live rendering
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from enviorment_wrapper import VelocityAntEnv
from models import PolicyNetwork, ValueNetwork
from trpo import TRPOAgent, comput_gae

# wandb is optional
wandb = None

def _init_wandb(args):
    """ Initialise Weights and Biases"""
    global wandb
    import wandb as _wandb
    wandb = _wandb

    wandb.init(
        project="trpo_env_ant_model",
        config={
            "algorithm": "TRPO",
            "env": "Ant-v5-velocity",
            "iterations": args.iterations,
            "batch_size": args.batch_size,
            "max_kl": args.max_kl,
            "damping": args.damping,
            "gamma": args.gamma,
            "lam": args.lam,
            "value_lr": args.value_lr,
            "value_epochs": args.value_epochs,
            "cg_iters": args.cg_iters,
            "hidden": args.hidden,
            "seed": args.seed
        },
    )
    wandb.define_metric("iteration")
    wandb.define_metric("*", step_metric="iteration")
    return wandb


# ---------------------------------------------------------------------------------------
# Save / Load Checkpoints
def save_checkpoint(path, policy, value_fn, value_optimizer,
                    iteration, total_steps, best_reward, args):
    # Persist everything needed to fully resume training later

    torch.save(
        {
            "policy_state": policy.state_dict(),
            "value_fn_state": value_fn.state_dict(),
            "value_optimizer_state": value_optimizer.state_dict(),
            "iteration": iteration,
            "total_steps": total_steps,
            "best_reward": best_reward,
            # Store architecture info so we can rebuild nets on load
            "obs_dim": list(policy.mean_net[0].weight.shape)[1],
            "act_dim": list(policy.mean_net[-1].weight.shape)[0],
            "hidden": list(policy.mean_net[0].weight.shape)[0],
            # Keep the full CLI args so that test script knows every setting
            "args": vars(args)
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    # Load a checkpoint dist, mapping tensors to *device
    return torch.load(path, map_location=device, weights_only=False)


# -----------------------------------------------------------------------------
# Trajectory Collection. This single path method from Section 5.

def collect_trajectories(env, policy, value_fn, batch_size, gamma, lam, device):
    # Roll out the policy, returning a batch dict and episode statistics
    observations, actions, rewards = [], [], []
    dones, log_probs, values = [], [], []
    episode_returns = []
    episode_lengths = []
    val_errors = []

    obs, _ = env.reset()
    ep_return = 0.0
    ep_length = 0
    steps = 0

    while steps < batch_size:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            action, lp = policy.act(obs_t)
            val = value_fn(obs_t)

        action_np = action.squeeze(0).cpu().numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        observations.append(obs)
        actions.append(action_np)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(lp.item())
        values.append(val.item())
        
        ep_return += reward
        ep_length += 1
        steps += 1
        obs = next_obs

        if "veloctiy_error_xy" in info:
            val_errors.append(info["velocity_error_xy"])

        if done:
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0.0
            obs, _ = env.reset()

    advantages, returns = comput_gae(
        np.array(rewards, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        gamma=gamma,
        lam=lam,
    )

    batch = {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(action, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "advantages": advantages,
        "returns": returns,
        "log_probs": np.array(log_probs, dtype=np.float32)        
    }

    stats = {
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "std_return": np.std(episode_returns) if episode_returns else 0.0,
        "min_return": np.min(episode_returns) if episode_returns else 0.0,
        "max_return": np.max(episode_returns) if episode_returns else 0.0,
        "mean_ep_len": np.mean(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(episode_lengths),
        "mean_vel_error": np.mean(val_errors) if val_errors else 0.0,
    }

    return batch, stats

# ------------------------------------------------------------------------------------
def train(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _init_wandb(args)

    # Enviornment
    render_mode = "human" if args.render else None
    env = VelocityAntEnv(render_mode=render_mode)

    obs_dim = env.observation_space.shape[0] # 27 base + 4 cmd = 31
    act_dim = env.action_space.shape[0]     # 8
    print(f"Obs dim: {obs_dim}  |  Act Dim: {act_dim}")

    # Networks
    policy = PolicyNetwork(obs_dim, act_dim, hidden_sizes=(args.hidden, args.hidden)).to(device)
    value_fn = ValueNetwork(obs_dim, hidden_sizes=(args.hidden, args.hidden)).to(device)

    agent = TRPOAgent(
        policy = policy,
        value_fn = value_fn,
        max_kl = args.max_kl,
        damping = args.damping,
        gamma =args.gamma,
        lam = args.lam,
        value_lr = args.value_lr,
        value_epochs = args.value_epochs,
        cg_iters = args.cg_iters,
        device = device
    )

    # Bookkeeping
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    best_reward = -np.inf
    start_iter = 1

    # Resume from checkpoint if requested
    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = load_checkpoint(args.resume, device=device)
        policy.load_state_dict(ckpt["policy_state"])
        value_fn.load_state_dict(ckpt["value_fn_state"])
        agent.value_optimizer.load_state_dict(ckpt["value_optimizer_state"])
        start_iter = ckpt["iteration"]+1
        total_steps = ckpt["total_steps"]
        best_reward = ckpt["best_rewards"]
        print(f"  -> continuing from iteration {start_iter}, "
               f"total_steps={total_steps:,}, best_reward={best_reward:.2f}")
        
    # Main Loop
    for it in range(start_iter, args.iterations + 1):
        t0 = time.time()

        batch, stats = collect_trajectories(
            env, policy, value_fn,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lam=args.lam,
            device=device
        )
        total_steps += len(batch["rewards"])

        info = agent.update(batch)
        dt = time.time() - t0
        mr = stats["mean_return"]

        # Logging
        wandb.log(
            {
                "iteration": it,
                "total_env_steps": total_steps,
                # Rewards
                "return/mean": mr,
                "return/std": stats["std_return"],
                "return/min": stats["min_return"],
                "return/max": stats["max_return"],
                # Velocity Tracking
                "velocity_error_xy": stats["mean_vel_error"],
                # TRPO Diagnostics
                "trpo/surrogate_loss": info["surrogate"],
                "trpo/kl_divergence": info["kl"],
                "trpo/value_loss": info["value_loss"],
                "trpo/step_accepted": int(info["accepted"]),
                # Episode Info
                "episode/mean_length": stats["mean_ep_len"],
                "episode/count": stats["num_episodes"],
                # Wallclock
                "timing/iter_seconds": dt,
            },
        )

        # Checkpoint
        if mr > best_reward:
            best_reward = mr
            save_checkpoint(
                save_dir / "best_model.pt",
                policy, value_fn, agent.value_optimizer,
                it, total_steps, best_reward, args,
            )

        if it % args.save_every == 0:
            save_checkpoint(
                save_dir / f"ckpt_{it}.pt",
                policy, value_fn, agent.value_optimizer,
                args.iterations, total_steps, best_reward, args,
            )

    env.close()

    wandb.finish()

    print(f"\nDone. Best mean return {best_reward:.2f}")
    print("Models saved in {save_dir.resolve()}")

    return policy, value_fn

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="TRPO for Velocity-Conditioned Ant")

    # algorithm
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=15000)
    p.add_argument("--max_kl", type=float, default=0.003)
    p.add_argument("--damping", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--value_lr", type=float, default=3e-5)
    p.add_argument("--value_epochs", type=int, default=5)
    p.add_argument("--cg_iters", type=int, default=10)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    # checkpointing
    p.add_argument("--save_dir", type=str, default="checkpoints")
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume training from")
    p.add_argument("--render", action="store_true")

    # Weights & Biases
    p.add_argument("--wandb", action="store_true",
                   help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", type=str, default="trpo-ant",
                   help="W&B project name")
    p.add_argument("--wandb_entity", type=str, default=None,
                   help="W&B entity (team or username)")
    p.add_argument("--wandb_name", type=str, default=None,
                   help="W&B run name (auto-generated if omitted)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
