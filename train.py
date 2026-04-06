"""
Training script for TRPO on the velocity conditioned Ant Env
Collects experiences from multple parallel CPU environments via AsyncVectorEnv,
then performs a single TRPO update on a merged batch.
Domain randomization is applied independently in each sub-environment

"""

import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym

import numpy as np
import torch

from enviorment_wrapper import VelocityAntEnv
from models import PolicyNetwork, ValueNetwork
from trpo import TRPOAgent, RunningMeanStd

# wandb is optional
wandb = None

def _init_wandb(args):
    """ Initialise Weights and Biases"""
    global wandb
    import wandb as _wandb
    wandb = _wandb

    wandb.init(
        project="trpo_env_ant_model",
        config=vars(args),
    )
    wandb.define_metric("iteration")
    wandb.define_metric("*", step_metric="iteration")
    return wandb

# ---------------------------------------------------------------------------------------
# Make environments

def make_env(**kwargs):
    return VelocityAntEnv(**kwargs)

def _env_kwards(args, *, render_mode=None, seed_offset=0):
    # Build the kwargs dict for a sigle VelocityAntEnv instance
    return {
        "render_mode": render_mode,
        "cmd_vx_range": (args.cmd_vx_min, args.cmd_vx_max),
        "cmd_vy_range": (args.cmd_vy_min, args.cmd_vy_max),
        "cmd_yaw_rate_range": (args.cmd_yaw_rate_min, args.cmd_yaw_rate_max),
        "randomize_mass": args.randomize_mass,
        "mass_scale_range": (args.mass_min_scale, args.mass_max_scale),
        "randomize_friction": args.randomize_friction,
        "friction_scale_range": (args.friction_min_scale, args.friction_max_scale),
        "randomize_action_delay": args.randomize_action_delay,
        "action_delay_range": (args.action_delay_min, args.action_delay_max),
        "randomize_obs_delay": args.randomize_obs_delay,
        "obs_delay_range": (args.obs_delay_min, args.obs_delay_max),
        "randomization_seed": args.seed + seed_offset,
    }

# ---------------------------------------------------------------------------------------
# Save / Load Checkpoints
def save_checkpoint(path, policy, value_fn, value_optimizer,
                    iteration, total_steps, best_reward, args,
                    obs_rms=None, ret_rms=None):
    # Persist everything needed to fully resume training later

    ckpt = {
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
    }
    if obs_rms is not None:
        ckpt["obs_rms"] = {"mean": obs_rms.mean, "var": obs_rms.var, "count": obs_rms.count}
    if ret_rms is not None:
        ckpt["ret_rms"] = {"mean": ret_rms.mean, "var": ret_rms.var, "count": ret_rms.count}
    torch.save(ckpt, path)

    print("... Saving Model ...")


def load_checkpoint(path, device="cpu"):
    # Load a checkpoint dist, mapping tensors to *device
    return torch.load(path, map_location=device, weights_only=False)


# --------------------------------------------------------------------------------
"""
    Generalized Advantage Estimation

    Â+t = ∑_{l=0}^{∞} (γλ)^l · δ_{t+l}
    where δ_t = t_t + γ V(s_{t+1} - V(s_t))
"""

def comput_vectorzed_gae(rewards, values, dones, last_values, gamma=0.99, lam=0.97):
    # GAE for batched rollouts. All arrats have shape [T, N]

    T = rewards.shape[0]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros(rewards.shape[1], dtype=np.float32)

    for t in reversed(range(T)):
        next_val = last_values if t == T - 1 else values[t + 1]
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * non_terminal * values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# -----------------------------------------------------------------------------
# Trajectory Collection. This single path method from Section 5.

def collect_trajectories(env, policy, value_fn, batch_size, gamma, lam, device,
                    obs_rms=None, ret_rms=None):
    # Roll out the policy, returning a batch dict and episode statistics
    # Return a single merged batch suitable for TRPO update

    num_envs = env.num_envs
    obs_dim = env.single_observation_space.shape[0]
    act_dim = env.single_action_space.shape[0]

    observations, actions, rewards_list = [], [], []
    dones_list, log_probs_list, values_list = [], [], []
    raw_obs_buffer = []

    episode_returns = []
    episode_lengths = []
    vel_errors = []
    yaw_rate_errors = []
    mass_scales = []
    friction_scales = []
    action_delays = []

    obs_raw, _ = env.reset()
    ep_returns = np.zeros(num_envs, dtype=np.float64)
    ep_lengths = np.zeros(num_envs, dtype=np.int32)
    total_steps = 0


    while total_steps < batch_size:
        raw_obs_buffer.append(obs_raw.copy())

        if obs_rms is not None:
            obs = obs_rms.normalize(obs_raw).astype(np.float32)
        else:
            obs = obs_raw.astype(np.float32)

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            action, lp = policy.act(obs_t)
            val = value_fn(obs_t)

        action_np = action.cpu().numpy()
        action_clipped = np.clip(action_np, env.single_action_space.low, env.single_action_space.high)

        next_obs_raw, reward, terminated, truncated, infos = env.step(action_clipped)
        done = np.logical_or(terminated,truncated)

        observations.append(obs.copy())
        actions.append(action_np.copy()) # Store unclipped so log probs stay consistent
        rewards_list.append(reward.astype(np.float32))
        dones_list.append(done.astype(np.float32))
        log_probs_list.append(lp.cpu().numpy().astype(np.float32))
        values_list.append(val.cpu().numpy().astype(np.float32))
        
        ep_returns += reward
        ep_lengths += 1
        total_steps += num_envs
        

        if "velocity_error" in infos:
            vel_errors.extend(np.asarray(infos["velocity_error"]).ravel().tolist())
        if "yaw_rate_error" in infos:
            yaw_rate_errors.extend(np.asarray(infos["yaw_rate_error"]).ravel().tolist())
        if "mass_scale" in infos:
            mass_scales.extend(np.asarray(infos["mass_scale"]).ravel().tolist())
        if "friction_scale" in infos:
            friction_scales.extend(np.asarray(infos["friction_scale"]).ravel().tolist())
        if "action_delay_steps" in infos:
            action_delays.extend(np.asarray(infos["action_delay_steps"]).ravel().tolist())

        for idx in np.nonzero(done)[0]:
            episode_returns.append(float(ep_returns[idx]))
            episode_lengths.append(int(ep_lengths[idx]))
            ep_returns[idx] = 0.0
            ep_lengths[idx] = 0
        
        obs_raw = next_obs_raw

    # Update obs normalizer with all raaw observations seen this rollout
    if obs_rms is not None:
        obs_rms.update(np.asarray(raw_obs_buffer, dtype=np.float64).reshape(-1, obs_dim))

    # Bootstrap value for the last observation
    if obs_rms is not None:
        last_obs = obs_rms.normalize(obs_raw).astype(np.float32)
    else:
        last_obs = obs_raw.astype(np.float32)
    with torch.no_grad():
        last_values = value_fn(
            torch.as_tensor(last_obs, dtype=torch.float32, device=device)
        ).cpu().numpy().astype(np.float32)

    rewards_arr = np.asarray(rewards_list, dtype=np.float32)    # [T, N]
    values_arr = np.asarray(values_list, dtype=np.float32)
    dones_arr = np.asarray(dones_list, dtype=np.float32)


    advantages, returns = comput_vectorzed_gae(
        rewards_arr,
        values_arr,
        dones_arr,
        last_values,
        gamma=gamma,
        lam=lam,
    )

    # Flatten [T, N, ...] -> [T*N, ....]
    flat = lambda a, cols: np.asarray(a, dtype=np.float32).reshape(-1, cols) if cols > 0 else np.asarray(a, dtype=np.float32).reshape(-1)

    flat_returns = flat(returns, 0)
    if ret_rms is not None:
        ret_rms.update(flat_returns)
        flat_returns = ret_rms.normalize(flat_returns).astype(np.float32)

    batch = {
        "observations": flat(observations, obs_dim),
        "actions": flat(actions, act_dim),
        "rewards": flat(rewards_arr, 0),
        "advantages": flat(advantages, 0),
        "returns": flat_returns,
        "log_probs": flat(log_probs_list, 0)        
    }

    stats = {
        "mean_return": np.mean(episode_returns) if episode_returns else 0.0,
        "std_return": np.std(episode_returns) if episode_returns else 0.0,
        "min_return": np.min(episode_returns) if episode_returns else 0.0,
        "max_return": np.max(episode_returns) if episode_returns else 0.0,
        "mean_ep_len": np.mean(episode_lengths) if episode_lengths else 0.0,
        "num_episodes": len(episode_lengths),
        "mean_vel_error": np.mean(vel_errors) if vel_errors else 0.0,
        "mean_yaw_rate_error": np.mean(yaw_rate_errors) if yaw_rate_errors else 0.0,
        "mean_mass_scale": np.mean(mass_scales) if mass_scales else 0.0,
        "mean_friction_scale": np.mean(friction_scales) if friction_scales else 0.0,
        "mean_action_delay": np.mean(action_delays) if action_delays else 0.0,
    }

    return batch, stats

# ------------------------------------------------------------------------------------
# Training Loop
def train(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    _init_wandb(args)

    # Vectorized Enviornment
    render_mode = "human" if args.render else None
    env_fns = []
    for i in range(args.num_envs):
        kw = _env_kwards(
            args,
            render_mode=render_mode if i == 0 else None,
            seed_offset=i
        )
        env_fns.append(partial(make_env, **kw))

    env = gym.vector.AsyncVectorEnv(
        env_fns,
        shared_memory=False,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    print(f"Parallel Envs: {env.num_envs}")

    obs_dim = env.single_observation_space.shape[0] # 27 base
    act_dim = env.single_action_space.shape[0]     # 8
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

    obs_rms = RunningMeanStd(shape=(obs_dim,))
    ret_rms = RunningMeanStd(shape=())

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
        if "obs_rms" in ckpt:
            obs_rms.mean = ckpt["obs_rms"]["mean"]
            obs_rms.var = ckpt["obs_rms"]["var"]
            obs_rms.count = ckpt["obs_rms"]["count"]
        if "ret_rms" in ckpt:
            ret_rms.mean = ckpt["ret_rms"]["mean"]
            ret_rms.var = ckpt["ret_rms"]["var"]
            ret_rms.count = ckpt["ret_rms"]["count"]
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
            device=device,
            obs_rms=obs_rms,
            ret_rms=ret_rms
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
                "velocity_error": stats["mean_vel_error"],
                "yaw_rate_error": stats["mean_yaw_rate_error"],
                # TRPO Diagnostics
                "trpo/surrogate_loss": info["surrogate"],
                "trpo/kl_divergence": info["kl"],
                "trpo/value_loss": info["value_loss"],
                "trpo/step_accepted": int(info["accepted"]),
                # Episode Info
                "episode/mean_length": stats["mean_ep_len"],
                "episode/count": stats["num_episodes"],
                # Domain Info
                "domain/mass_scale": stats["mean_mass_scale"],
                "domain/friction_scale": stats["mean_friction_scale"],
                "domain/action_delay_scale": stats["mean_action_delay"],
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
                obs_rms=obs_rms,
                ret_rms=ret_rms
            )

        if it % args.save_every == 0:
            save_checkpoint(
                save_dir / f"ckpt_{it}.pt",
                policy, value_fn, agent.value_optimizer,
                args.iterations, total_steps, best_reward, args,
                obs_rms=obs_rms,
                ret_rms=ret_rms
            )

    env.close()

    wandb.finish()

    print(f"\nDone. Best mean return {best_reward:.2f}")
    print("Models saved in {save_dir.resolve()}")

    return policy, value_fn

# CLI
CONFIG = SimpleNamespace(
    # Algorithm
    iterations = 3000,
    batch_size = 24576,
    max_kl = 0.03,
    damping = 0.1,
    gamma = 0.99,
    lam = 0.97,
    value_lr = 1e-3,
    value_epochs = 5,
    cg_iters = 10,
    hidden=256,
    seed=42,
    num_envs = 10,

    # Checkpoints
    save_dir = "checkpoints",
    save_every = 50,
    resume = None,
    render = False,

    # Command Range
    cmd_vx_min = 0.0,
    cmd_vx_max = 1.5,
    cmd_vy_min = 0.0,
    cmd_vy_max = 1.5,
    cmd_yaw_rate_min = -0.5,
    cmd_yaw_rate_max = 0.5,

    # Domain Randomization
    randomize_mass = True,
    mass_min_scale = 0.9,
    mass_max_scale = 1.1,
    randomize_friction = True,
    friction_min_scale = 0.7,
    friction_max_scale = 1.3,
    randomize_action_delay = False,
    action_delay_min  = 0,
    action_delay_max = 2,
    randomize_obs_delay = False,
    obs_delay_min = 0,
    obs_delay_max = 0,

    # Weights and Biases
    wandb = True,
    wandb_project = "trpo_env_ant_model",
    wandb_entity = None,
    wandb_name  = None,
)

if __name__ == "__main__":
    train(CONFIG)
