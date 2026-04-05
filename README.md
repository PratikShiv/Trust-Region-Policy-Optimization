# Trust-Region Policy Optimization for Velocity-Conditioned Ant

This repository contains an implementation of Trust Region Policy Optimization (TRPO) applied to a custom velocity-conditioned Ant locomotion task.
The implementation is based on
Schulman et al., "Trust Region Policy Optimization," ICML 2015.

## Project Summary

- A custom Gymnasium wrapper around `Ant-v5` (`VelocityAntEnv`) extends the observation with commanded velocity inputs and replaces the default reward with a multi-component natural gait objective.
- The policy model is a diagonal Gaussian actor with state-independent log standard deviation; the value model is a separate neural network baseline.
- Training uses a TRPO update loop with conjugate gradient, Fisher-vector product, backtracking line search, and generalized advantage estimation.
- Checkpoint artifacts, evaluation behavior, and training diagnostics are saved under `checkpoints/` and `results/`.

## Repository Structure

- `enviorment_wrapper.py` - custom Ant environment wrapper that:
  - appends a velocity command vector `[Vx, Vy, Vz, yaw_rate]` to observations,
  - resamples command velocity periodically,
  - computes a reward that encourages command tracking, stable posture, energy efficiency, smooth joint motion, gait symmetry, and an alive bonus.
- `models.py` - policy and value network architectures:
  - `PolicyNetwork`: Gaussian policy over actions,
  - `ValueNetwork`: scalar state-value estimator.
- `trpo.py` - TRPO algorithm implementation:
  - conjugate gradient solver,
  - fisher vector product via KL Hessian-vector product,
  - surrogate objective and KL regularization,
  - generalized advantage estimation.
- `train.py` - training pipeline, trajectory collection, checkpointing, and W&B logging.
- `test.py` - evaluation script for rendering and offline policy playback.
- `checkpoints/` - saved model weights and training checkpoints.
- `results/` - training diagnostics and behavior artifacts.

## Current Results

### Checkpoint Behavior
- `results/checkpoint_1_output.gif` shows the Ant behavior at checkpoint 1 under velocity command conditions.

![Checkpoint 1 Behavior](results/checkpoint_1_output.gif)


### Training Diagnostics
- `results/reward_mean_during_training.png` shows mean episode return over training iterations.
- `results/velocity_error_during_training.png` shows the velocity tracking error over iterations.
- `results/trpo_kl_divergence_during_training.png` shows the KL divergence during policy updates.

### Observations from current artifacts
- Mean return is around `~2200` and shows large episode-to-episode noise.
- Velocity tracking error is around `0.75 - 0.85` and trends upward later in training.
- KL divergence remains stable near the chosen trust-region radius, indicating the update constraint is active.

## Usage

### Training

```bash
python3 train.py --wandb
python3 train.py --render
python3 train.py --resume checkpoints/ckpt_100.pt
```

### Evaluation

```bash
python3 test.py --checkpoint checkpoints/best_model.pt
python3 test.py --checkpoint checkpoints/best_model.pt --cmd_vx 1.5 --cmd_vy 0.0
python3 test.py --checkpoint checkpoints/best_model.pt --stochastic
```

> Note: Current training code initializes Weights & Biases unconditionally, so `wandb` must be installed even if `--wandb` is not provided.

## Limitations

The current repository is a strong foundation, but the implementation has some important gaps that should be addressed before this work can be considered fully reliable
