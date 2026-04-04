"""
Trust Region Policy Optimization (Schulman et al., 2015)

Algorithm outline from paper (Section 4, Practical Algorithm):
    1. Collect trajectories under the current policy π_θ_old

    2. Esimate advantages Â_t via GAE(γ, λ)

    3. Compute the polict gradient g = ∇_θ L(θ)|_{θ_old}

        Where L(θ) = E_t[ π_θ(a_t|s_t) / π_θ_old(a_t|s_t) · Â_t ]

    4. Solve for the 'natural gradient' direction s = F⁻¹ g
        via the conjugate-gradient algorith, where F is the Fisher
        information matrix (~ Hessian of the KL divergence)

    5. Compute the maximum step size   β = √(2δ / sᵀFs)
        where δ is the trust region radius (max allowed KL)

    6. backtracking line search:
        θ_new = θ_old + α · β · s
        Accept the largest α ∈ {1, p, p², ...} such that
            KL(π_θ_old | π_θ_new) ≤ δ  AND  L(θ_new) > L(θ_old)

    7. Fit V_ϕ to the empirical returns by regression
"""

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Utility Header

def flat_params(model):
    # Concatenate all model parameters into a single flat vector
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])

def set_flat_params(model, flat):
    # Write a flat vector back into model parameters
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].reshape(p.shape))
        idx += n

def flat_grad(grads):
    # Concatenate a tuple of gradient tensors into 1 flat vector
    return torch.cat([g.contiguous().reshape(-1) for g in grads])


# ------------------------------------------------------------------------------
"""
    Conjugate Gradient (Section 4, Appendix C)

    Solves Fx = g without ever forming F explicitly
    Only required a function fvp(v) that return the matrix-vector
    product Fv (the *Fisher-vector product*)
"""

def conjugate_gradient(fvp_fn, b, max_iter=10, residual_tol=1e-10):
    # Solves Fx=b via CG, using fvp_fn to compute F·v

    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)

    for _ in range(max_iter):
        Fp = fvp_fn(p)
        pFp = p.dot(Fp)
        alpha = rdotr / (pFp + 1e-8)
        x += alpha * p
        r -= alpha * Fp
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        
        p = r + (new_rdotr / (rdotr + 1e-8)) * p
        rdotr = new_rdotr

    return x


# --------------------------------------------------------------------------------
"""
    Generalized Advantage Estimation

    Â+t = ∑_{l=0}^{∞} (γλ)^l · δ_{t+l}
    where δ_t = t_t + γ V(s_{t+1} - V(s_t))
"""

def comput_gae(rewards, values, dones, gamma=0.99, lam=0.97):
    # Return the (advantage, returns) as numpy arrays
    n  = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    last_gae = 0

    for t in reversed(range(n)):
        next_value = 0.0 if (t == n - 1 or dones[t]) else values[t+1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * lam * next_non_terminal * values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ----------------------------------------------------------------------------------

class TRPOAgent:
    """
    Full TRPO update loop based on the paper.

    Parameters
    ----------
    policy          : PolicyNetwork     - Gaussian Policy to update
    value_fn        : ValueNetwork      - State-value baseline
    max_kl          : float             - δ, the trust region radius
    damping         : float             - Tikhonov damping added to F for stability
    gammma, lam     : float             - discount / GAE parameter
    value_lr        : float             - Adam LR for the value function
    value_epochs    : int               - SGD passes over the batch for the value fn
    cg_iters        : int               - Max CG iterations
    ls_steps        : int               - Max Backtracking line search steps
    ls_decay        : float             - Geometric decay factor for the line search
    """

    def __init__(
        self,
        policy,
        value_fn,
        max_kl = 0.01,
        damping = 0.1,
        gamma = 0.99,
        lam = 0.97,
        value_lr = 1e-3,
        value_epochs = 5,
        cg_iters = 10,
        ls_steps = 10,
        ls_decay = 10,
        device = "cpu",
    ):
        self.policy = policy
        self.value_fn = value_fn
        self.max_kl = max_kl
        self.damping = damping
        self.gamma = gamma
        self.lam = lam
        self.cg_iters = cg_iters
        self.ls_steps = ls_steps
        self.ls_decay = ls_decay
        self.device = device

        self.value_optimizer = torch.optim.Adam(value_fn.parameters(), lr=value_lr)
        self.value_epochs = value_epochs

    # Surrogate Objective L(θ)
    def _surrogate_loss(self, obs, actions, advantages, old_log_probs):
        # L(θ) = E_t[ π_θ(a|s) / π_θ_old(a|s) · Â_t ]
        new_log_probs, _ = self.policy.evaluate(obs, actions)
        ratio = (new_log_probs - old_log_probs).exp()
        return (ratio * advantages).mean()
    
    # KL Divergence for diagonal Gaussian
    def _mean_kl(self, obs, old_means, old_log_stds):
        """
        KL( π_old | π_new ) averaged over the batch
            KL = ∑_i [ log(σ_new_i/σ_old_i)
                        + (σ_old_i² + (μ_old_i - μ_new_i)²) / (2 σ_new_i²)
                        - 0.5]
        """
        dist = self.policy(obs)
        new_means = dist.loc
        new_log_stds = dist.scale.log()

        old_var = (2.0 * old_log_stds).exp()
        new_var = (2.0 * new_log_stds).exp()

        kl = (
            new_log_stds - old_log_stds
            + (old_var + (old_means - new_means).pow(2)) / (2.0 * new_var)
            - 0.5
        )
        return kl.sum(dim=-1).mean()
        
    # Fisher Vetor Product Fv
    # Two back-props through the KL graph: First to get ∇KL
    # Second to get ∇(∇KL · v) which equals Fv + damping·v
    def _fisher_vector_product(self, obs, old_means, old_log_stds, v):
        kl = self._mean_kl(obs, old_means, old_log_stds)
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_kl_grad = flat_grad(grads)

        kl_dot_v = flat_kl_grad.dot(v)
        grad2 = torch.autograd.grad(kl_dot_v, self.policy.parameters())
        
        return flat_grad(grad2) + self.damping*v
    
    # Single TRPO Update
    def update(self, batch):
        # Perform one full TRPO iteration (Step 3-7)
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], dtype=torch.float32, device=self.device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Snapshot the old distribution
        old_means, old_log_stds = self.policy.get_distribution_params(obs)

        # Step 3: Polict Gradient g
        loss = self._surrogate_loss(obs, actions, advantages, old_log_probs)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        g = flat_grad(grads)
    
        if g.norm() < 1e-8:
            self._update_value(obs, returns)
            return {"surrogate": loss.item(),
                    "kl": 0.0,
                    "value_loss": 0.0,
                    "accepted": False}
        
        # Step 4: Natural Gradient Direction s = F⁻¹g (via CG)
        def fvp(v):
            return self._fisher_vector_product(obs, old_means, old_log_stds, v)
        
        step_dir = conjugate_gradient(fvp, g, max_iter=self.cg_iters)

        # Step 5: Max Step Size   β = √(2δ / sᵀFs)
        sFs = step_dir.dot(fvp(step_dir))
        beta = torch.sqrt(2.0 * self.max_kl / (sFs + 1e-8))
        full_step = beta * step_dir

        # Step 6: Backtracking line search
        old_params = flat_params(self.policy)
        expected_improve = g.dot(full_step)
        accepted = False

        for i in range(self.ls_steps):
            alpha = self.ls_decay **i
            set_flat_params(self.policy, old_params + alpha * full_step)

            with torch.no_grad():
                new_loss = self._surrogate_loss(obs, actions, advantages, old_log_probs)
                new_kl = self._mean_kl(obs, old_means, old_log_stds)

            if new_kl <= self.max_kl and (new_loss - loss) > 0:
                accepted = True
                break

        if not accepted:
            set_flat_params(self.policy, old_params)

        # Step 7: Update value function
        vloss = self._update_value(obs, returns)

        final_kl = self._mean_kl(obs, old_means, old_log_stds).item()

        return {
            "surrogate": loss.item(),
            "kl": final_kl,
            "value_loss": vloss,
            "accepted": accepted,
        }
    
    def _update_value(self, obs, returns):
        total_loss = 0.0
        
        for _ in range(self.value_epochs):
            pred = self.value_fn(obs)
            loss = (returns - pred).pow(2).mean()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
            total_loss += loss.item()
        
        return total_loss / self.value_epochs