"""
Policy and Value Networks for TRPO

Policy: Diagnoal-Gaussian π(a|s) = N(μ_θ(s), diag(σ²))
        μ is the neural network, log(σ) is the free parameter vector.

Value: V_ϕ(s) - A seperate network trained by regression on returns
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

def build_mlp(in_dim, out_dim, hidden_size, activation=nn.Tanh):
    """
    Build the NN layer
    """

    layers = []
    prev = in_dim
    
    for h in hidden_size:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))

    return nn.Sequential(*layers)


class PolicyNetwork(nn.Module):
    """
    Diagonal Gaussian Policy with state independent log-std
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.mean_net = build_mlp(obs_dim, act_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

        # Small initial output -> Action start near zero
        with torch.no_grad():
            self.mean_net[-1].weight.data.mul_(0.01)
            self.mean_net[-1].bias.zero_()

    LOG_STD_MIN = -2.0
    LOG_STD_MAX = 0.5
    def forward(self, obs):
        # Return the normal distribution over actions
        mean = self.mean_net(obs)
        log_std = self.log_std.clamp(self.LOG_STD_MAX, self.LOG_STD_MIN)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std)
    
    @torch.no_grad()
    def act(self, obs):
        # Sample an action and return (action, log_prob)
        dist = self.forward(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)

        return action, log_prob
    
    def evaluate(self, obs, actions):
        # Compute log probabilites and entropy for given (obs, action) pairs
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)

        return log_prob, entropy
    
    def get_distribution_params(self, obs):
        # Reutrn (mean, std) detached - Used as the 'old' distribution
        dist = self.forward(obs)
        
        return dist.loc.detach().clone(), dist.scale.log().detach().clone()
    
class ValueNetwork(nn.Module):
    """
    State Value Network
    """

    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = build_mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)
