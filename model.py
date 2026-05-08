import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np   # for np.log(2.) in the correction

LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
EPS = 1e-6

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        def layer_init(layer, gain=1.0):
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)
            return layer

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, action_dim), gain=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1)),
        )

        # Learnable log_std (one per action dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mu = self.actor(x)
        value = self.critic(x)
        return mu, value

    def _dist(self, mu):
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        std = std.expand_as(mu)
        return Normal(mu, std)

    def get_action(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

        mu, value = self.forward(obs)
        dist = self._dist(mu)

        if deterministic:
            raw_action = mu  
            action = torch.tanh(mu)
        else:
            raw_action = dist.rsample()                # unbounded sample
            action = torch.tanh(raw_action)            # squashed to [-1, 1]

        # Log‑prob using the change‑of‑variables formula
        # log_prob = dist.log_prob(raw_action) - sum(log(1 - tanh(raw_action)^2) + EPS)
        # where the second term is: 2 * (log(2) - raw_action - softplus(-2*raw_action))
        log_prob = dist.log_prob(raw_action).sum(-1)
        correction = (2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))).sum(-1)
        log_prob = log_prob - correction

        log_prob = torch.nan_to_num(log_prob, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # Return squashed action (not raw)
        return action.detach().cpu().numpy()[0], float(value.item()), float(log_prob.item())

    def evaluate(self, obs, action):
        # Sanitize observation
        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

        mu, value = self.forward(obs)
        dist = self._dist(mu)

        # Stability epsilon: prevents the log(0) issue at the boundaries of tanh
        eps = 1e-5 
        action_clamped = torch.clamp(action, -1.0 + eps, 1.0 - eps)
        
        # Recover raw Gaussian action for log_prob calculation
        raw_action = 0.5 * (torch.log(1.0 + action_clamped) - torch.log(1.0 - action_clamped))

        # Log-prob with the tanh squashing correction
        log_prob = dist.log_prob(raw_action).sum(-1)
        # This is the change of variables correction: log(1 - tanh^2(x))
        correction = torch.sum(torch.log(1.0 - action_clamped**2 + 1e-6), dim=-1)
        log_prob = log_prob - correction

        entropy = dist.entropy().sum(-1)

        return log_prob, value.squeeze(-1), entropy