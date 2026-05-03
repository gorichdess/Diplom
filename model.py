import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

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

        # Actor outputs mean in *pre-tanh* space
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), gain=0.01),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )

        # Learnable log_std (one per action dim)
        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        mu_raw = self.actor(x)
        value = self.critic(x)
        return mu_raw, value

    def _dist(self, mu_raw):
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        std = std.expand_as(mu_raw)
        return Normal(mu_raw, std)

    @staticmethod
    def _tanh_squash(action_raw):
        return torch.tanh(action_raw)

    @staticmethod
    def _tanh_logprob_correction(action_raw):
        a = torch.tanh(action_raw)
        return torch.log(1.0 - a * a + EPS).sum(-1)

    def get_action(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)
        obs = torch.clamp(obs, -50.0, 50.0)

        mu_raw, value = self.forward(obs)
        dist = self._dist(mu_raw)

        if deterministic:
            action_raw = mu_raw
        else:
            action_raw = dist.rsample()

        action = self._tanh_squash(action_raw)

        log_prob_raw = dist.log_prob(action_raw).sum(-1)
        log_prob = log_prob_raw - self._tanh_logprob_correction(action_raw)

        log_prob = torch.nan_to_num(log_prob, nan=-20.0, posinf=20.0, neginf=-20.0)
        log_prob = torch.clamp(log_prob, -20.0, 20.0)

        return action.detach().cpu().numpy()[0], float(value.item()), float(log_prob.item())

    def evaluate(self, obs, action):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0)
        obs = torch.clamp(obs, -50.0, 50.0)

        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = torch.clamp(action, -0.999, 0.999)

        mu_raw, value = self.forward(obs)
        dist = self._dist(mu_raw)

        # Invert tanh to get raw action
        action_raw = 0.5 * (torch.log1p(action + EPS) - torch.log1p(-action + EPS))

        log_prob_raw = dist.log_prob(action_raw).sum(-1)
        log_prob = log_prob_raw - self._tanh_logprob_correction(action_raw)

        entropy = dist.entropy().sum(-1)

        log_prob = torch.nan_to_num(log_prob, nan=-20.0, posinf=20.0, neginf=-20.0)
        entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)

        return log_prob, value.squeeze(-1), entropy