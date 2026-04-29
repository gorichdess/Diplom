import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()

        def layer_init(layer):
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
            return layer

        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim)),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1)),
        )

        self.log_std = nn.Parameter(torch.ones(action_dim) * -0.5)

    def forward(self, x):
        mu = self.actor(x)
        mu = torch.tanh(mu)
        value = self.critic(x)
        return mu, value

    def get_action(self, obs, deterministic=False):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        mu, value = self.forward(obs)

        # clamp exploration
        log_std = torch.clamp(self.log_std, -2, 1)
        std = log_std.exp()

        dist = Normal(mu, std)

        action = mu if deterministic else dist.sample()

        log_prob = dist.log_prob(action).sum(-1)

        return (
            action.detach().cpu().numpy()[0],
            value.item(),
            log_prob.item()
        )

    def evaluate(self, obs, action):
        mu, value = self.forward(obs)

        log_std = torch.clamp(self.log_std, -2, 1)
        dist = Normal(mu, log_std.exp())

        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)

        return logp, value.squeeze(-1), entropy