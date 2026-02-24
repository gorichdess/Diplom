import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        self.common = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor, mu for mean action output
        self.actor_mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  #Diagonal action space [-1, 1]
        )
        
        # Critic value function output
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Standard deviation for action distribution
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.common(x)
        mu = self.actor_mu(x)
        value = self.critic(x)
        return mu, value

    def get_action(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0)
        
        with torch.no_grad():
            mu, value = self.forward(obs)
            
            if deterministic:
                return mu.numpy()[0], value.item(), 0.0
            
            std = self.log_std.exp()
            dist = Normal(mu, std)
            
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.numpy()[0], value.item(), log_prob.item()
    
    def evaluate(self, states, actions):
        mu, values = self.forward(states)
        std = self.log_std.exp()
        dist = Normal(mu, std)
        
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, values.squeeze(), entropy