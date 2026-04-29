import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic


class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.lr = 3e-4
        self.epochs = 8
        self.batch_size = 256

        self.ac = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def update(self):
        if len(self.memory) == 0:
            return 0

        states = torch.tensor(np.array([m[0] for m in self.memory]), dtype=torch.float32)
        actions = torch.tensor(np.array([m[1] for m in self.memory]), dtype=torch.float32)
        old_logp = torch.tensor(np.array([m[2] for m in self.memory]), dtype=torch.float32)

        rewards = [m[3] for m in self.memory]
        dones = [m[4] for m in self.memory]
        values = [m[5] for m in self.memory]

        values = np.array(values + [0])

        # -------- GAE --------
        gae = 0
        returns = []

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])

        returns = torch.tensor(returns, dtype=torch.float32)

        values = torch.tensor(values[:-1], dtype=torch.float32)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0

        for _ in range(self.epochs):

            new_logp, v_pred, entropy = self.ac.evaluate(states, actions)

            ratio = torch.exp(new_logp - old_logp)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            # clipped value loss (VERY IMPORTANT FIX)
            v_clipped = values + (v_pred - values).clamp(-0.2, 0.2)
            value_loss = 0.5 * torch.max(
                (v_pred - returns) ** 2,
                (v_clipped - returns) ** 2
            ).mean()

            entropy_bonus = entropy.mean()

            loss = policy_loss + value_loss - 0.01 * entropy_bonus

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

        self.memory = []
        return total_loss / self.epochs