import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic


class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2

        self.lr = 1e-4
        self.epochs = 8
        self.batch_size = 256

        self.entropy_coef = 0.05     # Increased for more exploration
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        self.ac = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr, eps=1e-5)

        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def set_entropy_coef(self, coef):
        """Allow decay of entropy coefficient over time."""
        self.entropy_coef = coef

    def update(self):
        if len(self.memory) == 0:
            return 0.0

        states = torch.tensor(np.array([m[0] for m in self.memory]), dtype=torch.float32)
        actions = torch.tensor(np.array([m[1] for m in self.memory]), dtype=torch.float32)
        old_logp = torch.tensor(np.array([m[2] for m in self.memory]), dtype=torch.float32).detach()

        rewards = np.array([m[3] for m in self.memory], dtype=np.float32)
        dones = np.array([m[4] for m in self.memory], dtype=np.float32)
        values = np.array([m[5] for m in self.memory], dtype=np.float32)

        # Compute GAE
        values_ext = np.concatenate([values, np.array([0.0], dtype=np.float32)])
        returns = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1.0 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            returns[t] = gae + values_ext[t]

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)

        advantages = returns - values_t
        adv_std = advantages.std()
        if torch.isfinite(adv_std) and adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        else:
            advantages = advantages - advantages.mean()

        # Safety
        states = torch.nan_to_num(states, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        old_logp = torch.nan_to_num(old_logp, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
        returns = torch.nan_to_num(returns, nan=0.0)
        advantages = torch.nan_to_num(advantages, nan=0.0)

        n = states.shape[0]
        idx = torch.arange(n)

        total_loss = 0.0
        num_updates = 0

        for _ in range(self.epochs):
            perm = idx[torch.randperm(n)]
            for start in range(0, n, self.batch_size):
                mb = perm[start:start + self.batch_size]

                s = states[mb]
                a = actions[mb]
                oldlp = old_logp[mb]
                ret = returns[mb]
                adv = advantages[mb]
                v_old = values_t[mb]

                new_logp, v_pred, entropy = self.ac.evaluate(s, a)

                new_logp = torch.nan_to_num(new_logp, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                v_pred = torch.nan_to_num(v_pred, nan=0.0)
                entropy = torch.nan_to_num(entropy, nan=0.0)

                ratio = torch.exp(new_logp - oldlp)
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value clipping
                v_clipped = v_old + (v_pred - v_old).clamp(-0.2, 0.2)
                value_loss = 0.5 * torch.max((v_pred - ret) ** 2, (v_clipped - ret) ** 2).mean()

                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                num_updates += 1

        self.memory = []
        return total_loss / max(1, num_updates)