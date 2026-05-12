import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic

class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.15

        self.lr = 1e-4
        self.epochs = 12                
        self.batch_size = 1024          

        self.entropy_coef = 0.02
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        self.reward_scale = 0.1        

        self.ac = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        self.memory = []

    def store(self, transition):
        self.memory.append(transition)

    def set_entropy_coef(self, coef):
        self.entropy_coef = float(coef)

    def set_lr(self, lr):
        self.lr = float(lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    @torch.no_grad()
    def value_of(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        v = self.ac.critic(obs).squeeze(-1)
        return float(v.item())

    def update(self, last_value=0.0):
        if len(self.memory) == 0:
            return 0.0

        # ---- unpack ----
        states  = torch.tensor(np.array([m[0] for m in self.memory]), dtype=torch.float32)
        actions = torch.tensor(np.array([m[1] for m in self.memory]), dtype=torch.float32)  # raw actions
        old_logp= torch.tensor(np.array([m[2] for m in self.memory]), dtype=torch.float32).detach()

        rewards = np.array([m[3] for m in self.memory], dtype=np.float32) * self.reward_scale
        dones   = np.array([m[4] for m in self.memory], dtype=np.float32)
        values  = np.array([m[5] for m in self.memory], dtype=np.float32) * self.reward_scale  # also scale value estimates

        # safety
        states   = torch.nan_to_num(states,   nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        actions  = torch.nan_to_num(actions,  nan=0.0, posinf=10.0, neginf=-10.0)
        old_logp = torch.nan_to_num(old_logp, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # ---- GAE ----
        values_ext = np.append(values, float(last_value) * self.reward_scale)
        returns    = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t+1] * (1.0 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t]    = gae + values_ext[t]

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)

        # normalize advantages
        advantages_t = returns - values_t
        adv_std = advantages_t.std()
        if torch.isfinite(adv_std) and adv_std > 1e-6:
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-8)
        else:
            advantages_t = advantages_t - advantages_t.mean()
        advantages_t = torch.nan_to_num(advantages_t, nan=0.0).clamp(-5.0, 5.0)

        # clamp returns for value loss stability
        returns = torch.clamp(returns, -10.0, 10.0)

        n = states.shape[0]
        idx = torch.arange(n)

        total_loss = 0.0
        num_updates = 0

        for _ in range(self.epochs):
            perm = idx[torch.randperm(n)]
            for start in range(0, n, self.batch_size):
                mb = perm[start:start+self.batch_size]

                s = states[mb]
                a = actions[mb]          # raw action
                oldlp = old_logp[mb]
                ret = returns[mb]
                adv = advantages_t[mb]
                v_old = values_t[mb]

                # evaluate now expects raw action
                new_logp, v_pred, entropy = self.ac.evaluate(s, a)

                new_logp = torch.nan_to_num(new_logp, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                v_pred   = torch.nan_to_num(v_pred,   nan=0.0)
                entropy  = torch.nan_to_num(entropy,  nan=0.0)

                # PPO ratio
                ratio = torch.exp(torch.clamp(new_logp - oldlp, -10, 10))
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)

                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                v_clipped = v_old + (v_pred - v_old).clamp(-0.2, 0.2)
                vf1 = (v_pred - ret) ** 2
                vf2 = (v_clipped - ret) ** 2
                value_loss = 0.5 * torch.max(vf1, vf2).mean()

                entropy_bonus = entropy.mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                num_updates += 1

        self.memory = []
        return total_loss / max(1, num_updates)