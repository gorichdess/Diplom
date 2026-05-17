import torch
import torch.optim as optim
import numpy as np
from model import ActorCritic

# Proximal Policy Optimization (PPO) agent
# implementing clipped policy-gradient training
# for continuous robot control.
class PPOAgent:
    def __init__(self, obs_dim, action_dim):
        # PPO and GAE hyperparameters:
        # gamma - reward discount factor
        # lam   - Generalized Advantage Estimation decay
        # clip  - PPO policy update clipping threshold
        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.15

        # Optimization parameters for PPO training.
        self.lr = 1e-4
        self.epochs = 12                
        self.batch_size = 1024          

        # Entropy regularization coefficient encouraging exploration.
        self.entropy_coef = 0.02
        self.value_coef = 0.5
        self.max_grad_norm = 0.5

        # Reward normalization factor improving numerical stability
        # during value estimation and policy optimization.
        self.reward_scale = 0.1        

        self.ac = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.lr)

        # Temporary rollout buffer storing transitions
        # collected from environment interaction.
        self.memory = []

    # Store one environment transition in PPO memory buffer.
    def store(self, transition):
        self.memory.append(transition)

    # Dynamically adjust exploration strength during training.
    def set_entropy_coef(self, coef):
        self.entropy_coef = float(coef)

    # Update optimizer learning rate during curriculum training.
    def set_lr(self, lr):
        self.lr = float(lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    # Disable gradient tracking during value inference.
    @torch.no_grad()
    # Estimate critic value for bootstrap return calculation.
    def value_of(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        v = self.ac.critic(obs).squeeze(-1)
        return float(v.item())

    # Perform PPO policy and value network optimization
    # using collected rollout trajectories.
    def update(self, last_value=0.0):
        # Perform PPO policy and value network optimization
        # using collected rollout trajectories.
        if len(self.memory) == 0:
            return 0.0

        # Convert rollout buffer into batched tensors
        # for PPO optimization.
        states  = torch.tensor(np.array([m[0] for m in self.memory]), dtype=torch.float32)
        actions = torch.tensor(np.array([m[1] for m in self.memory]), dtype=torch.float32)  # raw actions
        old_logp= torch.tensor(np.array([m[2] for m in self.memory]), dtype=torch.float32).detach()

        # Scale rewards to stabilize critic targets
        # and reduce gradient variance.
        rewards = np.array([m[3] for m in self.memory], dtype=np.float32) * self.reward_scale
        dones   = np.array([m[4] for m in self.memory], dtype=np.float32)
        values  = np.array([m[5] for m in self.memory], dtype=np.float32) * self.reward_scale  # also scale value estimates

        # Prevent invalid numerical values during training.
        states   = torch.nan_to_num(states,   nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)
        actions  = torch.nan_to_num(actions,  nan=0.0, posinf=10.0, neginf=-10.0)
        old_logp = torch.nan_to_num(old_logp, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        values_ext = np.append(values, float(last_value) * self.reward_scale)
        returns    = np.zeros_like(rewards, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)

        # Compute Generalized Advantage Estimation (GAE),
        # which reduces variance while preserving
        # low-bias policy gradient estimates.
        gae = 0.0
        for t in reversed(range(len(rewards))):
            # Temporal-difference residual used in GAE recursion.
            delta = rewards[t] + self.gamma * values_ext[t+1] * (1.0 - dones[t]) - values_ext[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae
            returns[t]    = gae + values_ext[t]

        returns = torch.tensor(returns, dtype=torch.float32)
        values_t = torch.tensor(values, dtype=torch.float32)

        # Normalize advantages to improve optimization stability.
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

        # Perform multiple epochs of mini-batch PPO updates
        # using the same collected rollout.
        for _ in range(self.epochs):
            perm = idx[torch.randperm(n)]
            for start in range(0, n, self.batch_size):
                mb = perm[start:start+self.batch_size]

                s = states[mb]
                a = actions[mb]       
                oldlp = old_logp[mb]
                ret = returns[mb]
                adv = advantages_t[mb]
                v_old = values_t[mb]

                # Recompute policy statistics under current network parameters.
                new_logp, v_pred, entropy = self.ac.evaluate(s, a)

                new_logp = torch.nan_to_num(new_logp, nan=-20.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
                v_pred   = torch.nan_to_num(v_pred,   nan=0.0)
                entropy  = torch.nan_to_num(entropy,  nan=0.0)

                # Importance sampling ratio between
                # old and updated policy distributions.
                ratio = torch.exp(torch.clamp(new_logp - oldlp, -10, 10))
                ratio = torch.nan_to_num(ratio, nan=1.0, posinf=10.0, neginf=0.0).clamp(0.0, 10.0)

                # PPO clipped surrogate objective prevents
                # excessively large policy updates.
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Clip critic updates similarly to PPO policy clipping
                # for more stable value learning.
                v_clipped = v_old + (v_pred - v_old).clamp(-0.2, 0.2)
                vf1 = (v_pred - ret) ** 2
                vf2 = (v_clipped - ret) ** 2
                value_loss = 0.5 * torch.max(vf1, vf2).mean()

                # Encourage stochastic exploration during training.
                entropy_bonus = entropy.mean()

                # Combined PPO objective:
                # policy loss + value loss - entropy bonus.
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_bonus

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # Prevent exploding gradients during optimization.
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step() # Apply parameter update using Adam optimizer.

                total_loss += float(loss.item())
                num_updates += 1

        self.memory = [] # Clear rollout buffer after PPO update.
        return total_loss / max(1, num_updates)