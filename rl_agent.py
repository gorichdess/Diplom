import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from model import ActorCritic

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return (np.array(self.states, dtype=np.float32),
                np.array(self.actions, dtype=np.float32),
                np.array(self.probs, dtype=np.float32),
                np.array(self.values, dtype=np.float32),
                np.array(self.rewards, dtype=np.float32),
                np.array(self.dones, dtype=np.bool_),
                batches)

    def store_memory(self, state, action, prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []

class PPOAgent:
    def __init__(self, obs_dim, action_dim, 
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 critic_coef=0.5,
                 entropy_coef=0.01,
                 max_grad_norm=0.5,
                 epochs=10,
                 batch_size=64,
                 horizon=2048):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.horizon = horizon
        
        # Сети
        self.actor_critic = ActorCritic(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        # Память
        self.memory = PPOMemory(batch_size)
        
        # Счетчики
        self.steps = 0
        self.episodes = 0
        
        # Для логирования
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
    
    def get_gae(self, rewards, dones, values):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        # Переворачиваем для обратного прохода
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # для последнего шага нет следующего значения
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = np.array(advantages) + np.array(values)
        advantages = np.array(advantages)
        
        # Нормализация преимуществ
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def collect_trajectory(self, env):
        """Сбор одной траектории"""
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done and episode_length < self.horizon:
            # Получаем действие
            action, value, log_prob = self.actor_critic.get_action(state)
            
            # Выполняем действие
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Сохраняем в память
            self.memory.store_memory(state, action, log_prob, value, reward, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            self.steps += 1
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episodes += 1
        
        return episode_reward, episode_length
    
    def update(self):
        """Обновление политики"""
        # Получаем данные из памяти
        states, actions, old_probs, values, rewards, dones, batches = self.memory.generate_batches()
        
        # Вычисляем преимущества и returns
        advantages, returns = self.get_gae(rewards, dones, values)
        
        # Преобразуем в тензоры
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_probs = torch.FloatTensor(old_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Несколько эпох обучения
        total_loss = 0
        for _ in range(self.epochs):
            for batch in batches:
                # Получаем батч
                state_batch = states[batch]
                action_batch = actions[batch]
                old_prob_batch = old_probs[batch]
                advantage_batch = advantages[batch]
                return_batch = returns[batch]
                
                # Оцениваем текущую политику
                new_probs, values_pred, entropy = self.actor_critic.evaluate(state_batch, action_batch)
                
                # PPO Clip loss
                ratio = torch.exp(new_probs - old_prob_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = F.mse_loss(values_pred, return_batch)
                
                # Entropy loss (for exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss
                total_loss += loss.item()
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Очищаем память
        self.memory.clear_memory()
        
        return total_loss / (self.epochs * len(batches) + 1e-8)
    
    def save(self, path):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
    
    def load(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']