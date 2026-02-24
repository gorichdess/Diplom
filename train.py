import numpy as np
import torch
import time
from robot_env import RobotEnv
from rl_agent import PPOAgent

def train_agent(env, total_timesteps=500000, save_path="ppo_robot_model.pth"):

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        critic_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        epochs=10,
        batch_size=64,
        horizon=2048
    )
    
    print("=" * 60)
    print("Starting PPO Training")
    print("=" * 60)
    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 60)
    
    start_time = time.time()
    
    while agent.steps < total_timesteps:
        episode_reward, episode_length = agent.collect_trajectory(env)
        
        # Update the policy
        loss = agent.update()
        
        if agent.episodes % 10 == 0:
            avg_reward = np.mean(agent.episode_rewards)
            avg_length = np.mean(agent.episode_lengths)
            elapsed_time = time.time() - start_time
            
            print(f"\nEpisode {agent.episodes}")
            print(f"Steps: {agent.steps}/{total_timesteps} ({agent.steps/total_timesteps*100:.1f}%)")
            print(f"Time elapsed: {elapsed_time:.1f}s")
            print(f"Avg Reward (last 100): {avg_reward:.2f}")
            print(f"Avg Episode Length: {avg_length:.2f}")
            print(f"Loss: {loss:.4f}")
            print("-" * 50)
        
        if agent.episodes % 100 == 0:
            checkpoint_path = f"checkpoint_ep{agent.episodes}.pth"
            agent.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    agent.save(save_path)
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training finished!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Final model saved to: {save_path}")
    print("=" * 60)
     
    return agent

if __name__ == "__main__":
    env = RobotEnv(size=20, render=False)
    
    #Training the agent
    agent = train_agent(env, total_timesteps=600000)
    
    env.close()