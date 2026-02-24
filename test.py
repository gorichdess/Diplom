import numpy as np
import torch
import time
from robot_env import RobotEnv
from rl_agent import PPOAgent

def test_agent(model_path="ppo_robot_model.pth", num_episodes=10, render=True, size=20):
    
    env = RobotEnv(size=size, render=render)
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = PPOAgent(obs_dim=obs_dim, action_dim=action_dim)
    agent.load(model_path)
    
    print("=" * 60)
    print(f"Testing agent from: {model_path}")
    print(f"Number of episodes: {num_episodes}")
    print("=" * 60)
    
    all_rewards = []
    all_lengths = []
    successes = 0
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _, _ = agent.actor_critic.get_action(state, deterministic=True)
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
   
            if render:
                time.sleep(0.01)
            
            if terminated and reward > 100: 
                successes += 1
        
        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Average Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
    print(f"Success Rate: {successes}/{num_episodes} ({successes/num_episodes*100:.1f}%)")
    print("=" * 60)
    
    env.close()
    return all_rewards, all_lengths, successes

if __name__ == "__main__":

    test_agent("ppo_robot_model.pth", 10, render=True, size=20)