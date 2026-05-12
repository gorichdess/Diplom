import torch
import time
import pybullet as p
import numpy as np
from robot_env import RobotEnv
from model import ActorCritic

def test():
    # 1. Initialize Environment
    # You can change the difficulty here (e.g., 0.1 for easy, 0.8 for hard)
    # Default is 0.3 if not specified
    target_difficulty = 0.5
    env = RobotEnv(render=True, difficulty=target_difficulty)
    
    # 2. Load Model
    model = ActorCritic(env.obs_dim, 2)
    try:
        model.load_state_dict(torch.load("upds/checkpoint_upd_1080.pth", map_location="cpu"))
        print(f"Successfully loaded best_model.pth")
    except FileNotFoundError:
        print("Warning: best_model.pth not found. Testing with random weights.")
    
    model.eval()

    # 3. Access and Display Difficulty
    current_diff = env.difficulty
    print(f"\n--- TEST SETTINGS ---")
    print(f"Difficulty: {current_diff}")
    print(f"Observation Dimension: {env.obs_dim}")
    
    # 4. Run Test Episode
    state, _ = env.reset()
    
    # Add a visual label in the PyBullet window
    p.addUserDebugText(
        f"Difficulty: {current_diff:.2f}", 
        [0, 0, 2.5], 
        textColorRGB=[1, 0, 0], 
        textSize=2
    )

    done = False
    total_reward = 0
    
    print("Starting simulation...")
    while not done:
        with torch.no_grad():
            # deterministic=True returns (squashed_action, value, logp, raw_action)
            squashed_action, *_ = model.get_action(state, deterministic=True)

        state, reward, term, trunc, _ = env.step(squashed_action)
        total_reward += reward
        done = term or trunc

        # Keep the simulation at a watchable speed
        time.sleep(1/60)

    print(f"Episode Finished. Total Reward: {total_reward:.2f}")
    print("---------------------\n")
    
    time.sleep(3)
    env.close()

if __name__ == "__main__":
    test()