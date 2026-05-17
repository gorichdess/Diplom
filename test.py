import torch
import time
import pybullet as p
import numpy as np
from robot_env import RobotEnv
from model import ActorCritic

# Evaluate trained PPO policy in the PyBullet environment.
def test():
    # Select terrain complexity level for policy evaluation.
    target_difficulty = 1.0
    env = RobotEnv(render=True, difficulty=target_difficulty) # Create simulation environment with GUI rendering enabled.
    
    # Initialize Actor-Critic neural network
    # with environment-specific observation size.
    model = ActorCritic(env.obs_dim, 2) # Load previously trained PPO policy parameters.
    try:
        model.load_state_dict(torch.load("upds/upds_no_curr_1 — копия/best_model.pth", map_location="cpu"))
        print(f"Successfully loaded best_model.pth")
    except FileNotFoundError:
        print("Warning: best_model.pth not found. Testing with random weights.")
    
    # Switch network to inference mode.
    model.eval()

    current_diff = env.difficulty
    print(f"\n--- TEST SETTINGS ---")
    print(f"Difficulty: {current_diff}")
    print(f"Observation Dimension: {env.obs_dim}")
    
    # Initialize new evaluation episode.
    state, _ = env.reset()
    
    # Display current curriculum difficulty in simulation window.
    p.addUserDebugText(
        f"Difficulty: {current_diff:.2f}", 
        [0, 0, 2.5], 
        textColorRGB=[1, 0, 0], 
        textSize=2
    )

    done = False
    total_reward = 0
    
    print("Starting simulation...")
    # Run policy inference and environment interaction
    # until episode termination.
    while not done:
        # Disable gradient computation during evaluation
        # to improve inference efficiency.
        with torch.no_grad():
            # Use deterministic policy actions during evaluation
            # for reproducible robot behavior.
            squashed_action, *_ = model.get_action(state, deterministic=True)

        state, reward, term, trunc, _ = env.step(squashed_action) # Execute selected action in the simulation environment.
        total_reward += reward # Accumulate total episodic reward for performance evaluation.
        done = term or trunc # Episode ends on success, failure, or timeout truncation.

        time.sleep(1/60) # Slow simulation to approximate real-time visualization.

    print(f"Episode Finished. Total Reward: {total_reward:.2f}")
    print("---------------------\n")
    
    time.sleep(3)
    env.close() # Properly disconnect PyBullet simulation resources.

if __name__ == "__main__":
    test()