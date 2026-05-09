import torch
import time
import pybullet as p
import numpy as np
import pandas as pd
from robot_env import RobotEnv
from model import ActorCritic

def run_evaluation(num_episodes=100, target_difficulty=0.65):
    print(f"Starting evaluation of {num_episodes} episodes...")
    env = RobotEnv(render=False, difficulty=target_difficulty)

    model = ActorCritic(env.obs_dim, 2)
    try:
        model.load_state_dict(torch.load("upds/checkpoint_upd_380.pth", map_location="cpu"))
        model.eval()
    except FileNotFoundError:
        print("Error: best_model.pth not found.")
        return

    episode_data = []
    best_reward = -float('inf')
    best_seed = 0

    for i in range(num_episodes):
        obs, _ = env.reset(seed=i)
        done = False
        ep_reward = 0
        ep_steps = 0

        while not done:
            with torch.no_grad():
                squashed_action, *_ = model.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(squashed_action)  # capture info
            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

        # Use the ground-truth success flag from the environment if available,
        # otherwise fall back to distance check.
        success = 1 if info.get("reach_goal", False) else 0

        episode_data.append({
            "episode": i,
            "reward": round(ep_reward, 2),
            "steps": ep_steps,
            "success": success,
            "dist_to_goal": 0.0  # you could still compute if needed, but success is enough
        })

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_seed = i

        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{num_episodes}...")

    env.close()

    # --- Save to CSV and summary (unchanged) ---
    df = pd.DataFrame(episode_data)
    df.to_csv("evaluation_results.csv", index=False)
    
    success_rate = (df['success'].sum() / num_episodes) * 100
    avg_reward = df['reward'].mean()
    avg_steps = df['steps'].mean()

    summary_text = (
        f"EVALUATION SUMMARY\n"
        f"==================\n"
        f"Difficulty:      {target_difficulty}\n"
        f"Total Episodes:  {num_episodes}\n"
        f"Success Rate:    {success_rate:.1f}%\n"
        f"Average Reward:  {avg_reward:.2f}\n"
        f"Average Steps:   {avg_steps:.1f}\n"
        f"Best Reward:     {best_reward:.2f} (Episode {best_seed})\n"
    )

    with open("evaluation_summary.txt", "w") as f:
        f.write(summary_text)

    print("\n" + summary_text)
    print(f"Results saved to 'evaluation_results.csv' and 'evaluation_summary.txt'")

    # --- Replay Best Episode ---
    print(f"\nReplaying Best Episode (Seed {best_seed}) for visualization...")
    replay_env = RobotEnv(render=True, difficulty=target_difficulty)
    obs, _ = replay_env.reset(seed=best_seed)
    
    p.addUserDebugText(f"BEST EPISODE REPLAY (Reward: {best_reward:.2f})", [0,0,3], [0,1,0], 2)
    
    done = False
    while not done:
        with torch.no_grad():
            squashed_action, *_ = model.get_action(obs, deterministic=True)   # corrected unpacking
        obs, _, term, trunc, _ = replay_env.step(squashed_action)            # you can ignore info here
        done = term or trunc
        time.sleep(1/60)
    
    replay_env.close()

if __name__ == "__main__":
    run_evaluation()