# Evaluation script for the trained PPO agent.
# The script runs the trained policy on several terrain difficulty levels
# and saves quantitative results for diploma analysis.

import argparse
import os
import torch
import numpy as np
import pandas as pd

from robot_env import RobotEnv
from rl_agent import PPOAgent


def evaluate_trained_agent(checkpoint_path, difficulty, episodes=100, render=False):
    """
    Evaluate a trained PPO policy on a fixed terrain difficulty.

    Parameters:
        checkpoint_path (str): path to the saved trained model checkpoint.
        difficulty (float): terrain difficulty level in RobotEnv.
        episodes (int): number of evaluation episodes.
        render (bool): whether to show PyBullet GUI.

    Returns:
        list[dict]: episode-level evaluation results.
    """

    # Create evaluation environment.
    # draw_path=False is used to avoid unnecessary debug visualization
    # during statistical evaluation.
    env = RobotEnv(render=render, difficulty=difficulty, draw_path=False)

    # Initialize PPO agent with the same observation and action dimensions
    # as during training.
    agent = PPOAgent(env.obs_dim, env.action_space.shape[0])

    # Load trained Actor-Critic parameters.
    # map_location="cpu" makes loading possible even without GPU.
    agent.ac.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    # Switch neural network to evaluation mode.
    # This disables training-specific behavior if such layers are added later.
    agent.ac.eval()

    results = []

    for seed in range(episodes):
        # Fixed seed makes evaluation more reproducible.
        obs, _ = env.reset(seed=seed)

        done = False
        ep_reward = 0.0
        ep_steps = 0
        last_info = {}

        while not done:
            # Deterministic action is used for final evaluation.
            # This measures the learned policy itself, not exploration noise.
            with torch.no_grad():
                action, *_ = agent.ac.get_action(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_steps += 1
            done = terminated or truncated
            last_info = info

        # Success means the robot reached the goal.
        success = 1 if last_info.get("reach_goal", False) else 0

        results.append({
            "method": "PPO agent",
            "difficulty": difficulty,
            "episode": seed,
            "reward": round(ep_reward, 2),
            "steps": ep_steps,
            "success": success
        })

        print(
            f"Episode {seed:03d} | "
            f"Diff {difficulty:.2f} | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {ep_steps:4d} | "
            f"Success {success}"
        )

    env.close()
    return results


def summarize_results(df):
    """
    Create diploma-ready summary table for the trained PPO agent.

    The output columns match baseline_eval.py, so the results
    can be directly combined and compared in one table.
    """

    summary = (
        df.groupby(["method", "difficulty"])
        .agg(
            episodes=("episode", "count"),
            successful_episodes=("success", "sum"),
            success_rate_percent=("success", lambda x: round(x.mean() * 100.0, 2)),
            avg_reward=("reward", lambda x: round(x.mean(), 2)),
            std_reward=("reward", lambda x: round(x.std(), 2)),
            min_reward=("reward", lambda x: round(x.min(), 2)),
            max_reward=("reward", lambda x: round(x.max(), 2)),
            avg_steps=("steps", lambda x: round(x.mean(), 2)),
            std_steps=("steps", lambda x: round(x.std(), 2)),
        )
        .reset_index()
    )

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="upds2/best_model.pth",
        help="Path to trained PPO model checkpoint"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per difficulty"
    )

    parser.add_argument(
        "--difficulties",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.4, 0.5, 0.55],
        help="List of terrain difficulty levels for evaluation"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable PyBullet GUI rendering"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    all_results = []

    for difficulty in args.difficulties:
        print(f"\n=== Evaluating PPO agent at difficulty {difficulty:.2f} ===")

        results = evaluate_trained_agent(
            checkpoint_path=args.checkpoint,
            difficulty=difficulty,
            episodes=args.episodes,
            render=args.render
        )

        all_results.extend(results)

    # Save detailed episode-by-episode results.
    df = pd.DataFrame(all_results)
    detailed_name = "ppo_evaluation_detailed.csv"
    df.to_csv(detailed_name, index=False)

    # Save summary table for diploma.
    summary = summarize_results(df)
    summary_name = "ppo_evaluation_summary.csv"
    summary.to_csv(summary_name, index=False)

    print("\n=== PPO Evaluation Summary ===")
    print(summary.to_string(index=False))

    print(f"\nSaved detailed results to: {detailed_name}")
    print(f"Saved summary results to: {summary_name}")