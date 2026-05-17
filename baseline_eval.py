# Baseline evaluation script for diploma comparison.
# This script evaluates non-learning baseline policies in the same RobotEnv
# environment that is used for PPO training.
#
# The goal is to compare the trained PPO agent with:
# 1) random policy
# 2) proportional heuristic controller
#
# The script saves:
# - detailed episode-by-episode results
# - summary table with success rate, reward statistics, and step statistics

import argparse
import os
import numpy as np
import pandas as pd

from robot_env import RobotEnv


def proportional_controller(obs, env):
    """
    Simple proportional controller baseline.

    The controller uses handcrafted rules:
    - linear velocity depends on distance to the goal;
    - angular velocity depends on heading error.

    This baseline is useful for comparison because it does not learn,
    but still uses meaningful navigation information from observations.
    """

    angle_diff = obs[3]       # Heading error relative to the goal direction
    dist = obs[2] * 10.0      # Approximate distance to the goal

    lin_vel = np.clip(dist * 0.5, -1.0, 1.0)
    ang_vel = np.clip(angle_diff * 2.0, -1.0, 1.0)

    return np.array([lin_vel, ang_vel], dtype=np.float32)


def random_policy(obs, env):
    """
    Random policy baseline.

    The action is sampled randomly from the environment action space.
    This baseline shows how the agent performs without any control strategy.
    """

    return env.action_space.sample()


def evaluate_policy(policy_function, policy_name, difficulty, episodes=100, render=False):
    """
    Evaluate one baseline policy on one terrain difficulty level.

    Parameters:
        policy_function: function that maps observation to action.
        policy_name: name of the evaluated baseline policy.
        difficulty: terrain difficulty level.
        episodes: number of evaluation episodes.
        render: whether to enable PyBullet GUI.

    Returns:
        list of dictionaries with episode-level results.
    """

    env = RobotEnv(render=render, difficulty=difficulty, draw_path=False)

    results = []

    for seed in range(episodes):
        # Fixed seed improves reproducibility of evaluation.
        obs, _ = env.reset(seed=seed)

        done = False
        ep_reward = 0.0
        ep_steps = 0
        last_info = {}

        while not done:
            action = policy_function(obs, env)

            obs, reward, terminated, truncated, info = env.step(action)

            ep_reward += float(reward)
            ep_steps += 1
            done = terminated or truncated
            last_info = info

        success = 1 if last_info.get("reach_goal", False) else 0

        results.append({
            "method": policy_name,
            "difficulty": difficulty,
            "episode": seed,
            "reward": round(ep_reward, 2),
            "steps": ep_steps,
            "success": success
        })

        print(
            f"{policy_name:14s} | "
            f"Diff {difficulty:.2f} | "
            f"Episode {seed:03d} | "
            f"Reward {ep_reward:8.2f} | "
            f"Steps {ep_steps:4d} | "
            f"Success {success}"
        )

    env.close()
    return results


def summarize_results(df):
    """
    Create diploma-ready summary table.

    The table contains:
    - number of episodes;
    - number of successful episodes;
    - success rate;
    - average reward;
    - reward standard deviation;
    - average number of steps;
    - step standard deviation.
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
        "--policy",
        choices=["proportional", "random", "both"],
        default="both",
        help="Baseline policy to evaluate"
    )

    parser.add_argument(
        "--difficulties",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.40, 0.50, 0.55],
        help="Terrain difficulty levels for evaluation"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per difficulty"
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable PyBullet GUI rendering"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="evaluation_results",
        help="Directory for saving evaluation CSV files"
    )

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    policies = {}

    if args.policy in ["proportional", "both"]:
        policies["Proportional controller"] = proportional_controller

    if args.policy in ["random", "both"]:
        policies["Random policy"] = random_policy

    all_results = []

    for difficulty in args.difficulties:
        for policy_name, policy_function in policies.items():
            print(f"\n=== Evaluating {policy_name} at difficulty {difficulty:.2f} ===")

            results = evaluate_policy(
                policy_function=policy_function,
                policy_name=policy_name,
                difficulty=difficulty,
                episodes=args.episodes,
                render=args.render
            )

            all_results.extend(results)

    # Save detailed episode-level results.
    detailed_df = pd.DataFrame(all_results)
    detailed_path = os.path.join(args.out_dir, "baseline_detailed_results.csv")
    detailed_df.to_csv(detailed_path, index=False)

    # Save summary table for diploma.
    summary_df = summarize_results(detailed_df)
    summary_path = os.path.join(args.out_dir, "baseline_summary_results.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Baseline Summary Table ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved detailed baseline results to: {detailed_path}")
    print(f"Saved summary baseline results to: {summary_path}")