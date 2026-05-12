import numpy as np
import pandas as pd
from robot_env import RobotEnv

def proportional_controller(obs, env):
    """Рух на ціль: лінійна швидкість пропорційна відстані, кутова – помилці за кутом."""
    # obs структура: [to_goal_x*0.1, to_goal_y*0.1, dist*0.1, angle_diff, sin_yaw, cos_yaw, ...]
    angle_diff = obs[3]   # різниця кута до цілі та орієнтації
    dist = obs[2] * 10.0  # масштаб назад: ми множили на 0.1
    # Прості коефіцієнти
    lin_vel = np.clip(dist * 0.5, -1.0, 1.0)
    ang_vel = np.clip(angle_diff * 2.0, -1.0, 1.0)
    return np.array([lin_vel, ang_vel], dtype=np.float32)

def random_policy(obs, env):
    return env.action_space.sample()

def evaluate_baseline(policy_function, env, episodes=100):
    results = []
    for seed in range(episodes):
        obs, _ = env.reset(seed=seed)
        done = False
        ep_reward = 0.0
        ep_steps = 0
        while not done:
            action = policy_function(obs, env)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_steps += 1
            done = term or trunc
        success = 1 if info.get("reach_goal", False) else 0
        results.append({"episode": seed, "reward": round(ep_reward,2),
                        "steps": ep_steps, "success": success})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=float, default=0.3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--policy", choices=["proportional", "random"], default="proportional")
    args = parser.parse_args()

    env = RobotEnv(render=False, difficulty=args.difficulty)
    if args.policy == "proportional":
        policy = proportional_controller
    else:
        policy = random_policy

    results = evaluate_baseline(policy, env, episodes=args.episodes)
    df = pd.DataFrame(results)
    fname = f"baseline_{args.policy}_diff{args.difficulty:.2f}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved {fname}")
    print(f"Success rate: {df['success'].mean()*100:.1f}%, Avg reward: {df['reward'].mean():.2f}")
    env.close()