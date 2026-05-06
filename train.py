import time
import torch
import logging
import numpy as np
from collections import deque

from robot_env import RobotEnv
from rl_agent import PPOAgent


def setup_logger():
    logging.basicConfig(
        filename="training.log",
        filemode="w",
        level=logging.INFO,
        format="%(message)s"
    )


def train():
    setup_logger()

    # -------- Curriculum --------
    difficulty = 0.0
    max_difficulty = 0.2
    diff_step = 0.05
    diff_every_updates = 10
    target_avg_reward = 60.0

    # -------- PPO --------
    rollout_steps = 4096

    # -------- Env / Agent --------
    env = RobotEnv(render=False, difficulty=difficulty, draw_path=False)
    agent = PPOAgent(env.obs_dim, env.action_space.shape[0])

    # -------- Tracking --------
    ep = 0
    ep_steps = 0
    ep_reward = 0.0
    num_of_ep=5000

    best_reward = -1e9
    rewards_100 = deque(maxlen=100)

    updates = 0
    total_steps = 0

    obs, _ = env.reset()
    start_time = time.time()

    print("Training started...")

    while ep < num_of_ep:
        # ================= ROLLOUT =================
        for _ in range(rollout_steps):
            action, value, logp = agent.ac.get_action(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store((obs, action, logp, reward, float(done), value))

            obs = next_obs
            ep_reward += float(reward)
            ep_steps += 1
            total_steps += 1

            if done:
                rewards_100.append(ep_reward)

                print(f"EP {ep:05d} | Reward {ep_reward:8.2f} | Steps {ep_steps}")

                if ep_reward > best_reward:
                    best_reward = ep_reward
                    torch.save(agent.ac.state_dict(), "best_model.pth")
                    print(f"Saved BEST model: {best_reward:.2f}")

                ep += 1
                ep_steps = 0
                ep_reward = 0.0

                obs, _ = env.reset()

        # ================= PPO UPDATE =================
        last_value = agent.value_of(obs)
        loss = agent.update(last_value=last_value)
        updates += 1

        # ================= STATS =================
        elapsed = time.time() - start_time
        avg100 = np.mean(rewards_100) if len(rewards_100) > 0 else 0.0

        log_line = (
            f"UPD {updates:04d} | "
            f"EP {ep:05d} | "
            f"Avg100 {avg100:8.2f} | "
            f"Best {best_reward:8.2f} | "
            f"Loss {loss:8.4f} | "
            f"Diff {difficulty:4.2f} | "
            f"Ent {agent.entropy_coef:6.4f} | "
            f"Steps {ep_steps:8d} | "
            f"Time {elapsed:8.1f}s"
        )

        print(log_line)
        logging.info(log_line)

        # ================= CHECKPOINT =================
        if updates % 10 == 0:
            torch.save(agent.ac.state_dict(), f"checkpoint_upd_{updates}.pth")
            print(f"Checkpoint saved at update {updates}")

        # ================= ENTROPY DECAY =================
        agent.entropy_coef = max(0.005, agent.entropy_coef * 0.999)

        # ================= CURRICULUM =================
        if (
            updates % diff_every_updates == 0
            and len(rewards_100) == rewards_100.maxlen
            and avg100 > target_avg_reward
        ):
            if difficulty < max_difficulty:
                difficulty = min(max_difficulty, difficulty + diff_step)

                env.close()
                env = RobotEnv(render=False, difficulty=difficulty, draw_path=False)
                obs, _ = env.reset()

                msg = f"Difficulty increased to {difficulty:.2f}"
                print(msg)
                logging.info(msg)


if __name__ == "__main__":
    train()