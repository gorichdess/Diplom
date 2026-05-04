import torch
import logging
import time
import numpy as np
from tqdm import tqdm

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

    # Start with a very easy terrain
    env = RobotEnv(render=False, difficulty=0.2)
    agent = PPOAgent(env.obs_dim, 2)

    try:
        agent.ac.load_state_dict(torch.load("best_model.pth"))
        print("Loaded best_model.pth. Continuing training...")
    except FileNotFoundError:
        print("best_model.pth not found, starting from scratch.")

    best_reward = -1e9
    reward_window = []

    start_time = time.time()
    print("Training started...")

    total_episodes = 4000

    for ep in tqdm(range(total_episodes)):
        # Curriculum: difficulty rises to 1.3 over 2000 episodes
        difficulty = min(1.0, 0.2 + 0.8 * (ep / 2000))   # reaches 1.5 at episode 2500
        env.difficulty = difficulty

        # Entropy decay (keeps exploration early, more greedy later)
        ent_coef = 0.05 * (1.0 - ep / total_episodes) + 0.01
        agent.set_entropy_coef(ent_coef)

        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_steps = 0

        while not done:
            action, value, logp = agent.ac.get_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done_flag = term or trunc
            agent.store((
                state,
                action,
                logp,
                reward,
                done_flag,
                value
            ))
            state = next_state
            ep_reward += reward
            ep_steps += 1
            done = done_flag

        loss = agent.update()

        reward_window.append(ep_reward)
        if len(reward_window) > 100:
            reward_window.pop(0)

        avg100 = np.mean(reward_window)

        if ep_reward > best_reward:
            best_reward = ep_reward
            torch.save(agent.ac.state_dict(), "best_model.pth")

        if ep % 250 == 0:
            torch.save(agent.ac.state_dict(), f"checkpoint_{ep}.pth")

        elapsed = time.time() - start_time

        log_line = (
            f"EP {ep:04d} | "
            f"Reward {ep_reward:8.2f} | "
            f"Avg100 {avg100:8.2f} | "
            f"Best {best_reward:8.2f} | "
            f"Steps {ep_steps:4d} | "
            f"Loss {loss:8.4f} | "
            f"Diff {difficulty:4.2f} | "
            f"Time {elapsed:8.1f}s"
        )

        logging.info(log_line)

        if ep % 20 == 0:
            print(log_line)

    torch.save(agent.ac.state_dict(), "final_model.pth")

    print("Training finished.")
    print("Saved:")
    print(" - training.log")
    print(" - best_model.pth")
    print(" - final_model.pth")


if __name__ == "__main__":
    train()