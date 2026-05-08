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

def evaluate_policy(agent, env_eval, eval_episodes=3):
    """Evaluate the agent deterministically without exploring noise."""
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs, _ = env_eval.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                action, _, _ = agent.ac.get_action(obs, deterministic=True)
            obs, reward, term, trunc, _ = env_eval.step(action)
            ep_ret += reward
            done = term or trunc
        avg_reward += ep_ret
    return avg_reward / eval_episodes


def train():
    setup_logger()

    # -------- Curriculum --------
    difficulty = 0.0
    max_difficulty = 0.8
    diff_step = 0.05
    diff_every_updates = 10
    target_avg_reward = 60.0

    # -------- PPO --------
    rollout_steps = 4096
    initial_lr = 1e-4

    # -------- Env / Agent --------
    env = RobotEnv(render=False, difficulty=difficulty, draw_path=False)
    # Secondary environment solely for deterministic evaluation
    env_eval = RobotEnv(render=False, difficulty=difficulty, draw_path=False)
    
    agent = PPOAgent(env.obs_dim, env.action_space.shape[0])

    # -------- Tracking --------
    ep = 0
    ep_steps = 0
    ep_reward = 0.0
    num_of_ep = 10000

    best_eval_reward = -1e9
    rewards_100 = deque(maxlen=100)
    steps_100 = deque(maxlen=100) 

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
                steps_100.append(ep_steps)

                print(f"EP {ep:05d} | Reward {ep_reward:8.2f} | Steps {ep_steps}")

                ep += 1
                ep_steps = 0
                ep_reward = 0.0

                obs, _ = env.reset()

        # ================= PPO UPDATE =================
        last_value = agent.value_of(obs)
        loss = agent.update(last_value=last_value)
        updates += 1

        # ================= EVALUATION & CHECKPOINTING =================
        if updates % 5 == 0:
            eval_reward = evaluate_policy(agent, env_eval, eval_episodes=3)
            print(f"--- Eval Reward: {eval_reward:.2f} ---")
            
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                torch.save(agent.ac.state_dict(), "upds/best_model.pth")
                print(f"Saved BEST model: {best_eval_reward:.2f}")
                
        if updates % 10 == 0:
            torch.save(agent.ac.state_dict(), f"upds/checkpoint_upd_{updates}.pth")

        # ================= STATS =================
        elapsed = time.time() - start_time
        avg100 = np.mean(rewards_100) if len(rewards_100) > 0 else 0.0
        avg_steps = np.mean(steps_100) if len(steps_100) > 0 else 0.0

        log_line = (
            f"UPD {updates:04d} | "
            f"EP {ep:05d} | "
            f"Avg100 {avg100:8.2f} | "
            f"AvgSteps {avg_steps:7.1f} | "  
            f"Loss {loss:8.4f} | "
            f"Diff {difficulty:4.2f} | "
            f"Ent {agent.entropy_coef:6.4f} | "
            f"LR {agent.lr:.6f} | "
            f"Time {elapsed:8.1f}s"
        )

        if updates % 1 == 0:
            print(log_line)
            logging.info(log_line)

        # ================= HYPERPARAMETER DECAY =================
        agent.entropy_coef = max(0.005, agent.entropy_coef * 0.9998)
        
        # Linear learning rate decay based on target num_of_ep
        frac = 1.0 - (ep / float(num_of_ep))
        new_lr = max(0.0, initial_lr * frac)
        agent.set_lr(new_lr)

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
                
                env_eval.close()
                env_eval = RobotEnv(render=False, difficulty=difficulty, draw_path=False)
                
                obs, _ = env.reset()

                msg = f"Difficulty increased to {difficulty:.2f}"
                print(msg)
                logging.info(msg)


if __name__ == "__main__":
    train()