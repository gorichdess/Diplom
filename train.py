import time
import torch
import logging
import numpy as np
import argparse
from collections import deque

from robot_env import RobotEnv
from rl_agent import PPOAgent

# Configure file-based logging for training statistics.
def setup_logger(log_file_name="training.log"):
    logging.basicConfig(
        filename=log_file_name,
        filemode="w",
        level=logging.INFO,
        format="%(message)s"
    )

# Evaluate deterministic policy performance
# without exploration noise.
def evaluate_policy(agent, env_eval, eval_episodes=3):
    avg_reward = 0.0 # Accumulate rewards across evaluation episodes.
    for _ in range(eval_episodes):
        obs, _ = env_eval.reset()
        done = False
        ep_ret = 0.0
        while not done:
            with torch.no_grad():
                # Use deterministic policy during evaluation
                # for reproducible performance measurement.
                squashed_action, *_ = agent.ac.get_action(obs, deterministic=True) 
            obs, reward, term, trunc, _ = env_eval.step(squashed_action)
            ep_ret += reward
            done = term or trunc
        avg_reward += ep_ret
    return avg_reward / eval_episodes

# Define adaptive curriculum success threshold.
# Higher terrain difficulty requires lower expected success rate.
def get_target_success(difficulty: float) -> float:
    target = 0.8 - 0.5 * difficulty
    return max(0.2, min(0.8, target))


def train():
    # Parse command-line training configuration parameters.
    parser = argparse.ArgumentParser()

    # Training can operate in:
    # curriculum mode (adaptive difficulty)
    # or fixed-difficulty mode.
    parser.add_argument("--mode", choices=["curriculum", "fixed"], default="curriculum",
                        help="curriculum: поступове ускладнення; fixed: постійна складність")
    parser.add_argument("--fixed_difficulty", type=float, default=0.5,
                        help="Складність для режиму fixed (за замовчуванням 0.5)")
    parser.add_argument("--log_file", type=str, default="training.log",
                        help="Ім'я файлу для логування")
    args = parser.parse_args()

    setup_logger(args.log_file)

    if args.mode == "fixed":
        difficulty = args.fixed_difficulty
    else:
        difficulty = 0.0 # Curriculum learning starts from simplest terrain configuration.

    # Parameters controlling adaptive curriculum progression.
    max_difficulty = 1.0
    diff_step = 0.05
    diff_every_updates = 10
    success_threshold = 0.8         
    success_lower_bound = 0.2       

    # Number of environment interactions collected
    # before each PPO optimization phase.
    rollout_steps = 4096
    initial_lr = 1e-4

    # Separate environments are used for
    # training and evaluation.
    env = RobotEnv(render=False, difficulty=difficulty, draw_path=False)

    agent = PPOAgent(env.obs_dim, env.action_space.shape[0])

    ep = 0
    ep_steps = 0
    ep_reward = 0.0
    max_updates = 1200
    num_of_ep=10000

    best_eval_reward = -1e9

    # Sliding performance statistics over recent episodes.
    rewards_100 = deque(maxlen=100)
    steps_100 = deque(maxlen=100)
    success_100 = deque(maxlen=100)

    updates = 0
    total_steps = 0

    # Initialize first training episode.
    obs, _ = env.reset()
    start_time = time.time()

    print("Training started...")
    if args.mode == "fixed":
        print(f"Mode: fixed difficulty = {difficulty:.2f}")
    else:
        print("Mode: curriculum learning")

    while updates < max_updates: # Main reinforcement learning training loop.
        for _ in range(rollout_steps): # Collect environment trajectories for PPO update.
            squashed_action, value, logp, raw_action = agent.ac.get_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, info = env.step(squashed_action)
            done = terminated or truncated

            # Store rollout transition for PPO optimization.
            agent.store((obs, raw_action, logp, reward, float(done), value))

            obs = next_obs
            ep_reward += float(reward) # Track cumulative reward and episode length.
            ep_steps += 1
            total_steps += 1

            if done:
                rewards_100.append(ep_reward) 
                steps_100.append(ep_steps)

                # Track recent navigation success rate
                # for curriculum adaptation.
                success_100.append(1 if info.get("reach_goal", False) else 0) 
                print(f"EP {ep:05d} | Reward {ep_reward:8.2f} | Steps {ep_steps}")
                ep += 1
                ep_steps = 0
                ep_reward = 0.0
                obs, _ = env.reset()

        last_value = agent.value_of(obs)

        # Optimize PPO policy using collected rollout data.
        loss = agent.update(last_value=last_value)
        updates += 1

        if updates % 10 == 0:
            # Periodically save training checkpoints
            # for recovery and analysis.
            torch.save(agent.ac.state_dict(), f"upds/upds_curr_0-1-new/checkpoint_upd_{updates}.pth")

        elapsed = time.time() - start_time

        # Compute moving-average training statistics.
        avg100 = np.mean(rewards_100) if len(rewards_100) > 0 else 0.0
        avg_steps = np.mean(steps_100) if len(steps_100) > 0 else 0.0

        if len(success_100) > 0:
            succ_rate = (sum(success_100) / len(success_100)) * 100.0
        else:
            succ_rate = 0.0

        # Aggregate training metrics into formatted log output.
        log_line = (
            f"UPD {updates:04d} | "
            f"EP {ep:05d} | "
            f"Avg100 {avg100:8.2f} | "
            f"AvgSteps {avg_steps:7.1f} | "
            f"Succ {succ_rate:.1f}% | "
            f"Loss {loss:8.4f} | "
            f"Diff {difficulty:4.2f} | "
            f"Ent {agent.entropy_coef:6.4f} | "
            f"LR {agent.lr:.6f} | "
            f"Time {elapsed:8.1f}s"
        )

        if updates % 1 == 0:
            print(log_line)
            logging.info(log_line)

        # Gradually reduce exploration strength
        # as training progresses.
        agent.entropy_coef = max(0.005, agent.entropy_coef * 0.999)

        # Linearly decay learning rate during training
        # for more stable convergence.
        frac = 1.0 - (ep / float(num_of_ep))
        new_lr = max(0.0, initial_lr * frac)
        agent.set_lr(new_lr)

        # Adapt terrain difficulty based on recent agent performance.
        if args.mode == "curriculum":
            if (
                updates % diff_every_updates == 0
                and len(rewards_100) == rewards_100.maxlen
                and len(success_100) == success_100.maxlen
            ):
                current_success = sum(success_100) / len(success_100)
                target_success = get_target_success(difficulty)

                # Increase difficulty once agent consistently
                # achieves target success rate.
                if current_success >= target_success and difficulty < max_difficulty:
                    difficulty = min(max_difficulty, difficulty + diff_step)

                    # Regenerate procedural terrain
                    # using updated curriculum difficulty.
                    env.set_difficulty(difficulty) 
                    obs, _ = env.reset()

                    # Clear historical metrics after curriculum transition.
                    success_100.clear()
                    rewards_100.clear()
                    steps_100.clear()
                    msg = (f"Difficulty increased to {difficulty:.2f} "
                           f"(target {target_success*100:.1f}%, actual {current_success*100:.1f}%)")
                    print(msg)
                    logging.info(msg)

                # Reduce difficulty if agent performance collapses,
                # preventing catastrophic curriculum failure.
                elif current_success <= success_lower_bound and difficulty > 0.0:
                    difficulty = max(0.0, difficulty - diff_step)
                    env.set_difficulty(difficulty)
                    obs, _ = env.reset()
                    success_100.clear()
                    rewards_100.clear()
                    steps_100.clear()
                    msg = f"Difficulty decreased to {difficulty:.2f} (success {current_success*100:.1f}%)"
                    print(msg)
                    logging.info(msg)


if __name__ == "__main__":
    train()