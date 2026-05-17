# PPO-Based Robot Navigation in 3D Terrain

This project implements reinforcement learning for autonomous robot navigation in a three-dimensional PyBullet simulation environment.  
The agent controls a Husky-like wheeled robot and learns to reach a target position on procedurally generated rough terrain.

The main method is Proximal Policy Optimization (PPO) with an Actor-Critic neural network. The project also includes curriculum learning, baseline controllers, trained-agent evaluation, and training log visualization.

## Project goal

The goal of the project is to train and evaluate a reinforcement learning agent for robot movement in a 3D environment with variable terrain difficulty.

The environment supports:
- flat and rough terrain;
- procedural heightfield generation;
- obstacles such as rocks and logs;
- different terrain difficulty levels;
- robot state and terrain observations;
- goal-reaching reward evaluation.

## Technologies

- Python
- PyTorch
- Gymnasium
- PyBullet
- NumPy
- Pandas
- Matplotlib
- SciPy

## Project structure
├── upds                      # Folder with checkpoints of training sorted by PPO updates + log file
  ├──base_model               # Folder with checkpoints of base model training
  ├──upds_curr_0-1            # Folder with checkpoints of curriculum training
  ├──upds_no_curr_0.5         # Folder with checkpoints of stable difficulty = 0.5 training
  ├──upds_no_curr_1           # Folder with checkpoints of stable difficulty = 1.0 training
├── robot_env.py              # Custom PyBullet/Gymnasium robot environment
├── model.py                  # Actor-Critic neural network
├── rl_agent.py               # PPO agent implementation
├── train.py                  # PPO training script
├── evaluate_trained.py       # Evaluation of trained PPO model
├── baseline_eval.py          # Evaluation of random and proportional baselines
├── plot_training.py          # Visualization of training logs
├── test.py                   # Visual testing of trained model
└── README.md
Environment

The simulation environment is implemented in robot_env.py.
It uses a Husky differential-drive robot model in PyBullet. The robot receives continuous actions: [linear_velocity, angular_velocity]

Both actions are normalized to the range [-1, 1].

The observation vector contains robot navigation features, dynamic state information, terrain ray samples, and the current terrain difficulty. The terrain difficulty ranges from 0.0 for flat terrain to 1.0 for highly irregular terrain.

PPO agent

The PPO agent uses an Actor-Critic neural network.
The actor predicts continuous control actions, while the critic estimates the value of the current state. The model uses Gaussian action sampling with tanh squashing to keep actions inside the valid environment range.

The PPO implementation includes:

clipped policy optimization;
Generalized Advantage Estimation;
entropy regularization;
reward scaling;
gradient clipping;
mini-batch updates.
Training

Training is started with:

python train.py

Curriculum learning mode is used by default. In this mode, training starts from simple terrain and gradually increases difficulty when the agent reaches the required success rate.

To train on a fixed difficulty:

python train.py --mode fixed --fixed_difficulty 0.55

To save logs to a specific file:

python train.py --log_file training.log

The training script logs reward, number of steps, success rate, loss, difficulty, entropy coefficient, learning rate, and elapsed time.

Evaluation

To evaluate the trained PPO agent:

python evaluate_trained.py --checkpoint upds/best_model.pth

The evaluation script runs the trained policy on several difficulty levels and saves detailed and summary results. The output includes success rate, average reward, reward standard deviation, average number of steps, and step standard deviation.

Baseline comparison

The project includes two baseline methods:

Random policy
Proportional controller

Run baseline evaluation with:

python baseline_eval.py

Or evaluate only one baseline:

python baseline_eval.py --policy proportional
python baseline_eval.py --policy random

The baseline results are saved in CSV format and can be compared with the PPO evaluation results.

Training visualization

Training logs can be visualized using:

python plot_training.py training.log --labels "PPO with curriculum" --output training_plot

The script creates plots for:

average reward;
success rate;
terrain difficulty;
loss;
entropy;
learning rate.
Visual testing

To run a trained model with PyBullet GUI:

python test.py

This opens the simulation window and shows the robot moving in the environment using the trained deterministic policy.

Results

The trained PPO agent is evaluated across several terrain difficulty levels.
The main metrics are:

method
difficulty
episodes
successful_episodes
success_rate_percent
avg_reward
std_reward
avg_steps
std_steps

These metrics are used to compare the trained reinforcement learning policy with non-learning baseline controllers.

Author:Kateryna Lisna
