from robot_env import RobotEnv
import numpy as np

env = RobotEnv()

obs, _ = env.reset()

for i in range(1000):
    action = np.array([0.5, 0.5])
    obs, r, term, trunc, _ = env.step(action)

env.reset()

for i in range(1000):
    action = np.array([0.5, 0.5])
    obs, r, term, trunc, _ = env.step(action)
