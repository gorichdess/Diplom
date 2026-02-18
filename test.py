from robot_env import RobotEnv
import numpy as np

env = RobotEnv()

obs, _ = env.reset()

while True:

    left = float(input("left speed: "))
    right = float(input("right speed: "))

    for _ in range(200):
        env.step([left, right])
