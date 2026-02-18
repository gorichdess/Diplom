import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
from enum import IntEnum

class TerrainType(IntEnum):
    EMPTY = 0
    ROUGH = 1
    MOUNTAIN = 2
    SWAMP = 3
    START = 4
    GOAL = 5

class RobotEnv(gym.Env):
    def __init__(self,size=20, render=True):
        super(RobotEnv, self).__init__()

        self.size = size  # размер карты
        self.grid = None
        self.start = None
        self.goal = None

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # Connect to PyBullet
        if render:
            self.physics_client = p.connect(p.GUI)
            # отключаем слои GUI
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Генерируем логическую карту
        self.generate_random_grid()

        # Генерируем плавный рельеф на основе карты
        heightfieldData = self.generate_terrain_from_grid().flatten()

        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[0.5, 0.5, 1],  # размер клетки XY и масштаб Z
            heightfieldTextureScaling=20,
            heightfieldData=heightfieldData,
            numHeightfieldRows=self.size,
            numHeightfieldColumns=self.size
        )

        self.terrain = p.createMultiBody(0, terrainShape)

        # Ставим робота на старт
        start_x = self.start[0]*0.5
        start_y = self.start[1]*0.5
        start_h = heightfieldData[self.start[0]*self.size + self.start[1]]  # высота на клетке старта
        self.robot_id = p.loadURDF("husky/husky.urdf", [start_x, start_y, start_h + 0.2])


    # Этап 1: логическая карта
    def generate_random_grid(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                r = random.random()
                if r < 0.1:
                    self.grid[i,j] = TerrainType.ROUGH.value
                elif r < 0.15:
                    self.grid[i,j] = TerrainType.MOUNTAIN.value
                elif r < 0.2:
                    self.grid[i,j] = TerrainType.SWAMP.value

        if self.start is None:
            self.start = (0,0)
        if self.goal is None:
            self.goal = (self.size-1, self.size-1)

        self.grid[self.start[0], self.start[1]] = TerrainType.START.value
        self.grid[self.goal[0], self.goal[1]] = TerrainType.GOAL.value

    # Этап 2: плавный рельеф
    def generate_terrain_from_grid(self):
        base_height = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i,j]
                if cell == TerrainType.MOUNTAIN.value:
                    base_height[i,j] = 1.3
                elif cell == TerrainType.ROUGH.value:
                    base_height[i,j] = 0.6
                elif cell == TerrainType.SWAMP.value:
                    base_height[i,j] = -0.4
                else:
                    base_height[i,j] = 0.0

        # Плавное сглаживание (переходы между клетками)
        smooth_height = base_height.copy()
        for _ in range(1):  # повторяем сглаживание несколько раз
            smooth_height[1:-1,1:-1] = (
            2*smooth_height[1:-1,1:-1] +  # основная клетка в 2 раза важнее
            smooth_height[2:,1:-1] +
            smooth_height[:-2,1:-1] +
            smooth_height[1:-1,2:] +
            smooth_height[1:-1,:-2]
        ) / 6.0


        # Добавляем небольшой шум для неровностей
        smooth_height += 0.05 * np.random.randn(self.size, self.size)

        return smooth_height

    # Стандартные методы Gym
    def reset(self):
        start_x = self.start[0]*0.5
        start_y = self.start[1]*0.5
        # высота на старте
        h = np.max([p.getBasePositionAndOrientation(self.robot_id)[0][2], 0.2])
        p.resetBasePositionAndOrientation(self.robot_id, [start_x, start_y, h], [0,0,0,1])
        obs = self._get_observation()
        return obs, {}

    def step(self, action):

        left_speed = action[0]
        right_speed = action[1]

        p.setJointMotorControl2(self.robot_id, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=100)
        p.setJointMotorControl2(self.robot_id, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=100)
        p.setJointMotorControl2(self.robot_id, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=100)
        p.setJointMotorControl2(self.robot_id, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=100)

        p.stepSimulation()
        time.sleep(1/120)
        obs = self._get_observation()
        reward = -np.linalg.norm(obs[:2]) 
        terminated = False
        truncated = False
       
        return obs, reward, terminated, truncated, {}


    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos + orn)