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
    scale = 0.65  # size of one cell in meters

    def __init__(self, size=20, render=True):
        super(RobotEnv, self).__init__()
        self.size = size
        self.render = render
        
        self.grid = None
        self.start = None
        self.goal = None
        self.terrain = None
        self.robot_id = None

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        # Connect to PyBullet
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        #Reset the environment to initialize everything
        self.reset()

    def reset(self, seed=None, options=None):
        #Full reset of the environment
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        #Generate terrain
        self.generate_random_grid()
        smooth_height = self.generate_terrain_from_grid()
        self.heightfieldData = smooth_height.T.flatten().astype(np.float32)

        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1],
            heightfieldData=self.heightfieldData,
            numHeightfieldRows=self.size,
            numHeightfieldColumns=self.size
        )
        
        total_length = self.size * self.scale
        # Center the terrain at (0,0) in world coordinates
        offset = total_length / 2.0 - (self.scale / 2.0)

        self.terrain = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrainShape,
            basePosition=[offset, offset, 0]
        )
        p.changeVisualShape(self.terrain, -1, rgbaColor=[0.5, 0.5, 0.5, 1], specularColor=[1, 1, 1])

        # Robot spawn
        start_x = self.start[0] * self.scale + offset # offset нужен, чтобы центр карты был в (0,0) мира
        start_y = self.start[1] * self.scale + offset
        # Height
        start_h = self.heightfieldData[self.start[0] * self.size + self.start[1]]
        
        self.robot_id = p.loadURDF("husky/husky.urdf", [start_x, start_y, start_h + 0.5])

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        
        self.left_wheels = [2, 4]
        self.right_wheels = [3, 5]
        
        #Goal position in world coordinates
        self.goal_pos = (self.goal[0] * self.scale, self.goal[1] * self.scale)
        self.prev_dist = np.linalg.norm(np.array(pos[:2]) - self.goal_pos)

        # Camera setup
        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=10.0, 
                cameraYaw=45, 
                cameraPitch=-30, 
                cameraTargetPosition=[start_x, start_y, start_h]
            )

        return self._get_observation(), {}


    #Generate random grid with different terrain types
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

    #Smoothly generate terrain heights based on the grid types, with transitions between cells
    def generate_terrain_from_grid(self):
        base_height = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i,j]
                if cell == TerrainType.MOUNTAIN.value:
                    base_height[i,j] = 1.5
                elif cell == TerrainType.ROUGH.value:
                    base_height[i,j] = 0.5
                elif cell == TerrainType.SWAMP.value:
                    base_height[i,j] = -0.5
                else:
                    base_height[i,j] = 0.0

        #Smooth the heightmap by averaging with neighbors to create gradual transitions
        smooth_height = base_height.copy()
        for _ in range(1):  # repeat smoothing multiple times if needed
            smooth_height[1:-1,1:-1] = (
            2*smooth_height[1:-1,1:-1] +  # Main cell has more weight
            smooth_height[2:,1:-1] +
            smooth_height[:-2,1:-1] +
            smooth_height[1:-1,2:] +
            smooth_height[1:-1,:-2]
        ) / 6.0


        #Add some random noise to make it more natural
        smooth_height += 0.05 * np.random.randn(self.size, self.size)

        return smooth_height

    def step(self, action):
        self.apply_action(action)

        for _ in range(10):
            p.stepSimulation()

        obs = self._get_observation()
        reward, done = self.compute_reward()

        terminated = done
        truncated = False
        return obs, reward, terminated, truncated, {}

    
    def apply_action(self, action):

        left = action[0]
        right = action[1]

        for wheel in self.left_wheels:
            p.setJointMotorControl2(self.robot_id, wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=left,
                                    force=20)

        for wheel in self.right_wheels:
            p.setJointMotorControl2(self.robot_id, wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=right,
                                    force=20)
            
    def compute_reward(self):

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)

        dist = np.linalg.norm(np.array(pos[:2]) - self.goal_pos)

        reward = (self.prev_dist - dist) * 1000   # приблизился → плюс

        self.prev_dist = dist

        if dist < 0.5:
            reward += 100
            done = True
        else:
            done = False

        return reward, done


    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)

        x, y = pos[0], pos[1]

        goal_vec = np.array(self.goal_pos) - np.array([x, y])
        dist = np.linalg.norm(goal_vec)

        local_patch = self.get_local_heightmap(x, y)

        obs = np.concatenate([
            [x, y, dist],
            local_patch.flatten()
        ])

        return obs
    
    def get_local_heightmap(self, x, y, patch_size=3):
    
        cell_size = 0.65

        cx = int(x / cell_size)
        cy = int(y / cell_size)

        half = patch_size // 2

        patch = np.zeros((patch_size, patch_size))

        for i in range(-half, half+1):
            for j in range(-half, half+1):

                gx = cx + i
                gy = cy + j

                if 0 <= gx < self.size and 0 <= gy < self.size:
                    idx = gx * self.size + gy
                    patch[i+half, j+half] = self.heightfieldData[idx]

        return patch
