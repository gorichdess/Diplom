import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import random
from enum import IntEnum
import os
import sys

class SuppressAllOutput:
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_fds = [os.dup(1), os.dup(2)]
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        os.close(self.null_fd)
        os.close(self.save_fds[0])
        os.close(self.save_fds[1])

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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(55,), dtype=np.float32)

        # Connect to PyBullet
        if self.render:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.left_wheels = [2, 4]
        self.right_wheels = [3, 5]


        #Reset the environment to initialize everything
        self.reset()

    def reset(self, seed=None, options=None):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        self.generate_random_grid()
        smooth_height = self.generate_terrain_from_grid()
        self.heightfieldData = smooth_height.flatten().astype(np.float32)

        terrainShape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1],
            heightfieldData=self.heightfieldData,
            numHeightfieldRows=self.size,
            numHeightfieldColumns=self.size
        )
        
        self.terrain = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=terrainShape)
        p.changeVisualShape(self.terrain, -1, rgbaColor=[0.5, 0.5, 0.5, 1])

        total_len = self.size * self.scale
        half_len = total_len / 2.0 

        def grid_to_world(i, j):
            # (i + 0.5) чтобы попасть в центр ячейки
            x = (i + 0.5) * self.scale - half_len
            y = (j + 0.5) * self.scale - half_len
            return x, y

        start_x, start_y = grid_to_world(self.start[0], self.start[1])
        start_h = smooth_height[self.start[0], self.start[1]]
        
        with SuppressAllOutput():
            self.robot_id = p.loadURDF("husky/husky.urdf", [start_x, start_y, start_h + 0.5])

        # Camera setup
        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=10.0, 
                cameraYaw=45, 
                cameraPitch=-30, 
                cameraTargetPosition=[start_x, start_y, start_h]
            )

        goal_x, goal_y = grid_to_world(self.goal[0], self.goal[1])
        goal_z = smooth_height[self.goal[0], self.goal[1]] + 0.5
        self.goal_pos = np.array([goal_x, goal_y])

        visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=[1, 0, 0, 1])
        self.goal_marker = p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape, 
                                            basePosition=[goal_x, goal_y, goal_z])

        self.prev_dist = np.linalg.norm(np.array([start_x, start_y]) - self.goal_pos)
        self.step_count = 0
        
        if self.render:
            p.resetDebugVisualizerCamera(10, 45, -30, [0, 0, 0])

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

        margin = 2
        self.start = (margin, margin)
        self.goal = (self.size - margin - 1, self.size - margin - 1)

        self.grid[self.start[0], self.start[1]] = TerrainType.START.value
        self.grid[self.goal[0], self.goal[1]] = TerrainType.GOAL.value

    #Smoothly generate terrain heights based on the grid types, with transitions between cells
    def generate_terrain_from_grid(self):
        base_height = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                cell = self.grid[i,j]
                if cell == TerrainType.MOUNTAIN.value:
                    base_height[i,j] = 1.7
                elif cell == TerrainType.ROUGH.value:
                    base_height[i,j] = 0.5
                elif cell == TerrainType.SWAMP.value:
                    base_height[i,j] = -0.6
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

        for _ in range(20):
            p.stepSimulation()

        obs = self._get_observation()
        reward, done = self.compute_reward()

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = p.getBaseVelocity(self.robot_id)
        
        terminated = done
        truncated = False

        current_speed = np.linalg.norm(lin_vel[:2])
        
        if current_speed < 0.01:
            self.stuck_steps = getattr(self, 'stuck_steps', 0) + 1
        else:
            self.stuck_steps = 0

        if self.stuck_steps > 100:
            # print("Robot is stuck!")
            terminated = True
            reward -= 50  

        _, orn = p.getBasePositionAndOrientation(self.robot_id)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        if abs(roll) > 1.4 or abs(pitch) > 1.4:
            # print("Robot flipped!")
            terminated = True
            reward -= 150 

        if pos[2] < -2: 
            terminated = True
            reward -= 200

        boundary = (self.size * self.scale) / 2.0
        if abs(pos[0]) > boundary + 0.5 or abs(pos[1]) > boundary + 0.5:
            terminated = True
            reward -= 100

        self.step_count += 1
        if self.step_count >= 2000:
            truncated = True

        return obs, reward, terminated, truncated, {}

    
    def apply_action(self, action):
        max_velocity = 5.0 
        left = action[0] * max_velocity
        right = action[1] * max_velocity

        for wheel in self.left_wheels:
            p.setJointMotorControl2(self.robot_id, wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=left,
                                    force=100) 

        for wheel in self.right_wheels:
            p.setJointMotorControl2(self.robot_id, wheel,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=right,
                                    force=100)
            
    def compute_reward(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = p.getBaseVelocity(self.robot_id)
        
        current_dist = np.linalg.norm(np.array(pos[:2]) - self.goal_pos)
        progress = self.prev_dist - current_dist
        self.prev_dist = current_dist

        reward = progress * 100.0 

        goal_vec = self.goal_pos - np.array(pos[:2])
        goal_vec = goal_vec / (np.linalg.norm(goal_vec) + 1e-6)
        velocity_towards_goal = np.dot(np.array(lin_vel[:2]), goal_vec)
        
        reward += velocity_towards_goal * 2.0

        speed = np.linalg.norm(lin_vel[:2])
        if speed < 0.05:
            reward -= 0.1 

        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        if abs(roll) > 0.6 or abs(pitch) > 0.6:
            reward -= 1.0 
        
        done = False
        if abs(roll) > 1.4 or abs(pitch) > 1.4:
            reward -= 50.0

        if current_dist < 0.8:
            reward += 1000.0
            done = True

        return reward, done


    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        x, y = pos[0], pos[1]
        
        _, _, yaw = p.getEulerFromQuaternion(orn)
        
        goal_vec = self.goal_pos - np.array([x, y])
        dist = np.linalg.norm(goal_vec)
        
        abs_target_angle = np.arctan2(goal_vec[1], goal_vec[0])
        
        rel_target_angle = abs_target_angle - yaw
        
        target_sin = np.sin(rel_target_angle)
        target_cos = np.cos(rel_target_angle)

        local_patch = self.get_local_heightmap(x, y)

        obs = np.concatenate([
            [x / 10.0, y / 10.0, dist / 10.0], 
            [target_sin, target_cos],          
            [yaw / np.pi],                     
            local_patch.flatten()              
        ]).astype(np.float32)

        return obs
    
    def get_local_heightmap(self, x, y, patch_size=7):

        total_len = self.size * self.scale
        half_len = total_len / 2.0
        
        cx = int((x + half_len) / self.scale)
        cy = int((y + half_len) / self.scale)

        half = patch_size // 2

        patch = np.zeros((patch_size, patch_size))

        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        robot_z = pos[2]

        patch = np.zeros((patch_size, patch_size))

        for i in range(-half, half+1):
            for j in range(-half, half+1):
                gx = cx + i
                gy = cy + j
                if 0 <= gx < self.size and 0 <= gy < self.size:
                    idx = gx * self.size + gy
                    patch[i+half, j+half] = self.heightfieldData[idx] - robot_z
                else:
                    patch[i+half, j+half] = -2.0 


        return patch
