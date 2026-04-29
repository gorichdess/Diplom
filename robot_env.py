import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import scipy.ndimage


class RobotEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False):
        super().__init__()

        self.render = render

        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        if render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # ====================================================
        # ACTION SPACE ([-1, 1])
        # ====================================================
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

        # ====================================================
        # OBS SPACE
        # ====================================================
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        # ====================================================
        # TERRAIN SETTINGS
        # ====================================================
        self.size = 64
        self.scale = 0.35

        world = self.size * self.scale
        self.half = world / 2

        self.start_xy = np.array([-self.half + 3.5, -self.half + 3.5])
        self.goal_xy = np.array([self.half - 3.5, self.half - 3.5])

        # ====================================================
        # OBS NORMALIZATION
        # ====================================================
        self.obs_mean = np.zeros(14)
        self.obs_var = np.ones(14)
        self.obs_count = 1e-4

        # runtime
        self.robot = None
        self.goal_marker = None
        self.steps = 0
        self.prev_dist = 0

    # ====================================================
    # OBS NORMALIZATION
    # ====================================================
    def normalize_obs(self, obs):
        self.obs_count += 1
        delta = obs - self.obs_mean
        self.obs_mean += delta / self.obs_count
        self.obs_var += delta * (obs - self.obs_mean)

        std = np.sqrt(self.obs_var / self.obs_count + 1e-8)
        return (obs - self.obs_mean) / std

    # ====================================================
    # TERRAIN
    # ====================================================
    def _create_terrain(self):

        z = np.random.randn(self.size, self.size)
        z = scipy.ndimage.gaussian_filter(z, sigma=2.5)

        x = np.linspace(0, 3*np.pi, self.size)
        y = np.linspace(0, 3*np.pi, self.size)
        xx, yy = np.meshgrid(x, y)

        z += 0.35 * np.sin(xx) * np.cos(yy)

        z -= z.min()
        z /= (z.max() + 1e-8)

        # smoother terrain (important for learning)
        z *= 0.25

        # SAFE AREAS
        z[0:10, 0:10] = 0.03
        z[-10:, -10:] = 0.03

        shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1],
            heightfieldData=z.flatten(),
            numHeightfieldRows=self.size,
            numHeightfieldColumns=self.size
        )

        p.createMultiBody(0, shape)

    # ====================================================
    # WALLS (anti falling exploit)
    # ====================================================
    def _create_walls(self):

        h = 2
        t = 0.4
        s = self.half

        def wall(x, y, sx, sy):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, h])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, h],
                                      rgbaColor=[0.6, 0.6, 0.6, 1])

            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, h]
            )

        wall(s, 0, t, s)
        wall(-s, 0, t, s)
        wall(0, s, s, t)
        wall(0, -s, s, t)

    # ====================================================
    # GOAL VISUAL
    # ====================================================
    def _create_goal(self):

        vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.8,
            length=0.3,
            rgbaColor=[0, 1, 0, 1]
        )

        self.goal_marker = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=vis,
            basePosition=[self.goal_xy[0], self.goal_xy[1], 0.4]
        )

    # ====================================================
    # OBSERVATION
    # ====================================================
    def _get_obs(self):

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang = p.getBaseVelocity(self.robot)

        yaw = p.getEulerFromQuaternion(orn)[2]

        to_goal = self.goal_xy - np.array(pos[:2])
        dist = np.linalg.norm(to_goal)

        obs = np.array([
            pos[0],
            pos[1],
            vel[0],
            vel[1],
            ang[2],
            np.sin(yaw),
            np.cos(yaw),
            to_goal[0],
            to_goal[1],
            dist,
            pos[2],
            self.prev_dist,
            self.steps / 1000.0,
            1.0
        ], dtype=np.float32)

        return self.normalize_obs(obs)

    # ====================================================
    # RESET
    # ====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        self._create_terrain()
        self._create_walls()
        self._create_goal()

        self.robot = p.loadURDF(
            "husky/husky.urdf",
            [self.start_xy[0], self.start_xy[1], 0.6]
        )

        self.steps = 0
        self.prev_dist = np.linalg.norm(self.goal_xy - self.start_xy)

        for _ in range(20):
            p.stepSimulation()

        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=14,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0]
            )

        return self._get_obs(), {}

    # ====================================================
    # STEP
    # ====================================================
    def step(self, action):

        # IMPORTANT: stable scaling
        throttle = float(action[0]) * 3.0
        steer = float(action[1]) * 1.0

        left = throttle - steer
        right = throttle + steer

        # wheel control
        for w in [2, 3]:
            p.setJointMotorControl2(self.robot, w,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=left,
                                    force=160)

        for w in [4, 5]:
            p.setJointMotorControl2(self.robot, w,
                                    p.VELOCITY_CONTROL,
                                    targetVelocity=right,
                                    force=160)

        for _ in range(5):
            p.stepSimulation()

        self.steps += 1

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, _ = p.getBaseVelocity(self.robot)

        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        dist = np.linalg.norm(self.goal_xy - np.array(pos[:2]))

        # ====================================================
        # REWARD (IMPORTANT FIX)
        # ====================================================
        progress = self.prev_dist - dist
        speed = np.linalg.norm(vel[:2])

        reward = 20.0 * progress

        # encourage motion only if useful
        if progress > 0:
            reward += 0.05 * speed

        # discourage wasting energy
        reward -= 0.01 * (abs(left) + abs(right))

        # small time penalty
        reward -= 0.02

        self.prev_dist = dist

        terminated = False
        truncated = False

        # ====================================================
        # SUCCESS
        # ====================================================
        if dist < 1.5:
            reward += 800
            terminated = True

        # ====================================================
        # FLIP
        # ====================================================
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            reward -= 200
            terminated = True

        # ====================================================
        # FALL
        # ====================================================
        if pos[2] < -1:
            reward -= 300
            terminated = True

        # ====================================================
        # OUT OF BOUNDS
        # ====================================================
        lim = self.half - 0.5
        if abs(pos[0]) > lim or abs(pos[1]) > lim:
            reward -= 300
            terminated = True

        # ====================================================
        # TIME LIMIT
        # ====================================================
        if self.steps >= 900:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    # ====================================================
    def close(self):
        p.disconnect()