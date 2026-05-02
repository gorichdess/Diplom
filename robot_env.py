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

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # 14 base features + 8 terrain heights
        self.obs_dim = 14 + 8
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        self.size = 64
        self.scale = 0.35
        world = self.size * self.scale
        self.half = world / 2

        # ----- Define safe flat zones ----------
        self.flat_size = 15          # first 15×15 cells are flat
        self.smooth_width = 3        # cells over which we ramp from flat to terrain

        half_extent = (self.size - 1) * self.scale / 2
        start_idx = self.flat_size // 2
        goal_idx = self.size - 1 - start_idx

        self.start_xy = np.array([
            -half_extent + start_idx * self.scale,
            -half_extent + start_idx * self.scale
        ])
        self.goal_xy = np.array([
            -half_extent + goal_idx * self.scale,
            -half_extent + goal_idx * self.scale
        ])

        self.robot = None          # will be created in reset
        self.goal_marker = None
        self.steps = 0
        self.prev_dist = 0

        # Stuck detection (still useful)
        self.stuck_counter = 0
        self.stuck_threshold = 0.02
        self.max_stuck_steps = 50

    # ------------------------------------------------------------------
    #  Utility: terrain height under (x,y)
    # ------------------------------------------------------------------
    def _get_terrain_height(self, x, y):
        start = [x, y, 3.0]
        end   = [x, y, -0.5]
        result = p.rayTestBatch([start], [end], numThreads=0)
        if result[0][0] != -1:
            return result[0][3][2]
        return 0.0

    # ------------------------------------------------------------------
    #  Terrain creation (unchanged)
    # ------------------------------------------------------------------
    def _create_terrain(self):
        z = np.random.randn(self.size, self.size)
        z = scipy.ndimage.gaussian_filter(z, sigma=2.0)
        x = np.linspace(0, 3 * np.pi, self.size)
        y = np.linspace(0, 3 * np.pi, self.size)
        xx, yy = np.meshgrid(x, y)
        z += 0.4 * np.sin(xx) * np.cos(yy)
        z -= z.min()
        z /= (z.max() + 1e-8)
        z *= 2.2

        raw_z = z.copy()
        flat_h = 0.4  # высота платформ

        # ──── Start zone (top-left corner) ────
        # Ядро плоской зоны
        z[0:self.flat_size, 0:self.flat_size] = flat_h
        
        # Переходная зона вокруг старта
        for i in range(self.flat_size + self.smooth_width):
            for j in range(self.flat_size + self.smooth_width):
                # Пропускаем ядро (уже flat_h)
                if i < self.flat_size and j < self.flat_size:
                    continue
                # Расстояние от края плоской зоны (может быть 0 для прилегающих клеток)
                di = max(0, i - self.flat_size + 1)
                dj = max(0, j - self.flat_size + 1)
                t = min(1.0, max(di, dj) / self.smooth_width)
                z[i, j] = (1 - t) * flat_h + t * raw_z[i, j]

        # ──── Goal zone (bottom-right corner) ────
        goal_start = self.size - self.flat_size  # индекс начала плоского ядра финиша
        trans_start = self.size - self.flat_size - self.smooth_width  # индекс начала перехода
        
        # Ядро плоской зоны
        z[goal_start:, goal_start:] = flat_h
        
        # Переходная зона вокруг финиша
        for i in range(trans_start, self.size):
            for j in range(trans_start, self.size):
                # Пропускаем ядро (уже flat_h)
                if i >= goal_start and j >= goal_start:
                    continue
                # Расстояние от края плоской зоны (считаем снаружи внутрь)
                di = max(0, goal_start - i)
                dj = max(0, goal_start - j)
                t = min(1.0, max(di, dj) / self.smooth_width)
                z[i, j] = (1 - t) * flat_h + t * raw_z[i, j]

        shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1],
            heightfieldData=z.flatten(),
            numHeightfieldRows=self.size,
            numHeightfieldColumns=self.size
        )
        p.createMultiBody(0, shape)

    # ------------------------------------------------------------------
    #  Walls (unchanged)
    # ------------------------------------------------------------------
    def _create_walls(self):
        h = 2.5
        t = 0.4
        s = self.half
        def wall(x, y, sx, sy):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, h])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, h],
                                      rgbaColor=[0.6, 0.6, 0.6, 1])
            p.createMultiBody(0, col, vis, basePosition=[x, y, 0.3])
        wall(s, 0, t, s + 1)
        wall(-s, 0, t, s + 1)
        wall(0, s, s + 1, t)
        wall(0, -s, s + 1, t)

    # ------------------------------------------------------------------
    #  Goal marker (unchanged)
    # ------------------------------------------------------------------
    def _create_goal(self):
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.8, length=0.3,
                                  rgbaColor=[0, 1, 0, 1])
        self.goal_marker = p.createMultiBody(0, -1, vis,
                                             basePosition=[self.goal_xy[0],
                                                           self.goal_xy[1], 0.4])

    # ------------------------------------------------------------------
    #  Observation scaling (unchanged)
    # ------------------------------------------------------------------
    def _scale_obs(self, raw_obs):
        scaled = raw_obs.copy()
        scaled[0] = raw_obs[0] * 2.0       # to_goal_x
        scaled[1] = raw_obs[1] * 2.0       # to_goal_y
        scaled[2] = raw_obs[2] * 5.0       # distance
        scaled[3] = raw_obs[3] / np.pi     # angle_diff
        scaled[6] = raw_obs[6] / 5.0       # vel_x
        scaled[7] = raw_obs[7] / 5.0       # vel_y
        scaled[8] = raw_obs[8] / 2.0       # ang_z
        scaled[9] = raw_obs[9] / 2.0       # z
        scaled[10] = raw_obs[10] / 1.5     # roll
        scaled[11] = raw_obs[11] / 1.5     # pitch
        scaled[14:] = raw_obs[14:] / 2.0   # terrain heights
        return scaled

    # ------------------------------------------------------------------
    #  Reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self._create_terrain()
        self._create_walls()
        self._create_goal()

        # ------------------------------------------------------------------
        #  ★★★  NEW simple robot: a box with velocity control  ★★★
        # ------------------------------------------------------------------
        # Collision shape: 0.8 x 0.6 x 0.3 m (low, stable)
        col_shape = p.createCollisionShape(p.GEOM_BOX,
                                           halfExtents=[0.4, 0.3, 0.15])
        vis_shape = p.createVisualShape(p.GEOM_BOX,
                                        halfExtents=[0.4, 0.3, 0.15],
                                        rgbaColor=[0.2, 0.6, 0.9, 1])
        self.robot = p.createMultiBody(
            baseMass=1.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=[self.start_xy[0], self.start_xy[1], 0.15]  # half-height
        )
        # Slight damping to make it feel a bit more natural (optional)
        p.changeDynamics(self.robot, -1, linearDamping=0.5, angularDamping=0.5)

        self.steps = 0
        self.stuck_counter = 0
        self.prev_dist = np.linalg.norm(self.goal_xy - self.start_xy)

        for _ in range(20):
            p.stepSimulation()

        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=14, cameraYaw=45, cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0]
            )
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    #  Observation (unchanged, works for any base)
    # ------------------------------------------------------------------
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang = p.getBaseVelocity(self.robot)
        euler = p.getEulerFromQuaternion(orn)
        yaw = euler[2]
        to_goal = self.goal_xy - np.array(pos[:2])
        dist = np.linalg.norm(to_goal)
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        angle_diff = np.arctan2(np.sin(angle_to_goal - yaw),
                                np.cos(angle_to_goal - yaw))
        raw_obs = np.array([
            to_goal[0] * 0.1,
            to_goal[1] * 0.1,
            dist * 0.1,
            angle_diff,
            np.sin(yaw),
            np.cos(yaw),
            vel[0],
            vel[1],
            ang[2],
            pos[2],
            euler[0],
            euler[1],
            self.steps / 1000.0,
            1.0
        ], dtype=np.float32)

        sample_offsets = [
            (1.0,  0.0), (0.7,  0.5), (0.7, -0.5), (0.3,  0.7),
            (0.3, -0.7), (-0.3, 0.7), (-0.3,-0.7), (0.0,  0.0)
        ]
        heights = []
        for dx, dy in sample_offsets:
            wx = pos[0] + dx * np.cos(yaw) - dy * np.sin(yaw)
            wy = pos[1] + dx * np.sin(yaw) + dy * np.cos(yaw)
            h = self._get_terrain_height(wx, wy)
            heights.append(h - pos[2])
        heights = np.array(heights, dtype=np.float32)
        raw_obs = np.concatenate([raw_obs, heights])
        return self._scale_obs(raw_obs)

    # ------------------------------------------------------------------
    #  Step (velocity control instead of wheel joints)
    # ------------------------------------------------------------------
    def step(self, action):
        # ---------------------------------------------------------
        #  ★  Map action to linear + angular velocity directly  ★
        # ---------------------------------------------------------
        linear_speed  = float(action[0]) * 3.0    # m/s forward (robot's x)
        angular_speed = float(action[1]) * 2.5    # rad/s around vertical

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(orn)[2]

        forward_dir = np.array([np.cos(yaw), np.sin(yaw), 0])
        desired_lin_vel = forward_dir * linear_speed
        desired_ang_vel = np.array([0, 0, angular_speed])

        # Apply velocity directly – the simplest way to move the robot
        p.resetBaseVelocity(self.robot,
                            linearVelocity=desired_lin_vel,
                            angularVelocity=desired_ang_vel)

        # Step simulation
        for _ in range(5):
            p.stepSimulation()

        self.steps += 1
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        dist = np.linalg.norm(self.goal_xy - np.array(pos[:2]))

        # ----------- Reward (unchanged) -----------
        reward = (self.prev_dist - dist) * 1.2

        to_goal = self.goal_xy - np.array(pos[:2])
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        angle_diff = abs(np.arctan2(np.sin(angle_to_goal - yaw),
                                    np.cos(angle_to_goal - yaw)))
        reward += 2.0 * (1.0 - angle_diff / np.pi)

        if dist > 0.5:
            vel_norm = np.linalg.norm(vel[:2])
            if vel_norm > 0.01:
                vel_dir = np.array([vel[0], vel[1]]) / vel_norm
                goal_dir = to_goal / (dist + 1e-6)
                align = np.dot(vel_dir, goal_dir)
                reward += 0.1 * max(0, align) * vel_norm

        reward -= 0.2 * (roll**2 + pitch**2)
        reward += 0.02

        # ----------- Stuck detection -----------
        if self.prev_dist - dist < self.stuck_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= self.max_stuck_steps:
            reward -= 20.0
            truncated = True
            self.stuck_counter = 0
        else:
            truncated = False

        self.prev_dist = dist
        terminated = False

        # ----------- Terminal conditions (unchanged) -----------
        if dist < 1.5:
            reward += 50.0
            terminated = True
        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            reward -= 100.0
            terminated = True
        if pos[2] < -0.3:
            reward -= 100.0
            terminated = True
        lim = self.half - 0.5
        if abs(pos[0]) > lim or abs(pos[1]) > lim:
            reward -= 100.0
            terminated = True

        if self.steps >= 1000:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    def close(self):
        p.disconnect()