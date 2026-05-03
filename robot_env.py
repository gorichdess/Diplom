import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import scipy.ndimage


class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render=False, difficulty=0.3):
        """
        difficulty: 0.0 = flat, 1.0 = full roughness (curriculum parameter)
        """
        super().__init__()
        self.render = render
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # action: [throttle, steer] in [-1, 1]
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

        self.flat_size = 15
        self.smooth_width = 3
        self.flat_h = 0.4

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

        self.difficulty = difficulty

        self.robot = None
        self.goal_marker = None
        self.steps = 0
        self.prev_dist = 0

        # Stuck detection
        self.stuck_counter = 0
        self.stuck_threshold = 0.02
        self.max_stuck_steps = 50

        self.wheel_joints = []
        self.steer_joints = []

        self.max_wheel_speed = 25.0   # reduced for stability
        self.max_motor_force = 300.0
        self.max_steer_angle = 0.5
        self.max_steer_force = 80.0

        self.physics_steps_per_env_step = 20

    # -------------------------- safety helpers --------------------------
    def _safe_obs(self, obs, clip=50.0):
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        return np.clip(obs, -clip, clip)

    def _get_terrain_height(self, x, y):
        if not np.isfinite(x) or not np.isfinite(y):
            return 0.0
        start = [float(x), float(y), 3.0]
        end = [float(x), float(y), -0.5]
        result = p.rayTestBatch([start], [end], numThreads=0)
        if result and result[0][0] != -1:
            h = float(result[0][3][2])
            return h if np.isfinite(h) else 0.0
        return 0.0

    # -------------------------- terrain --------------------------
    def _create_terrain(self):
        z = np.random.randn(self.size, self.size)
        z = scipy.ndimage.gaussian_filter(z, sigma=2.0)
        x = np.linspace(0, 3 * np.pi, self.size)
        y = np.linspace(0, 3 * np.pi, self.size)
        xx, yy = np.meshgrid(x, y)
        z += 0.4 * np.sin(xx) * np.cos(yy)

        z -= z.min()
        z /= (z.max() + 1e-8)
        z *= 1.4 * self.difficulty    # curriculum scaling

        raw_z = z.copy()

        # Start flat core
        z[0:self.flat_size, 0:self.flat_size] = self.flat_h

        # Start transition
        for i in range(self.flat_size + self.smooth_width):
            for j in range(self.flat_size + self.smooth_width):
                if i < self.flat_size and j < self.flat_size:
                    continue
                di = max(0, i - self.flat_size + 1)
                dj = max(0, j - self.flat_size + 1)
                t = min(1.0, max(di, dj) / self.smooth_width)
                z[i, j] = (1 - t) * self.flat_h + t * raw_z[i, j]

        # Goal flat core
        goal_start = self.size - self.flat_size
        trans_start = self.size - self.flat_size - self.smooth_width
        z[goal_start:, goal_start:] = self.flat_h

        # Goal transition
        for i in range(trans_start, self.size):
            for j in range(trans_start, self.size):
                if i >= goal_start and j >= goal_start:
                    continue
                di = max(0, goal_start - i)
                dj = max(0, goal_start - j)
                t = min(1.0, max(di, dj) / self.smooth_width)
                z[i, j] = (1 - t) * self.flat_h + t * raw_z[i, j]

        shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1],
            heightfieldData=z.flatten(),
            numHeightfieldRows=self.size,
            numHeightfieldColumns=self.size
        )
        terrain_id = p.createMultiBody(0, shape)
        p.changeDynamics(terrain_id, -1, lateralFriction=1.6, restitution=0.0)
        p.setPhysicsEngineParameter(enableConeFriction=1)

    # -------------------------- walls --------------------------
    def _create_walls(self):
        h = 2.5
        t = 0.4
        s = self.half

        def wall(x, y, sx, sy):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, h])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, h],
                                      rgbaColor=[0.6, 0.6, 0.6, 1])
            wid = p.createMultiBody(0, col, vis, basePosition=[x, y, 0.3])
            p.changeDynamics(wid, -1, lateralFriction=1.5, restitution=0.0)

        wall(s, 0, t, s + 1)
        wall(-s, 0, t, s + 1)
        wall(0, s, s + 1, t)
        wall(0, -s, s + 1, t)

    # -------------------------- goal --------------------------
    def _create_goal(self):
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.8, length=0.3,
                                  rgbaColor=[0, 1, 0, 1])
        self.goal_marker = p.createMultiBody(
            0, -1, vis,
            basePosition=[self.goal_xy[0], self.goal_xy[1], self.flat_h + 0.15]
        )

    # -------------------------- obs scaling --------------------------
    def _scale_obs(self, raw_obs):
        scaled = raw_obs.copy()
        scaled[0] = raw_obs[0] * 2.0
        scaled[1] = raw_obs[1] * 2.0
        scaled[2] = raw_obs[2] * 5.0
        scaled[3] = raw_obs[3] / np.pi
        scaled[6] = raw_obs[6] / 5.0
        scaled[7] = raw_obs[7] / 5.0
        scaled[8] = raw_obs[8] / 2.0
        scaled[9] = raw_obs[9] / 2.0
        scaled[10] = raw_obs[10] / 1.5
        scaled[11] = raw_obs[11] / 1.5
        scaled[14:] = raw_obs[14:] / 2.0
        return scaled

    # -------------------------- joint detection --------------------------
    def _detect_joints(self):
        self.wheel_joints = []
        self.steer_joints = []
        num_joints = p.getNumJoints(self.robot)

        for j in range(num_joints):
            info = p.getJointInfo(self.robot, j)
            name = info[1].decode("utf-8", errors="ignore").lower()
            jtype = info[2]

            if jtype != p.JOINT_REVOLUTE:
                continue

            if "steer" in name:
                self.steer_joints.append(j)
            elif "wheel" in name:
                self.wheel_joints.append(j)

        if len(self.wheel_joints) < 2:
            candidates = []
            for j in range(num_joints):
                info = p.getJointInfo(self.robot, j)
                name = info[1].decode("utf-8", errors="ignore").lower()
                jtype = info[2]
                if jtype == p.JOINT_REVOLUTE and ("steer" not in name):
                    candidates.append(j)
            self.wheel_joints = candidates[:4]

    # -------------------------- reset --------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=150,
            solverResidualThreshold=1e-7,
            numSubSteps=2,
            enableConeFriction=1
        )

        self._create_terrain()
        self._create_walls()
        self._create_goal()

        spawn_z = self.flat_h + 0.35

        self.robot = p.loadURDF(
            "racecar/racecar.urdf",
            basePosition=[self.start_xy[0], self.start_xy[1], spawn_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            flags=p.URDF_USE_INERTIA_FROM_FILE
        )

        num_joints = p.getNumJoints(self.robot)
        for link in range(-1, num_joints):
            p.changeDynamics(
                self.robot, link,
                lateralFriction=2.0,
                spinningFriction=0.10,
                rollingFriction=0.06,
                restitution=0.0,
                linearDamping=0.04,
                angularDamping=0.10
            )

        self._detect_joints()

        for j in self.wheel_joints:
            p.changeDynamics(
                self.robot, j,
                lateralFriction=1.6,
                rollingFriction=0.0,
                spinningFriction=0.0,
                restitution=0.0
            )

        if len(self.wheel_joints) > 0:
            p.setJointMotorControlArray(
                self.robot,
                self.wheel_joints,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[0.0] * len(self.wheel_joints),
                forces=[self.max_motor_force] * len(self.wheel_joints)
            )

        if len(self.steer_joints) > 0:
            p.setJointMotorControlArray(
                self.robot,
                self.steer_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0.0] * len(self.steer_joints),
                forces=[self.max_steer_force] * len(self.steer_joints)
            )

        self.steps = 0
        self.stuck_counter = 0
        self.prev_dist = np.linalg.norm(self.goal_xy - self.start_xy)

        for _ in range(120):
            p.stepSimulation()

        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=14, cameraYaw=45, cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0]
            )

        return self._get_obs(), {}

    # -------------------------- observation --------------------------
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
            (0.3, -0.7), (-0.3, 0.7), (-0.3, -0.7), (0.0,  0.0)
        ]
        heights = []
        for dx, dy in sample_offsets:
            wx = pos[0] + dx * np.cos(yaw) - dy * np.sin(yaw)
            wy = pos[1] + dx * np.sin(yaw) + dy * np.cos(yaw)
            h = self._get_terrain_height(wx, wy)
            heights.append(h - pos[2])

        heights = np.array(heights, dtype=np.float32)
        raw_obs = np.concatenate([raw_obs, heights])

        obs = self._scale_obs(raw_obs)
        return self._safe_obs(obs, clip=50.0)

    # -------------------------- step (improved reward) --------------------------
    def step(self, action):
        throttle = float(action[0])
        steer = float(action[1])

        steer_angle = np.clip(steer, -1.0, 1.0) * self.max_steer_angle
        if len(self.steer_joints) > 0:
            p.setJointMotorControlArray(
                self.robot,
                self.steer_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[steer_angle] * len(self.steer_joints),
                forces=[self.max_steer_force] * len(self.steer_joints)
            )

        target_w = np.clip(throttle, -1.0, 1.0) * self.max_wheel_speed
        if len(self.wheel_joints) > 0:
            p.setJointMotorControlArray(
                self.robot,
                self.wheel_joints,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[target_w] * len(self.wheel_joints),
                forces=[self.max_motor_force] * len(self.wheel_joints)
            )

        for _ in range(self.physics_steps_per_env_step):
            p.stepSimulation()

        self.steps += 1

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        dist = np.linalg.norm(self.goal_xy - np.array(pos[:2]))

        # --- Improved reward signal ---
        # 1) Distance progress
        reward = (self.prev_dist - dist) * 2.5

        # 2) Wall proximity penalty (discourage wall-hugging)
        lim = self.half - 0.5
        wall_dist = min(lim - abs(pos[0]), lim - abs(pos[1]))
        wall_penalty = 0.0
        if wall_dist < 1.5:   # meters from boundary
            wall_penalty = -0.5 * (1.5 - wall_dist)

        # 3) Speed penalty (prevent flipping)
        speed = np.linalg.norm(vel[:2])
        speed_penalty = -0.05 * speed

        # 4) Gentle heading bonus (encourage moving toward the goal)
        to_goal = self.goal_xy - np.array(pos[:2])
        angle_to_goal = np.arctan2(to_goal[1], to_goal[0])
        angle_diff = np.arctan2(np.sin(angle_to_goal - yaw),
                                np.cos(angle_to_goal - yaw))
        forward_speed = vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw)
        heading_bonus = 0.0
        if forward_speed > 0.1:   # only when moving forward
            heading_bonus = 0.1 * forward_speed * np.cos(angle_diff)

        reward += wall_penalty + speed_penalty + heading_bonus

        # Living cost
        reward -= 0.03

        # Goal reached
        if dist < 1.2:
            reward += 150.0
            terminated = True
        else:
            terminated = False

        # Terminal penalties
        if abs(roll) > 1.5 or abs(pitch) > 1.5:
            reward -= 50.0
            terminated = True

        if pos[2] < -1.0:
            reward -= 100.0
            terminated = True

        if abs(pos[0]) > lim or abs(pos[1]) > lim:
            reward -= 100.0
            terminated = True

        # Stuck detection
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

        if self.steps >= 1000:
            truncated = True

        return self._get_obs(), float(reward), terminated, truncated, {}

    def close(self):
        p.disconnect()