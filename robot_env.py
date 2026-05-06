import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import scipy.ndimage
import heapq


class RobotEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render: bool = False, difficulty: float = 0.3, draw_path: bool = True):
        super().__init__()

        self.render = render
        self.difficulty = float(np.clip(difficulty, 0.0, 1.0))
        self.draw_path = bool(draw_path)

        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setRealTimeSimulation(0)
        if render:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        # Action: [throttle, steer] in [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Create a denser local grid + some long-range "look ahead"
        self.sample_offsets = []
    
        # Rows at 0.2m, 0.5m (Close), 1.5m (Medium), and 3.5m (Far)
        for x in [0.2, 0.5, 1.5, 3.5]: 
            # Spread Y based on distance: wider vision for further rows
            y_spread = 0.5 if x < 1.0 else 1.2
            for y in np.linspace(-y_spread, y_spread, 5):
                self.sample_offsets.append((x, float(y)))

        self.sample_offsets.append((0.0, 0.0))

        # Obs: 14 base + 8 terrain samples
        self.obs_dim = 14 + len(self.sample_offsets)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Map / heightfield settings
        self.size = 64
        self.scale = 0.35
        world = self.size * self.scale
        self.half = world / 2

        self.flat_size = 12
        self.smooth_width = 3
        self.flat_h = 0.4

        half_extent = (self.size - 1) * self.scale / 2
        start_idx = self.flat_size // 2
        goal_idx = self.size - 1 - start_idx

        self.start_xy = np.array(
            [-half_extent + start_idx * self.scale, -half_extent + start_idx * self.scale],
            dtype=np.float32,
        )
        self.goal_xy = np.array(
            [-half_extent + goal_idx * self.scale, -half_extent + goal_idx * self.scale],
            dtype=np.float32,
        )

        # Robot + episode state
        self.robot = None
        self.goal_marker = None
        self.terrain_id = None

        self.steps = 0
        self.prev_dist = 0.0

        # Stuck logic
        self.stuck_counter = 0
        self.stuck_threshold = 0.005
        self.max_stuck_steps = 500

        # Joint lists
        self.wheel_joints = []
        self.steer_joints = []

        # Controls
        self.max_wheel_speed = 20.0
        self.max_motor_force = 300.0
        self.max_steer_angle = 0.35
        self.max_steer_force = 80.0

        self.physics_steps_per_env_step = 10

        # Terrain debug/state
        self.heightmap = None
        self.path_cells = None
        self._debug_ids = []
        self.terrain_obstacles = []

    # -------------------------- utils --------------------------
    def _safe_obs(self, obs, clip: float = 50.0):
        obs = np.asarray(obs, dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=clip, neginf=-clip)
        return np.clip(obs, -clip, clip)

    def _get_terrain_height(self, x: float, y: float) -> float:
        if not np.isfinite(x) or not np.isfinite(y):
            return 0.0
        start = [float(x), float(y), 3.0]
        end = [float(x), float(y), -2.0]
        res = p.rayTestBatch([start], [end], numThreads=0)
        if res and res[0][0] != -1:
            h = float(res[0][3][2])
            return h if np.isfinite(h) else 0.0
        return 0.0

    # -------------------------- debug draw --------------------------
    def _clear_debug(self):
        if not self.render:
            return
        for did in self._debug_ids:
            try:
                p.removeUserDebugItem(did)
            except Exception:
                pass
        self._debug_ids = []

    def _debug_draw_path(self, path_cells, life=0):
        if not self.render or not self.draw_path or path_cells is None or len(path_cells) < 2:
            return

        self._clear_debug()
        half_extent = (self.size - 1) * self.scale / 2.0

        def cell_to_world(ix, iy):
            x = -half_extent + ix * self.scale
            y = -half_extent + iy * self.scale

            z = self._get_terrain_height(x, y)
            if not np.isfinite(z):
                z = self.flat_h

            return (float(x), float(y), float(z + 0.15))

        for a, b in zip(path_cells[:-1], path_cells[1:]):
            p0 = cell_to_world(*a)
            p1 = cell_to_world(*b)
            did = p.addUserDebugLine(
                p0, p1,
                lineColorRGB=[1, 0, 0],
                lineWidth=3.0,
                lifeTime=life
            )
            self._debug_ids.append(did)

    def _create_terrain(self):
        size = self.size
        diff = float(np.clip(self.difficulty, 0.0, 1.0))

        # ---------- cleanup old ----------
        if self.terrain_id is not None:
            try:
                p.removeBody(self.terrain_id)
            except Exception:
                pass
            self.terrain_id = None

        for bid in self.terrain_obstacles:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        self.terrain_obstacles = []

        # ---------- helpers ----------
        def normalize01(a):
            a = a - a.min()
            return a / (a.max() + 1e-8)

        def fbm(shape, sigmas, weights):
            acc = np.zeros(shape, dtype=np.float32)
            for s, w in zip(sigmas, weights):
                n = np.random.randn(*shape).astype(np.float32)
                n = scipy.ndimage.gaussian_filter(n, sigma=s)
                acc += w * n
            return acc

        def stamp_gaussian(z, cx, cy, radius, amp):
            x0 = max(0, cx - radius)
            x1 = min(size, cx + radius + 1)
            y0 = max(0, cy - radius)
            y1 = min(size, cy + radius + 1)
            xs = np.arange(x0, x1)[:, None]
            ys = np.arange(y0, y1)[None, :]
            dx = xs - cx
            dy = ys - cy
            d2 = dx * dx + dy * dy
            sigma = max(1.0, radius * 0.55)
            bump = np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
            z[x0:x1, y0:y1] += amp * bump

        def clamp_slope_iter(z, max_dz_per_cell, passes=4):
            zc = z.copy()
            for _ in range(passes):
                dxp = np.roll(zc, -1, axis=0) - zc
                dxm = np.roll(zc, +1, axis=0) - zc
                dyp = np.roll(zc, -1, axis=1) - zc
                dym = np.roll(zc, +1, axis=1) - zc

                dxp = np.clip(dxp, -max_dz_per_cell, max_dz_per_cell)
                dxm = np.clip(dxm, -max_dz_per_cell, max_dz_per_cell)
                dyp = np.clip(dyp, -max_dz_per_cell, max_dz_per_cell)
                dym = np.clip(dym, -max_dz_per_cell, max_dz_per_cell)

                zc = 0.25 * (
                    (np.roll(zc, -1, axis=0) - dxp)
                    + (np.roll(zc, +1, axis=0) - dxm)
                    + (np.roll(zc, -1, axis=1) - dyp)
                    + (np.roll(zc, +1, axis=1) - dym)
                )
            return zc

        def astar_path(z, start, goal, max_step):
            nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
            sx, sy = start
            gx, gy = goal

            def h(x, y):
                dx = abs(x - gx)
                dy = abs(y - gy)
                return (dx + dy) + (np.sqrt(2) - 2.0) * min(dx, dy)

            g = np.full((size, size), np.inf, dtype=np.float32)
            g[sx, sy] = 0.0
            came = np.full((size, size, 2), -1, dtype=np.int32)

            pq = []
            heapq.heappush(pq, (h(sx, sy), 0.0, sx, sy))

            while pq:
                _, gc, x, y = heapq.heappop(pq)
                if (x, y) == (gx, gy):
                    path = [(x, y)]
                    while (x, y) != (sx, sy):
                        px, py = came[x, y]
                        if px < 0:
                            break
                        x, y = int(px), int(py)
                        path.append((x, y))
                    path.reverse()
                    return path

                if gc > g[x, y] + 1e-6:
                    continue

                z0 = float(z[x, y])
                for dx, dy in nbrs:
                    nx, ny = x + dx, y + dy
                    if nx < 0 or nx >= size or ny < 0 or ny >= size:
                        continue

                    z1 = float(z[nx, ny])
                    dz = abs(z1 - z0)
                    if dz > max_step:
                        continue

                    step_cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1.0
                    slope_pen = 1.0 + 8.0 * (dz / (max_step + 1e-6)) ** 2
                    ng = gc + step_cost * slope_pen

                    if ng < g[nx, ny]:
                        g[nx, ny] = ng
                        came[nx, ny] = (x, y)
                        heapq.heappush(pq, (ng + h(nx, ny), ng, nx, ny))

            return None

        def carve_corridor(z, path, width_cells, smooth_sigma, drop, keep_rough, berm):
            core = np.zeros((size, size), dtype=np.uint8)
            for x, y in path:
                core[x, y] = 1
            dist = scipy.ndimage.distance_transform_edt(core == 0).astype(np.float32)

            w = float(width_cells)
            band = np.clip(1.0 - dist / w, 0.0, 1.0)
            band = band * band * (3.0 - 2.0 * band)  # smoothstep

            z_s = scipy.ndimage.gaussian_filter(z, sigma=smooth_sigma)
            z_trail = z_s - drop

            z_new = (1.0 - band) * z + band * ((1.0 - keep_rough) * z_trail + keep_rough * z)
            if berm > 0.0:
                z_new = z_new + (1.0 - band) * berm  # raise outside -> trail looks carved

            return z_new, band

        def world_from_cell(ix, iy):
            half_extent = (size - 1) * self.scale / 2.0
            x = -half_extent + ix * self.scale
            y = -half_extent + iy * self.scale
            return float(x), float(y)

        # Start/goal indices
        start_idx = self.flat_size // 2
        goal_idx = size - 1 - start_idx
        start = (start_idx, start_idx)
        goal = (goal_idx, goal_idx)

        margin = self.flat_size + self.smooth_width + 4

        max_step = 0.13 + 0.07 * (1.0 - diff)  # easy ~0.18 hard ~0.11
        clamp_limit = max_step * (1.10 + 0.10 * (1.0 - diff))

        # --------- attempt loop until solvable ---------
        for _attempt in range(50):
            # 1) Macro hills that are visible
            macro = fbm((size, size), sigmas=[22.0, 14.0], weights=[1.0, 0.55])
            mid = fbm((size, size), sigmas=[9.0, 5.0], weights=[0.9, 0.55])
            macro01 = normalize01(macro)
            mid01 = normalize01(mid)

            macro_amp = 0.75 + 1.10 * diff   # bigger hills
            mid_amp   = 0.25 + 0.60 * diff   # more noticeable secondary structure

            z = self.flat_h + (macro01 - 0.5) * 2.0 * macro_amp
            z = z + (mid01 - 0.5) * 2.0 * mid_amp

            # 2) Discrete features: mounds and pits (Made more distinct!)
            n_mounds = int(6 + 12 * diff)
            n_pits = int(5 + 10 * diff)

            for _ in range(n_mounds):
                cx = np.random.randint(margin, size - margin)
                cy = np.random.randint(margin, size - margin)
                # Tighter radius + taller height = steeper hills
                r = np.random.randint(4, 9)  
                h = (0.55 + 0.85 * diff) * np.random.uniform(0.9, 1.3)
                stamp_gaussian(z, cx, cy, r, +h)

            for _ in range(n_pits):
                cx = np.random.randint(margin, size - margin)
                cy = np.random.randint(margin, size - margin)
                # Tighter radius + deeper depth = sharper holes
                r = np.random.randint(4, 10)
                d = (0.50 + 0.85 * diff) * np.random.uniform(0.9, 1.3)
                stamp_gaussian(z, cx, cy, r, -d)

            # 3) Tiny micro roughness only
            micro = fbm((size, size), sigmas=[2.2, 1.2], weights=[0.015 + 0.02 * diff, 0.01 + 0.015 * diff])
            z = z + micro

            # 4) Clamp slopes
            z = clamp_slope_iter(z, max_dz_per_cell=clamp_limit, passes=4)

            # 5) Start/goal pads + transitions
            z[0:self.flat_size, 0:self.flat_size] = self.flat_h
            z[-self.flat_size:, -self.flat_size:] = self.flat_h

            raw = z.copy()
            for i in range(self.flat_size + self.smooth_width):
                for j in range(self.flat_size + self.smooth_width):
                    if i < self.flat_size and j < self.flat_size:
                        continue
                    di = max(0, i - self.flat_size + 1)
                    dj = max(0, j - self.flat_size + 1)
                    t = min(1.0, max(di, dj) / self.smooth_width)
                    z[i, j] = (1 - t) * self.flat_h + t * raw[i, j]

            raw = z.copy()
            goal_start = size - self.flat_size
            trans_start = size - self.flat_size - self.smooth_width
            for i in range(trans_start, size):
                for j in range(trans_start, size):
                    if i >= goal_start and j >= goal_start:
                        continue
                    di = max(0, goal_start - i)
                    dj = max(0, goal_start - j)
                    t = min(1.0, max(di, dj) / self.smooth_width)
                    z[i, j] = (1 - t) * self.flat_h + t * raw[i, j]

            # 6) A* path
            path = astar_path(z, start, goal, max_step=max_step)
            if path is None:
                continue

            # 7) Carve corridor (visible)
            corridor_w = int(10 - 6 * diff)      # easy wide, hard narrow
            smooth_sigma = 2.6 - 1.2 * diff
            drop = 0.05 + 0.10 * diff
            keep_rough = 0.03 + 0.10 * diff
            berm = 0.06 + 0.08 * diff

            z, band = carve_corridor(z, path, corridor_w, smooth_sigma, drop, keep_rough, berm)

            # final clamp
            z = clamp_slope_iter(z, max_dz_per_cell=clamp_limit, passes=2)

            self.heightmap = z.astype(np.float32)
            self.path_cells = path

            # 8) Build heightfield
            shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                meshScale=[self.scale, self.scale, 1.0],
                heightfieldData=self.heightmap.flatten(),
                numHeightfieldRows=size,
                numHeightfieldColumns=size,
            )
            self.terrain_id = p.createMultiBody(0, shape)
            p.changeDynamics(self.terrain_id, -1, lateralFriction=1.6, restitution=0.0)
            p.setPhysicsEngineParameter(enableConeFriction=1)

            # 9) Uniform Obstacle Placement Algorithm
            placed = []
            
            # Define a padding distance to keep rocks out of the spawn/goal flat zones
            pad = self.flat_size + self.smooth_width + 2 

            def pick_uniform_point(min_dist_cells):
                """Pure uniform sampling over the entire map."""
                for _ in range(200): # max tries
                    # FIX: Sample from the ENTIRE map (0 to size), not the restricted middle 'margin'
                    ix = np.random.randint(0, size)
                    iy = np.random.randint(0, size)

                    # Keep away from start/goal pads
                    if ix < pad and iy < pad: continue
                    if ix > size - pad and iy > size - pad: continue

                    # Keep off the main path (band is 1.0 on path, 0.0 far outside)
                    if band[ix, iy] > 0.15:
                        continue

                    # Ensure spacing from other placed objects
                    too_close = False
                    for px, py in placed:
                        if (ix - px)**2 + (iy - py)**2 < min_dist_cells**2:
                            too_close = True
                            break
                    
                    if not too_close:
                        placed.append((ix, iy))
                        return ix, iy
                
                # Fallback if map gets too crowded (also fixed to use full map)
                ix = np.random.randint(pad, size - pad)
                iy = np.random.randint(pad, size - pad)
                placed.append((ix, iy))
                return ix, iy

            # 10) Fallen logs
            n_logs = int(2 + 4 * diff) 
            for _ in range(n_logs):
                ix, iy = pick_uniform_point(min_dist_cells=6)
                x, y = world_from_cell(ix, iy)

                z0 = self._get_terrain_height(x, y)
                if not np.isfinite(z0):
                    z0 = self.flat_h

                length = 1.2 + 1.8 * np.random.rand()
                radius = 0.10 + 0.10 * np.random.rand()
                yaw = float(np.random.uniform(0, np.pi))

                col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=radius, length=length,
                    rgbaColor=[0.35, 0.25, 0.12, 1]
                )

                quat = p.getQuaternionFromEuler([0.0, 1.5708, yaw])

                bid = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, z0 + radius * 0.8], 
                    baseOrientation=quat,
                )

                p.changeDynamics(bid, -1, lateralFriction=1.2, restitution=0.0)
                self.terrain_obstacles.append(bid)

            # 11) Rocks
            n_rocks = int(15 + 25 * diff) 
            for _ in range(n_rocks):
                ix, iy = pick_uniform_point(min_dist_cells=4)
                x, y = world_from_cell(ix, iy)

                z0 = self._get_terrain_height(x, y)
                if not np.isfinite(z0):
                    z0 = self.flat_h

                sx = 0.10 + 0.25 * np.random.rand()
                sy = 0.10 + 0.25 * np.random.rand()
                sz = 0.08 + 0.18 * np.random.rand()

                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, sz])
                vis = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[sx, sy, sz],
                    rgbaColor=[0.4, 0.4, 0.4, 1]
                )

                yaw = float(np.random.uniform(0, np.pi))
                quat = p.getQuaternionFromEuler([0.0, 0.0, yaw])

                bid = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[x, y, z0 + sz * 0.8],  
                    baseOrientation=quat,
                )

                p.changeDynamics(bid, -1, lateralFriction=1.3, restitution=0.0)
                self.terrain_obstacles.append(bid)

            # 12) Debug draw safe path
            self._debug_draw_path(self.path_cells, life=0)

            return self.terrain_id

        # fallback flat
        self.heightmap = np.full((size, size), self.flat_h, dtype=np.float32)
        self.path_cells = None
        shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=[self.scale, self.scale, 1.0],
            heightfieldData=self.heightmap.flatten(),
            numHeightfieldRows=size,
            numHeightfieldColumns=size,
        )
        self.terrain_id = p.createMultiBody(0, shape)
        p.changeDynamics(self.terrain_id, -1, lateralFriction=1.6, restitution=0.0)
        p.setPhysicsEngineParameter(enableConeFriction=1)
        return self.terrain_id

    # -------------------------- walls --------------------------
    def _create_walls(self):
        h = 2.5  # Height
        t = 0.4  # Thickness
        # Correct bound calculation to match the Heightfield mesh exactly
        bound = (self.size - 1) * self.scale / 2.0 

        def wall(x, y, sx, sy):
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, h])
            vis = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[sx, sy, h], rgbaColor=[0.6, 0.6, 0.6, 1]
            )
            # Offset Z by 1.0 so the wall sits firmly in the ground
            wid = p.createMultiBody(0, col, vis, basePosition=[x, y, 1.0])
            p.changeDynamics(wid, -1, lateralFriction=1.5, restitution=0.0)

        # Place walls exactly at the edges of the terrain
        wall(bound + t, 0, t, bound + t)  # Right
        wall(-(bound + t), 0, t, bound + t) # Left
        wall(0, bound + t, bound + t, t)  # Top
        wall(0, -(bound + t), bound + t, t) # Bottom

    # -------------------------- goal --------------------------
    def _create_goal(self):
        vis = p.createVisualShape(
            p.GEOM_CYLINDER, radius=0.8, length=0.3, rgbaColor=[0, 1, 0, 1]
        )
        self.goal_marker = p.createMultiBody(
            0,
            -1,
            vis,
            basePosition=[float(self.goal_xy[0]), float(self.goal_xy[1]), self.flat_h + 0.15],
        )

    # -------------------------- obs scaling --------------------------
    def _scale_obs(self, obs: np.ndarray) -> np.ndarray:
        # IMPORTANT: disable tanh squashing for now (debug / learning stability)
        # You can bring normalization later once PPO works.
        return np.asarray(obs, dtype=np.float32)
    # -------------------------- joint detection --------------------------
    def _detect_joints(self):
        self.wheel_joints = [2, 3, 5, 7]
        self.steer_joints = [4, 6]

    # -------------------------- reset --------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=80,
            solverResidualThreshold=1e-6,
            numSubSteps=1,
            enableConeFriction=1,
        )

        # Build world
        self._create_terrain()
        self._create_walls()
        self._create_goal()

        # Spawn a little above flat pad
        spawn_z = self.flat_h + 0.35

        # Face the goal (major help for early learning)
        yaw = float(
            np.arctan2(
                self.goal_xy[1] - self.start_xy[1],
                self.goal_xy[0] - self.start_xy[0],
            )
        )

        self.robot = p.loadURDF(
            "racecar/racecar.urdf",
            basePosition=[float(self.start_xy[0]), float(self.start_xy[1]), float(spawn_z)],
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, yaw]),
            flags=p.URDF_USE_INERTIA_FROM_FILE,
        )

        # Dynamics tuning (keep your values; these are safe defaults)
        num_joints = p.getNumJoints(self.robot)
        for link in range(-1, num_joints):
            p.changeDynamics(
                self.robot,
                link,
                lateralFriction=2.0,
                spinningFriction=0.10,
                rollingFriction=0.06,
                restitution=0.0,
                linearDamping=0.04,
                angularDamping=0.10,
            )

        # HARD-CODE correct joints (from your printed joint list)
        self.wheel_joints = [2, 3, 5, 7]
        self.steer_joints = [4, 6]

        # Stop any default motors first (prevents weird URDF defaults)
        for j in self.wheel_joints + self.steer_joints:
            try:
                p.setJointMotorControl2(
                    self.robot,
                    j,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=0.0,
                    force=0.0,
                )
            except Exception:
                pass

        # Apply wheel motors (velocity)
        if self.wheel_joints:
            p.setJointMotorControlArray(
                self.robot,
                self.wheel_joints,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[0.0] * len(self.wheel_joints),
                forces=[self.max_motor_force] * len(self.wheel_joints),
            )

        # Apply steering motors (position)
        if self.steer_joints:
            p.setJointMotorControlArray(
                self.robot,
                self.steer_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[0.0] * len(self.steer_joints),
                forces=[self.max_steer_force] * len(self.steer_joints),
            )

        # Episode state
        self.steps = 0
        self.stuck_counter = 0
        self.prev_dist = float(np.linalg.norm(self.goal_xy - self.start_xy))

        # Let the car settle
        for _ in range(80):
            p.stepSimulation()

        if self.render:
            p.resetDebugVisualizerCamera(
                cameraDistance=14,
                cameraYaw=45,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0],
            )

        return self._get_obs(), {}

    # -------------------------- observation --------------------------
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        pos = np.asarray(pos, dtype=np.float32)
        vel = np.asarray(vel, dtype=np.float32)
        ang = np.asarray(ang, dtype=np.float32)

        to_goal = self.goal_xy - pos[:2]
        dist = float(np.linalg.norm(to_goal))
        angle_to_goal = float(np.arctan2(to_goal[1], to_goal[0]))
        angle_diff = float(np.arctan2(np.sin(angle_to_goal - yaw),
                                    np.cos(angle_to_goal - yaw)))

        # Base observation features
        raw_obs = [
            to_goal[0] * 0.1, to_goal[1] * 0.1, dist * 0.1,
            angle_diff, np.sin(yaw), np.cos(yaw),
            vel[0], vel[1], ang[2], pos[2],
            roll, pitch, self.steps / 1000.0, 1.0
        ]

        # Calculate terrain heights for the expanded sample_offsets[cite: 3]
        heights = []
        for dx, dy in self.sample_offsets:
            # Rotate offsets based on car's current yaw
            wx = pos[0] + dx * np.cos(yaw) - dy * np.sin(yaw)
            wy = pos[1] + dx * np.sin(yaw) + dy * np.cos(yaw)
            
            h = self._get_terrain_height(float(wx), float(wy))
            heights.append(h - float(pos[2])) # Height relative to car

        raw_obs.extend(heights)
        obs = np.array(raw_obs, dtype=np.float32)
        
        return self._safe_obs(obs, clip=50.0)
    
    def _distance_to_path(self, pos):
        if self.path_cells is None:
            return 0.0

        px, py = pos[0], pos[1]

        half_extent = (self.size - 1) * self.scale / 2.0

        min_dist = 1e9
        for ix, iy in self.path_cells[::5]:  # sample every 5 cells (fast)
            x = -half_extent + ix * self.scale
            y = -half_extent + iy * self.scale
            d = (px - x) ** 2 + (py - y) ** 2
            if d < min_dist:
                min_dist = d

        return np.sqrt(min_dist)

    # -------------------------- step --------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        throttle = float(action[0])
        steer = float(action[1])

        steer_angle = np.clip(steer, -1.0, 1.0) * self.max_steer_angle
        if self.steer_joints:
            p.setJointMotorControlArray(
                self.robot,
                self.steer_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=[steer_angle] * len(self.steer_joints),
                forces=[self.max_steer_force] * len(self.steer_joints),
            )

        target_w = np.clip(throttle, -1.0, 1.0) * self.max_wheel_speed
        if self.wheel_joints:
            p.setJointMotorControlArray(
                self.robot,
                self.wheel_joints,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=[target_w] * len(self.wheel_joints),
                forces=[self.max_motor_force] * len(self.wheel_joints),
            )

        for _ in range(self.physics_steps_per_env_step):
            p.stepSimulation()

        self.steps += 1

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, ang = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        pos = np.asarray(pos, dtype=np.float32)
        vel = np.asarray(vel, dtype=np.float32)
        ang = np.asarray(ang, dtype=np.float32)

        dist = float(np.linalg.norm(self.goal_xy - pos[:2]))

        # ---------------- reward ----------------
        progress = self.prev_dist - dist
        reward = progress * 6.0   

        # heading
        to_goal = self.goal_xy - pos[:2]
        angle_to_goal = float(np.arctan2(to_goal[1], to_goal[0]))
        angle_diff = float(np.arctan2(np.sin(angle_to_goal - yaw),
                                    np.cos(angle_to_goal - yaw)))

        forward_speed = float(vel[0] * np.cos(yaw) + vel[1] * np.sin(yaw))
        reward += 0.4 * forward_speed * np.cos(angle_diff)

        # alive bonus
        #reward += 0.02

        # reduce steering penalty
        reward -= 0.005 * abs(steer)

        # wall penalty (keep)
        lim = self.half - 0.5
        wall_dist = min(lim - abs(pos[0]), lim - abs(pos[1]))
        if wall_dist < 1.5:
            reward -= (1.5 - wall_dist) * 0.6

        # tilt penalty (reduced)
        reward -= 0.05 * (abs(roll) + abs(pitch))

        # step penalty (small)
        reward -= 0.005

        # path penalty (weaker)
        path_dist = self._distance_to_path(pos)
        reward -= 0.05 * path_dist

        if self.stuck_counter > 0:
            reward += 0.1 * forward_speed
        # ---------------- termination/truncation ----------------
        terminated = False
        truncated = False

        if dist < 2.0:
            reward += 150.0
            terminated = True

        if abs(roll) > 1.5 or abs(pitch) > 1.5:
            reward -= 50.0
            terminated = True

        if pos[2] < -1.0:
            reward -= 100.0
            terminated = True

        if abs(pos[0]) > lim or abs(pos[1]) > lim:
            reward -= 100.0
            terminated = True

        if (self.prev_dist - dist) < self.stuck_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        if self.stuck_counter >= self.max_stuck_steps:
            reward -= 100.0
            truncated = True
            self.stuck_counter = 0

        if self.steps >= 1000:
            truncated = True

        self.prev_dist = dist
        return self._get_obs(), float(reward), terminated, truncated, {}

    def close(self):
        self._clear_debug()
        for bid in self.terrain_obstacles:
            try:
                p.removeBody(bid)
            except Exception:
                pass
        self.terrain_obstacles = []
        p.disconnect()