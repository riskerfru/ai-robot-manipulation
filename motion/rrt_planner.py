# =============================================================
#   RRT PATH PLANNER - 6 Strategy Obstacle Avoidance
#   Strategy 1: Direct path
#   Strategy 2: Rise over obstacle
#   Strategy 3: Expand sideways
#   Strategy 4: Expand + arc around
#   Strategy 5: Full RRT random search
#   Strategy 6: Joint reconfiguration (redundancy resolution)
# =============================================================

import numpy as np
import random


class RRTPlanner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.max_iterations    = config["rrt_max_iterations"]
        self.step_size         = config["rrt_step_size"]
        self.goal_threshold    = config["rrt_goal_threshold"]
        self.smooth            = config["smooth_trajectory"]
        self.trajectory_points = config["trajectory_points"]

        self.x_limits = (-0.8, 0.8)
        self.y_limits = (-0.8, 0.8)
        self.z_limits = (0.05, 1.2)

        self.obstacles = []

        self.min_safe_height    = 0.40
        self.max_safe_height    = 0.80
        self.safe_height_margin = 0.15

        self.robot = None
        self.carrying_clearance = 0.0

        # Verbose mode — set False to suppress strategy search prints
        # Only final outcome prints when verbose=False
        self.verbose = False

        self.logger.info("RRT Planner initialized")

    # ----------------------------------------------------------
    # Workspace and obstacle management
    # ----------------------------------------------------------

    def set_workspace_limits(self, x_limits, y_limits, z_limits):
        self.x_limits = x_limits
        self.y_limits = y_limits
        self.z_limits = z_limits

    def add_obstacle(self, center, size, name="obstacle"):
        self.obstacles.append({
            "name":   name,
            "center": np.array(center),
            "size":   np.array(size)
        })

    def clear_obstacles(self):
        self.obstacles = []

    def update_obstacles_from_objects(self, objects):
        self.obstacles = [
            o for o in self.obstacles
            if not o["name"].startswith("obj_")
        ]
        for name, pos in objects.items():
            if pos is not None:
                self.add_obstacle(
                    center=(pos[0], pos[1], pos[2] + 0.04),
                    size=(0.06, 0.06, 0.06),
                    name=f"obj_{name}"
                )

    def _active(self, ignore_name=None):
        if ignore_name is None:
            return self.obstacles
        return [
            o for o in self.obstacles
            if o["name"] != f"obj_{ignore_name}"
            and o["name"] != ignore_name
        ]

    def _get_obstacles_in_corridor(self, start, goal, obstacles,
                                    corridor_width=0.25):
        relevant  = []
        start     = np.array(start)
        goal      = np.array(goal)
        direction = goal[:2] - start[:2]
        distance  = np.linalg.norm(direction)
        if distance < 0.001:
            return relevant
        unit = direction / distance
        for obs in obstacles:
            obs_xy    = obs["center"][:2]
            to_obs    = obs_xy - start[:2]
            proj      = np.dot(to_obs, unit)
            if 0 < proj < distance + 0.1:
                closest   = start[:2] + unit * proj
                dist_perp = np.linalg.norm(obs_xy - closest)
                if dist_perp < np.max(obs["size"][:2]) + corridor_width:
                    relevant.append(obs)
        return relevant

    # ----------------------------------------------------------
    # Main plan - tries all 6 strategies
    # ----------------------------------------------------------

    def plan(self, start, goal, ignore_name=None):
        start  = np.array(start)
        goal   = np.array(goal)
        active = self._active(ignore_name)

        if self.verbose:
            print(f"\n  [PLAN] {self._fmt(start)} -> {self._fmt(goal)}")

        # Strategy 1: Direct
        if self._path_clear(start, goal, active):
            print(f"  [PLAN] Strategy 1 (Direct): SUCCESS")
            return self._interpolate(start, goal), "success", []

        if self.verbose:
            print(f"  [PLAN] Strategy 1 (Direct): blocked")
        blocking = self._find_blocking(start, goal, active)
        if self.verbose:
            self._print_blocking(blocking)

        # Strategy 2: Rise over
        if self.verbose:
            print(f"  [PLAN] Strategy 2 (Rise over)...")
        path = self._strategy_rise(start, goal, active, blocking)
        if path:
            print(f"  [PLAN] Strategy 2 (Rise over): SUCCESS")
            return path, "success", []
        if self.verbose:
            print(f"  [PLAN] Strategy 2 (Rise over): failed")

        # Strategy 3: Expand sideways
        if self.verbose:
            print(f"  [PLAN] Strategy 3 (Expand sideways)...")
        path = self._strategy_sideways(start, goal, active, blocking)
        if path:
            print(f"  [PLAN] Strategy 3 (Expand sideways): SUCCESS")
            return path, "success", []
        if self.verbose:
            print(f"  [PLAN] Strategy 3 (Expand sideways): failed")

        # Strategy 4: Arc around
        if self.verbose:
            print(f"  [PLAN] Strategy 4 (Arc around)...")
        path = self._strategy_arc(start, goal, active, blocking)
        if path:
            print(f"  [PLAN] Strategy 4 (Arc around): SUCCESS")
            return path, "success", []
        if self.verbose:
            print(f"  [PLAN] Strategy 4 (Arc around): failed")

        # Strategy 5: Full RRT
        if self.verbose:
            print(f"  [PLAN] Strategy 5 (Full RRT)...")
        path, status, _ = self._rrt(start, goal, active)
        if status == "success":
            print(f"  [PLAN] Strategy 5 (Full RRT): SUCCESS")
            return path, "success", []
        if self.verbose:
            print(f"  [PLAN] Strategy 5 (Full RRT): failed")

        # Strategy 6: Joint reconfiguration
        if self.verbose:
            print(f"  [PLAN] Strategy 6 (Joint reconfiguration)...")
        path = self._strategy_joint_reconfig(start, goal, active, ignore_name)
        if path:
            print(f"  [PLAN] Strategy 6 (Joint reconfiguration): SUCCESS")
            return path, "success", []

        print(f"  [PLAN] All strategies failed for {self._fmt(goal)}")
        return None, "blocked", blocking

    # ----------------------------------------------------------
    # Strategy 2: Rise over (tries multiple heights)
    # ----------------------------------------------------------

    def _strategy_rise(self, start, goal, active, blocking):
        safe_h = self._safe_height(blocking)

        for height_bonus in [0.0, 0.10, 0.20, 0.30]:
            h = safe_h + height_bonus
            waypoints = [
                start,
                np.array([start[0], start[1], h]),
                np.array([goal[0],  goal[1],  h]),
                goal
            ]
            path = self._validate(waypoints, active)
            if path:
                if height_bonus > 0 and self.verbose:
                    print(f"  [PLAN] Rise succeeded at +{height_bonus:.2f}m")
                return path
        return None

    # ----------------------------------------------------------
    # Strategy 3: Expand sideways
    # ----------------------------------------------------------

    def _strategy_sideways(self, start, goal, active, blocking):
        safe_h = self._safe_height(blocking)

        start_arr = np.array(start)
        goal_arr  = np.array(goal)

        travel = goal_arr[:2] - start_arr[:2]
        dist   = np.linalg.norm(travel)
        if dist < 0.001:
            return None

        travel_unit = travel / dist
        perp_left   = np.array([-travel_unit[1],  travel_unit[0]])
        perp_right  = np.array([ travel_unit[1], -travel_unit[0]])

        if not blocking:
            return None

        obs_center = np.mean([obs["center"][:2] for obs in blocking], axis=0)
        mid_point  = (start_arr[:2] + goal_arr[:2]) / 2
        to_obs     = obs_center - mid_point

        max_r = max(np.max(obs["size"][:2]) for obs in blocking)

        for perp, side in [(perp_right if np.dot(to_obs, perp_left) > 0
                            else perp_left, "primary"),
                           (perp_left if np.dot(to_obs, perp_left) > 0
                            else perp_right, "opposite")]:
            for expand_mult in [1.0, 1.5, 2.0, 2.5]:
                for h_bonus in [0.0, 0.10, 0.20]:
                    h           = safe_h + h_bonus
                    expand_dist = max_r + 0.20 * expand_mult
                    expand_xy   = mid_point + perp * expand_dist
                    expand_wp   = np.array([
                        np.clip(expand_xy[0], self.x_limits[0], self.x_limits[1]),
                        np.clip(expand_xy[1], self.y_limits[0], self.y_limits[1]),
                        h
                    ])

                    waypoints = [
                        start,
                        np.array([start_arr[0], start_arr[1], h]),
                        expand_wp,
                        np.array([goal_arr[0],  goal_arr[1],  h]),
                        goal
                    ]
                    path = self._validate(waypoints, active)
                    if path:
                        if self.verbose:
                            print(f"  [PLAN] Sideways {side} x{expand_mult:.1f} "
                                  f"at h={h:.2f}m")
                        return path
        return None

    # ----------------------------------------------------------
    # Strategy 4: Arc around
    # ----------------------------------------------------------

    def _strategy_arc(self, start, goal, active, blocking):
        if not blocking:
            return None
        safe_h     = self._safe_height(blocking)
        obs_center = np.mean([obs["center"] for obs in blocking], axis=0)
        arc_points = self._generate_arc(
            start, goal, obs_center, safe_h, blocking, active
        )
        if not arc_points:
            return None
        waypoints = (
            [start]
            + [np.array([start[0], start[1], safe_h])]
            + arc_points
            + [np.array([goal[0], goal[1], safe_h])]
            + [goal]
        )
        return self._validate(waypoints, active)

    def _generate_arc(self, start, goal, obs_center, safe_h,
                       blocking, active, n_points=3):
        start = np.array(start)
        goal  = np.array(goal)
        travel = goal[:2] - start[:2]
        dist   = np.linalg.norm(travel)
        if dist < 0.001:
            return None
        travel_unit = travel / dist
        perp_left   = np.array([-travel_unit[1],  travel_unit[0]])
        perp_right  = np.array([ travel_unit[1], -travel_unit[0]])
        max_r = max(np.max(obs["size"][:2]) for obs in blocking) + 0.20

        for perp in [perp_left, perp_right]:
            arc   = []
            valid = True
            for i in range(1, n_points + 1):
                t      = i / (n_points + 1)
                mid_xy = start[:2] + t * travel
                arc_xy = mid_xy + perp * max_r * np.sin(t * np.pi)
                arc_pt = np.array([
                    np.clip(arc_xy[0], self.x_limits[0], self.x_limits[1]),
                    np.clip(arc_xy[1], self.y_limits[0], self.y_limits[1]),
                    safe_h
                ])
                if not self._point_clear(arc_pt, active):
                    valid = False
                    break
                arc.append(arc_pt)
            if valid and arc:
                if self.verbose:
                    side = "left" if np.array_equal(perp, perp_left) else "right"
                    print(f"  [PLAN] Arc on {side} ({len(arc)} waypoints)")
                return arc
        return None

    # ----------------------------------------------------------
    # Strategy: Transport-aware path (used when carrying object)
    # ----------------------------------------------------------

    def plan_transport(self, start, goal, ignore_name=None):
        start  = np.array(start)
        goal   = np.array(goal)
        active = self._active(ignore_name)

        if not active:
            return self.plan(start, goal, ignore_name)

        if self.verbose:
            print(f"  [TRANSPORT] Finding max-clearance path...")

        best_path      = None
        best_clearance = -1.0

        mid    = (start + goal) / 2
        radius = np.linalg.norm(goal - start) * 0.6

        for angle_deg in range(0, 360, 30):
            angle  = np.radians(angle_deg)
            via_xy = mid[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
            via_z  = max(start[2], goal[2], 0.55)

            via = np.array([
                np.clip(via_xy[0], self.x_limits[0], self.x_limits[1]),
                np.clip(via_xy[1], self.y_limits[0], self.y_limits[1]),
                via_z
            ])

            waypoints = [start, via, goal]
            path = self._validate(waypoints, active)

            if path:
                min_clear = self._path_min_clearance(path, active)
                if min_clear > best_clearance:
                    best_clearance = min_clear
                    best_path      = path

        if best_path:
            print(f"  [TRANSPORT] Best path clearance: {best_clearance:.3f}m")
            return best_path, "success", []

        if self.verbose:
            print(f"  [TRANSPORT] No clear angle found - using standard planner")
        return self.plan(start, goal, ignore_name)

    def _path_min_clearance(self, path, obstacles):
        min_dist = float("inf")
        for point in path[::5]:
            point = np.array(point)
            for obs in obstacles:
                dist = np.linalg.norm(point - obs["center"]) - np.max(obs["size"])
                if dist < min_dist:
                    min_dist = dist
        return max(0.0, min_dist)

    # ----------------------------------------------------------
    # Strategy 6: Joint Reconfiguration
    # ----------------------------------------------------------

    def _strategy_joint_reconfig(self, start, goal, active, ignore_name):
        if self.robot is None:
            return None

        import pybullet as p
        import time

        saved_joints = []
        for joint_idx in self.robot.arm_joints:
            state = p.getJointState(self.robot.robot, joint_idx)
            saved_joints.append(state[0])

        configs = [
            {"name": "elbow up",       "joints": {3: -0.5}},
            {"name": "elbow down",     "joints": {3:  0.5}},
            {"name": "elbow up more",  "joints": {3: -1.0}},
            {"name": "upper arm in",   "joints": {2: -0.4, 3: -0.4}},
            {"name": "upper arm out",  "joints": {2:  0.4, 3:  0.4}},
            {"name": "base left",      "joints": {0: -0.3}},
            {"name": "base right",     "joints": {0:  0.3}},
            {"name": "wrist flip",     "joints": {5:  0.5}},
            {"name": "reach extend",   "joints": {1: -0.3, 3: -0.3}},
            {"name": "reach retract",  "joints": {1:  0.3, 3:  0.3}},
            {"name": "combined 1",     "joints": {0: -0.3, 3: -0.5}},
            {"name": "combined 2",     "joints": {0:  0.3, 3: -0.5}},
            {"name": "combined 3",     "joints": {2:  0.3, 3: -0.8}},
            {"name": "high reach",     "joints": {1: -0.5, 3: -0.8}},
        ]

        for config in configs:
            if self.verbose:
                print(f"  [RECONFIG] Trying '{config['name']}'...")

            new_joints = list(saved_joints)
            for arm_idx, delta in config["joints"].items():
                if arm_idx < len(new_joints):
                    new_joints[arm_idx] = saved_joints[arm_idx] + delta

            for i, joint_idx in enumerate(self.robot.arm_joints):
                p.setJointMotorControl2(
                    self.robot.robot, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=new_joints[i],
                    force=self.robot.robot_config["max_force"],
                    maxVelocity=2.0
                )

            for _ in range(80):
                p.stepSimulation()
                time.sleep(self.robot.sim_config["timestep"])

            new_ee = np.array(self.robot.get_end_effector_position())

            if self.verbose:
                print(f"  [RECONFIG] New EE position: {self._fmt(new_ee)}")

            if self._path_clear(new_ee, goal, active):
                print(f"  [RECONFIG] '{config['name']}' → direct path clear")
                path = self._interpolate(new_ee, goal)
                return path

            blocking_new = self._find_blocking(new_ee, goal, active)
            safe_h = self._safe_height(blocking_new) if blocking_new \
                else self.min_safe_height

            rise_waypoints = [
                new_ee,
                np.array([new_ee[0], new_ee[1], safe_h]),
                np.array([goal[0],   goal[1],   safe_h]),
                goal
            ]
            path = self._validate(rise_waypoints, active)
            if path:
                print(f"  [RECONFIG] '{config['name']}' → rise-over path found")
                return path

            path = self._strategy_sideways(
                new_ee, goal, active, blocking_new or []
            )
            if path:
                print(f"  [RECONFIG] '{config['name']}' → sideways path found")
                return path

        # Restore original joints
        for i, joint_idx in enumerate(self.robot.arm_joints):
            p.setJointMotorControl2(
                self.robot.robot, joint_idx,
                p.POSITION_CONTROL,
                targetPosition=saved_joints[i],
                force=self.robot.robot_config["max_force"],
                maxVelocity=1.0
            )
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.robot.sim_config["timestep"])

        return None

    # ----------------------------------------------------------
    # Shared utilities
    # ----------------------------------------------------------

    def _expand_sideways(self, start, goal, safe_h, blocking):
        start = np.array(start)
        goal  = np.array(goal)
        travel = goal[:2] - start[:2]
        dist   = np.linalg.norm(travel)
        if dist < 0.001:
            return None
        travel_unit = travel / dist
        perp_left   = np.array([-travel_unit[1],  travel_unit[0]])
        perp_right  = np.array([ travel_unit[1], -travel_unit[0]])
        if not blocking:
            return None
        obs_center = np.mean([obs["center"][:2] for obs in blocking], axis=0)
        mid_point  = (start[:2] + goal[:2]) / 2
        to_obs     = obs_center - mid_point
        if np.dot(to_obs, perp_left) > 0:
            expand_dir = perp_right
            side = "right"
        else:
            expand_dir = perp_left
            side = "left"
        max_r       = max(np.max(obs["size"][:2]) for obs in blocking) + 0.20
        expand_xy   = mid_point + expand_dir * max_r
        expand_pos  = np.array([
            np.clip(expand_xy[0], self.x_limits[0], self.x_limits[1]),
            np.clip(expand_xy[1], self.y_limits[0], self.y_limits[1]),
            safe_h
        ])
        if self.verbose:
            print(f"  [PLAN] Expand {side}: {self._fmt(expand_pos)}")
        return expand_pos

    def _validate(self, waypoints, active):
        full_path = []
        for i in range(len(waypoints) - 1):
            a = np.array(waypoints[i])
            b = np.array(waypoints[i + 1])
            if self._path_clear(a, b, active):
                full_path.extend(self._interpolate(a, b))
            else:
                seg_path, status, _ = self._rrt(a, b, active, max_iter=300)
                if status == "success":
                    full_path.extend(seg_path)
                else:
                    return None
        if self.smooth and len(full_path) > 2:
            full_path = self._smooth(full_path, active)
        return full_path if full_path else None

    def _safe_height(self, blocking):
        if not blocking:
            return self.min_safe_height
        max_top = max(obs["center"][2] + obs["size"][2] for obs in blocking)
        margin  = self.safe_height_margin + self.carrying_clearance
        return float(np.clip(
            max_top + margin,
            self.min_safe_height,
            self.max_safe_height
        ))

    def _print_blocking(self, blocking):
        if blocking:
            print(f"  [PLAN] Obstacles in path:")
            for obs in blocking:
                c = obs["center"]
                print(f"         - {obs['name']} "
                      f"({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")

    def _fmt(self, p):
        return (f"({float(p[0]):.2f},"
                f"{float(p[1]):.2f},"
                f"{float(p[2]):.2f})")

    # ----------------------------------------------------------
    # RRT* fallback
    # ----------------------------------------------------------

    def _rrt(self, start, goal, active, max_iter=None):
        if max_iter is None:
            max_iter = self.max_iterations
        start = np.array(start)
        goal  = np.array(goal)

        root = {"pos": start, "parent": None, "cost": 0.0}
        tree = [root]
        rewire_radius = self.step_size * 2.5

        for iteration in range(max_iter):
            sample  = self._sample(goal)
            nearest = self._nearest(tree, sample)
            new_pos = self._steer(nearest["pos"], sample)

            if not self._point_clear(new_pos, active):
                continue
            if not self._path_clear(nearest["pos"], new_pos, active):
                continue

            new_cost    = nearest["cost"] + np.linalg.norm(new_pos - nearest["pos"])
            best_parent = nearest
            best_cost   = new_cost

            for node in tree:
                dist = np.linalg.norm(node["pos"] - new_pos)
                if dist < rewire_radius:
                    candidate_cost = node["cost"] + dist
                    if (candidate_cost < best_cost and
                            self._path_clear(node["pos"], new_pos, active)):
                        best_parent = node
                        best_cost   = candidate_cost

            new_node = {"pos": new_pos, "parent": best_parent, "cost": best_cost}
            tree.append(new_node)

            for node in tree:
                if node is new_node or node is root:
                    continue
                dist = np.linalg.norm(node["pos"] - new_pos)
                if dist < rewire_radius:
                    new_cost_via = best_cost + dist
                    if (new_cost_via < node["cost"] and
                            self._path_clear(new_pos, node["pos"], active)):
                        node["parent"] = new_node
                        node["cost"]   = new_cost_via

            if np.linalg.norm(new_pos - goal) < self.goal_threshold:
                path = self._extract(tree, new_node, goal)
                if self.verbose:
                    cost = round(new_node["cost"], 3)
                    print(f"  [RRT*] Found path: cost={cost}, "
                          f"iter={iteration}, nodes={len(tree)}")
                if self.smooth:
                    path = self._smooth(path, active)
                return path, "success", []

        return None, "blocked", []

    # ----------------------------------------------------------
    # Collision checking
    # ----------------------------------------------------------

    def _point_clear(self, point, obstacles):
        point = np.array(point)
        if not (self.x_limits[0] <= point[0] <= self.x_limits[1] and
                self.y_limits[0] <= point[1] <= self.y_limits[1] and
                self.z_limits[0] <= point[2] <= self.z_limits[1]):
            return False
        for obs in obstacles:
            if self.carrying_clearance > 0 and not obs["name"].startswith("obj_"):
                buffer = 0.05 + self.carrying_clearance
            else:
                buffer = 0.05
            if np.all(np.abs(point - obs["center"]) <= obs["size"] + buffer):
                return False
        return True

    def _path_clear(self, p1, p2, obstacles, steps=10):
        p1 = np.array(p1)
        p2 = np.array(p2)
        for i in range(steps + 1):
            t = i / steps
            if not self._point_clear(p1 + t * (p2 - p1), obstacles):
                return False
        return True

    def _find_blocking(self, start, goal, obstacles):
        blocking = []
        start    = np.array(start)
        goal     = np.array(goal)
        d        = goal - start
        dist     = np.linalg.norm(d)
        if dist < 0.001:
            return []
        unit = d / dist
        for obs in obstacles:
            proj = np.dot(obs["center"] - start, unit)
            if 0 < proj < dist:
                closest = start + unit * proj
                if np.linalg.norm(obs["center"] - closest) < np.max(obs["size"]) + 0.1:
                    blocking.append(obs)
        return blocking

    # ----------------------------------------------------------
    # Path utilities
    # ----------------------------------------------------------

    def _sample(self, goal, bias=0.1):
        if random.random() < bias:
            return np.array(goal)
        return np.array([
            random.uniform(*self.x_limits),
            random.uniform(*self.y_limits),
            random.uniform(*self.z_limits)
        ])

    def _nearest(self, tree, point):
        dists = [np.linalg.norm(n["pos"] - point) for n in tree]
        return tree[np.argmin(dists)]

    def _steer(self, src, dst):
        src = np.array(src)
        dst = np.array(dst)
        d   = dst - src
        n   = np.linalg.norm(d)
        return src + (d / n) * self.step_size if n > self.step_size else dst

    def _extract(self, tree, node, goal):
        path = [np.array(goal)]
        while node:
            path.append(node["pos"])
            node = node["parent"]
        path.reverse()
        return path

    def _interpolate(self, start, goal):
        start = np.array(start)
        goal  = np.array(goal)
        n     = self.trajectory_points
        return [start + (i / n) * (goal - start) for i in range(n + 1)]

    def _smooth(self, path, obstacles):
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._path_clear(smoothed[-1], path[j], obstacles):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        if self.verbose:
            print(f"  [PLAN] Smoothed: {len(path)} -> {len(smoothed)} waypoints")
        return smoothed