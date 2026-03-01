# =============================================================
#   RRT PATH PLANNER - Rapidly Exploring Random Tree
#   Plans collision-free paths in 3D space
#   Reports which obstacles are blocking if path not found
# =============================================================

import numpy as np
import random
import time


class RRTPlanner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.max_iterations  = config["rrt_max_iterations"]
        self.step_size       = config["rrt_step_size"]
        self.goal_threshold  = config["rrt_goal_threshold"]
        self.smooth          = config["smooth_trajectory"]
        self.trajectory_points = config["trajectory_points"]

        self.x_limits = (-0.8, 0.8)
        self.y_limits = (-0.8, 0.8)
        self.z_limits = (0.05, 1.2)

        self.obstacles = []
        self.logger.info("RRT Planner initialized")

    # ----------------------------------------------------------
    # Obstacle management
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
        self.logger.debug(f"RRT obstacle added: {name} at {center}")

    def clear_obstacles(self):
        self.obstacles = []

    def update_obstacles_from_objects(self, objects, robot_base=(0, 0, 0)):
        """Add scene objects as obstacles - keeps named obstacles"""
        # Remove old object obstacles but keep named ones (walls etc)
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

    # ----------------------------------------------------------
    # Collision checking
    # ----------------------------------------------------------

    def _is_in_workspace(self, point):
        x, y, z = point
        return (self.x_limits[0] <= x <= self.x_limits[1] and
                self.y_limits[0] <= y <= self.y_limits[1] and
                self.z_limits[0] <= z <= self.z_limits[1])

    def _is_collision_free(self, point):
        if not self._is_in_workspace(point):
            return False
        p = np.array(point)
        for obs in self.obstacles:
            diff = np.abs(p - obs["center"])
            if np.all(diff <= obs["size"] + 0.05):
                return False
        return True

    def _is_path_collision_free(self, p1, p2, steps=10):
        for i in range(steps + 1):
            t     = i / steps
            point = p1 + t * (p2 - p1)
            if not self._is_collision_free(point):
                return False
        return True

    # ----------------------------------------------------------
    # RRT Core
    # ----------------------------------------------------------

    def _random_sample(self, goal, goal_bias=0.1):
        if random.random() < goal_bias:
            return np.array(goal)
        return np.array([
            random.uniform(*self.x_limits),
            random.uniform(*self.y_limits),
            random.uniform(*self.z_limits)
        ])

    def _nearest_node(self, tree, point):
        distances = [np.linalg.norm(node["pos"] - point) for node in tree]
        return tree[np.argmin(distances)]

    def _steer(self, from_pos, to_pos):
        direction = to_pos - from_pos
        distance  = np.linalg.norm(direction)
        if distance < self.step_size:
            return to_pos
        return from_pos + (direction / distance) * self.step_size

    def plan(self, start, goal, ignored_objects=None):
        """
        Plan a collision-free path.
        Returns (path, status, blocking_obstacles)
        status: "success" | "blocked"
        """
        start = np.array(start)
        goal  = np.array(goal)

        self.logger.debug(
            f"RRT: {tuple(start.round(3))} -> {tuple(goal.round(3))}"
        )

        # Check straight line first
        if self._is_path_collision_free(start, goal):
            self.logger.debug("Straight line path clear")
            return self._interpolate_path(start, goal), "success", []

        # Build RRT tree
        tree = [{"pos": start, "parent": None}]

        for iteration in range(self.max_iterations):
            random_point = self._random_sample(goal)
            nearest      = self._nearest_node(tree, random_point)
            new_pos      = self._steer(nearest["pos"], random_point)

            if not self._is_collision_free(new_pos):
                continue
            if not self._is_path_collision_free(nearest["pos"], new_pos):
                continue

            new_node = {"pos": new_pos, "parent": nearest}
            tree.append(new_node)

            if np.linalg.norm(new_pos - goal) < self.goal_threshold:
                path = self._extract_path(tree, new_node, goal)
                if self.smooth:
                    path = self._smooth_path(path)
                self.logger.debug(
                    f"RRT found path in {iteration} iterations"
                )
                return path, "success", []

        # Failed - find what's blocking
        blocking = self._find_blocking_obstacles(start, goal)
        self.logger.warning(
            f"RRT failed - blocked by: {[o['name'] for o in blocking]}"
        )
        return None, "blocked", blocking

    def _find_blocking_obstacles(self, start, goal):
        """Find obstacles between start and goal"""
        blocking  = []
        direction = goal - start
        distance  = np.linalg.norm(direction)

        if distance < 0.001:
            return []

        unit = direction / distance

        for obs in self.obstacles:
            to_obs     = obs["center"] - start
            projection = np.dot(to_obs, unit)

            if 0 < projection < distance:
                closest_point = start + unit * projection
                dist_to_path  = np.linalg.norm(
                    obs["center"] - closest_point
                )
                if dist_to_path < np.max(obs["size"]) + 0.1:
                    blocking.append(obs)

        return blocking

    def _extract_path(self, tree, final_node, goal):
        path = [goal]
        node = final_node
        while node is not None:
            path.append(node["pos"])
            node = node["parent"]
        path.reverse()
        return path

    def _interpolate_path(self, start, goal):
        path = []
        for i in range(self.trajectory_points + 1):
            t     = i / self.trajectory_points
            point = start + t * (goal - start)
            path.append(point)
        return path

    def _smooth_path(self, path):
        if len(path) <= 2:
            return path
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                if self._is_path_collision_free(
                    np.array(smoothed[-1]),
                    np.array(path[j])
                ):
                    break
                j -= 1
            smoothed.append(path[j])
            i = j
        self.logger.debug(
            f"Path smoothed: {len(path)} -> {len(smoothed)} waypoints"
        )
        return smoothed

    def plan_pick_approach(self, robot_pos, object_pos):
        above_pos = (object_pos[0], object_pos[1], object_pos[2] + 0.3)
        approach_path, status, blocking = self.plan(robot_pos, above_pos)
        grasp_pos   = (object_pos[0], object_pos[1], object_pos[2] + 0.06)
        descend_path = self._interpolate_path(
            np.array(above_pos), np.array(grasp_pos)
        )
        return approach_path, descend_path, status, blocking