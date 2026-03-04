# =============================================================
#   ROBOT TOOLS - Functions Claude can call directly
#   These give Claude real data from the simulation
#   Claude uses these to make physics-aware decisions
# =============================================================

import numpy as np


class RobotTools:
    def __init__(self, robot, logger):
        self.robot  = robot
        self.logger = logger

    # ----------------------------------------------------------
    # Tool definitions for Claude API
    # ----------------------------------------------------------

    def get_tool_definitions(self):
        """Returns tool schemas for Anthropic API"""
        return [
            {
                "name": "get_robot_position",
                "description": (
                    "Get the current 3D position of the robot end effector. "
                    "Returns x, y, z coordinates in metres."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_object_position",
                "description": (
                    "Get the exact 3D position of a named object in the scene. "
                    "Returns x, y, z coordinates. "
                    "Use this before planning any pick operation."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Object name e.g. 'red', 'blue', 'green'"
                        }
                    },
                    "required": ["name"]
                }
            },
            {
                "name": "get_all_objects",
                "description": (
                    "Get positions of all objects currently in the scene. "
                    "Returns a dict of name -> (x, y, z)."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_all_obstacles",
                "description": (
                    "Get all static obstacles in the scene with their "
                    "positions and sizes. Use this to understand what "
                    "might be blocking paths."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "check_path_clear",
                "description": (
                    "Check if a straight line path between two 3D points "
                    "is free of obstacles. Returns 'clear' or a description "
                    "of what is blocking the path. "
                    "Use this before commanding any movement."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "start_x": {"type": "number"},
                        "start_y": {"type": "number"},
                        "start_z": {"type": "number"},
                        "end_x":   {"type": "number"},
                        "end_y":   {"type": "number"},
                        "end_z":   {"type": "number"}
                    },
                    "required": [
                        "start_x", "start_y", "start_z",
                        "end_x", "end_y", "end_z"
                    ]
                }
            },
            {
                "name": "get_safe_height",
                "description": (
                    "Calculate the minimum safe height the robot must rise to "
                    "in order to travel over all obstacles between two points. "
                    "Returns height in metres."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "from_x": {"type": "number"},
                        "from_y": {"type": "number"},
                        "to_x":   {"type": "number"},
                        "to_y":   {"type": "number"}
                    },
                    "required": ["from_x", "from_y", "to_x", "to_y"]
                }
            },
            {
                "name": "is_point_reachable",
                "description": (
                    "Check if a 3D point is within the robot workspace limits. "
                    "Returns true/false and reason if unreachable."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "z": {"type": "number"}
                    },
                    "required": ["x", "y", "z"]
                }
            },
            {
                "name": "get_gripper_state",
                "description": "Get current state of the gripper - open or closed.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "get_holding",
                "description": (
                    "Get the name of the object currently held by the robot, "
                    "or null if not holding anything."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "plan_pick_sequence",
                "description": (
                    "Generate a validated, collision-aware pick sequence for "
                    "a named object. Returns exact waypoints the robot should "
                    "follow to successfully pick the object, accounting for "
                    "all obstacles. Use this when you need to pick something."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "object_name": {
                            "type": "string",
                            "description": "Name of object to pick"
                        }
                    },
                    "required": ["object_name"]
                }
            },
            {
                "name": "plan_place_sequence",
                "description": (
                    "Generate a validated place sequence to put the currently "
                    "held object at a target x, y position."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "target_x": {"type": "number"},
                        "target_y": {"type": "number"}
                    },
                    "required": ["target_x", "target_y"]
                }
            }
        ]

    # ----------------------------------------------------------
    # Tool execution
    # ----------------------------------------------------------

    def execute_tool(self, tool_name, tool_input):
        """
        Execute a tool call from Claude.
        Returns result as string.
        """
        print(f"  [TOOL] {tool_name}({self._fmt_input(tool_input)})", end=" ")

        try:
            if tool_name == "get_robot_position":
                result = self._get_robot_position()

            elif tool_name == "get_object_position":
                result = self._get_object_position(tool_input["name"])

            elif tool_name == "get_all_objects":
                result = self._get_all_objects()

            elif tool_name == "get_all_obstacles":
                result = self._get_all_obstacles()

            elif tool_name == "check_path_clear":
                result = self._check_path_clear(
                    tool_input["start_x"], tool_input["start_y"],
                    tool_input["start_z"],
                    tool_input["end_x"],   tool_input["end_y"],
                    tool_input["end_z"]
                )

            elif tool_name == "get_safe_height":
                result = self._get_safe_height(
                    tool_input["from_x"], tool_input["from_y"],
                    tool_input["to_x"],   tool_input["to_y"]
                )

            elif tool_name == "is_point_reachable":
                result = self._is_point_reachable(
                    tool_input["x"], tool_input["y"], tool_input["z"]
                )

            elif tool_name == "get_gripper_state":
                result = self._get_gripper_state()

            elif tool_name == "get_holding":
                result = self._get_holding()

            elif tool_name == "plan_pick_sequence":
                result = self._plan_pick_sequence(tool_input["object_name"])

            elif tool_name == "plan_place_sequence":
                result = self._plan_place_sequence(
                    tool_input["target_x"], tool_input["target_y"]
                )

            else:
                result = f"Unknown tool: {tool_name}"

        except Exception as e:
            result = f"Tool error: {str(e)}"

        print(f"= {result}")
        self.logger.debug(f"Tool {tool_name}: {result}")
        return str(result)

    # ----------------------------------------------------------
    # Tool implementations
    # ----------------------------------------------------------

    def _get_robot_position(self):
        pos = self.robot.get_end_effector_position()
        return f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

    def _get_object_position(self, name):
        pos = self.robot.get_object_position(name)
        if pos is None:
            return f"Object '{name}' not found in scene"
        return f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"

    def _get_all_objects(self):
        positions = self.robot.get_all_object_positions()
        if not positions:
            return "No objects in scene"
        result = []
        for name, pos in positions.items():
            if pos:
                result.append(
                    f"{name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                )
        return " | ".join(result)

    def _get_all_obstacles(self):
        if not self.robot.obstacles:
            return "No obstacles in scene"
        result = []
        for name, data in self.robot.obstacles.items():
            pos  = data["position"]
            size = data["size"]
            top  = pos[2] + size[2]
            result.append(
                f"{name}: center=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                f"size={[round(s,2) for s in size]} top={top:.2f}m"
            )
        return " | ".join(result)

    def _check_path_clear(self, sx, sy, sz, ex, ey, ez):
        start   = np.array([sx, sy, sz])
        end     = np.array([ex, ey, ez])
        active  = self.robot.rrt.obstacles

        blocked = []
        steps   = 15
        for i in range(steps + 1):
            t     = i / steps
            point = start + t * (end - start)
            for obs in active:
                if np.all(np.abs(point - obs["center"]) <= obs["size"] + 0.05):
                    blocked.append(obs["name"])

        if not blocked:
            return "clear"

        unique = list(dict.fromkeys(blocked))
        return f"BLOCKED by: {', '.join(unique)}"

    def _get_safe_height(self, fx, fy, tx, ty):
        start   = np.array([fx, fy, 0])
        end     = np.array([tx, ty, 0])
        active  = self.robot.rrt.obstacles

        relevant = self.robot.rrt._get_obstacles_in_corridor(
            start, end, active
        ) if hasattr(self.robot.rrt, '_get_obstacles_in_corridor') else active

        if not relevant:
            relevant = active

        if not relevant:
            return f"{self.robot.rrt.min_safe_height:.2f}m"

        max_top = max(
            obs["center"][2] + obs["size"][2] for obs in relevant
        )
        safe_h = float(np.clip(
            max_top + 0.15,
            self.robot.rrt.min_safe_height,
            self.robot.rrt.max_safe_height
        ))
        return f"{safe_h:.2f}m (tallest obstacle top: {max_top:.2f}m)"

    def _is_point_reachable(self, x, y, z):
        xl = self.robot.rrt.x_limits
        yl = self.robot.rrt.y_limits
        zl = self.robot.rrt.z_limits

        if not (xl[0] <= x <= xl[1]):
            return f"unreachable: x={x} outside limits {xl}"
        if not (yl[0] <= y <= yl[1]):
            return f"unreachable: y={y} outside limits {yl}"
        if not (zl[0] <= z <= zl[1]):
            return f"unreachable: z={z} outside limits {zl}"
        return f"reachable: ({x:.2f}, {y:.2f}, {z:.2f}) is within workspace"

    def _get_gripper_state(self):
        if self.robot.grasped_object:
            return f"closed (holding '{self.robot.grasped_object}')"
        return "open"

    def _get_holding(self):
        if self.robot.grasped_object:
            return self.robot.grasped_object
        return "null"

    def _plan_pick_sequence(self, object_name):
        pos = self.robot.get_object_position(object_name)
        if pos is None:
            return f"Cannot plan: '{object_name}' not found"

        # Get safe height
        robot_pos = self.robot.get_end_effector_position()
        safe_h_str = self._get_safe_height(
            robot_pos[0], robot_pos[1], pos[0], pos[1]
        )
        safe_h = float(safe_h_str.split("m")[0])

        # Check path to above object
        path_check = self._check_path_clear(
            robot_pos[0], robot_pos[1], safe_h,
            pos[0], pos[1], safe_h
        )

        steps = [
            f"1. Rise to safe height {safe_h:.2f}m",
            f"2. Move to above object: ({pos[0]:.3f}, {pos[1]:.3f}, {safe_h:.2f})",
            f"3. Descend to grasp: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]+0.06:.3f})",
            f"4. Close gripper",
            f"5. Lift to: ({pos[0]:.3f}, {pos[1]:.3f}, {safe_h:.2f})"
        ]

        path_status = "horizontal path is clear" if path_check == "clear" \
            else f"horizontal path: {path_check}"

        return (
            f"Pick sequence for '{object_name}' at {self._get_object_position(object_name)}: "
            f"{path_status}. "
            f"Steps: {' | '.join(steps)}"
        )

    def _plan_place_sequence(self, tx, ty):
        if not self.robot.grasped_object:
            return "Cannot plan place: not holding any object"

        robot_pos = self.robot.get_end_effector_position()
        safe_h_str = self._get_safe_height(
            robot_pos[0], robot_pos[1], tx, ty
        )
        safe_h = float(safe_h_str.split("m")[0])

        path_check = self._check_path_clear(
            robot_pos[0], robot_pos[1], safe_h,
            tx, ty, safe_h
        )

        steps = [
            f"1. Rise to {safe_h:.2f}m",
            f"2. Move to above target: ({tx:.3f}, {ty:.3f}, {safe_h:.2f})",
            f"3. Descend to: ({tx:.3f}, {ty:.3f}, 0.13)",
            f"4. Open gripper",
            f"5. Lift to: ({tx:.3f}, {ty:.3f}, {safe_h:.2f})"
        ]

        path_status = "path clear" if path_check == "clear" \
            else f"path: {path_check}"

        return (
            f"Place sequence to ({tx:.2f}, {ty:.2f}): {path_status}. "
            f"Steps: {' | '.join(steps)}"
        )

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _fmt_input(self, tool_input):
        if not tool_input:
            return ""
        parts = [f"{k}={v}" for k, v in tool_input.items()]
        return ", ".join(parts)