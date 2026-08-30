# =============================================================
#   TASK PLANNER - AI Hierarchical Task Planning + Memory
#   Breaks high level commands into executable steps
#   Remembers what happened previously in the session
# =============================================================

import anthropic
import json
import time


class TaskPlanner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.client = anthropic.Anthropic(api_key=config["api_key"])
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]

        # Memory: stores past commands and outcomes
        self.memory = []
        self.memory_length = config["memory_length"]

        # Current world state
        self.world_state = {
            "robot_position": [0.0, 0.0, 0.5],
            "gripper_state": "open",
            "holding_object": None,
            "objects": {}
        }

        self.system_prompt = """You are an advanced AI brain controlling a Franka Panda robot arm.

Your job is to break down user commands into a sequence of executable steps.

You will receive:
1. Current robot state (position, gripper, what it is holding)
2. Objects in the scene with their coordinates
3. Memory of recent actions
4. User command

You must respond ONLY with a valid JSON object in this exact format:
{
    "task_name": "short description of overall task",
    "steps": [
        {
            "action": "move",
            "x": 0.4,
            "y": 0.0,
            "z": 0.5,
            "description": "moving above red block"
        },
        {
            "action": "pick",
            "object": "red",
            "description": "picking up red block"
        },
        {
            "action": "place",
            "x": 0.0,
            "y": 0.4,
            "description": "placing red block at target"
        },
        {
            "action": "open_gripper",
            "description": "opening gripper"
        },
        {
            "action": "close_gripper",
            "description": "closing gripper"
        },
        {
            "action": "home",
            "description": "returning to home position"
        }
    ],
    "reasoning": "brief explanation of why you chose these steps"
}

Available actions:
- move: move to x,y,z coordinates
- pick: pick up a named object
- place: place held object at x,y coordinates
- open_gripper: open the gripper
- close_gripper: close the gripper
- home: return to home position

Workspace limits:
- x: -0.8 to 0.8
- y: -0.8 to 0.8
- z: 0.05 to 1.2

Important rules:
- Always move ABOVE an object before picking (z + 0.3)
- Never go below z = 0.05 (that is the floor)
- If holding an object, complete the place before picking another
- For stacking, place each block slightly higher than the last
- Respond ONLY with JSON, no extra text whatsoever"""

    # ----------------------------------------------------------
    # World state management
    # ----------------------------------------------------------

    def update_world_state(self, robot_position=None, gripper_state=None,
                           holding_object=None, objects=None):
        """Update the current world state"""
        if robot_position:
            self.world_state["robot_position"] = list(robot_position)
        if gripper_state:
            self.world_state["gripper_state"] = gripper_state
        if holding_object is not None:
            self.world_state["holding_object"] = holding_object
        if objects:
            self.world_state["objects"] = {
                name: list(pos) for name, pos in objects.items()
            }

    def update_objects(self, detected_objects):
        """Update object positions from camera detection"""
        self.world_state["objects"] = {
            name: list(pos) for name, pos in detected_objects.items()
        }

    # ----------------------------------------------------------
    # Memory management
    # ----------------------------------------------------------

    def add_to_memory(self, command, steps, outcome="completed"):
        """Add a completed task to memory"""
        self.memory.append({
            "command": command,
            "steps_count": len(steps),
            "outcome": outcome,
            "timestamp": time.time()
        })
        # Keep only recent memory
        if len(self.memory) > self.memory_length:
            self.memory = self.memory[-self.memory_length:]

    def get_memory_summary(self):
        """Format memory for the AI prompt"""
        if not self.memory:
            return "No previous actions this session."

        summary = "Recent actions:\n"
        for i, mem in enumerate(self.memory[-5:]):  # Last 5 actions
            summary += f"  {i+1}. '{mem['command']}' → {mem['outcome']}\n"
        return summary

    # ----------------------------------------------------------
    # Core planning
    # ----------------------------------------------------------

    def plan(self, user_input, detected_objects=None):
        """
        Main planning function.
        Takes user input, returns list of executable steps.
        """
        self.logger.log_command(user_input)

        # Update objects if provided
        if detected_objects:
            self.update_objects(detected_objects)

        # Build prompt with full context
        prompt = self._build_prompt(user_input)

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            self.logger.debug(f"AI raw response: {response_text}")

            # Parse JSON response
            plan = json.loads(response_text)

            # Validate plan structure
            if "steps" not in plan:
                raise ValueError("AI response missing 'steps' field")

            self.logger.log_task(
                plan.get("task_name", user_input),
                [s.get("description", s["action"]) for s in plan["steps"]]
            )

            self.logger.debug(f"AI reasoning: {plan.get('reasoning', 'none')}")

            return plan

        except json.JSONDecodeError as e:
            self.logger.log_error(e, "JSON parsing")
            # Fallback: simple move command
            return self._fallback_plan(user_input)

        except Exception as e:
            self.logger.log_error(e, "Task planning")
            return self._fallback_plan(user_input)

    def _build_prompt(self, user_input):
        """Build the full prompt with world state and memory"""
        robot_pos = self.world_state["robot_position"]
        objects = self.world_state["objects"]

        # Format object positions
        if objects:
            obj_str = "Objects in scene:\n"
            for name, pos in objects.items():
                obj_str += f"  - {name}: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}\n"
        else:
            obj_str = "No objects detected in scene.\n"

        prompt = f"""CURRENT ROBOT STATE:
Position: x={robot_pos[0]:.3f}, y={robot_pos[1]:.3f}, z={robot_pos[2]:.3f}
Gripper: {self.world_state['gripper_state']}
Holding: {self.world_state['holding_object'] or 'nothing'}

{obj_str}
MEMORY:
{self.get_memory_summary()}

USER COMMAND: {user_input}

Plan the steps to execute this command."""

        return prompt

    def _fallback_plan(self, user_input):
        """Simple fallback if AI fails"""
        self.logger.warning("Using fallback plan")
        return {
            "task_name": "fallback",
            "steps": [
                {
                    "action": "move",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.5,
                    "description": "moving to safe position"
                }
            ],
            "reasoning": "fallback due to planning error"
        }

    # ----------------------------------------------------------
    # Pre-built complex task plans
    # ----------------------------------------------------------

    def plan_stack_all(self, objects):
        """
        Pre-built plan to stack all objects into a tower.
        Used as fallback or when explicitly requested.
        """
        steps = []
        base_x, base_y = 0.0, 0.3
        stack_height = 0.05

        for i, (name, pos) in enumerate(objects.items()):
            place_z = stack_height + (i * 0.09)

            steps.append({
                "action": "pick",
                "object": name,
                "description": f"picking {name} block"
            })
            steps.append({
                "action": "place",
                "x": base_x,
                "y": base_y,
                "description": f"stacking {name} at height {place_z:.2f}"
            })

        steps.append({
            "action": "home",
            "description": "returning home after stacking"
        })

        return {
            "task_name": "stack all blocks",
            "steps": steps,
            "reasoning": "picking each block and stacking at base position"
        }

    def plan_sort_by_color(self, objects):
        """
        Pre-built plan to sort objects to different zones.
        Red → left, Blue → center, Green → right
        """
        color_zones = {
            "red":   (-0.4, 0.4),
            "blue":  (0.0,  0.4),
            "green": (0.4,  0.4)
        }

        steps = []
        for name, pos in objects.items():
            if name in color_zones:
                tx, ty = color_zones[name]
                steps.append({
                    "action": "pick",
                    "object": name,
                    "description": f"picking {name} block"
                })
                steps.append({
                    "action": "place",
                    "x": tx,
                    "y": ty,
                    "description": f"placing {name} in {name} zone"
                })

        steps.append({
            "action": "home",
            "description": "returning home after sorting"
        })

        return {
            "task_name": "sort blocks by color",
            "steps": steps,
            "reasoning": "sorting each color to its designated zone"
        }