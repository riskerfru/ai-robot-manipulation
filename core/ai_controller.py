# =============================================================
#   AI CONTROLLER - Claude Powered Robot Brain
#   Handles command interpretation with full conversation memory
#   Error recovery and retry logic built in
#   Bridges user commands to task planner
# =============================================================

import anthropic
import json
import time


class AIController:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.client = anthropic.Anthropic(api_key=config["api_key"])
        self.model = config["model"]
        self.max_tokens = config["max_tokens"]

        # Conversation memory — stores full history
        self.conversation_history = []
        self.memory_length = config["memory_length"]

        # Current understood world state
        self.known_objects = {}
        self.robot_position = [0.0, 0.0, 0.5]
        self.holding = None

        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1.0

        self.system_prompt = """You are an intelligent AI controller for a Franka Panda robot arm.

You receive:
1. A user command in natural language
2. Current robot state
3. Objects detected in the scene with coordinates
4. Memory of recent actions

You must respond ONLY with valid JSON in this exact format:
{
    "intent": "pick|place|move|home|scan|stack|sort|custom",
    "target_object": "red|blue|green|null",
    "target_position": {"x": 0.0, "y": 0.0, "z": 0.5},
    "steps": [
        {"action": "pick", "object": "red", "description": "picking red block"},
        {"action": "place", "x": 0.0, "y": 0.4, "description": "placing at target"}
    ],
    "response": "Natural language confirmation of what you will do",
    "confidence": 0.95
}

Action types in steps:
- pick: pick up named object
- place: place at x,y coordinates  
- move: move end effector to x,y,z
- home: return to home position
- open_gripper: open gripper
- close_gripper: close gripper
- scan: scan scene with camera

Workspace limits: x(-0.8,0.8), y(-0.8,0.8), z(0.05,1.2)

Rules:
- Always add z offset of 0.3 above objects before picking
- Never command z below 0.05
- If holding object, place before picking another
- For stacking: place each block 0.09m higher than last
- Sort zones: red=(-0.4,0.4), blue=(0.0,0.4), green=(0.4,0.4)
- Confidence below 0.5 means you are unsure — say so in response
- ONLY return JSON, absolutely no other text"""

    # ----------------------------------------------------------
    # State management
    # ----------------------------------------------------------

    def update_state(self, robot_position=None, holding=None,
                     detected_objects=None):
        """Update AI's understanding of current world state"""
        if robot_position:
            self.robot_position = list(robot_position)
        if holding is not None:
            self.holding = holding
        if detected_objects:
            self.known_objects = {
                name: list(pos)
                for name, pos in detected_objects.items()
            }

    # ----------------------------------------------------------
    # Memory management
    # ----------------------------------------------------------

    def add_to_history(self, role, content):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
        # Keep history within memory limit
        if len(self.conversation_history) > self.memory_length * 2:
            # Keep system context + recent messages
            self.conversation_history = (
                self.conversation_history[-self.memory_length * 2:]
            )

    def get_context_summary(self):
        """Build context string from recent history"""
        if not self.conversation_history:
            return "No previous actions."

        recent = self.conversation_history[-6:]  # Last 3 exchanges
        summary = "Recent conversation:\n"
        for msg in recent:
            if msg["role"] == "user":
                summary += f"  User: {msg['content'][:100]}\n"
            else:
                try:
                    parsed = json.loads(msg["content"])
                    summary += f"  Robot: {parsed.get('response', '')[:100]}\n"
                except Exception:
                    summary += f"  Robot: {msg['content'][:100]}\n"
        return summary

    def clear_memory(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("AI memory cleared")

    # ----------------------------------------------------------
    # Core command processing
    # ----------------------------------------------------------

    def process_command(self, user_input, detected_objects=None):
        """
        Main function. Takes user text, returns structured command.
        Has retry logic for API failures.
        """
        if detected_objects:
            self.update_state(detected_objects=detected_objects)

        # Build full context prompt
        prompt = self._build_prompt(user_input)

        # Add to history
        self.add_to_history("user", user_input)

        # Try with retries
        for attempt in range(self.max_retries):
            try:
                response = self._call_api(prompt)
                command = self._parse_response(response)

                # Add response to history
                self.add_to_history("assistant", response)

                # Log
                self.logger.log_command(user_input, command)
                self.logger.info(
                    f"AI: {command.get('response', 'executing command')}"
                )

                return command

            except json.JSONDecodeError as e:
                self.logger.warning(
                    f"JSON parse failed (attempt {attempt+1}): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

            except anthropic.APIError as e:
                self.logger.error(f"API error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                break

        # All retries failed — return safe fallback
        self.logger.warning("All retries failed, using fallback command")
        return self._fallback_command(user_input)

    def _call_api(self, prompt):
        """Make API call to Claude"""
        messages = [{"role": "user", "content": prompt}]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=messages
        )

        return response.content[0].text.strip()

    def _build_prompt(self, user_input):
        """Build full prompt with world state and memory"""
        # Object positions
        if self.known_objects:
            obj_str = "Detected objects:\n"
            for name, pos in self.known_objects.items():
                obj_str += (
                    f"  {name}: x={pos[0]:.3f}, "
                    f"y={pos[1]:.3f}, z={pos[2]:.3f}\n"
                )
        else:
            obj_str = "No objects detected yet.\n"

        prompt = f"""ROBOT STATE:
Position: x={self.robot_position[0]:.3f}, y={self.robot_position[1]:.3f}, z={self.robot_position[2]:.3f}
Holding: {self.holding or 'nothing'}
Gripper: {'closed' if self.holding else 'open'}

{obj_str}
MEMORY:
{self.get_context_summary()}

USER COMMAND: "{user_input}"

Respond with JSON only."""

        return prompt

    def _parse_response(self, response_text):
        """Parse and validate JSON response from Claude"""
        # Clean response
        text = response_text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        command = json.loads(text)

        # Validate required fields
        required = ["intent", "steps", "response"]
        for field in required:
            if field not in command:
                raise ValueError(f"Missing required field: {field}")

        # Validate steps
        for step in command["steps"]:
            if "action" not in step:
                raise ValueError("Step missing action field")

        return command

    def _fallback_command(self, user_input):
        """Safe fallback when AI fails"""
        return {
            "intent": "move",
            "target_object": None,
            "target_position": {"x": 0.0, "y": 0.0, "z": 0.5},
            "steps": [
                {
                    "action": "move",
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.5,
                    "description": "moving to safe position"
                }
            ],
            "response": "I had trouble understanding that. Moving to safe position.",
            "confidence": 0.0
        }

    # ----------------------------------------------------------
    # Direct command shortcuts
    # ----------------------------------------------------------

    def ask(self, question):
        """
        Ask Claude a general question about the scene or task.
        Returns natural language answer — not a robot command.
        """
        try:
            obj_str = ""
            if self.known_objects:
                obj_str = "Current objects: " + ", ".join(
                    f"{n} at ({p[0]:.2f},{p[1]:.2f})"
                    for n, p in self.known_objects.items()
                )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system="You are a helpful robot assistant. Answer questions about the robot workspace concisely.",
                messages=[{
                    "role": "user",
                    "content": f"{obj_str}\n\nQuestion: {question}"
                }]
            )
            answer = response.content[0].text.strip()
            self.logger.info(f"Q: {question} → A: {answer}")
            return answer

        except Exception as e:
            self.logger.error(f"Ask failed: {e}")
            return "I couldn't answer that question right now."

    def suggest_next_action(self):
        """
        Ask Claude to suggest what the robot should do next
        based on current scene state.
        """
        if not self.known_objects:
            return "Scan the scene first to see what objects are available."

        obj_summary = ", ".join(
            f"{n} at ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
            for n, p in self.known_objects.items()
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                system="You are a robot task advisor. Suggest one useful next action concisely.",
                messages=[{
                    "role": "user",
                    "content": (
                        f"Robot is holding: {self.holding or 'nothing'}. "
                        f"Objects: {obj_summary}. "
                        f"What should the robot do next?"
                    )
                }]
            )
            suggestion = response.content[0].text.strip()
            self.logger.info(f"Suggestion: {suggestion}")
            return suggestion

        except Exception as e:
            self.logger.error(f"Suggest failed: {e}")
            return "Unable to generate suggestion."