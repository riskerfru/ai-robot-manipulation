# =============================================================
#   AI CONTROLLER - Claude with Tool Use
#   Claude now queries the simulation before planning
#   Grounded reasoning - no more blind guessing
# =============================================================

import anthropic
import json


SYSTEM_PROMPT = """You are an AI controller for a Franka Panda 7-axis robot arm in a PyBullet simulation.

You have tools to query the real simulation state. ALWAYS use tools before planning any movement.

RULES:
1. Before picking anything - call get_object_position AND check_path_clear
2. Before moving anywhere - call is_point_reachable
3. If path is blocked - call get_safe_height to find how high to go
4. Always call get_all_obstacles first to understand the scene
5. Use plan_pick_sequence for pick operations - it gives you validated steps

RESPONSE FORMAT - always return valid JSON:
{
  "response": "human readable explanation of what you're doing",
  "intent": "pick" | "place" | "move" | "home" | "scan" | "none",
  "target_object": "object name if picking",
  "steps": [
    {"action": "pick", "object": "red"},
    {"action": "place", "x": 0.2, "y": 0.4},
    {"action": "place", "x": 0.3, "y": -0.3, "stack_on": "green"},
    {"action": "move", "x": 0.3, "y": 0.0, "z": 0.5}
  ],
  "confidence": 0.95,
  "reasoning": "brief explanation of why you chose this plan"
}

For pick operations, return a single step: {"action": "pick", "object": "name"}
For place operations, return a single step: {"action": "place", "x": 0.2, "y": 0.4}
The robot code handles all motion details internally."""


class AIController:
    def __init__(self, config, logger):
        self.config  = config
        self.logger  = logger
        self.client  = anthropic.Anthropic(api_key=config["api_key"])
        self.model   = config["model"]
        self.max_tokens   = config["max_tokens"]
        self.memory_length = config["memory_length"]

        self.conversation_history = []
        self.robot_state = {
            "position":        (0, 0, 0),
            "holding":         None,
            "detected_objects": {},
        }

        # Robot tools instance - set after robot is created
        self.tools_instance = None

        self.logger.info("AI Controller initialized with tool use")

    def set_tools(self, robot_tools_instance):
        """Connect robot tools to this controller"""
        self.tools_instance = robot_tools_instance
        self.logger.info("Robot tools connected to AI")

    def update_state(self, robot_position=None, holding=None,
                     detected_objects=None):
        if robot_position:
            self.robot_state["position"] = robot_position
        if holding is not None:
            self.robot_state["holding"] = holding
        if detected_objects:
            self.robot_state["detected_objects"] = detected_objects

    def _build_context(self, user_command, detected_objects):
        """Build context message for Claude"""
        pos = self.robot_state["position"]
        holding = self.robot_state["holding"] or "nothing"

        obj_str = ""
        if detected_objects:
            parts = []
            for name, pos_obj in detected_objects.items():
                if pos_obj:
                    parts.append(
                        f"{name} at ({pos_obj[0]:.3f}, {pos_obj[1]:.3f}, {pos_obj[2]:.3f})"
                    )
            obj_str = ", ".join(parts)

        context = (
            f"Robot position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})\n"
            f"Holding: {holding}\n"
            f"Visible objects: {obj_str or 'none'}\n"
            f"Command: {user_command}\n\n"
            f"Use your tools to query the simulation, then return a JSON plan."
        )
        return context

    def process_command(self, user_command, detected_objects=None):
        """
        Process a command using Claude with tool use.
        Claude will call simulation tools before planning.
        """
        if not self.tools_instance:
            self.logger.warning("No tools connected - falling back to basic mode")
            return self._fallback_plan(user_command)

        context = self._build_context(user_command, detected_objects)

        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": context
        })

        # Trim history
        if len(self.conversation_history) > self.memory_length * 2:
            self.conversation_history = self.conversation_history[
                -self.memory_length * 2:
            ]

        tools = self.tools_instance.get_tool_definitions()

        self.logger.info(f"USER COMMAND: '{user_command}'")
        print(f"\n  Claude thinking with tools...")

        # Agentic loop - Claude keeps calling tools until done
        messages = list(self.conversation_history)
        max_tool_rounds = 8

        for round_num in range(max_tool_rounds):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages
            )

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Claude finished - extract final JSON
                break

            elif response.stop_reason == "tool_use":
                # Claude wants to call tools
                tool_results = []
                assistant_content = []

                for block in response.content:
                    assistant_content.append(block)

                    if block.type == "tool_use":
                        result = self.tools_instance.execute_tool(
                            block.name, block.input
                        )
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result
                        })

                # Add assistant message with tool calls
                messages.append({
                    "role":    "assistant",
                    "content": assistant_content
                })

                # Add tool results
                messages.append({
                    "role":    "user",
                    "content": tool_results
                })

            else:
                break

        # Extract final text response
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text = block.text
                break

        # Parse JSON from response
        plan = self._parse_response(final_text, user_command)

        # Add assistant response to history
        self.conversation_history.append({
            "role":    "assistant",
            "content": final_text
        })

        self.logger.info(f"AI: {plan.get('response', '')}")
        return plan

    def _parse_response(self, text, original_command):
        """Parse JSON from Claude response"""
        import re

        # Try to extract JSON block
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                plan = json.loads(json_match.group())
                return plan
            except json.JSONDecodeError:
                pass

        # Fallback - build basic plan from text
        self.logger.warning("Could not parse JSON from AI response")
        return self._fallback_plan(original_command, text)

    def _fallback_plan(self, command, ai_text=""):
        """Basic fallback when tool use fails"""
        cmd = command.lower()
        response = ai_text or f"Processing: {command}"

        if "pick" in cmd or "grab" in cmd or "get" in cmd:
            for color in ["red", "blue", "green"]:
                if color in cmd:
                    return {
                        "response": response,
                        "intent": "pick",
                        "target_object": color,
                        "steps": [{"action": "pick", "object": color}],
                        "confidence": 0.7
                    }

        if "home" in cmd or "reset" in cmd:
            return {
                "response": "Going home",
                "intent": "home",
                "steps": [{"action": "home"}],
                "confidence": 1.0
            }

        return {
            "response": response or "Command not understood",
            "intent": "none",
            "steps": [],
            "confidence": 0.3
        }

    def ask(self, question):
        """Ask Claude a question about the scene"""
        if not self.tools_instance:
            return "Tools not connected"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system="You are a robot assistant. Answer questions about the robot simulation concisely.",
            tools=self.tools_instance.get_tool_definitions(),
            messages=[{"role": "user", "content": question}]
        )

        # Handle tool calls in question answering too
        messages = [{"role": "user", "content": question}]

        for _ in range(4):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                system="Answer questions about the robot simulation concisely.",
                tools=self.tools_instance.get_tool_definitions(),
                messages=messages
            )

            if response.stop_reason == "end_turn":
                break
            elif response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = list(response.content)
                for block in response.content:
                    if block.type == "tool_use":
                        result = self.tools_instance.execute_tool(
                            block.name, block.input
                        )
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result
                        })
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": tool_results})

        for block in response.content:
            if hasattr(block, "text"):
                return block.text

        return "No response"

    def suggest_next_action(self):
        return self.ask(
            "Based on the current scene, what should the robot do next? "
            "Check object and obstacle positions first."
        )

    def clear_memory(self):
        self.conversation_history = []
        self.logger.info("Conversation memory cleared")