# =============================================================
#   AGENT LOOP - Autonomous Warehouse Order Agent
#   Mirrors Humanoid's KinetIQ System 2 architecture:
#     - Receives high-level goal
#     - Decomposes into subtasks
#     - Executes: observe → think → act → check → repeat
#     - Replans on failure
#     - Reports final status
#
#   Usage:
#     from agent.agent_loop import WarehouseAgent
#     agent = WarehouseAgent(robot, ai, logger)
#     result = agent.run("pick red, stack blue on green")
# =============================================================

import time
import json
import anthropic


# =============================================================
#   SYSTEM PROMPT — Claude acts as System 2 (task orchestrator)
#   This mirrors KinetIQ's VLM reasoning layer
# =============================================================

ORCHESTRATOR_PROMPT = """You are an autonomous warehouse robot agent — the task orchestrator (System 2).

You control a Franka Panda robot arm via these low-level capabilities (System 1):
  - pick(object_name)         pick up a named object
  - place(x, y)               place held object at coordinates
  - stack(object_name)        place held object on top of another
  - scan()                    scan scene and get all object positions
  - home()                    return robot to home position

CONVEYOR SLOTS — numbered, no colour coding:
  Slot 1: x=0.72, y= 0.30
  Slot 2: x=0.72, y= 0.00
  Slot 3: x=0.72, y=-0.30

SLOT RULES:
  Slots are PLACE destinations only — never pick from x=0.72
  Objects to pick are at x=0.50-0.55 (table area)
  If user specifies a slot number → use that slot
  If no slot specified → use next empty slot (1 first, then 2, then 3)
  Never place two objects in the same slot
  Use action "place_slot" with param "slot": 1, 2, or 3

Your job:
1. Decompose the warehouse order into a sequence of subtasks
2. Execute each subtask one at a time
3. Check the result (success / failed / reason)
4. If failed — replan using the failure reason
5. Continue until all subtasks are done or truly impossible

RESPONSE FORMAT — always return valid JSON:
{
  "thinking": "your reasoning about the current situation",
  "action": "pick" | "place" | "stack" | "scan" | "home" | "done" | "failed",
  "params": {
    "object": "red",         (for pick)
    "on": "green",           (for stack — the object to stack ON TOP OF)
    "x": 0.4,               (for place)
    "y": 0.1                (for place)
  },
  "message": "human readable description of what you are doing"
}

When the full order is complete, return: {"action": "done", "message": "order complete"}
When a subtask is truly impossible after retrying, return: {"action": "failed", "message": "reason"}

CRITICAL RULES:
- Execute ONE action per response
- NEVER pick an object if you are already holding something — place first
- Always scan first if you don't know where objects are
- If pick fails with "unreachable" — the object is out of the safe zone
  Safe zone: 0.30m to 0.65m from robot base (origin)
  If object is too close (<0.25m): report failed — cannot rescue without human
  If object is too far (>0.75m): report failed — out of reach
- If pick fails for other reason — try again once, then report failed
- If place fails — try a slightly different position (shift x or y by 0.05)
- Never give up after first failure — try at least 2 alternatives
- Be concise in thinking — max 2 sentences
- Track what you are holding — the observation tells you under "Holding:"
- Check reachability badge in observation (✅ ok / ⚠️ edge / ❌ unreachable)
"""


# =============================================================
#   EMERGENCY STOP — shared flag checked every loop iteration
# =============================================================

class EmergencyStop:
    """
    Shared flag visible to all parts of the system.
    Set stop=True from anywhere to halt the agent loop.

    Usage:
      e_stop = EmergencyStop()
      e_stop.trigger("user pressed stop button")
      e_stop.reset()
    """
    def __init__(self):
        self.stop   = False
        self.reason = None

    def trigger(self, reason="emergency stop"):
        self.stop   = True
        self.reason = reason
        print(f"\n  🛑 EMERGENCY STOP: {reason}")

    def reset(self):
        self.stop   = False
        self.reason = None
        print("  ✅ Emergency stop cleared")

    def is_active(self):
        return self.stop


# Global instance — importable from anywhere
EMERGENCY_STOP = EmergencyStop()


# =============================================================
#   OBSERVATION BUILDER
#   What the agent sees at each step — this is what they judge
# =============================================================

def build_observation(robot, completed_tasks, failed_tasks,
                       remaining_order, last_result=None):
    """
    Build a structured observation string for Claude.

    Design choices (for design note):
    - Include exact positions so agent doesn't need to guess
    - Include reachability status to prevent wasted attempts
    - Include task history so agent knows what it has done
    - Include last result so agent can react to failures
    - Keep format consistent — Claude learns the pattern quickly
    """

    # Get current scene state — use robot.objects directly (always accurate)
    positions   = robot.get_all_object_positions()   # from PyBullet directly
    holding     = robot.grasped_object or "nothing"
    robot_pos   = robot.get_end_effector_position()
    obstacles   = list(robot.obstacles.keys())
    # Note: positions comes from PyBullet physics — not camera — always correct

    # Build object summary with reachability
    object_lines = []
    for name, pos in positions.items():
        if pos:
            reach = robot.check_reachability(pos[0], pos[1], pos[2])
            status_icon = {
                "ok":      "✅",
                "warning": "⚠️",
                "error":   "❌"
            }.get(reach["severity"], "?")

            object_lines.append(
                f"  {status_icon} {name}: "
                f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                f"reach={reach['severity']} ({reach['distance']}m)"
            )
        else:
            object_lines.append(f"  ? {name}: position unknown")

    objects_str = "\n".join(object_lines) if object_lines else "  none visible"

    # Last result summary
    if last_result:
        result_str = (
            f"Last action: {last_result.get('action')} "
            f"→ {last_result.get('status')} "
            f"({last_result.get('detail', '')})"
        )
    else:
        result_str = "Last action: none (starting fresh)"

    # Build slot occupancy string
    slot_status = ""
    if hasattr(robot, "slot_occupants"):
        for sn, (sx, sy) in robot.conveyor_slots.items():
            occupant = robot.slot_occupants.get(sn, "empty")
            slot_status += f"  Slot {sn}: ({sx}, {sy:+.2f}) — {occupant}\n"
    else:
        slot_status = "  Slot 1: (0.72, +0.30)\n  Slot 2: (0.72, 0.00)\n  Slot 3: (0.72, -0.30)\n"

    # Build full observation
    obs = f"""=== CURRENT OBSERVATION ===
CONVEYOR SLOTS (place destinations — x=0.72, never pick these):
{slot_status}
Robot position: ({robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f})
Holding: {holding}

Objects in scene:
{objects_str}

Obstacles: {', '.join(obstacles) if obstacles else 'none'}

{result_str}

Completed tasks: {completed_tasks if completed_tasks else 'none yet'}
Failed tasks:    {failed_tasks if failed_tasks else 'none'}

Remaining order: {remaining_order}
==========================="""

    return obs


# =============================================================
#   TASK DECOMPOSER
#   Breaks a natural language order into subtask list
# =============================================================

def decompose_order(order, client, model, known_objects=None):
    """
    Use Claude to decompose a natural language order into subtasks.
    known_objects: list of object names that exist in scene.
    """
    print(f"\n  🧠 Decomposing order: '{order}'")

    obj_note = ""
    if known_objects:
        obj_note = f"\nONLY use these objects (no others): {known_objects}"

    system = (
        "Decompose a warehouse robot order into subtasks.\n"
        "Return ONLY a JSON array of subtasks, nothing else.\n\n"
        "Subtask formats:\n"
        '{"action": "pick",  "object": "red"}\n'
        '{"action": "place_slot", "slot": 1}\n'
        '{"action": "place", "x": 0.4, "y": 0.1}\n'
        '{"action": "stack", "object": "blue", "on": "green"}\n'
        '{"action": "scan"}\n'
        '{"action": "home"}\n\n'
        "CONVEYOR SLOTS:\n"
        "  Slot 1: (0.68, +0.30)\n"
        "  Slot 2: (0.68,  0.00)\n"
        "  Slot 3: (0.68, -0.30)\n"
        "Use place_slot for dispatch/conveyor tasks.\n"
        + obj_note + "\n\n"
        "Rules:\n"
        "- Pick before place or stack\n"
        "- NEVER invent objects not in the known list\n"
        "- Return ONLY the JSON array"
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Decompose this order: {order}"
        }]
    )

    text = response.content[0].text.strip()

    # Parse JSON array
    try:
        # Handle code blocks
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        subtasks = json.loads(text)
        print(f"  📋 Subtasks ({len(subtasks)}):")
        for i, t in enumerate(subtasks):
            print(f"     {i+1}. {t}")
        return subtasks
    except json.JSONDecodeError:
        print(f"  ⚠️  Could not parse decomposition, using simple fallback")
        # Simple fallback — parse common patterns
        return _simple_decompose(order)


def _simple_decompose(order):
    """Fallback decomposer for common patterns"""
    subtasks = [{"action": "scan"}]
    order_lower = order.lower()

    for color in ["red", "blue", "green"]:
        if f"pick {color}" in order_lower or f"grab {color}" in order_lower:
            subtasks.append({"action": "pick", "object": color})

    if "stack" in order_lower:
        for color in ["red", "blue", "green"]:
            if color in order_lower:
                subtasks.append({"action": "stack", "object": color,
                                  "on": "green"})
                break

    return subtasks


# =============================================================
#   MAIN AGENT LOOP
# =============================================================

class WarehouseAgent:
    """
    Autonomous warehouse order fulfilment agent.

    Architecture mirrors Humanoid KinetIQ:
      System 3: natural language order (input)
      System 2: this class — Claude VLM reasoning
      System 1: robot.pick(), place(), stack() — capabilities
      System 0: PyBullet physics — execution

    Each capability reports success/failed/reason back to System 2
    so the agent can track progress and replan.
    """

    def __init__(self, robot, ai_config, logger):
        self.robot      = robot
        self.logger     = logger
        self.client     = anthropic.Anthropic(api_key=ai_config["api_key"])
        self.model      = ai_config["model"]
        self.e_stop     = EMERGENCY_STOP
        self.max_retries = 3   # max retries per subtask
        self._place_count = {}  # track placements per zone to offset

    def run(self, order):
        """
        Run the autonomous agent loop for a given order.

        Args:
            order: natural language string
                   e.g. "pick red and place at dispatch,
                          then stack blue on green"

        Returns:
            {
              "success": True/False,
              "completed": [...],
              "failed": [...],
              "steps_taken": int,
              "summary": "human readable result"
            }
        """
        print(f"\n{'='*50}")
        print(f"  🤖 WAREHOUSE AGENT STARTING")
        print(f"  📦 Order: {order}")
        print(f"{'='*50}\n")

        self.e_stop.reset()

        # Reset slot occupancy for fresh run
        if hasattr(self.robot, "slot_occupants"):
            self.robot.slot_occupants = {}
            print("  [SLOT] Slot occupancy reset for new order")

        # Step 1 — Decompose order into subtasks
        # Use robot.objects directly — always accurate, no camera errors
        known_objects = list(self.robot.objects.keys())
        print(f"  📦 Known objects in scene: {known_objects}")
        subtasks = decompose_order(order, self.client, self.model,
                                   known_objects=known_objects)
        remaining = list(subtasks)   # copy
        completed = []
        failed    = []
        steps     = 0
        last_result = None

        # Step 2 — Execute subtasks in loop
        while remaining:

            # Emergency stop check
            if self.e_stop.is_active():
                print(f"\n  🛑 Agent stopped: {self.e_stop.reason}")
                self.robot.home()
                return self._build_result(False, completed, failed, steps,
                                          f"stopped: {self.e_stop.reason}")

            current_subtask = remaining[0]
            retries = 0

            print(f"\n  📌 Current subtask: {current_subtask}")

            # Inner retry loop for each subtask
            while retries < self.max_retries:

                # Emergency stop check inside retry loop
                if self.e_stop.is_active():
                    break

                steps += 1

                # Step 3 — Build observation
                obs = build_observation(
                    robot=self.robot,
                    completed_tasks=completed,
                    failed_tasks=failed,
                    remaining_order=[t for t in remaining],
                    last_result=last_result
                )

                # Step 4 — Claude reasons and chooses action
                print(f"\n  👁️  Observing scene...")
                print(f"  🧠 Claude reasoning (attempt {retries + 1})...")

                decision = self._get_decision(obs, current_subtask, retries)

                if not decision:
                    retries += 1
                    continue

                print(f"  💭 Thinking: {decision.get('thinking', '')}")
                print(f"  ▶️  Action:   {decision.get('message', '')}")

                # Check for done/failed signals
                if decision["action"] == "done":
                    completed.append(str(current_subtask))
                    remaining.pop(0)
                    last_result = {"action": "done", "status": "success",
                                   "detail": "subtask complete"}
                    break

                if decision["action"] == "failed":
                    print(f"  ❌ Agent gave up: {decision.get('message')}")
                    failed.append({
                        "task": str(current_subtask),
                        "reason": decision.get("message", "unknown")
                    })
                    remaining.pop(0)
                    last_result = {"action": str(current_subtask),
                                   "status": "failed",
                                   "detail": decision.get("message")}
                    break

                # Step 5 — Safety check: if subtask is pick but holding something
                # inject a place action first rather than letting agent thrash
                action_to_run = decision.get("action")
                if action_to_run == "pick" and self.robot.grasped_object:
                    held = self.robot.grasped_object
                    print(f"  ⚠️  Holding '{held}' — placing aside before pick")
                    aside_result = self.robot.place_object(0.45, 0.15)
                    if isinstance(aside_result, bool):
                        aside_ok = aside_result
                    else:
                        aside_ok = aside_result.get("status") == "success"
                    if not aside_ok:
                        # Try different aside position
                        aside_result = self.robot.place_object(0.45, -0.15)

                # Step 5 — Execute action
                last_result = self._execute_action(decision)

                print(f"  {'✅' if last_result['status'] == 'success' else '❌'} "
                      f"Result: {last_result['status']} — {last_result['detail']}")

                # Step 6 — Check result
                if last_result["status"] == "success":
                    # Check if this completes the current subtask
                    if self._subtask_complete(current_subtask, last_result):
                        completed.append(str(current_subtask))
                        remaining.pop(0)
                        last_result["detail"] += " ✓ subtask complete"
                        break
                else:
                    retries += 1
                    print(f"  ⚠️  Attempt {retries}/{self.max_retries} failed. "
                          f"Reason: {last_result.get('detail')}")

            else:
                # Exhausted all retries
                print(f"  ❌ Subtask failed after {self.max_retries} attempts")
                failed.append({
                    "task": str(current_subtask),
                    "reason": last_result.get("detail", "max retries exceeded")
                                if last_result else "unknown"
                })
                remaining.pop(0)

        # Done
        success = len(failed) == 0
        summary = self._build_summary(completed, failed, steps)
        print(f"\n{'='*50}")
        print(f"  {'✅ ORDER COMPLETE' if success else '⚠️  ORDER PARTIAL'}")
        print(f"  {summary}")
        print(f"{'='*50}\n")

        return self._build_result(success, completed, failed, steps, summary)

    def _get_decision(self, observation, current_subtask, retry_count):
        """
        Ask Claude to decide the next action given the observation.
        Returns parsed decision dict or None on failure.
        """
        user_msg = (
            f"{observation}\n\n"
            f"Current subtask to complete: {current_subtask}\n"
            f"Retry attempt: {retry_count}\n"
            f"{'PREVIOUS ATTEMPT FAILED — try a different approach' if retry_count > 0 else ''}\n\n"
            f"What is your next action? Return JSON only."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                system=ORCHESTRATOR_PROMPT,
                messages=[{"role": "user", "content": user_msg}]
            )

            text = response.content[0].text.strip()

            # Parse JSON
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return None

        except Exception as e:
            print(f"  ⚠️  Decision error: {e}")
            return None

    def _execute_action(self, decision):
        """
        Execute a single action and return rich status dict.
        Maps Claude's decision to robot methods.
        """
        action = decision.get("action")
        params = decision.get("params", {})

        try:
            if action == "scan":
                # Use robot.objects directly — always accurate
                # camera-based scan can miss objects at table edges
                positions = self.robot.get_all_object_positions()
                return {
                    "action": "scan",
                    "status": "success",
                    "detail": f"found {len(positions)} objects: {list(positions.keys())}"
                }

            elif action == "pick":
                obj = params.get("object", "")
                if not obj:
                    return {"action": "pick", "status": "failed",
                            "detail": "no object specified"}

                # Reachability gate — check before attempting
                pos = self.robot.get_object_position(obj)
                if pos:
                    reach = self.robot.check_reachability(pos[0], pos[1])
                    dist  = reach.get("distance", 0)

                    if not reach["reachable"]:
                        # Hard block — too close or too far
                        print(f"  [REACH] ❌ Cannot pick {obj}: {reach['message']}")
                        return {
                            "action": "pick",
                            "status": "failed",
                            "target": obj,
                            "detail": f"unreachable: {reach['message']} — "
                                      f"object must be moved to safe zone (0.30-0.65m)"
                        }
                    elif reach["severity"] == "warning":
                        # Warn but allow attempt
                        print(f"  [REACH] ⚠️  {obj} at edge of reach ({dist}m) — attempting")
                    else:
                        print(f"  [REACH] ✅ {obj} reachable ({dist}m)")

                result = self.robot.pick_object(obj)
                # Handle both old bool returns and new dict returns
                if isinstance(result, bool):
                    ok = result
                    result = {
                        "action": "pick",
                        "status": "success" if ok else "failed",
                        "target": obj,
                        "detail": f"picked {obj}" if ok else f"failed to pick {obj}"
                    }
                # Clear slot occupancy if this object was in a slot
                if result.get("status") == "success":
                    if hasattr(self.robot, "slot_occupants"):
                        for sn, occupant in list(self.robot.slot_occupants.items()):
                            if occupant == obj:
                                del self.robot.slot_occupants[sn]
                                print(f"  [SLOT] Slot {sn} cleared (picked {obj})")
                return result

            elif action in ("place", "place_slot"):
                # Check if placing in a numbered slot
                slot_num = params.get("slot") or params.get("slot_number")

                # Ensure slot state exists (defensive — works with old robot.py too)
                if not hasattr(self.robot, "conveyor_slots"):
                    self.robot.conveyor_slots = {
                        1: (0.72,  0.30),
                        2: (0.72,  0.00),
                        3: (0.72, -0.30),
                    }
                if not hasattr(self.robot, "slot_occupants"):
                    self.robot.slot_occupants = {}

                slots    = self.robot.conveyor_slots
                occupied = self.robot.slot_occupants

                if slot_num is not None:
                    slot_num = int(slot_num)
                    if slot_num not in slots:
                        return {"action": "place", "status": "failed",
                                "detail": f"slot {slot_num} does not exist (valid: 1,2,3)"}
                    occupant = occupied.get(slot_num)
                    if occupant:
                        return {"action": "place", "status": "failed",
                                "detail": f"slot {slot_num} already occupied by {occupant}"}
                    x_final, y_final = slots[slot_num]
                    print(f"  [SLOT] Placing in slot {slot_num} at ({x_final}, {y_final})")

                else:
                    x = float(params.get("x", 0.4))
                    y = float(params.get("y", 0.0))

                    # Auto-assign slot if target is conveyor area (x >= 0.68)
                    if x >= 0.58 or abs(x - 0.63) < 0.08:
                        slot_num = None
                        for sn in sorted(slots.keys()):
                            if sn not in occupied or occupied[sn] is None:
                                slot_num = sn
                                break
                        if slot_num is None:
                            return {"action": "place", "status": "failed",
                                    "detail": "all conveyor slots are occupied"}
                        x_final, y_final = slots[slot_num]
                        print(f"  [SLOT] Auto-assigned slot {slot_num} "
                              f"at ({x_final}, {y_final})")
                    else:
                        x_final, y_final = x, y
                        slot_num = None

                result = self.robot.place_object(x_final, y_final)

                # Update slot occupancy on success
                if slot_num is not None:
                    ok = result.get("status") == "success" if isinstance(result, dict) else bool(result)
                    if ok:
                        # Record what was just placed (before grasped_object clears)
                        self.robot.slot_occupants[slot_num] = placed_name = (
                            result.get("target") if isinstance(result, dict) else "object"
                        )
                        print(f"  [SLOT] ✅ Slot {slot_num} occupied by {placed_name}")
                if isinstance(result, bool):
                    return {
                        "action": "place",
                        "status": "success" if result else "failed",
                        "target": f"({x}, {y})",
                        "detail": f"placed at ({x}, {y})" if result
                                  else f"failed to place at ({x}, {y})"
                    }
                return result

            elif action == "stack":
                # Accept all key variants Claude might use
                obj = (params.get("object")
                       or params.get("item")
                       or decision.get("object", ""))
                # Accept target under any key Claude might send
                target = (params.get("on")
                          or params.get("target")
                          or params.get("stack_on")
                          or params.get("onto")
                          or decision.get("on", "")
                          or decision.get("target", ""))

                if not obj or not target:
                    # Last resort: if holding something and only one target in scene
                    if self.robot.grasped_object:
                        obj = self.robot.grasped_object
                    all_objs = list(self.robot.objects.keys())
                    if not target and len(all_objs) > 0:
                        # Try to infer from subtask definition
                        target = current_subtask.get("on", current_subtask.get("target", ""))
                    if not obj or not target:
                        return {"action": "stack", "status": "failed",
                                "detail": "missing object or target for stack"}

                # KEY FIX: only pick if not already holding the object
                if self.robot.grasped_object == obj:
                    print(f"  [STACK] Already holding {obj} — skipping pick")
                    pick_ok = True
                elif self.robot.grasped_object and self.robot.grasped_object != obj:
                    # Holding something else — place it aside first
                    held = self.robot.grasped_object
                    print(f"  [STACK] Holding {held}, placing aside before stack")
                    self.robot.place_object(0.45, 0.15)
                    pick_result = self.robot.pick_object(obj)
                    pick_ok = pick_result.get("status") == "success"                               if isinstance(pick_result, dict) else pick_result
                else:
                    # Not holding anything — pick normally
                    pick_result = self.robot.pick_object(obj)
                    pick_ok = pick_result.get("status") == "success"                               if isinstance(pick_result, dict) else pick_result

                if not pick_ok:
                    return {"action": "stack", "status": "failed",
                            "target": obj,
                            "detail": f"could not pick {obj} for stacking"}

                # Place on target
                target_pos = self.robot.get_object_position(target)
                if not target_pos:
                    return {"action": "stack", "status": "failed",
                            "target": target,
                            "detail": f"stack target '{target}' not found"}

                place_result = self.robot.place_object(
                    target_pos[0], target_pos[1], stack_on=target
                )
                place_ok = place_result.get("status") == "success"                            if isinstance(place_result, dict) else place_result

                # Check verified position from feedback system
                if isinstance(place_result, dict):
                    place_status = place_result.get("status")
                    place_detail = place_result.get("detail", "")
                else:
                    place_status = "success" if place_ok else "failed"
                    place_detail = f"stacked {obj} on {target}"

                return {
                    "action": "stack",
                    "status": place_status,
                    "target": f"{obj} on {target}",
                    "detail": place_detail if place_status == "failed"
                              else f"stacked {obj} on {target} ✅ verified"
                }

            elif action == "home":
                self.robot.home()
                return {"action": "home", "status": "success",
                        "detail": "robot returned to home position"}

            else:
                return {"action": action, "status": "failed",
                        "detail": f"unknown action: {action}"}

        except Exception as e:
            return {"action": action, "status": "failed",
                    "detail": f"exception: {str(e)}"}

    def _subtask_complete(self, subtask, last_result):
        """
        Check if the last successful action completed the current subtask.
        """
        if last_result["status"] != "success":
            return False

        action    = subtask.get("action")
        completed = last_result.get("action")

        # Direct action match
        if action == completed:
            return True

        # Stack completes when place succeeds (stack = pick + place)
        if action == "stack" and completed in ["stack", "place"]:
            return True

        return False

    def _build_summary(self, completed, failed, steps):
        total = len(completed) + len(failed)
        failed_tasks = [f["task"] for f in failed]
        outcome = "All tasks done!" if not failed else f"Failed: {failed_tasks}"
        return (
            f"Completed {len(completed)}/{total} subtasks "
            f"in {steps} action steps. {outcome}"
        )

    def _build_result(self, success, completed, failed, steps, summary):
        return {
            "success":     success,
            "completed":   completed,
            "failed":      failed,
            "steps_taken": steps,
            "summary":     summary
        }