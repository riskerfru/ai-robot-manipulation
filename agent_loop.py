# =============================================================
#   AGENT LOOP - Autonomous Warehouse Order Agent
#   Resolves the six agent failure modes documented in
#   FAILURE_MODES.md:
#
#   Bug 1: Subtask pointer not advancing
#   Bug 2: Agent picks from occupied slots
#   Bug 3: Objects drift out of reach
#   Bug 4: Observation format confuses Claude
#   Bug 5: Slot auto-assignment conflict
#   Bug 6: Stack instead of next slot
# =============================================================

import time
import json
import math
import re
import anthropic
from place_verifier import PlacementVerifier
from core.robot_tools import RobotTools


from cell.layout import CONVEYOR_SLOTS   # single source of truth
from audit.trail import NullAudit

_SLOT_PROMPT_LINES = "\n".join(
    f"  Slot {n}: ({x:.2f}, {y:+.2f})"
    for n, (x, y) in CONVEYOR_SLOTS.items()
)


ORCHESTRATOR_PROMPT = f"""You are an autonomous warehouse robot agent.

You control a Franka Panda robot arm via these actions:
  pick(object)        - pick up named object from TABLE ONLY
  place_slot(slot)    - place held object in conveyor slot 1, 2 or 3
  place(x, y)         - place at exact coordinates
  stack(object, on)   - stack object on top of another
  scan()              - scan scene
  home()              - return to home

CONVEYOR SLOTS — DESTINATIONS ONLY, NEVER PICK FROM THESE:
{_SLOT_PROMPT_LINES}

You also have read-only tools to inspect the scene (object/robot
positions, obstacles, reachability, and pick/place path planning).
Call as many of these as you need before deciding. When you are
ready to act, reply with the JSON action described below as plain
text (not as a tool call).

=== CRITICAL RULES ===
1. READ "DO NOW" CAREFULLY - execute ONLY that one task
2. NEVER pick an object from a conveyor slot position (x > 0.60)
3. NEVER pick if already holding something - place first
4. Objects to pick are ALWAYS on the table (x < 0.60)
5. If a slot is occupied - use the next empty slot
6. If object is unreachable - report failed immediately

RESPONSE FORMAT - always return valid JSON:
{{
  "thinking": "one sentence about current situation",
  "action": "pick" | "place_slot" | "place" | "stack" | "scan" | "home" | "done" | "failed",
  "params": {{
    "object": "red",    (for pick, stack)
    "slot": 2,          (for place_slot)
    "on": "green",      (for stack - object to stack ON TOP OF)
    "x": 0.4,           (for place)
    "y": 0.1            (for place)
  }},
  "message": "what you are doing"
}}

When current task is complete: {{"action": "done", "message": "task done"}}
When truly impossible: {{"action": "failed", "message": "specific reason"}}
"""


class EmergencyStop:
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


EMERGENCY_STOP = EmergencyStop()


# =============================================================
#   FIXED OBSERVATION BUILDER
#   Key fix: clearly separates DONE / DO NOW / NOT YET
#   So Claude cannot confuse completed with pending tasks
# =============================================================

def _fit_field(value, width=36):
    """Render value into a fixed-width field, truncating with an ellipsis."""
    s = str(value)
    if len(s) > width:
        return s[:width - 1] + "…"
    return s.ljust(width)


def build_observation(robot, completed_tasks, failed_tasks,
                      current_subtask, future_subtasks,
                      last_result=None):
    """
    Build crystal clear observation for Claude.

    FIX FOR BUG 1 & 4:
    Instead of showing all remaining tasks at once,
    we show exactly three sections:
      DONE     - do not repeat these
      DO NOW   - execute only this one task
      NOT YET  - do not do these yet

    This prevents Claude from jumping ahead or
    repeating completed tasks.
    """

    # Get scene state
    positions = robot.get_all_object_positions()
    holding   = robot.grasped_object or "nothing"
    robot_pos = robot.get_end_effector_position()
    obstacles = list(robot.obstacles.keys())

    # Build object summary
    # FIX FOR BUG 2: clearly label which objects are on table vs slots
    slot_positions = set()
    if hasattr(robot, 'conveyor_slots'):
        for sn, (sx, sy) in robot.conveyor_slots.items():
            slot_positions.add(sn)

    object_lines   = []
    table_objects  = []  # objects safe to pick
    slot_objects   = []  # objects in slots - do not pick

    for name, pos in positions.items():
        if pos:
            reach  = robot.check_reachability(pos[0], pos[1], pos[2])
            icon   = {"ok": "✅", "warning": "⚠️",
                      "error": "❌"}.get(reach["severity"], "?")
            dist   = reach.get("distance", 0)

            # Check if object is in a slot position
            in_slot = False
            slot_num = None
            if hasattr(robot, 'slot_occupants'):
                for sn, occupant in robot.slot_occupants.items():
                    if occupant == name:
                        in_slot  = True
                        slot_num = sn
                        break

            if in_slot:
                slot_objects.append(
                    f"  🚫 {name}: IN SLOT {slot_num} "
                    f"— DO NOT PICK THIS"
                )
            else:
                table_objects.append(
                    f"  {icon} {name}: "
                    f"pos=({pos[0]:.2f}, {pos[1]:.2f}) "
                    f"dist={dist}m reach={reach['severity']}"
                )

    # Build slot status
    slot_lines = []
    if hasattr(robot, 'conveyor_slots'):
        for sn, (sx, sy) in robot.conveyor_slots.items():
            occupant = robot.slot_occupants.get(sn, "EMPTY")
            status   = f"occupied by {occupant}" if occupant != "EMPTY" else "EMPTY ← can place here"
            slot_lines.append(f"  Slot {sn}: ({sx}, {sy:+.2f}) — {status}")

    # Last result
    if last_result:
        result_str = (
            f"Last action result: {last_result.get('status', '?').upper()} "
            f"— {last_result.get('detail', '')}"
        )
    else:
        result_str = "Last action result: none (starting fresh)"

    # Fixed-width fields for the DONE / DO NOW / NOT YET box.
    # Computed before the f-string so truncation always keeps
    # the box border intact, regardless of list length.
    done_field   = _fit_field(completed_tasks if completed_tasks else 'none yet')
    donow_field  = _fit_field(current_subtask)
    notyet_field = _fit_field(future_subtasks[:2] if future_subtasks else 'none')

    # Build observation with clear DO NOW section
    obs = f"""=== ROBOT STATE ===
Holding: {holding}
Robot at: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f})

=== CONVEYOR SLOTS (DESTINATIONS — NEVER PICK FROM THESE) ===
{chr(10).join(slot_lines) if slot_lines else "  Slot 1/2/3 available"}

=== OBJECTS ON TABLE (SAFE TO PICK) ===
{chr(10).join(table_objects) if table_objects else "  none on table"}

=== OBJECTS IN SLOTS (DO NOT PICK THESE) ===
{chr(10).join(slot_objects) if slot_objects else "  none in slots yet"}

=== OBSTACLES ===
{', '.join(obstacles) if obstacles else 'none'}

=== {result_str} ===

╔══════════════════════════════════════╗
║ TASKS ALREADY DONE (do not repeat): ║
║ {done_field} ║
╠══════════════════════════════════════╣
║ DO NOW (execute only this):          ║
║ {donow_field} ║
╠══════════════════════════════════════╣
║ NOT YET (do not do these yet):       ║
║ {notyet_field} ║
╚══════════════════════════════════════╝"""

    return obs


def decompose_order(order, client, model, known_objects=None):
    """Decompose natural language order into subtask list."""
    print(f"\n  🧠 Decomposing order: '{order}'")

    obj_note = ""
    if known_objects:
        obj_note = f"\nONLY use these objects: {known_objects}"

    system = (
        "Decompose a warehouse robot order into subtasks.\n"
        "Return ONLY a JSON array of subtasks, nothing else.\n\n"
        "Subtask formats:\n"
        '{"action": "pick", "object": "red"}\n'
        '{"action": "place_slot", "slot": 1}\n'
        '{"action": "place", "x": 0.4, "y": 0.1}\n'
        '{"action": "stack", "object": "blue", "on": "green"}\n'
        '{"action": "scan"}\n'
        '{"action": "home"}\n\n'
        "CONVEYOR SLOTS:\n"
        + _SLOT_PROMPT_LINES + "\n"
        "Use place_slot for dispatch/conveyor tasks.\n"
        "Each object goes to a DIFFERENT slot.\n"
        + obj_note + "\n\n"
        "Rules:\n"
        "- Pick before place or stack\n"
        "- Each object to different slot (not all to slot 2)\n"
        "- NEVER invent objects not in the known list\n"
        "- Return ONLY the JSON array"
    )

    response = client.messages.create(
        model=model,
        max_tokens=500,
        system=system,
        messages=[{"role": "user",
                   "content": f"Decompose this order: {order}"}]
    )

    text = response.content[0].text.strip()

    try:
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
        print(f"  ⚠️  Parse failed, using fallback")
        return [{"action": "scan"}]


def deterministic_decision(subtask):
    """
    Turn a recipe step straight into an action, with no language
    model. Used for PLC-selected recipes and in degraded mode.
    """
    action = subtask.get("action")
    params = {k: v for k, v in subtask.items() if k != "action"}
    return {
        "thinking": "pre-approved recipe step",
        "action":   action,
        "params":   params,
        "message":  f"recipe step: {subtask}",
    }


def match_recipe(order):
    """Map a free-text order onto a pre-approved recipe, or None."""
    from plc.recipes import RECIPES
    text = (order or "").lower()
    if "dispatch" in text and ("all" in text or "block" in text
                               or "part" in text):
        return list(RECIPES[1]["subtasks"])
    for recipe in RECIPES.values():
        if recipe["name"] in text:
            return list(recipe["subtasks"])
    return None


class WarehouseAgent:
    """
    Autonomous warehouse order fulfilment agent.
    Fixed version with clear subtask tracking.
    """

    def __init__(self, robot, ai_config, logger, audit=None,
                 verifier=None, llm_enabled=True):
        self.robot       = robot
        self.logger      = logger
        self.audit       = audit or NullAudit()

        # The language model is optional. Without it (no key, no
        # network, or llm_enabled=False) the agent runs pre-approved
        # recipes through the same gated, verified pipeline.
        #   llm_enabled=False : recipe mode, by design
        #   degraded=True     : the model was wanted but is unavailable
        self.client      = None
        self.llm_enabled = llm_enabled
        self.degraded    = False
        self.degraded_reason = None
        if llm_enabled:
            if ai_config.get("api_key"):
                try:
                    self.client = anthropic.Anthropic(
                        api_key=ai_config["api_key"],
                        timeout=ai_config.get("timeout_s", 20.0),
                        max_retries=ai_config.get("api_retries", 1),
                    )
                except Exception as e:
                    self._enter_degraded(f"client init failed: {e}")
            else:
                self._enter_degraded("no API key configured")

        self.model       = ai_config["model"]
        self.e_stop      = EMERGENCY_STOP
        self.max_retries = 3
        self.verifier = verifier or PlacementVerifier(robot)
        self.tools = RobotTools(robot, logger)
        self.max_tool_turns = 6

    def _enter_degraded(self, reason):
        """Switch to recipe-only operation. Logged, never silent."""
        if not self.degraded:
            print(f"  ⚠️  DEGRADED MODE: {reason} — "
                  f"continuing without the language model")
        self.degraded = True
        self.degraded_reason = reason
        self.audit.log("degraded_mode", reason=reason)

    # Candidate table spots to set a held object down aside, checked
    # in order for reachability and clearance from existing objects.
    PICK_ASIDE_CANDIDATES = [
        (0.45,  0.15), (0.45, -0.15),
        (0.35,  0.20), (0.35, -0.20),
        (0.30,  0.00), (0.50,  0.00),
    ]

    def _find_pick_aside_spot(self, clearance=0.08):
        """
        Find a free, reachable table spot to set a held object down.
        Returns (x, y) or None if no candidate is safe.
        """
        occupied = [
            pos for pos in self.robot.get_all_object_positions().values()
            if pos
        ]
        for x, y in self.PICK_ASIDE_CANDIDATES:
            reach = self.robot.check_reachability(x, y)
            if not reach["reachable"]:
                continue
            if any(math.hypot(x - p[0], y - p[1]) < clearance
                   for p in occupied):
                continue
            return (x, y)
        return None

    def run(self, order, subtasks=None):
        """
        order    : free-text order (decomposed by the language model)
        subtasks : pre-approved recipe steps; skips decomposition
        """
        print(f"\n{'='*50}")
        print(f"  🤖 WAREHOUSE AGENT STARTING")
        print(f"  📦 Order: {order}")
        print(f"{'='*50}\n")

        self.e_stop.reset()
        self.verifier.load_calibration()

        # A new order gets a fresh attempt at the language model.
        if self.llm_enabled and self.client is not None:
            self.degraded = False
            self.degraded_reason = None

        # Reset slot occupancy
        if hasattr(self.robot, "slot_occupants"):
            self.robot.slot_occupants = {}
            print("  [SLOT] Slot occupancy reset for new order")

        # Decompose
        known_objects = list(self.robot.objects.keys())
        print(f"  📦 Known objects in scene: {known_objects}")
        self.audit.log("order_received", order=order,
                       recipe=subtasks is not None,
                       known_objects=known_objects)
        if subtasks is None:
            subtasks = self._decompose(order, known_objects)
        if subtasks is None:
            reason = (f"language model unavailable "
                      f"({self.degraded_reason or 'recipe mode'}) and the "
                      f"order matches no pre-approved recipe")
            print(f"  ❌ {reason}")
            failed = [{"task": order, "reason": reason}]
            self.audit.log("order_result", success=False, failed=failed)
            return self._build_result(False, [], failed, 0, reason)
        self.audit.log("order_decomposed", subtasks=subtasks,
                       degraded=self.degraded)
        remaining  = list(subtasks)
        completed  = []
        failed     = []
        steps      = 0
        last_result = None

        while remaining:

            if self.e_stop.is_active():
                print(f"\n  🛑 Agent stopped: {self.e_stop.reason}")
                self.robot.home()
                return self._build_result(
                    False, completed, failed, steps,
                    f"stopped: {self.e_stop.reason}"
                )

            # FIX FOR BUG 1:
            # current_subtask is ALWAYS remaining[0]
            # future_subtasks is remaining[1:]
            # completed is the done list
            # Claude sees these separately - no confusion
            current_subtask  = remaining[0]
            future_subtasks  = remaining[1:]
            retries          = 0

            print(f"\n  📌 Current subtask: {current_subtask}")

            while retries < self.max_retries:

                if self.e_stop.is_active():
                    break

                steps += 1

                # Build observation with clear sections
                obs = build_observation(
                    robot           = self.robot,
                    completed_tasks = completed,
                    failed_tasks    = failed,
                    current_subtask = current_subtask,
                    future_subtasks = future_subtasks,
                    last_result     = last_result
                )

                print(f"\n  👁️  Observing scene...")
                decision = self._decide(obs, current_subtask, retries)

                if not decision:
                    retries += 1
                    continue

                print(f"  💭 Thinking: {decision.get('thinking', '')}")
                print(f"  ▶️  Action:   {decision.get('message', '')}")

                if decision["action"] == "done":
                    completed.append(str(current_subtask))
                    remaining.pop(0)
                    last_result = {"action": "done", "status": "success",
                                   "detail": "subtask complete"}
                    break

                if decision["action"] == "failed":
                    print(f"  ❌ Agent gave up: {decision.get('message')}")
                    failed.append({
                        "task":   str(current_subtask),
                        "reason": decision.get("message", "unknown")
                    })
                    remaining.pop(0)
                    break

                # Safety: if trying to pick but holding something
                if (decision.get("action") == "pick"
                        and self.robot.grasped_object):
                    held  = self.robot.grasped_object
                    aside = self._find_pick_aside_spot()
                    if aside is None:
                        last_result = {
                            "action": "pick", "status": "failed",
                            "detail": f"holding {held}, no free "
                                      f"reachable table spot to place aside"
                        }
                        retries += 1
                        print(f"  ⚠️  Holding '{held}' — no free spot to "
                              f"place aside. Attempt "
                              f"{retries}/{self.max_retries}")
                        continue
                    print(f"  ⚠️  Holding '{held}' — placing aside at "
                          f"({aside[0]:.2f}, {aside[1]:.2f})")
                    self.robot.place_object(*aside)

                last_result = self._execute_action(
                    decision, current_subtask
                )
                self.audit.log("action_result", subtask=current_subtask,
                               result=last_result)

                status_icon = "✅" if last_result["status"] == "success" else "❌"
                print(f"  {status_icon} Result: "
                      f"{last_result['status']} — {last_result['detail']}")

                if last_result["status"] == "success":
                    if self._subtask_complete(current_subtask, last_result):
                        completed.append(str(current_subtask))
                        remaining.pop(0)
                        break
                    else:
                        retries += 1
                        print(f"  ⚠️  Attempt {retries}/{self.max_retries}: "
                              f"action '{last_result.get('action')}' succeeded "
                              f"but did not complete subtask {current_subtask}")
                else:
                    retries += 1
                    print(f"  ⚠️  Attempt {retries}/{self.max_retries} failed. "
                          f"Reason: {last_result.get('detail')}")

            else:
                print(f"  ❌ Subtask failed after {self.max_retries} attempts")
                failed.append({
                    "task":   str(current_subtask),
                    "reason": last_result.get("detail", "max retries")
                              if last_result else "unknown"
                })
                remaining.pop(0)

        success = len(failed) == 0
        summary = self._build_summary(completed, failed, steps)
        print(f"\n{'='*50}")
        print(f"  {'✅ ORDER COMPLETE' if success else '⚠️  ORDER PARTIAL'}")
        print(f"  {summary}")
        print(f"{'='*50}\n")

        self.verifier.save_calibration()
        self.verifier.get_calibration_report()
        self.audit.log("order_result", success=success,
                       completed=completed, failed=failed,
                       steps=steps, degraded=self.degraded)
        return self._build_result(success, completed, failed, steps, summary)

    # ----------------------------------------------------------
    # Language model with deterministic fallback
    # ----------------------------------------------------------

    def _use_llm(self):
        return (self.llm_enabled and self.client is not None
                and not self.degraded)

    def _decompose(self, order, known_objects):
        """LLM decomposition; falls back to a matching recipe."""
        if self._use_llm():
            try:
                return decompose_order(order, self.client, self.model,
                                       known_objects=known_objects)
            except Exception as e:
                self._enter_degraded(f"decomposition failed: {e}")
        return match_recipe(order)

    def _decide(self, obs, current_subtask, retries):
        """Next action: language model, or the recipe step itself."""
        if self._use_llm():
            print(f"  🧠 Claude reasoning (attempt {retries + 1})...")
            self._llm_error = None
            decision = self._get_decision(obs, current_subtask, retries)
            if self._llm_error is None:
                if decision:
                    self.audit.log("llm_decision", subtask=current_subtask,
                                   decision=decision)
                return decision
            self._enter_degraded(f"decision call failed: {self._llm_error}")
        decision = deterministic_decision(current_subtask)
        self.audit.log("recipe_decision", subtask=current_subtask,
                       decision=decision)
        return decision

    def _get_decision(self, observation, current_subtask, retry_count):
        """
        Ask Claude for next action.

        Wires the RobotTools schemas into a real tool-use loop: Claude
        may call read-only inspection/planning tools any number of times
        (up to max_tool_turns) before responding with the final JSON
        action, which is parsed from its plain-text reply.
        """
        user_msg = (
            f"{observation}\n\n"
            f"YOUR TASK RIGHT NOW: {current_subtask}\n"
            f"{'⚠️  PREVIOUS ATTEMPT FAILED — try different approach' if retry_count > 0 else ''}\n\n"
            f"Return JSON only. Execute the DO NOW task."
        )
        messages = [{"role": "user", "content": user_msg}]

        for _ in range(self.max_tool_turns):
            try:
                response = self.client.messages.create(
                    model       = self.model,
                    max_tokens  = 500,
                    system      = ORCHESTRATOR_PROMPT,
                    tools       = self.tools.get_tool_definitions(),
                    messages    = messages
                )
            except Exception as e:
                print(f"  ⚠️  Decision error: {e}")
                self._llm_error = e
                return None

            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                text = "".join(
                    b.text for b in response.content if b.type == "text"
                ).strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        print(f"  ⚠️  Could not parse JSON from reply: {text[:200]}")
                        return None
                return None

            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in tool_uses:
                result_str = self.tools.execute_tool(block.name, block.input)
                self.audit.log("tool_call", tool=block.name,
                               input=block.input, result=result_str[:500])
                tool_results.append({
                    "type":        "tool_result",
                    "tool_use_id": block.id,
                    "content":     result_str
                })
            messages.append({"role": "user", "content": tool_results})

        print("  ⚠️  Too many tool calls without a final decision")
        return None

    def _execute_action(self, decision, current_subtask):
        """Execute action and return status dict."""
        action = decision.get("action")
        params = decision.get("params", {})

        try:
            # ──────────────────────────────────────────
            if action == "scan":
                positions = self.robot.get_all_object_positions()
                return {
                    "action": "scan",
                    "status": "success",
                    "detail": f"found {len(positions)} objects: "
                              f"{list(positions.keys())}"
                }

            # ──────────────────────────────────────────
            elif action == "pick":
                obj = params.get("object", "")
                if not obj:
                    return {"action": "pick", "status": "failed",
                            "detail": "no object specified"}

                # FIX FOR BUG 2:
                # Block picking from slot positions
                if hasattr(self.robot, "slot_occupants"):
                    for sn, occupant in self.robot.slot_occupants.items():
                        if occupant == obj:
                            return {
                                "action": "pick",
                                "status": "failed",
                                "detail": f"{obj} is in slot {sn} — "
                                          f"cannot pick from conveyor slot"
                            }

                # FIX FOR BUG 3:
                # Check object has not drifted out of reach
                pos = self.robot.get_object_position(obj)
                if pos:
                    reach = self.robot.check_reachability(
                        pos[0], pos[1], pos[2]
                    )
                    dist = reach.get("distance", 0)
                    if not reach["reachable"]:
                        print(f"  [REACH] ❌ {obj} unreachable ({dist}m)")
                        return {
                            "action": "pick",
                            "status": "failed",
                            "detail": f"{obj} is at {dist}m — "
                                      f"out of reach (safe zone: 0.30-0.65m)"
                        }
                    elif reach["severity"] == "warning":
                        print(f"  [REACH] ⚠️  {obj} at edge ({dist}m) — attempting")
                    else:
                        print(f"  [REACH] ✅ {obj} reachable ({dist}m)")

                result = self.robot.pick_object(obj)

                if isinstance(result, bool):
                    result = {
                        "action": "pick",
                        "status": "success" if result else "failed",
                        "target": obj,
                        "detail": f"picked {obj} from "
                                  f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})"
                                  if result and pos
                                  else f"failed to pick {obj}"
                    }

                # Clear slot if object was tracked there
                if result.get("status") == "success":
                    if hasattr(self.robot, "slot_occupants"):
                        for sn, occupant in list(
                            self.robot.slot_occupants.items()
                        ):
                            if occupant == obj:
                                del self.robot.slot_occupants[sn]
                                print(f"  [SLOT] Slot {sn} cleared "
                                      f"(picked {obj})")
                return result

            # ──────────────────────────────────────────
            elif action in ("place", "place_slot"):

                slot_num = params.get("slot") or params.get("slot_number")

                if not hasattr(self.robot, "conveyor_slots"):
                    self.robot.conveyor_slots = dict(CONVEYOR_SLOTS)
                if not hasattr(self.robot, "slot_occupants"):
                    self.robot.slot_occupants = {}

                slots    = self.robot.conveyor_slots
                occupied = self.robot.slot_occupants

                if slot_num is not None:
                    # Explicit slot requested (place_slot, or place with
                    # a slot param) — resolve to a real slot coordinate.
                    slot_num = int(slot_num)
                    if slot_num not in slots:
                        return {"action": "place", "status": "failed",
                                "detail": f"slot {slot_num} does not exist"}

                    occupant = occupied.get(slot_num)
                    if occupant:
                        # FIX FOR BUG 5:
                        # Instead of auto-assigning silently
                        # find next empty and tell Claude clearly
                        next_slot = None
                        for sn in sorted(slots.keys()):
                            if sn not in occupied:
                                next_slot = sn
                                break
                        if next_slot is None:
                            return {
                                "action": "place",
                                "status": "failed",
                                "detail": f"slot {slot_num} occupied by "
                                          f"{occupant} and no empty slots left"
                            }
                        print(f"  [SLOT] Slot {slot_num} occupied — "
                              f"using slot {next_slot} instead")
                        slot_num = next_slot

                    x_final, y_final = slots[slot_num]
                    print(f"  [SLOT] Placing in slot {slot_num} "
                          f"at ({x_final}, {y_final})")

                elif action == "place_slot":
                    # place_slot with no slot number given — fall back
                    # to the next empty slot. Still slot-only, never
                    # touches literal place(x, y) coordinates.
                    slot_num = None
                    for sn in sorted(slots.keys()):
                        if sn not in occupied:
                            slot_num = sn
                            break
                    if slot_num is None:
                        return {
                            "action": "place_slot",
                            "status": "failed",
                            "detail": "all conveyor slots are occupied"
                        }
                    x_final, y_final = slots[slot_num]
                    print(f"  [SLOT] Auto-assigned slot {slot_num} "
                          f"at ({x_final}, {y_final})")

                else:
                    # FIX FOR BUG E: action == "place" with explicit
                    # coordinates is a literal table placement — never
                    # silently hijacked into a conveyor slot.
                    x_final = float(params.get("x", 0.4))
                    y_final = float(params.get("y", 0.0))
                    slot_num = None

                # Check holding something
                if not self.robot.grasped_object:
                    return {"action": "place", "status": "failed",
                            "detail": "not holding anything"}

                held_obj = self.robot.grasped_object
                commanded = None
                if slot_num is not None:
                    # Send the robot to the calibrated position so a
                    # known slot offset is corrected on the first place.
                    correct = getattr(self.verifier, "corrected_target", None)
                    if correct:
                        commanded = correct(x_final, y_final, slot_num)
                        x_final, y_final = commanded
                result   = self.robot.place_object(x_final, y_final)

                # Update slot occupancy on success
                if slot_num is not None:
                    ok = (result.get("status") == "success"
                          if isinstance(result, dict) else bool(result))
                    if ok:
                        # Live verification - check and fix drift
                        sx, sy = slots[slot_num]
                        verify = self.verifier.verify_and_fix(
                            held_obj, sx, sy,
                            slot_num=slot_num, commanded=commanded
                        )
                        self.audit.log("verification", part=held_obj,
                                       slot=slot_num, result=verify)
                        if verify["status"] == "failed":
                            return {
                                "action": "place",
                                "status": "failed",
                                "target": held_obj,
                                "detail": verify["detail"]
                            }
                        self.robot.slot_occupants[slot_num] = held_obj
                        print(f"  [SLOT] ✅ Slot {slot_num} "
                              f"occupied by {held_obj}")

                if isinstance(result, bool):
                    return {
                        "action": "place",
                        "status": "success" if result else "failed",
                        "target": held_obj,
                        "detail": f"placed {held_obj} at "
                                  f"({x_final:.2f},{y_final:.2f})"
                                  if result
                                  else f"failed to place {held_obj}"
                    }
                return result

            # ──────────────────────────────────────────
            elif action == "stack":
                obj    = (params.get("object") or params.get("item")
                          or decision.get("object", ""))
                target = (params.get("on") or params.get("target")
                          or params.get("onto")
                          or decision.get("on", "")
                          or current_subtask.get("on", ""))

                if not obj or not target:
                    return {"action": "stack", "status": "failed",
                            "detail": "missing object or target for stack"}

                if self.robot.grasped_object == obj:
                    print(f"  [STACK] Already holding {obj} — skipping pick")
                    pick_ok = True
                elif self.robot.grasped_object:
                    held  = self.robot.grasped_object
                    aside = self._find_pick_aside_spot()
                    if aside is None:
                        return {
                            "action": "stack", "status": "failed",
                            "detail": f"holding {held}, no free "
                                      f"reachable table spot to place aside"
                        }
                    print(f"  [STACK] Holding {held}, placing aside at "
                          f"({aside[0]:.2f}, {aside[1]:.2f})")
                    self.robot.place_object(*aside)
                    pick_result = self.robot.pick_object(obj)
                    pick_ok = (pick_result.get("status") == "success"
                               if isinstance(pick_result, dict)
                               else pick_result)
                else:
                    pick_result = self.robot.pick_object(obj)
                    pick_ok = (pick_result.get("status") == "success"
                               if isinstance(pick_result, dict)
                               else pick_result)

                if not pick_ok:
                    return {"action": "stack", "status": "failed",
                            "detail": f"could not pick {obj}"}

                target_pos = self.robot.get_object_position(target)
                if not target_pos:
                    return {"action": "stack", "status": "failed",
                            "detail": f"target '{target}' not found"}

                place_result = self.robot.place_object(
                    target_pos[0], target_pos[1], stack_on=target
                )
                ok = (place_result.get("status") == "success"
                      if isinstance(place_result, dict)
                      else place_result)

                return {
                    "action": "stack",
                    "status": "success" if ok else "failed",
                    "target": f"{obj} on {target}",
                    "detail": f"stacked {obj} on {target} ✅ verified"
                              if ok else "stack failed"
                }

            # ──────────────────────────────────────────
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
        if last_result["status"] != "success":
            return False
        action    = subtask.get("action")
        completed = last_result.get("action")
        if action == completed:
            return True
        if action == "stack" and completed in ["stack", "place"]:
            return True
        if action == "place_slot" and completed in ["place", "place_slot"]:
            return True
        return False

    def _build_summary(self, completed, failed, steps):
        total       = len(completed) + len(failed)
        failed_list = [f["task"] for f in failed]
        outcome     = ("All tasks done!"
                       if not failed
                       else f"Failed: {failed_list}")
        return (f"Completed {len(completed)}/{total} subtasks "
                f"in {steps} action steps. {outcome}")

    def _build_result(self, success, completed, failed, steps, summary):
        return {
            "success":     success,
            "completed":   completed,
            "failed":      failed,
            "steps_taken": steps,
            "summary":     summary,
            "degraded":    self.degraded,
        }