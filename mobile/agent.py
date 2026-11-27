# =============================================================
#   AGENT - the language model drives the mobile robot, one
#   action at a time, through the guardian
#
#   Loop:
#       state  -> model proposes the next few actions
#       each action -> guardian ACCEPT / REJECT (code + reason)
#       ACCEPT -> robot does it, result recorded
#       REJECT -> stop, tell the model why, ask again
#
#   The robot only sees the station it is parked at, so the model
#   works the way a person on the floor would: go somewhere, look,
#   act, move on. It never sends a coordinate. It names a station,
#   a part number the camera gave, or a fitting ("fixture", "pad",
#   "bin", "tray", "table").
#
#   ScriptedPlanner does the same job without a language model, so
#   the cell can run and be tested offline. With mistakes=True it
#   deliberately proposes one illegal action of each common kind
#   first, to show the guardian refusing them. That is labelled in
#   the log as injected.
# =============================================================

import json
import time

import config as cfg
from mobile import cell as C
from mobile import floor as F
from mobile.guardian import Guardian, RULES

SYSTEM_PROMPT = """You control a mobile manipulator in a small factory cell:
a robot arm on an omnidirectional base that drives between four stations
along a lane: STORAGE, MACHINE, INSPECT, DISPATCH.

Process route for every part:
  STORAGE (RAW) -> MACHINE fixture (machined, ~4 s cycle, starts when the
  robot leaves) -> INSPECT pad (measured by camera -> PASS or FAIL)
  -> PASS goes to the DISPATCH tray; FAIL goes into the INSPECT reject bin.

You only see the station you are docked at; it is re-scanned by the mast
camera on every arrival. Parts are identified by number.

Actions (JSON objects):
  {"action":"goto","station":"MACHINE"}            optional "speed" m/s
  {"action":"pick","part":3}
  {"action":"place","into":"fixture"|"pad"|"bin"|"tray"|"table"}
  {"action":"inspect"}               measures the part on the pad
  {"action":"wait_machine"}          wait until the machine cycle ends
  {"action":"dispatch"}              send the DISPATCH tray out
  {"action":"say","text":"..."}
  {"action":"done"}                  the operator's request is complete

Inspection averages several camera frames and reports the size with its
uncertainty. A part too close to a limit to be sure is rejected on doubt.
Placing into a fixture or onto the pad is checked by the wrist force
sensor: if it touches too early (something in the way) or feels nothing,
the place stops, the part stays in the gripper, and that fitting is
blocked until the operator clears it. Then put the part on the table or
tell the operator.

Every action is checked by a fixed rule layer before it runs. If one is
rejected you are told the rule code and why; propose something else.
Do not argue with a rejection.

The operator may ask questions or give follow-up orders that refer to
earlier ones; earlier orders this session are listed. To answer a
question, use "say" and then "done".

Reply with JSON only:
{"thinking":"one short sentence","actions":[ ... up to 6 actions ... ]}
Prefer short batches; you will see the result and be asked again."""


def state_summary(cell):
    base = cell.base
    parts = []
    for ident, st in sorted(cell.state.items()):
        thing = cell.things.get(ident)
        where = ("gripper" if thing is not None and thing.held else
                 getattr(thing, "station", None) if thing else None)
        if st in (C.DISPATCHED, C.SCRAPPED):
            where = None
        inside = None
        if thing is not None and thing.inside is not None:
            box = cell.things.get(thing.inside)
            inside = getattr(box, "role", None)
        entry = {"part": ident, "state": st, "at": where}
        if inside:
            entry["in"] = inside
        if ident in cell.measured_mm:
            entry["measured_mm"] = cell.measured_mm[ident]
        parts.append(entry)
    visible = [{"part": t.id, "in": (getattr(cell.things.get(t.inside),
                                             "role", None)
                                     if t.inside else "table"),
                "size_mm": round(t.long * 1000)}
               for t in cell.at_station() if not t.container]
    fittings = sorted({getattr(t, "role", None) for t in cell.at_station()
                       if t.container} - {None})
    held = cell.held()
    return {
        "robot_at": base.station,
        "holding": held.id if held else None,
        "machine": {"status": cell.machine["status"],
                    "part": cell.machine["part"],
                    "fault": cell.machine["fault"]},
        "locked_out": sorted(cell.locked),
        "blocked_fittings": sorted(f"{r} at {s}" for s, r in
                                   getattr(cell, "blocked", ())),
        "seen_here": visible,
        "fittings_here": fittings,
        "parts": parts,
        "dispatched": cell.dispatched,
    }


# ---- doing ---------------------------------------------------------------
def execute(action, cell, log):
    """Carry out one accepted action. Returns (ok, result text)."""
    base, kind = cell.base, action["action"]
    if kind == "say":
        log("ROBOT", action.get("text", ""))
        return True, "said"
    if kind == "done":
        return True, "done"
    if kind == "goto":
        st = action["station"]
        if base.station == st:
            cell.scan()
            return True, f"already at {st}; rescanned"
        base.stow()
        out = base.go_to_station(st, speed=float(action.get("speed", 0.4)),
                                 log=log)
        if out is None:
            return False, f"no route to {st}"
        log("NAV", f"docked at {st}: {out['distance_m']:.2f} m driven, "
                   f"parked {out['parking_error_mm']:.1f} mm / "
                   f"{out['parking_error_deg']:+.1f} deg off the mark "
                   f"(unknown to the robot)")
        cell.scan()
        seen = [t for t in cell.at_station() if not t.container]
        log("CAMERA", f"mast camera re-measured {st}: " +
            (", ".join(f"part #{t.id}" for t in seen) if seen
             else "no parts"))
        return True, f"at {st}"
    if kind == "pick":
        thing = cell.things[action["part"]]
        r = cell.pick(thing)
        ok = r.get("status") == "success"
        grip = f", gripped at {r['grip_n']} N" if "grip_n" in r else ""
        return ok, (f"picked part {thing.id}{grip}" if ok
                    else r.get("detail", ""))
    if kind == "place":
        role = action["into"]
        ident = cell.held().id
        r = cell.place(role)
        ok = r.get("status") == "success"
        if not ok:
            return False, r.get("detail", "place failed")
        extra = (f", {r['placement_error_mm']} mm from fitting centre"
                 if "placement_error_mm" in r else "")
        if "force_n" in r and r.get("felt_contact"):
            extra += (f", touched down at {r['force_n']} N "
                      f"{1000 * (r['contact_z'] - r['expected_z']):+.1f} mm "
                      f"from expected")
        if role == "fixture" and ok:
            base.stow()
        return True, f"part {ident} placed in {role}{extra}"
    if kind == "inspect":
        r = cell.inspect()
        why = (" - too close to the limit to be sure, rejected on doubt"
               if r.get("doubt") else "")
        log("INSPECT", f"part {r['part']}: {r['size_mm']} mm "
                       f"+/- {r.get('U_mm', 0)} mm over {r.get('frames', 1)} "
                       f"frame(s) (nominal {F.NOMINAL_PART_MM:.0f} +/- "
                       f"{F.PART_TOLERANCE_MM:.0f}) -> {r['result']}{why}")
        return True, (f"part {r['part']} {r['result']} ({r['size_mm']} mm "
                      f"+/- {r.get('U_mm', 0)} mm{why})")
    if kind == "wait_machine":
        status = cell.wait_for_machine()
        return status == "done", f"machine {status}"
    if kind == "dispatch":
        n = cell.dispatch()
        log("DISPATCH", f"tray sent out with {n} part(s)")
        return True, f"dispatched {n}"
    return False, "unknown"


# ---- planners ----------------------------------------------------------------
class LLMPlanner:
    name = "language model"

    def __init__(self, client, model=None):
        self.client = client
        self.model = model or cfg.AI["model"]
        self.messages = []
        self.usage = {"input": 0, "output": 0, "calls": 0}
        self.history = []               # earlier orders this session

    def remember(self, command, outcome):
        self.history.append(f"{command!r} -> {outcome}")
        self.history = self.history[-6:]

    def start(self, command, cell):
        past = ("Earlier orders this session:\n" + "\n".join(
            f"- {h}" for h in self.history) + "\n\n") if self.history else ""
        self.messages = [{"role": "user", "content":
                          f"{past}Operator: {command}\n\nState:\n"
                          f"{json.dumps(state_summary(cell))}"}]

    def feedback(self, text, cell):
        self.messages.append({"role": "user", "content":
                              f"{text}\n\nState:\n"
                              f"{json.dumps(state_summary(cell))}"})

    def propose(self):
        reply = self.client.messages.create(
            model=self.model, max_tokens=900, system=SYSTEM_PROMPT,
            messages=self.messages)
        raw = reply.content[0].text
        u = getattr(reply, "usage", None)
        if u is not None:
            self.usage["input"] += getattr(u, "input_tokens", 0) or 0
            self.usage["output"] += getattr(u, "output_tokens", 0) or 0
        self.usage["calls"] += 1
        self.messages.append({"role": "assistant", "content": raw})
        text = raw.strip()
        start, end = text.find("{"), text.rfind("}")
        data = json.loads(text[start:end + 1])
        return data.get("thinking", ""), list(data.get("actions", []))


class ScriptedPlanner:
    """
    A fixed strategy, no language model: take each raw part through
    the route. With mistakes=True it first tries illegal actions, to
    exercise the guardian.
    """
    name = "scripted (no language model)"

    def __init__(self, mistakes=False):
        self.mistakes = mistakes
        self.cell = None
        self.tried = set()
        self.goal = "all"

    def start(self, command, cell):
        self.cell = cell
        self.goal = "one" if "one" in command.lower() else "all"

    def feedback(self, text, cell):
        pass

    def _inject(self, key, actions, why):
        if self.mistakes and key not in self.tried:
            self.tried.add(key)
            return f"[injected mistake] {why}", actions
        return None

    def propose(self):
        c = self.cell
        here, held, m = c.base.station, c.held(), c.machine
        st = c.state
        raw = [i for i, s in st.items() if s == C.RAW
               and getattr(c.things.get(i), "station", None) == "STORAGE"]
        machined = [i for i, s in st.items() if s == C.MACHINED]
        passed = [i for i, s in st.items() if s == C.PASS]
        failed = [i for i, s in st.items() if s == C.FAIL]
        in_fixture = m["part"] is not None
        if held is not None:
            s = st.get(held.id)
            if s == C.RAW:
                return "load the machine", self._go("MACHINE") + [
                    {"action": "place", "into": "fixture"},
                    {"action": "goto", "station": "INSPECT"}]
            if s == C.MACHINED:
                return "inspect it", self._go("INSPECT") + [
                    {"action": "place", "into": "pad"},
                    {"action": "inspect"}]
            if s == C.FAIL:
                bad = self._inject("tray_fail", self._go("DISPATCH") + [
                    {"action": "place", "into": "tray"}],
                    "try to ship a failed part")
                if bad:
                    return bad
                return "reject it", self._go("INSPECT") + [
                    {"action": "place", "into": "bin"}]
            if s == C.PASS:
                return "ship it", self._go("DISPATCH") + [
                    {"action": "place", "into": "tray"}]
        # Something on the pad after inspection.
        pad = [i for i in passed + failed
               if c.things.get(i) is not None
               and getattr(c.things[i], "station", None) == "INSPECT"]
        if pad:
            return "clear the pad", self._go("INSPECT") + [
                {"action": "pick", "part": pad[0]}]
        if m["status"] == "busy":
            bad = self._inject("enter_busy",
                               [{"action": "goto", "station": "MACHINE"}],
                               "drive into the machine mid-cycle")
            if bad:
                return bad
        if m["status"] == "done" or (m["status"] == "busy"
                                     and not raw):
            if m["status"] == "busy":
                return "wait for the machine", [{"action": "wait_machine"}]
            return "unload the machine", self._go("MACHINE") + [
                {"action": "pick", "part": m["part"]}]
        if raw and not in_fixture:
            bad = self._inject("fast", [{"action": "goto",
                                         "station": "STORAGE",
                                         "speed": 1.2}],
                               "ask for 1.2 m/s")
            if bad:
                return bad
            return "fetch a blank", self._go("STORAGE") + [
                {"action": "pick", "part": raw[0]}]
        if m["status"] == "busy":
            return "wait for the machine", [{"action": "wait_machine"}]
        trays = [i for i in passed if c.things.get(i) is not None
                 and getattr(c.things[i], "station", None) == "DISPATCH"]
        if trays:
            return "send the tray", self._go("DISPATCH") + [
                {"action": "dispatch"}]
        return "nothing left", [{"action": "done"}]

    def _go(self, station):
        if self.cell.base.station == station:
            return []
        return [{"action": "goto", "station": station}]


# ---- the loop ------------------------------------------------------------------
def run(command, cell, planner, log, max_rounds=40, guardian=None):
    guardian = guardian or Guardian(cell)
    planner.start(command, cell)
    log("OPERATOR", command)
    t0, rounds, executed = time.time(), 0, 0
    finished, report = False, []
    while rounds < max_rounds and not finished:
        rounds += 1
        try:
            thinking, actions = planner.propose()
        except Exception as e:          # bad JSON, network
            log("ERROR", f"planner reply unusable: {e}")
            planner.feedback(f"Your reply could not be used ({e}). "
                             f"Reply with JSON only.", cell)
            continue
        log("LLM" if isinstance(planner, LLMPlanner) else "PLAN",
            f"{thinking}  -> " + "; ".join(_short(a) for a in actions))
        if not actions:
            planner.feedback("You proposed no actions.", cell)
            continue
        report = []
        for action in actions:
            verdict = guardian.check(action)
            if not verdict:
                log("GUARD", f"{_short(action)} -> {verdict}")
                report.append(f"{_short(action)}: REJECTED {verdict.code} "
                              f"{RULES[verdict.code][0]} - {verdict.reason}")
                break
            log("GUARD", f"{_short(action)} -> ACCEPT")
            if action["action"] == "done":
                finished = True
                break
            ok, text = execute(action, cell, log)
            executed += 1
            if action["action"] not in ("goto",):
                log("ROBOT" if ok else "FAULT", text)
            report.append(f"{_short(action)}: {'ok' if ok else 'FAILED'}"
                          f" - {text}")
            if not ok:
                break
        planner.feedback("Results:\n" + "\n".join(report), cell)
    if hasattr(planner, "remember"):
        last = report[-1] if rounds and report else ""
        planner.remember(command, ("finished" if finished else "stopped")
                         + f" after {executed} actions"
                         + (f"; refused: {', '.join(r[1] for r in guardian.rejected)}"
                            if guardian.rejected else "")
                         + (f"; last: {last}" if last else ""))
    return {"finished": finished, "rounds": rounds, "executed": executed,
            "checked": guardian.checked, "rejected": guardian.rejected,
            "seconds": time.time() - t0, "sim_seconds": cell.now,
            "dispatched": cell.dispatched,
            "states": dict(cell.state), "driven_m": cell.base.odometer}


def _short(a):
    kind = a.get("action")
    if kind == "goto":
        sp = f" @{a['speed']}m/s" if "speed" in a else ""
        return f"goto {a.get('station')}{sp}"
    if kind == "pick":
        return f"pick #{a.get('part')}"
    if kind == "place":
        return f"place -> {a.get('into')}"
    if kind == "say":
        return f"say \"{a.get('text', '')[:40]}\""
    return str(kind)
