# =============================================================
#   GUARDIAN - fixed rules between the language model and the robot
#
#   The language model proposes one action at a time. Before the
#   base or the arm moves, every action is checked here against
#   rules that do not change and do not depend on understanding
#   the request:
#
#       is the action one the robot has?
#       is the station it touches open, or locked out?
#       is the machine mid-cycle?
#       is the part at the right point in its process route?
#       is the gripper in the right state?
#       is the speed within limits?
#
#   A refused action comes back with a code and a sentence, which
#   go to the model so it can propose something else. The model
#   is never trusted to know the rules; it is only told when it
#   broke one.
#
#   Each rule is written so it could be a rung of ladder logic on
#   a PLC: comparisons on a few integers and bits (station number,
#   part state code, machine busy, gripper closed). That mapping is
#   in RULES below. In this simulation they run in Python; they are
#   not safety-rated, and nothing here replaces a safety PLC.
# =============================================================

from mobile import floor as F
from mobile.base import MAX_SPEED
from mobile import cell as C

ACTIONS = {"goto", "pick", "place", "inspect", "wait_machine", "dispatch",
           "say", "done"}
ROLES = {"fixture", "pad", "bin", "tray", "table"}

# code: (short name, what the rung checks)
RULES = {
    "G01": ("UNKNOWN_ACTION", "action is one of the robot's skills"),
    "G02": ("BAD_ARGUMENT", "station / part / role exists"),
    "G03": ("STATION_LOCKED", "station lockout bit is off"),
    "G04": ("MACHINE_IN_CYCLE", "no approach or pick while MACHINE_BUSY"),
    "G05": ("NOT_AT_STATION", "base is docked where the part or fitting is"),
    "G06": ("GRIPPER_FULL", "gripper empty before a pick"),
    "G07": ("GRIPPER_EMPTY", "part in gripper before a place"),
    "G08": ("ROUTE_VIOLATION", "part state code matches what the fitting accepts"),
    "G09": ("FITTING_OCCUPIED", "fixture / pad empty before loading"),
    "G10": ("SPEED_LIMIT", f"commanded speed <= {MAX_SPEED} m/s"),
    "G11": ("NOT_INSPECTED", "only PASS parts may be dispatched"),
    "G12": ("NOTHING_TO_DO", "inspect / dispatch has something to act on"),
    "G13": ("STEP_LIMIT", "the plan has not run past its step budget"),
}


class Verdict:
    def __init__(self, ok, code=None, reason=""):
        self.ok, self.code, self.reason = ok, code, reason

    def __bool__(self):
        return self.ok

    def __str__(self):
        if self.ok:
            return "ACCEPT"
        return f"REJECT {self.code} {RULES[self.code][0]}: {self.reason}"


def _no(code, reason):
    return Verdict(False, code, reason)


class Guardian:

    def __init__(self, cell, max_steps=80):
        self.cell = cell
        self.max_steps = max_steps
        self.checked = 0
        self.rejected = []

    def check(self, action):
        v = self._check(action)
        self.checked += 1
        if not v:
            self.rejected.append((dict(action), v.code, v.reason))
        return v

    def _check(self, a):
        cell, base = self.cell, self.cell.base
        kind = a.get("action")
        if kind not in ACTIONS:
            return _no("G01", f"'{kind}' is not a skill")
        if self.checked >= self.max_steps:
            return _no("G13", f"{self.max_steps} actions already checked")
        here = base.station
        held = cell.held()
        busy = cell.machine["status"] == "busy"

        if kind in ("say", "done"):
            return Verdict(True)

        if kind == "wait_machine":
            if not busy and cell.machine["part"] is None:
                return _no("G12", "the machine is empty; nothing to wait for")
            return Verdict(True)

        if kind == "goto":
            st = a.get("station")
            if st not in F.STATIONS:
                return _no("G02", f"no station '{st}'")
            if st in cell.locked:
                return _no("G03", f"{st} is locked out")
            speed = float(a.get("speed", 0.4))
            if speed > MAX_SPEED:
                return _no("G10", f"{speed:.2f} m/s asked, limit "
                                  f"{MAX_SPEED:.2f}")
            if st == "MACHINE" and busy:
                return _no("G04", "the machine is mid-cycle; the robot may "
                                  "not enter its zone until it finishes")
            return Verdict(True)

        if here in cell.locked:
            return _no("G03", f"{here} is locked out")

        if kind == "pick":
            if held is not None:
                return _no("G06", f"already holding part {held.id}")
            ident = a.get("part")
            thing = cell.things.get(ident)
            if thing is None or thing.container:
                return _no("G02", f"no part {ident} known")
            if getattr(thing, "station", None) != here:
                return _no("G05", f"part {ident} is at "
                                  f"{getattr(thing, 'station', None)}, "
                                  f"robot is at {here}")
            if here == "MACHINE" and busy:
                return _no("G04", "the machine is mid-cycle")
            if cell.state.get(ident) in (C.DISPATCHED, C.SCRAPPED):
                return _no("G08", f"part {ident} is "
                                  f"{cell.state.get(ident)}")
            return Verdict(True)

        if kind == "place":
            role = a.get("into")
            if role not in ROLES:
                return _no("G02", f"no fitting '{role}'")
            if held is None:
                return _no("G07", "nothing in the gripper")
            if role == "table":
                return Verdict(True)
            if cell.equipment(role) is None:
                return _no("G05", f"there is no {role} at {here}")
            state = cell.state.get(held.id)
            accepts = C.ACCEPTS[role]
            if state not in accepts:
                if role == "tray":
                    return _no("G11", f"part {held.id} is {state}; only "
                                      f"PASS parts may be dispatched")
                return _no("G08", f"part {held.id} is {state}; the {role} "
                                  f"takes {'/'.join(sorted(accepts))}")
            if role in ("fixture", "pad") and cell.parts_in(role):
                return _no("G09", f"the {role} already holds part "
                                  f"{cell.parts_in(role)[0].id}")
            if role == "fixture" and busy:
                return _no("G04", "the machine is mid-cycle")
            return Verdict(True)

        if kind == "inspect":
            if here != "INSPECT":
                return _no("G05", "inspection happens at INSPECT")
            if not cell.parts_in("pad"):
                return _no("G12", "nothing on the inspection pad")
            return Verdict(True)

        if kind == "dispatch":
            if here != "DISPATCH":
                return _no("G05", "dispatch happens at DISPATCH")
            riders = cell.parts_in("tray")
            if not riders:
                return _no("G12", "the tray is empty")
            bad = [t.id for t in riders
                   if cell.state.get(t.id) != C.PASS]
            if bad:
                return _no("G11", f"tray holds parts {bad} that have not "
                                  f"passed inspection")
            return Verdict(True)

        return _no("G01", f"'{kind}' is not handled")
