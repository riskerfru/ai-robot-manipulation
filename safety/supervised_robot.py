# =============================================================
#   SUPERVISED ROBOT
#
#   Wraps any RobotInterface backend so that every motion call
#   is authorised by the SafetySupervisor first and recorded in
#   the audit trail. The agent and the placement verifier are
#   handed this wrapper, never the raw backend, so there is no
#   code path from the language model to an ungated motion.
# =============================================================

from audit.trail import NullAudit
from safety.supervisor import MotionCommand

_OWN = {"_robot", "_safety", "_audit", "_part_mass", "_source"}


class SupervisedRobot:

    def __init__(self, robot, safety, audit=None, part_mass_kg=0.5,
                 source="agent"):
        object.__setattr__(self, "_robot", robot)
        object.__setattr__(self, "_safety", safety)
        object.__setattr__(self, "_audit", audit or NullAudit())
        object.__setattr__(self, "_part_mass", part_mass_kg)
        object.__setattr__(self, "_source", source)

    # Everything that is not a gated motion goes to the backend.
    def __getattr__(self, name):
        return getattr(self._robot, name)

    def __setattr__(self, name, value):
        if name in _OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self._robot, name, value)

    @property
    def safety(self):
        return self._safety

    @property
    def backend(self):
        return self._robot

    # ---- gate --------------------------------------------------
    def _gate(self, command):
        decision = self._safety.authorize(command)
        self._audit.log("safety_decision",
                        kind=command.kind, target=command.target,
                        label=command.label, source=command.source,
                        **decision.as_dict())
        return decision

    def _denied(self, action, target, decision):
        return {
            "status": "failed", "action": action, "target": target,
            "safety_denied": True, "safety_code": decision.code.value,
            "detail": f"SAFETY DENIED [{decision.code.value} "
                      f"{decision.code.name}]: {decision.reason}",
        }

    def _after_motion(self, action, result):
        try:
            tcp = self._robot.get_end_effector_position()
            check = self._safety.monitor_tcp(tcp)
            if not check.allowed:
                self._audit.log("safety_trip", action=action,
                                tcp=tuple(tcp), **check.as_dict())
                try:
                    self._robot.stop()
                except Exception:
                    pass
        except Exception as exc:
            self._audit.log("monitor_error", action=action, error=str(exc))
        self._audit.log("motion_result", action=action,
                        result=_summarise(result))
        return result

    # ---- gated motions -----------------------------------------
    def move_to_position(self, x, y, z, *args, **kwargs):
        d = self._gate(MotionCommand("move", (x, y, z),
                                     source=self._source))
        if not d.allowed:
            return False
        return self._after_motion(
            "move", self._robot.move_to_position(x, y, z, *args, **kwargs))

    def pick_object(self, name):
        pos = self._robot.get_object_position(name)
        if not pos:
            self._audit.log("motion_result", action="pick",
                            result={"status": "failed",
                                    "detail": f"{name} not found"})
            return {"status": "failed", "action": "pick", "target": name,
                    "detail": f"{name} not found in the cell"}
        d = self._gate(MotionCommand("pick", tuple(pos), label=name,
                                     payload_kg=self._part_mass,
                                     source=self._source))
        if not d.allowed:
            return self._denied("pick", name, d)
        return self._after_motion("pick", self._robot.pick_object(name))

    def place_object(self, x, y, *args, **kwargs):
        held = getattr(self._robot, "grasped_object", None)
        tcp_z = self._safety.limits.workspace_z[0] + 0.10
        d = self._gate(MotionCommand("place", (x, y, tcp_z),
                                     label=str(held),
                                     payload_kg=self._part_mass,
                                     source=self._source))
        if not d.allowed:
            return self._denied("place", held, d)
        return self._after_motion(
            "place", self._robot.place_object(x, y, *args, **kwargs))

    def insert_part(self, name, pocket, *args, **kwargs):
        tcp_z = self._safety.limits.workspace_z[0] + 0.10
        d = self._gate(MotionCommand("place", (pocket["x"], pocket["y"],
                                               tcp_z),
                                     label=f"insert {name}",
                                     payload_kg=self._part_mass,
                                     source=self._source))
        if not d.allowed:
            return self._denied("insert", name, d)
        return self._after_motion(
            "insert", self._robot.insert_part(name, pocket, *args, **kwargs))

    def home(self):
        d = self._gate(MotionCommand("home", source=self._source))
        if not d.allowed:
            return False
        return self._after_motion("home", self._robot.home())


def _summarise(result):
    if isinstance(result, dict):
        keep = ("status", "action", "target", "detail", "position",
                "verified_position", "seated_error_mm", "peak_force_n",
                "searched", "search_offset_mm")
        return {k: result[k] for k in keep if k in result}
    return {"status": "success" if result else "failed"}
