# =============================================================
#   SAFETY SUPERVISOR
#
#   A deterministic gate between "something wants the robot to
#   move" and "the robot moves". Every motion command from the
#   agent, the verifier or an operator passes through
#   authorize(). The language model has no way to configure,
#   reset or bypass this layer.
#
#   IMPORTANT - what this is and is not
#   This is a software interlock layer. It is NOT a safety-rated
#   function and does not replace one. In a real cell the
#   emergency stop, guard interlocks and speed/force limits are
#   implemented in safety-rated hardware (the robot's safety
#   controller and a safety PLC) to the performance level the
#   risk assessment demands. This supervisor mirrors those
#   signals so the software never commands motion the safety
#   system would reject, and so every refusal is logged with a
#   reason code.
#
#   Design rules
#   - Fail closed: any missing signal or internal error denies.
#   - Emergency stop latches: only an explicit operator reset
#     clears it, and only when the cause has cleared.
#   - No stochastic input: decisions depend on limits, interlock
#     signals and the commanded target, nothing else.
# =============================================================

import math
import time
from dataclasses import dataclass, field
from enum import Enum


class SafetyState(str, Enum):
    RUN             = "RUN"
    PROTECTIVE_STOP = "PROTECTIVE_STOP"   # recoverable after reset
    EMERGENCY_STOP  = "EMERGENCY_STOP"    # latched


class Deny(str, Enum):
    """Reason codes. Stable identifiers for the audit trail and HMI."""
    OK                 = "S000"
    ESTOP_LATCHED      = "S001"
    ESTOP_CIRCUIT_OPEN = "S002"
    GUARD_OPEN         = "S003"
    NOT_IN_AUTO        = "S004"
    NO_PLC_PERMIT      = "S005"
    ROBOT_NOT_READY    = "S006"
    INTERLOCK_STALE    = "S007"
    PROTECTIVE_STOP    = "S008"
    OUTSIDE_WORKSPACE  = "S010"
    OUTSIDE_REACH      = "S011"
    KEEP_OUT_ZONE      = "S012"
    SPEED_LIMIT        = "S013"
    PAYLOAD_LIMIT      = "S014"
    INVALID_COMMAND    = "S020"
    INTERNAL_ERROR     = "S099"


@dataclass(frozen=True)
class Zone:
    """Axis-aligned keep-out volume in the cell frame (metres)."""
    name: str
    x: tuple
    y: tuple
    z: tuple = (-math.inf, math.inf)

    def contains(self, px, py, pz):
        return (self.x[0] <= px <= self.x[1]
                and self.y[0] <= py <= self.y[1]
                and self.z[0] <= pz <= self.z[1])


@dataclass(frozen=True)
class SafetyLimits:
    """Static limits. Frozen: nothing can edit them at run time."""
    workspace_x: tuple = (-0.10, 0.80)
    workspace_y: tuple = (-0.60, 0.60)
    workspace_z: tuple = (0.05, 1.00)
    padding: float = 0.02               # shrink the workspace by this
    reach_min: float = 0.25             # radial distance from base
    reach_max: float = 0.75
    max_tcp_speed: float = 0.25         # m/s, collaborative-speed default
    max_payload_kg: float = 1.0
    interlock_max_age_s: float = 0.5    # stale interlock data = deny
    keep_out: tuple = field(default_factory=tuple)


@dataclass
class Interlocks:
    """Snapshot of the cell signals the supervisor depends on."""
    estop_ok: bool = False
    guard_closed: bool = False
    mode_auto: bool = False
    plc_permit: bool = False
    robot_ready: bool = False
    timestamp: float = 0.0              # time.monotonic() of the read


@dataclass(frozen=True)
class MotionCommand:
    kind: str                           # move | pick | place | home
    target: tuple = None                # (x, y, z) in the cell frame
    speed: float = None                 # requested TCP speed, m/s
    payload_kg: float = 0.0
    source: str = "agent"               # agent | verifier | operator
    label: str = ""                     # part name, slot, etc.


@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: Deny
    reason: str

    def as_dict(self):
        return {"allowed": self.allowed, "code": self.code.value,
                "code_name": self.code.name, "reason": self.reason}


class StaticInterlocks:
    """
    Interlock source for a stand-alone simulation with no PLC:
    every signal is healthy until a test or operator changes it.
    """

    def __init__(self):
        self.estop_ok = True
        self.guard_closed = True
        self.mode_auto = True
        self.plc_permit = True
        self.robot_ready = True

    def read(self):
        return Interlocks(self.estop_ok, self.guard_closed, self.mode_auto,
                          self.plc_permit, self.robot_ready, time.monotonic())


class SafetySupervisor:

    def __init__(self, limits=None, interlock_source=None, clock=None):
        self.limits = limits or SafetyLimits()
        self.interlocks = interlock_source or StaticInterlocks()
        self._clock = clock or time.monotonic
        self.state = SafetyState.RUN
        self.last_trip = None           # (Deny code, reason)
        self.denied_count = 0
        self.authorized_count = 0
        self._listeners = []

    # ---- listeners (audit trail, agent stop flag) --------------
    def on_trip(self, callback):
        """callback(state, code, reason) on every state change to a stop."""
        self._listeners.append(callback)

    def _notify(self, code, reason):
        for cb in self._listeners:
            try:
                cb(self.state, code, reason)
            except Exception:
                pass

    # ---- stops -------------------------------------------------
    def emergency_stop(self, reason="emergency stop"):
        self.state = SafetyState.EMERGENCY_STOP
        self.last_trip = (Deny.ESTOP_LATCHED, reason)
        self._notify(Deny.ESTOP_LATCHED, reason)

    def protective_stop(self, code, reason):
        if self.state is SafetyState.EMERGENCY_STOP:
            return
        self.state = SafetyState.PROTECTIVE_STOP
        self.last_trip = (code, reason)
        self._notify(code, reason)

    def reset(self, operator):
        """
        Operator acknowledgement. Refused while the cause persists.
        `operator` is an identifier for the audit trail; the agent
        never calls this.
        """
        if not operator:
            return Decision(False, Deny.INVALID_COMMAND,
                            "reset needs an operator identity")
        check = self._check_interlocks(self._read_interlocks())
        if not check.allowed:
            return Decision(False, check.code,
                            f"cannot reset: {check.reason}")
        self.state = SafetyState.RUN
        self.last_trip = None
        return Decision(True, Deny.OK, f"reset by {operator}")

    # ---- the gate ----------------------------------------------
    def authorize(self, command):
        try:
            decision = self._authorize(command)
        except Exception as exc:                     # fail closed
            decision = Decision(False, Deny.INTERNAL_ERROR,
                                f"supervisor error: {exc}")
        if decision.allowed:
            self.authorized_count += 1
        else:
            self.denied_count += 1
        return decision

    def _authorize(self, cmd):
        if not isinstance(cmd, MotionCommand) or cmd.kind not in (
                "move", "pick", "place", "home"):
            return Decision(False, Deny.INVALID_COMMAND,
                            f"unknown command {getattr(cmd, 'kind', cmd)!r}")

        if self.state is SafetyState.EMERGENCY_STOP:
            return Decision(False, Deny.ESTOP_LATCHED,
                            "emergency stop latched - operator reset required")

        il = self._read_interlocks()
        check = self._check_interlocks(il)
        if not check.allowed:
            if check.code is Deny.ESTOP_CIRCUIT_OPEN:
                self.emergency_stop(check.reason)
            else:
                self.protective_stop(check.code, check.reason)
            return check

        if self.state is SafetyState.PROTECTIVE_STOP:
            return Decision(False, Deny.PROTECTIVE_STOP,
                            "protective stop active - operator reset required")

        lim = self.limits
        if cmd.speed is not None and cmd.speed > lim.max_tcp_speed:
            return Decision(False, Deny.SPEED_LIMIT,
                            f"requested {cmd.speed:.2f} m/s exceeds "
                            f"{lim.max_tcp_speed:.2f} m/s")
        if cmd.payload_kg > lim.max_payload_kg:
            return Decision(False, Deny.PAYLOAD_LIMIT,
                            f"payload {cmd.payload_kg:.2f} kg exceeds "
                            f"{lim.max_payload_kg:.2f} kg")

        if cmd.kind == "home":
            return Decision(True, Deny.OK, "home permitted")

        if cmd.target is None or len(cmd.target) != 3 or not all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in cmd.target):
            return Decision(False, Deny.INVALID_COMMAND,
                            f"{cmd.kind} needs a finite (x, y, z) target")

        return self.check_point(*cmd.target)

    def check_point(self, x, y, z):
        """Geometric checks only. Also used to monitor the live TCP."""
        lim = self.limits
        pad = lim.padding
        for axis, v, (lo, hi) in (("x", x, lim.workspace_x),
                                  ("y", y, lim.workspace_y),
                                  ("z", z, lim.workspace_z)):
            if not (lo + pad <= v <= hi - pad):
                return Decision(False, Deny.OUTSIDE_WORKSPACE,
                                f"{axis}={v:.3f} outside "
                                f"[{lo + pad:.2f}, {hi - pad:.2f}]")
        radial = math.hypot(x, y)
        if not (lim.reach_min <= radial <= lim.reach_max):
            return Decision(False, Deny.OUTSIDE_REACH,
                            f"radial distance {radial:.3f} m outside "
                            f"[{lim.reach_min:.2f}, {lim.reach_max:.2f}]")
        for zone in lim.keep_out:
            if zone.contains(x, y, z):
                return Decision(False, Deny.KEEP_OUT_ZONE,
                                f"target inside keep-out zone '{zone.name}'")
        return Decision(True, Deny.OK, "within limits")

    def monitor_tcp(self, position):
        """
        Call with the measured TCP position after/during motion.
        Leaving the workspace box or entering a keep-out zone trips a
        protective stop. (Reach limits apply to commanded targets.)
        """
        x, y, z = position
        lim = self.limits
        for axis, v, (lo, hi) in (("x", x, lim.workspace_x),
                                  ("y", y, lim.workspace_y),
                                  ("z", z, lim.workspace_z)):
            if not (lo <= v <= hi):
                d = Decision(False, Deny.OUTSIDE_WORKSPACE,
                             f"TCP {axis}={v:.3f} left the workspace")
                self.protective_stop(d.code, d.reason)
                return d
        for zone in lim.keep_out:
            if zone.contains(x, y, z):
                d = Decision(False, Deny.KEEP_OUT_ZONE,
                             f"TCP entered keep-out zone '{zone.name}'")
                self.protective_stop(d.code, d.reason)
                return d
        return Decision(True, Deny.OK, "TCP within limits")

    # ---- interlocks --------------------------------------------
    def _read_interlocks(self):
        try:
            return self.interlocks.read()
        except Exception:
            return Interlocks()          # all False, timestamp 0 -> deny

    def _check_interlocks(self, il):
        age = self._clock() - il.timestamp
        if il.timestamp <= 0 or age > self.limits.interlock_max_age_s:
            return Decision(False, Deny.INTERLOCK_STALE,
                            "interlock signals missing or stale")
        if not il.estop_ok:
            return Decision(False, Deny.ESTOP_CIRCUIT_OPEN,
                            "emergency stop circuit open")
        if not il.guard_closed:
            return Decision(False, Deny.GUARD_OPEN, "guard door open")
        if not il.mode_auto:
            return Decision(False, Deny.NOT_IN_AUTO, "cell not in AUTO")
        if not il.plc_permit:
            return Decision(False, Deny.NO_PLC_PERMIT,
                            "PLC has not granted motion permit")
        if not il.robot_ready:
            return Decision(False, Deny.ROBOT_NOT_READY,
                            "robot controller reports not ready")
        return Decision(True, Deny.OK, "interlocks healthy")

    def status(self):
        return {
            "state": self.state.value,
            "last_trip": ({"code": self.last_trip[0].value,
                           "reason": self.last_trip[1]}
                          if self.last_trip else None),
            "authorized": self.authorized_count,
            "denied": self.denied_count,
        }
