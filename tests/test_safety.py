"""Safety supervisor: deterministic, fail-closed, latching."""

import math

from audit.trail import AuditTrail
from core.robot_tools import RobotTools
from safety.supervised_robot import SupervisedRobot
from safety.supervisor import (Deny, Interlocks, MotionCommand, SafetyLimits,
                               SafetyState, SafetySupervisor, Zone)
from tests.fakes import FakeCellRobot, FakeLogger


def test_in_workspace_command_is_allowed():
    s = SafetySupervisor()
    d = s.authorize(MotionCommand("pick", (0.40, 0.0, 0.125), payload_kg=0.5))
    assert d.allowed and d.code is Deny.OK


def test_target_outside_workspace_is_denied_without_stopping_the_cell():
    s = SafetySupervisor()
    d = s.authorize(MotionCommand("place", (0.95, 0.95, 0.15)))
    assert not d.allowed and d.code is Deny.OUTSIDE_WORKSPACE
    assert s.state is SafetyState.RUN        # a bad target is not a fault


def test_target_beyond_reach_is_denied():
    s = SafetySupervisor()
    d = s.authorize(MotionCommand("place", (0.70, 0.50, 0.15)))
    assert d.code is Deny.OUTSIDE_REACH


def test_keep_out_zone_is_denied():
    limits = SafetyLimits(keep_out=(Zone("fixture", (0.45, 0.55),
                                         (-0.05, 0.05)),))
    s = SafetySupervisor(limits)
    assert s.authorize(MotionCommand("move", (0.50, 0.0, 0.30))).code \
        is Deny.KEEP_OUT_ZONE


def test_speed_and_payload_limits():
    s = SafetySupervisor()
    assert s.authorize(MotionCommand("move", (0.4, 0, 0.3), speed=1.0)).code \
        is Deny.SPEED_LIMIT
    assert s.authorize(MotionCommand("pick", (0.4, 0, 0.125),
                                     payload_kg=5.0)).code is Deny.PAYLOAD_LIMIT


def test_non_finite_or_missing_target_is_denied():
    s = SafetySupervisor()
    assert not s.authorize(MotionCommand("move", (math.nan, 0, 0.3))).allowed
    assert not s.authorize(MotionCommand("move", None)).allowed
    assert not s.authorize(MotionCommand("teleport", (0.4, 0, 0.3))).allowed
    assert not s.authorize("pick red").allowed


def test_guard_open_trips_protective_stop_and_needs_reset():
    s = SafetySupervisor()
    s.interlocks.guard_closed = False
    assert s.authorize(MotionCommand("home")).code is Deny.GUARD_OPEN
    assert s.state is SafetyState.PROTECTIVE_STOP
    # Reset is refused while the guard is still open.
    assert not s.reset("operator").allowed
    s.interlocks.guard_closed = True
    # Closing the guard alone does not restart motion.
    assert s.authorize(MotionCommand("home")).code is Deny.PROTECTIVE_STOP
    assert s.reset("operator").allowed
    assert s.authorize(MotionCommand("home")).allowed


def test_emergency_stop_latches():
    s = SafetySupervisor()
    s.interlocks.estop_ok = False
    assert s.authorize(MotionCommand("home")).code is Deny.ESTOP_CIRCUIT_OPEN
    assert s.state is SafetyState.EMERGENCY_STOP
    s.interlocks.estop_ok = True
    assert s.authorize(MotionCommand("home")).code is Deny.ESTOP_LATCHED
    assert not s.reset("").allowed           # anonymous reset refused
    assert s.reset("operator").allowed


def test_missing_interlock_source_fails_closed():
    class Broken:
        def read(self):
            raise IOError("fieldbus down")

    s = SafetySupervisor(interlock_source=Broken())
    d = s.authorize(MotionCommand("home"))
    assert not d.allowed and d.code is Deny.INTERLOCK_STALE


def test_stale_interlock_data_fails_closed():
    class Stale:
        def read(self):
            return Interlocks(True, True, True, True, True, timestamp=1.0)

    s = SafetySupervisor(interlock_source=Stale(), clock=lambda: 50.0)
    assert s.authorize(MotionCommand("home")).code is Deny.INTERLOCK_STALE


def test_supervisor_error_fails_closed():
    s = SafetySupervisor()
    s._authorize = lambda cmd: 1 / 0
    d = s.authorize(MotionCommand("home"))
    assert not d.allowed and d.code is Deny.INTERNAL_ERROR


def test_limits_cannot_be_edited_at_run_time():
    s = SafetySupervisor()
    try:
        s.limits.max_tcp_speed = 99.0
        edited = True
    except Exception:
        edited = False
    assert not edited


def test_supervised_robot_blocks_and_logs_denied_motion():
    backend = FakeCellRobot()
    audit = AuditTrail()
    robot = SupervisedRobot(backend, SafetySupervisor(), audit)
    backend.grasped_object = "red"
    out = robot.place_object(0.95, 0.95)
    assert out["status"] == "failed" and out["safety_denied"]
    assert out["safety_code"] == Deny.OUTSIDE_WORKSPACE.value
    assert ("place", 0.95, 0.95) not in backend.calls   # never reached it
    decisions = audit.events("safety_decision")
    assert decisions and decisions[-1]["data"]["allowed"] is False


def test_supervised_robot_passes_allowed_motion_and_state_through():
    backend = FakeCellRobot()
    robot = SupervisedRobot(backend, SafetySupervisor(), AuditTrail())
    assert robot.pick_object("red")["status"] == "success"
    assert robot.grasped_object == "red"
    robot.slot_occupants = {1: "red"}
    assert backend.slot_occupants == {1: "red"}


def test_tcp_leaving_workspace_trips_protective_stop():
    backend = FakeCellRobot()
    safety = SafetySupervisor()
    robot = SupervisedRobot(backend, safety, AuditTrail())
    backend.move_to_position = lambda *a, **k: setattr(
        backend, "tcp", (1.5, 0.0, 0.3)) or True
    robot.move_to_position(0.40, 0.0, 0.30)
    assert safety.state is SafetyState.PROTECTIVE_STOP
    assert backend.stopped


def test_language_model_tools_are_read_only():
    """No tool lets the model move the robot or touch the safety layer."""
    tools = RobotTools(FakeCellRobot(), FakeLogger()).get_tool_definitions()
    names = [t["name"] for t in tools]
    assert len(names) == 11
    forbidden = ("reset", "safety", "estop", "override", "limit", "permit",
                 "move", "execute", "set_")
    for name in names:
        assert name.startswith(("get_", "check_", "is_", "plan_")), name
        assert not any(word in name for word in forbidden), name
