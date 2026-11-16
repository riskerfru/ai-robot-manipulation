"""The verifier never reports a placement it could not confirm."""

from place_verifier import PlacementVerifier
from tests.fakes import FakeCellRobot


def test_exhausted_retries_report_failure_not_success():
    robot = FakeCellRobot(place_offsets=[(0.0, 0.12)] * 10)
    robot.parts["red"] = (0.63, 0.12, 0.14)          # 12 cm off slot 2
    v = PlacementVerifier(robot, calibration_path=None,
                          use_calibration=False)
    out = v.verify_and_fix("red", 0.63, 0.0, slot_num=2)
    assert out["status"] == "failed"
    assert out.get("terminal") is True
    assert "human intervention" in out["detail"]


def test_recovered_placement_is_marked_as_retried():
    robot = FakeCellRobot()
    robot.parts["red"] = (0.63, 0.12, 0.14)
    v = PlacementVerifier(robot, calibration_path=None)
    out = v.verify_and_fix("red", 0.63, 0.0, slot_num=2)
    assert out["status"] == "success" and out["retried"] is True


def test_tolerance_is_a_parameter():
    robot = FakeCellRobot()
    robot.parts["red"] = (0.63, 0.36, 0.14)          # 6 cm off
    loose = PlacementVerifier(robot, calibration_path=None)
    assert loose.verify_and_fix("red", 0.63, 0.30)["retried"] is False
    robot.parts["red"] = (0.63, 0.36, 0.14)
    tight = PlacementVerifier(robot, drift_warn=0.03, drift_retry=0.05,
                              calibration_path=None)
    assert tight.verify_and_fix("red", 0.63, 0.30)["retried"] is True


def test_no_measurement_means_no_verified_placement():
    robot = FakeCellRobot()
    robot.can_measure_parts = False
    out = PlacementVerifier(robot, calibration_path=None).verify_and_fix(
        "red", 0.63, 0.30)
    assert out["status"] == "failed" and out["unverified"]


def test_calibration_can_be_switched_off():
    v = PlacementVerifier(FakeCellRobot(), use_calibration=False,
                          calibration_path=None)
    v._record_drift(2, 0.60, 0.25, 0.55, 0.25)
    v._record_drift(2, 0.60, 0.25, 0.54, 0.25)
    assert v._apply_calibration(0.60, 0.25, 2) == (0.60, 0.25)


def test_calibration_corrects_a_repeatable_offset():
    # Same 12 cm error every time: the learned offset removes it.
    robot = FakeCellRobot(place_offsets=[(0.0, 0.12)] * 10)
    robot.parts["red"] = (0.63, 0.12, 0.14)
    v = PlacementVerifier(robot, calibration_path=None)
    out = v.verify_and_fix("red", 0.63, 0.0, slot_num=2)
    assert out["status"] == "success" and out["drift_cm"] < 1.0


def test_scattered_errors_are_not_learned_as_an_offset():
    # Errors in opposite directions are disturbances, not a slot offset.
    v = PlacementVerifier(FakeCellRobot(), calibration_path=None)
    v._record_drift(2, 0.63, 0.0, 0.63, 0.09)
    v._record_drift(2, 0.63, 0.0, 0.63, -0.08)
    v._record_drift(2, 0.63, 0.0, 0.70, 0.0)
    assert v.corrected_target(0.63, 0.0, 2) == (0.63, 0.0)


def test_offset_is_measured_against_the_commanded_position():
    # A corrected placement that lands on target must not dilute the offset.
    robot = FakeCellRobot(place_offsets=[(0.0, 0.06)] * 10)
    v = PlacementVerifier(robot, drift_warn=0.012, drift_retry=0.02,
                          calibration_path=None)
    for _ in range(3):
        robot.parts["red"] = (0.63, 0.06, 0.14)
        v.verify_and_fix("red", 0.63, 0.0, slot_num=2)
    dx, dy = v.calibration[2]
    assert abs(dx) < 0.005 and abs(dy + 0.06) < 0.005


def test_implausibly_large_offset_is_not_applied():
    v = PlacementVerifier(FakeCellRobot(), calibration_path=None)
    v._record_drift(2, 0.63, 0.0, 0.63, 0.30)
    v._record_drift(2, 0.63, 0.0, 0.63, 0.31)
    assert v.corrected_target(0.63, 0.0, 2) == (0.63, 0.0)
