"""
UR5e backend against a recording fake of the RTDE interfaces.
This checks the logic and call sequence only. It is not evidence
that the backend works on a robot: that needs URSim and then
supervised hardware commissioning.
"""

import math

from audit.trail import AuditTrail
from hardware.interface import conforms
from hardware.pose_source import CallbackPoseSource, FixturePoseSource
from hardware.ur_rtde import (ROBOT_MODE_RUNNING, SAFETY_MODE_NORMAL,
                              URCellConfig, URRobot)
from place_verifier import PlacementVerifier
from safety.supervised_robot import SupervisedRobot
from safety.supervisor import SafetySupervisor

SLOTS = {1: (0.60, 0.25), 2: (0.60, 0.0)}
NESTS = {"red": (0.40, 0.0), "blue": (0.40, 0.25)}


class FakeRTDE:
    """control + receive + io in one object."""

    def __init__(self):
        self.log = []
        self.tcp = [0.3, 0.0, 0.5, 0.0, math.pi, 0.0]
        self.safety_mode = SAFETY_MODE_NORMAL
        self.robot_mode = ROBOT_MODE_RUNNING
        self.protective = False
        self.estopped = False
        self.within_limits = True
        self.outputs = {}
        self.inputs = {}

    # control
    def isConnected(self): return True
    def setPayload(self, mass, cog): self.log.append(("payload", mass))
    def isPoseWithinSafetyLimits(self, pose): return self.within_limits

    def moveL(self, pose, speed, accel):
        self.log.append(("moveL", tuple(round(v, 3) for v in pose[:3]),
                         speed))
        self.tcp = list(pose)
        return True

    def moveJ(self, q, speed, accel):
        self.log.append(("moveJ", tuple(q)))
        return True

    def stopL(self, a): self.log.append(("stopL", a))
    def stopScript(self): self.log.append(("stopScript",))
    def disconnect(self): self.log.append(("disconnect",))

    # receive
    def getActualTCPPose(self): return list(self.tcp)
    def getSafetyMode(self): return self.safety_mode
    def getRobotMode(self): return self.robot_mode
    def isProtectiveStopped(self): return self.protective
    def isEmergencyStopped(self): return self.estopped
    def getDigitalInState(self, pin): return self.inputs.get(pin, False)

    # io
    def setStandardDigitalOut(self, pin, value):
        self.outputs[pin] = value
        self.log.append(("DO", pin, value))


def make(pose_source=None, **cfg):
    rtde = FakeRTDE()
    config = URCellConfig(conveyor_slots=SLOTS, **cfg)
    source = pose_source or FixturePoseSource(NESTS, part_z=0.125)
    robot = URRobot(config, source, control=rtde, receive=rtde, io=rtde,
                    sleep=lambda s: None)
    return robot, rtde


def test_backend_implements_the_robot_interface():
    robot, _ = make()
    assert conforms(robot)
    assert set(robot.objects) == {"red", "blue"}


def test_pick_sequence_and_gripper_output():
    robot, rtde = make()
    out = robot.pick_object("red")
    assert out["status"] == "success" and robot.grasped_object == "red"
    moves = [e[1] for e in rtde.log if e[0] == "moveL"]
    assert moves == [(0.4, 0.0, 0.325), (0.4, 0.0, 0.125), (0.4, 0.0, 0.325)]
    assert rtde.outputs[0] is True                 # gripper closed
    assert ("payload", 0.5) in rtde.log


def test_place_sequence_releases_and_retreats():
    robot, rtde = make()
    robot.pick_object("red")
    out = robot.place_object(0.60, 0.25)
    assert out["status"] == "success" and robot.grasped_object is None
    assert rtde.outputs[0] is False
    assert [e[1] for e in rtde.log if e[0] == "moveL"][-1] == \
        (0.6, 0.25, 0.325)
    assert out["verified_position"] is None        # fixtures do not measure


def test_speed_is_the_configured_commissioning_speed():
    robot, rtde = make(speed=0.08)
    robot.move_to_position(0.4, 0.1, 0.3)
    assert all(e[2] == 0.08 for e in rtde.log if e[0] == "moveL")


def test_no_motion_while_the_controller_is_stopped():
    for attr, value in (("protective", True), ("estopped", True),
                        ("safety_mode", 3), ("robot_mode", 5)):
        robot, rtde = make()
        setattr(rtde, attr, value)
        assert robot.pick_object("red")["status"] == "failed"
        assert robot.home() is False
        assert not [e for e in rtde.log if e[0] in ("moveL", "moveJ")]


def test_pose_outside_controller_safety_limits_is_refused():
    robot, rtde = make()
    rtde.within_limits = False
    assert robot.move_to_position(0.4, 0.0, 0.3) is False
    assert not [e for e in rtde.log if e[0] == "moveL"]


def test_missing_part_sensor_signal_fails_the_pick():
    robot, rtde = make(part_present_di=2)
    rtde.inputs[2] = False
    out = robot.pick_object("red")
    assert out["status"] == "failed" and robot.grasped_object is None
    assert rtde.outputs[0] is False                # gripper reopened


def test_cell_frame_transform_round_trips():
    robot, rtde = make(base_xyz=(0.1, -0.2, 0.05), base_yaw=math.pi / 2)
    robot.move_to_position(0.4, 0.1, 0.3)
    x, y, z = robot.get_end_effector_position()
    assert abs(x - 0.4) < 1e-9 and abs(y - 0.1) < 1e-9 and abs(z - 0.3) < 1e-9


def test_reachability_uses_ur5e_limits():
    robot, _ = make()
    assert robot.check_reachability(0.5, 0.0)["reachable"]
    assert not robot.check_reachability(0.9, 0.3)["reachable"]
    assert not robot.check_reachability(0.1, 0.0)["reachable"]


def test_fixture_cell_cannot_claim_a_verified_placement():
    robot, _ = make()
    supervised = SupervisedRobot(robot, SafetySupervisor(), AuditTrail())
    supervised.pick_object("red")
    supervised.place_object(0.60, 0.0)
    out = PlacementVerifier(supervised, calibration_path=None) \
        .verify_and_fix("red", 0.60, 0.0, slot_num=2)
    assert out["status"] == "failed" and out["unverified"]


def test_measuring_cell_detects_a_misplaced_part():
    seen = {"red": (0.40, 0.0, 0.125), "blue": (0.40, 0.25, 0.125)}
    source = CallbackPoseSource(seen, lambda name: seen.get(name))
    robot, _ = make(pose_source=source)
    supervised = SupervisedRobot(robot, SafetySupervisor(), AuditTrail())
    supervised.pick_object("red")
    supervised.place_object(0.60, 0.0)
    seen["red"] = (0.60, 0.03, 0.125)              # camera: 3 cm off
    out = PlacementVerifier(supervised, calibration_path=None) \
        .verify_and_fix("red", 0.60, 0.0, slot_num=2)
    assert out["status"] == "success" and 2.9 < out["drift_cm"] < 3.1
