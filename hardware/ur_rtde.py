# =============================================================
#   UR BACKEND - Universal Robots e-Series over RTDE
#
#   Implements RobotInterface for a UR5e using the `ur_rtde`
#   library (rtde_control / rtde_receive / rtde_io).
#
#   STATUS: exercised against a recording fake of the RTDE
#   interfaces (tests/test_ur_backend.py). It has NOT been run on
#   a robot or in URSim. Before first use on hardware:
#     1. Run it against URSim and confirm every call behaves.
#     2. Survey the cell and set URCellConfig (frame, heights).
#     3. Complete the risk assessment and configure the limits in
#        the robot's own safety configuration. Those limits, the
#        emergency stop and the guard circuit are what protect
#        people; this file only refuses to command motion the
#        controller reports it should not make.
#     4. First moves at reduced speed, in manual mode, with the
#        enabling device.
#
#   Motion here is straight-line moveL between approach points.
#   The collision-aware RRT* planner used in simulation is not
#   connected to this backend yet; the cell layout must keep the
#   approach corridors clear.
# =============================================================

import math
from dataclasses import dataclass, field

from hardware.interface import RobotInterface

SAFETY_MODE_NORMAL = 1          # UR safety mode: NORMAL
ROBOT_MODE_RUNNING = 7          # UR robot mode: RUNNING


@dataclass
class URCellConfig:
    ip: str = "192.168.1.100"
    # Cell frame -> robot base frame: translation then yaw (rad).
    base_xyz: tuple = (0.0, 0.0, 0.0)
    base_yaw: float = 0.0
    # Tool pointing straight down, as a rotation vector.
    tool_down_rotvec: tuple = (0.0, math.pi, 0.0)
    home_q: tuple = (0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0)
    approach_height: float = 0.20     # above the part / slot, metres
    grasp_offset_z: float = 0.0       # TCP z relative to part centre
    place_z: float = 0.125            # TCP z when releasing
    speed: float = 0.10               # m/s, commissioning default
    accel: float = 0.30               # m/s^2
    joint_speed: float = 0.50         # rad/s for the home move
    joint_accel: float = 0.50
    payload_kg: float = 0.5
    reach_min: float = 0.25
    reach_max: float = 0.80           # UR5e reach is 0.85 m
    gripper_do: int = 0               # digital output that closes the gripper
    gripper_settle_s: float = 0.4
    part_present_di: int = None       # digital input: part in gripper
    conveyor_slots: dict = field(default_factory=dict)


class DigitalGripper:
    """Two-state gripper on one controller digital output."""

    def __init__(self, io, receive, pin, settle_s, part_present_di=None,
                 sleep=None):
        import time
        self._io, self._rx, self._pin = io, receive, pin
        self._settle = settle_s
        self._di = part_present_di
        self._sleep = sleep or time.sleep

    def close(self):
        self._io.setStandardDigitalOut(self._pin, True)
        self._sleep(self._settle)

    def open(self):
        self._io.setStandardDigitalOut(self._pin, False)
        self._sleep(self._settle)

    def part_present(self):
        """True/False from the sensor, or None if no sensor is fitted."""
        if self._di is None:
            return None
        return bool(self._rx.getDigitalInState(self._di))


class URRobot(RobotInterface):

    def __init__(self, config, pose_source, logger=None, control=None,
                 receive=None, io=None, sleep=None):
        """
        control / receive / io can be injected (tests, URSim wrappers).
        When omitted they are created from the `ur_rtde` package.
        """
        self.cfg = config
        self.logger = logger
        self.pose_source = pose_source
        if control is None or receive is None or io is None:
            import rtde_control
            import rtde_io
            import rtde_receive
            control = control or rtde_control.RTDEControlInterface(config.ip)
            receive = receive or rtde_receive.RTDEReceiveInterface(config.ip)
            io = io or rtde_io.RTDEIOInterface(config.ip)
        self._c, self._r, self._io = control, receive, io
        self.gripper = DigitalGripper(io, receive, config.gripper_do,
                                      config.gripper_settle_s,
                                      config.part_present_di, sleep=sleep)
        self.grasped_object = None
        self.conveyor_slots = dict(config.conveyor_slots)
        self.slot_occupants = {}
        self._c.setPayload(0.0, [0.0, 0.0, 0.0])

    # ---- cell bookkeeping ---------------------------------------
    @property
    def objects(self):
        return {n: {} for n in self.pose_source.names()}

    @property
    def can_measure_parts(self):
        return bool(self.pose_source.measures)

    # ---- frames -------------------------------------------------
    def _to_base(self, x, y, z):
        c, s = math.cos(self.cfg.base_yaw), math.sin(self.cfg.base_yaw)
        dx, dy = x - self.cfg.base_xyz[0], y - self.cfg.base_xyz[1]
        return (c * dx + s * dy, -s * dx + c * dy,
                z - self.cfg.base_xyz[2])

    def _to_cell(self, bx, by, bz):
        c, s = math.cos(self.cfg.base_yaw), math.sin(self.cfg.base_yaw)
        return (c * bx - s * by + self.cfg.base_xyz[0],
                s * bx + c * by + self.cfg.base_xyz[1],
                bz + self.cfg.base_xyz[2])

    def _pose(self, x, y, z):
        return list(self._to_base(x, y, z)) + list(self.cfg.tool_down_rotvec)

    # ---- state --------------------------------------------------
    def controller_healthy(self):
        """Robot powered, running, in NORMAL safety mode, no stops."""
        try:
            return (self._c.isConnected()
                    and self._r.getSafetyMode() == SAFETY_MODE_NORMAL
                    and self._r.getRobotMode() == ROBOT_MODE_RUNNING
                    and not self._r.isProtectiveStopped()
                    and not self._r.isEmergencyStopped())
        except Exception:
            return False

    def get_end_effector_position(self):
        p = self._r.getActualTCPPose()
        return self._to_cell(p[0], p[1], p[2])

    def get_object_position(self, name):
        return self.pose_source.get(name)

    def get_all_object_positions(self):
        return {n: self.pose_source.get(n) for n in self.pose_source.names()}

    def check_reachability(self, x, y, z=0.65):
        bx, by, _ = self._to_base(x, y, z)
        dist = math.hypot(bx, by)
        if dist > self.cfg.reach_max:
            return {"reachable": False, "severity": "error",
                    "distance": round(dist, 3),
                    "message": f"OUT OF REACH — {dist:.2f}m "
                               f"(max {self.cfg.reach_max:.2f}m)",
                    "hint": "Move object closer to robot base"}
        if dist < self.cfg.reach_min:
            return {"reachable": False, "severity": "warning",
                    "distance": round(dist, 3),
                    "message": f"TOO CLOSE — {dist:.2f}m "
                               f"(min {self.cfg.reach_min:.2f}m)",
                    "hint": "Move object further from base"}
        return {"reachable": True, "severity": "ok",
                "distance": round(dist, 3),
                "message": f"Reachable — {dist:.2f}m", "hint": None}

    # ---- motion -------------------------------------------------
    def _move_l(self, x, y, z):
        if not self.controller_healthy():
            return False
        pose = self._pose(x, y, z)
        if not self._c.isPoseWithinSafetyLimits(pose):
            return False
        ok = self._c.moveL(pose, self.cfg.speed, self.cfg.accel)
        return bool(ok) and self.controller_healthy()

    def move_to_position(self, x, y, z, use_rrt=True, ignore_name=None):
        return self._move_l(x, y, z)

    def home(self):
        if not self.controller_healthy():
            return False
        return bool(self._c.moveJ(list(self.cfg.home_q),
                                  self.cfg.joint_speed, self.cfg.joint_accel))

    def pick_object(self, name):
        def fail(detail):
            return {"status": "failed", "action": "pick", "target": name,
                    "detail": detail}

        if self.grasped_object:
            return fail(f"already holding {self.grasped_object}")
        pos = self.pose_source.get(name)
        if not pos:
            return fail(f"{name} not found")
        x, y, z = pos
        up = z + self.cfg.approach_height
        self.gripper.open()
        if not self._move_l(x, y, up):
            return fail("approach move refused or stopped")
        if not self._move_l(x, y, z + self.cfg.grasp_offset_z):
            return fail("descent refused or stopped")
        self.gripper.close()
        if not self._move_l(x, y, up):
            self.gripper.open()
            return fail("lift refused or stopped")
        if self.gripper.part_present() is False:
            self.gripper.open()
            return fail(f"gripper closed but no part detected for {name}")
        self.grasped_object = name
        self._c.setPayload(self.cfg.payload_kg, [0.0, 0.0, 0.0])
        self.pose_source.notify_picked(name)
        return {"status": "success", "action": "pick", "target": name,
                "position": [x, y, z],
                "detail": f"picked {name} from ({x:.2f},{y:.2f},{z:.2f})"}

    def place_object(self, x, y, stack_on=None):
        held = self.grasped_object

        def fail(detail):
            return {"status": "failed", "action": "place", "target": held,
                    "detail": detail}

        if not held:
            return fail("not holding anything")
        if stack_on is not None:
            return fail("stacking is not implemented on the UR backend")
        z = self.cfg.place_z
        up = z + self.cfg.approach_height
        if not self._move_l(x, y, up):
            return fail("transport move refused or stopped")
        if not self._move_l(x, y, z):
            return fail("descent refused or stopped")
        self.gripper.open()
        self.grasped_object = None
        self._c.setPayload(0.0, [0.0, 0.0, 0.0])
        self.pose_source.notify_placed(held, x, y, z)
        retreated = self._move_l(x, y, up)
        measured = self.pose_source.get(held) if self.can_measure_parts else None
        return {"status": "success" if retreated else "failed",
                "action": "place", "target": held,
                "position": [x, y],
                "verified_position": list(measured) if measured else None,
                "detail": (f"placed {held} at ({x:.2f},{y:.2f})"
                           if retreated else "retreat refused or stopped")}

    # ---- stop / shutdown ----------------------------------------
    def stop(self):
        try:
            self._c.stopL(2.0)
        except Exception:
            pass

    def disconnect(self):
        for call in (self._c.stopScript, self._c.disconnect,
                     self._r.disconnect):
            try:
                call()
            except Exception:
                pass
