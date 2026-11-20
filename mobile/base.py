# =============================================================
#   MOBILE BASE - an omnidirectional platform carrying the arm
#
#   Mecanum-style: it can move in any direction without turning,
#   which is how industrial mobile manipulators such as KUKA's KMR
#   get into tight spaces in front of machines.
#
#   The base is moved kinematically: it follows the planned route
#   exactly, with no wheel slip or dynamics. Two things are real:
#
#   - Speed. It accelerates, holds a cruise speed, and is capped
#     to a crawl through the pedestrian walkway.
#   - Parking error. On arrival it stops a little off its mark
#     (default up to 15 mm and 1.5 degrees, seeded), as real bases
#     do. It does not know by how much: its own idea of where it
#     is (believed_pose) stays at the nominal dock. Only the camera
#     can tell it otherwise.
#
#   Everything mounted on the base - chassis, wheels, the mast with
#   its camera, the arm, and any part in the gripper - moves with
#   it. The safety supervisor and the arm's planner are told the
#   base's true pose, because those belong to the arm's own
#   controller and work in the arm's own frame.
# =============================================================

import math
import time

import pybullet as p

import config as cfg
from mobile import floor as F

CRUISE = 0.40           # m/s
MAX_SPEED = 0.60        # m/s, the most the base may be asked for
WALKWAY_SPEED = 0.15    # m/s through the pedestrian crossing
ACCEL = 0.50            # m/s^2

STOW = (0.30, 0.0, 0.55)        # arm pose for travel, base frame
SCAN_PARK = (0.10, 0.45, 0.70)  # arm clear of the mast camera
MAST_CAMERA = (0.48, 0.0, 1.45)  # camera on the mast, base frame


def compose(pose, x, y):
    """A point in a pose's frame, in the world."""
    bx, by, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return bx + x * c - y * s, by + x * s + y * c


def relative(pose, x, y):
    """A world point in a pose's frame."""
    bx, by, yaw = pose
    dx, dy = x - bx, y - by
    c, s = math.cos(-yaw), math.sin(-yaw)
    return dx * c - dy * s, dx * s + dy * c


class MobileBase:

    def __init__(self, robot, supervised, rng, docking_error_m=0.015,
                 docking_error_deg=1.5, follow_camera=False):
        self.robot = robot
        self.supervised = supervised
        self.rng = rng
        self.docking_error_m = docking_error_m
        self.docking_error_deg = docking_error_deg
        self.follow_camera = follow_camera
        self.true_pose = (0.0, 0.0, 0.0)
        self.believed_pose = (0.0, 0.0, 0.0)
        self.station = None             # where it is docked, if anywhere
        self.moving = False
        self.odometer = 0.0
        self.on_step = []               # callables run every physics step
        self._mounted = []              # (body, x, y, z) in base frame
        self._build()

    # ---- bodies -------------------------------------------------------
    def _visual(self, shape, rgba, **kw):
        vis = p.createVisualShape(shape, rgbaColor=rgba, **kw)
        return p.createMultiBody(baseVisualShapeIndex=vis,
                                 basePosition=[0, 0, 0])

    def _mount(self, body, x, y, z, yaw=0.0):
        self._mounted.append((body, x, y, z, yaw))

    def _build(self):
        dark, orange = [0.18, 0.19, 0.21, 1], [0.95, 0.55, 0.10, 1]
        cx = (F.FOOT_FRONT - F.FOOT_BACK) / 2
        hx = (F.FOOT_FRONT + F.FOOT_BACK) / 2
        self._mount(self._visual(p.GEOM_BOX, orange,
                                 halfExtents=[hx, F.FOOT_SIDE, 0.035]),
                    cx, 0, 0.055)
        self._mount(self._visual(p.GEOM_BOX, dark,
                                 halfExtents=[hx + 0.005, F.FOOT_SIDE + 0.005,
                                              0.006]), cx, 0, 0.094)
        for sx in (F.FOOT_FRONT - 0.07, -F.FOOT_BACK + 0.07):
            for sy in (F.FOOT_SIDE - 0.02, -F.FOOT_SIDE + 0.02):
                wheel = self._visual(p.GEOM_CYLINDER, dark, radius=0.04,
                                     length=0.04)
                self._mounted.append((wheel, sx, sy, 0.04, None))
        mx = -F.FOOT_BACK + 0.06
        top = MAST_CAMERA[2] + 0.05
        self._mount(self._visual(p.GEOM_BOX, dark,
                                 halfExtents=[0.02, 0.02, top / 2]),
                    mx, 0, top / 2)
        boom = (MAST_CAMERA[0] - mx) / 2
        self._mount(self._visual(p.GEOM_BOX, dark,
                                 halfExtents=[boom, 0.015, 0.015]),
                    mx + boom, 0, top)
        self._mount(self._visual(p.GEOM_BOX, [0.1, 0.1, 0.1, 1],
                                 halfExtents=[0.03, 0.04, 0.025]),
                    MAST_CAMERA[0], 0, MAST_CAMERA[2] + 0.02)

    # ---- pose -----------------------------------------------------------
    def set_pose(self, x, y, yaw):
        old = self.true_pose
        quat = p.getQuaternionFromEuler([0, 0, yaw])
        p.resetBasePositionAndOrientation(self.robot.robot, [x, y, 0], quat)
        for body, bx, by, bz, byaw in self._mounted:
            wx, wy = compose((x, y, yaw), bx, by)
            if byaw is None:            # wheel: axle across the base
                o = p.getQuaternionFromEuler([math.pi / 2, 0, yaw])
            else:
                o = p.getQuaternionFromEuler([0, 0, yaw + byaw])
            p.resetBasePositionAndOrientation(body, [wx, wy, bz], o)
        held = self.robot.grasped_object
        if held and held in self.robot.objects:
            body = self.robot.objects[held]["id"]
            (px, py, pz), porn = p.getBasePositionAndOrientation(body)
            rx, ry = relative(old, px, py)
            nx, ny = compose((x, y, yaw), rx, ry)
            prpy = p.getEulerFromQuaternion(porn)
            p.resetBasePositionAndOrientation(
                body, [nx, ny, pz],
                p.getQuaternionFromEuler([prpy[0], prpy[1],
                                          prpy[2] + yaw - old[2]]))
        self.true_pose = (x, y, yaw)
        self.supervised.safety.base_pose = self.true_pose
        self.robot.rrt.set_workspace_limits((x - 0.10, x + 0.80),
                                            (y - 0.60, y + 0.60),
                                            (0.05, 1.00))
        self.robot._tool_orn = None
        if self.follow_camera and self.robot.sim_config.get("gui"):
            p.resetDebugVisualizerCamera(2.2, 35, -38, [x + 0.25, y, 0.25])

    def _tick(self):
        p.stepSimulation()
        for fn in self.on_step:
            fn()
        if self.robot.sim_config["timestep"]:
            time.sleep(self.robot.sim_config["timestep"])

    def to_world(self, bx, by, bz=0.0):
        """A point in the base's frame, using where it really is."""
        x, y = compose(self.true_pose, bx, by)
        return x, y, bz

    def dead_reckoned(self, wx, wy):
        """
        Where the arm actually goes if it aims at a world point using
        the base's belief about its own pose - map coordinates without
        a camera. Off by exactly the parking error.
        """
        bx, by = relative(self.believed_pose, wx, wy)
        return compose(self.true_pose, bx, by)

    # ---- arm postures ---------------------------------------------------
    def tuck(self, steps=240):
        """
        Joint-space move to the arm's home posture. Cartesian moves
        alone let the elbow and wrist drift a little further each
        time; after enough of them a joint sits near its stop and the
        next place cannot hold the tool square. Mobile arms tuck in
        joint space before every drive for the same reason. Gated by
        the safety supervisor like any other motion.
        """
        from safety.supervisor import MotionCommand
        d = self.supervised._gate(MotionCommand("home", source="mobile"))
        if not d.allowed:
            return False
        r = self.robot
        ex, ey, ez = r.get_end_effector_position()
        if ez < 0.45:                   # lift clear of the table first
            self.supervised.move_to_position(ex, ey, 0.50, use_rrt=False)
        home = r.robot_config["home_position"]
        start = [p.getJointState(r.robot, j)[0] for j in r.arm_joints]
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)
            for k, j in enumerate(r.arm_joints):
                p.setJointMotorControl2(
                    r.robot, j, p.POSITION_CONTROL,
                    targetPosition=start[k] + (home[k] - start[k]) * t,
                    force=r.robot_config["max_force"])
            self._tick()
        for _ in range(30):
            self._tick()
        r._tool_orn = None
        return True

    def stow(self):
        if not self.tuck():
            return False
        x, y, z = self.to_world(*STOW)
        return self.supervised.move_to_position(x, y, z, use_rrt=False)

    def park_for_scan(self):
        x, y, z = self.to_world(*SCAN_PARK)
        return self.supervised.move_to_position(x, y, z, use_rrt=False)

    def camera_config(self):
        cam = dict(cfg.CELL_CAMERA)
        cx, cy, cz = self.to_world(*MAST_CAMERA)
        tx, ty, tz = self.to_world(MAST_CAMERA[0], 0.0, F.TABLE_Z)
        cam["position"], cam["target"] = [cx, cy, cz], [tx, ty, tz]
        return cam

    # ---- driving ---------------------------------------------------------
    def drive(self, waypoints, speed=CRUISE, log=None, final_yaw=None):
        """
        Follow a polyline. Speed is capped in the walkway. Returns a
        summary. The arm must already be stowed; the caller checks.
        """
        log = log or (lambda *a: None)
        speed = min(speed, MAX_SPEED)
        dt = cfg.SIM_TIMESTEP
        x, y, yaw = self.true_pose
        start_yaw = yaw
        target_yaw = yaw if final_yaw is None else final_yaw
        total = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(waypoints, waypoints[1:]))
        travelled, v, walkway_logged, slowest = 0.0, 0.0, False, None
        self.moving = True
        self.station = None
        try:
            for a, b in zip(waypoints, waypoints[1:]):
                seg = math.hypot(b[0] - a[0], b[1] - a[1])
                if seg < 1e-6:
                    continue
                ux, uy = (b[0] - a[0]) / seg, (b[1] - a[1]) / seg
                done = 0.0
                while done < seg - 1e-6:
                    remaining = total - travelled
                    limit = speed
                    if F.in_walkway(x, y):
                        limit = min(limit, WALKWAY_SPEED)
                        if not walkway_logged:
                            log("NAV", f"entering pedestrian walkway: speed "
                                       f"capped at {WALKWAY_SPEED:.2f} m/s")
                            walkway_logged = True
                        slowest = WALKWAY_SPEED
                    # brake for the end of the route
                    limit = min(limit, math.sqrt(2 * ACCEL * max(remaining,
                                                                 0.0005)))
                    if v < limit:
                        v = min(limit, v + ACCEL * dt)
                    else:
                        v = max(limit, v - ACCEL * 2 * dt)
                    stepd = min(max(v, 0.01) * dt, seg - done)
                    x += ux * stepd
                    y += uy * stepd
                    done += stepd
                    travelled += stepd
                    frac = travelled / total if total else 1.0
                    yaw = start_yaw + (target_yaw - start_yaw) * frac
                    self.set_pose(x, y, yaw)
                    self._tick()
        finally:
            self.moving = False
        self.odometer += travelled
        return {"distance_m": travelled, "slowed": slowest is not None}

    def go_to_station(self, station, speed=CRUISE, log=None,
                      parking_error=True):
        """
        Undock if docked, route along the lane to the station, dock.
        Returns a summary, or None if no route exists.
        """
        log = log or (lambda *a: None)
        sy = F.STATIONS[station]
        x, y, _ = self.true_pose
        legs = []
        if x > F.LANE_X + 0.01:
            legs.append([(x, y), (F.LANE_X, y)])        # back out
            x = F.LANE_X
        route = F.plan_route((x, y), (F.LANE_X, sy))
        if route is None:
            return None
        legs.append(route)
        if parking_error:
            r = self.docking_error_m
            ex = self.rng.uniform(-r, r)
            ey = self.rng.uniform(-r, r)
            eyaw = math.radians(self.rng.uniform(-self.docking_error_deg,
                                                 self.docking_error_deg))
        else:
            ex = ey = eyaw = 0.0
        legs.append([(F.LANE_X, sy), (F.DOCK_X + ex, sy + ey)])
        distance, slowed = 0.0, False
        for i, leg in enumerate(legs):
            last = i == len(legs) - 1
            out = self.drive(leg, speed=speed if not last else 0.20,
                             log=log, final_yaw=eyaw if last else None)
            distance += out["distance_m"]
            slowed = slowed or out["slowed"]
        self.station = station
        self.believed_pose = (F.DOCK_X, sy, 0.0)
        for _ in range(60):
            self._tick()
        return {"distance_m": distance, "slowed": slowed,
                "route": route,
                "parking_error_mm": math.hypot(ex, ey) * 1000,
                "parking_error_deg": math.degrees(eyaw)}
