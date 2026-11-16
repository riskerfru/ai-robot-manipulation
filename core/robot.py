# =============================================================
#   ROBOT CONTROLLER - Franka Panda
#   Fixed version:
#   - Heavier blocks (0.5kg) - less drift
#   - High friction on table and objects
#   - Correct slot positions (all different)
#   - Pre-place hover for accurate positioning
#   - Lower place height (0.10m)
#   - Reduced settle steps (50 not 150)
#   - Gradual constraint release
# =============================================================

import math
import random
import pybullet as p
import pybullet_data
import time
import numpy as np

from motion.rrt_planner import RRTPlanner
from motion.trajectory import TrajectoryPlanner
from perception.camera import CameraSystem
from motion.visual_servoing import VisualServoing
from cell.layout import CONVEYOR_SLOTS


class FrankaPandaRobot:

    # A part further than this from the gripper centre when the
    # fingers close is not between them: the pick is a miss.
    GRASP_CAPTURE_RADIUS = 0.02   # m, across the fingers
    GRASP_CAPTURE_HEIGHT = 0.03   # m, along the fingers
    GRASP_DEPTH = 0.012           # m, gripper point below part centre

    def __init__(self, config, logger):
        self.config        = config
        self._tool_orn     = None
        self._ik_limits    = None
        self._finger_yaw   = None
        self._insertion    = None
        self.on_force      = None     # optional callback(z, force_n)
        self.last_grasp_z  = None     # gripper height at the last grasp
        self.use_force_sensing = True # False: comparison runs only
        self.pocket_bias   = (0.0, 0.0)   # learned fixture offset (x, y)
        self.pocket_bias_samples = []
        self.use_pocket_bias = False      # packing cell turns this on
        self.perception_noise = 0.0       # sensor error, metres (1 sigma)
        self._perception_error = {}
        self.noise_rng     = random.Random(0)
        self.last_grip_width = None       # metres, set at each grasp
        self.part_gate     = None         # optional check after a pick
        self.pick_hints    = {}           # name -> finger_axis, open_to
        self._approach_finger_axis = None
        self.pockets       = {}       # part name -> nominal pocket
        self.stations      = {}       # part name -> supply station (x, y)
        self.logger        = logger
        self.robot_config  = config["robot_limits"]
        self.sim_config    = config["sim"]
        self.motion_config = config["motion"]
        self.camera_config = config["camera"]

        self._init_simulation()

        self.rrt        = RRTPlanner(self.motion_config, logger)
        self.trajectory = TrajectoryPlanner(self.motion_config, logger)
        self.servo      = VisualServoing(self.motion_config, logger)
        self.camera     = CameraSystem(self.camera_config, logger)
        self.camera.setup_overhead_camera()
        self.camera.setup_side_camera()
        self.camera.setup_wrist_camera(
            robot_id = self.robot,
            ee_link  = self.end_effector
        )
        self._wrist_frame = 0  # frame counter for saving

        self.objects          = {}
        self.obstacles        = {}
        self.grasp_constraint = None
        self.grasped_object   = None

        # Slot positions come from the cell layout (single source of truth)
        self.conveyor_slots = dict(CONVEYOR_SLOTS)
        self.slot_occupants = {}

        self.rrt.set_workspace_limits(
            self.robot_config["x"],
            self.robot_config["y"],
            self.robot_config["z"]
        )
        self.rrt.robot = self

        self.reset_to_home()
        self.logger.success("Franka Panda robot ready")

    # ----------------------------------------------------------
    # Simulation setup
    # ----------------------------------------------------------

    def _init_simulation(self):
        if self.sim_config["gui"]:
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8, cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0.2, 0, 0.2]
            )
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            p.addUserDebugLine([0,0,0], [0.3,0,0], [1,0,0], 2)
            p.addUserDebugLine([0,0,0], [0,0.3,0], [0,1,0], 2)
            p.addUserDebugLine([0,0,0], [0,0,0.3], [0,0,1], 2)
            print("  [VIEW] 3D view active")
            print("  [VIEW] Mouse controls:")
            print("         Left drag   = rotate view")
            print("         Right drag  = pan view")
            print("         Scroll      = zoom in/out")
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.sim_config["gravity"])
        p.setTimeStep(self.sim_config["timestep"])

        self.plane = p.loadURDF("plane.urdf")

        # FIX: High friction on table surface - objects stop sliding
        p.changeDynamics(
            self.plane, -1,
            lateralFriction  = 2.0,
            spinningFriction = 0.1,
            rollingFriction  = 0.1
        )

        self.robot = p.loadURDF(
            self.robot_config["urdf"],
            basePosition=[0, 0, 0],
            useFixedBase=True,
            flags=p.URDF_USE_SELF_COLLISION
        )

        self.num_joints    = p.getNumJoints(self.robot)
        self.arm_joints    = self.robot_config["arm_joints"]
        self.finger_joints = self.robot_config["finger_joints"]
        self.end_effector  = self.robot_config["end_effector_index"]
        self.logger.info(f"Simulation initialized: {self.num_joints} joints")

        # Camera control buttons in PyBullet GUI
        if self.sim_config["gui"]:
            self._btn_overhead = p.addUserDebugParameter(
                "Camera: Overhead", 0, 1, 0
            )
            self._btn_front = p.addUserDebugParameter(
                "Camera: Front", 0, 1, 0
            )
            self._btn_top = p.addUserDebugParameter(
                "Camera: Top", 0, 1, 0
            )
            self._btn_side = p.addUserDebugParameter(
                "Camera: Side", 0, 1, 0
            )
            self._btn_close = p.addUserDebugParameter(
                "Camera: Close-up", 0, 1, 0
            )
            self._btn_wrist = p.addUserDebugParameter(
                "Camera: Wrist View", 0, 1, 0
            )
            self._btn_wide = p.addUserDebugParameter(
                "Camera: Wide", 0, 1, 0
            )
            self._prev_btn_vals = {}
            print("  [CAMERA] GUI buttons added — click to switch view")

    def reset_to_home(self):
        home = self.robot_config["home_position"]
        for i, joint_idx in enumerate(self.arm_joints):
            p.resetJointState(self.robot, joint_idx, home[i])
        self.open_gripper()
        for _ in range(100):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])
        self.logger.info("Robot at home position")

    # ----------------------------------------------------------
    # Gripper
    # ----------------------------------------------------------

    def open_gripper(self, width=None):
        """
        Open the fingers. width (m, per finger from the centre) opens
        them only that far, for releasing next to another part.
        """
        if not self.finger_joints:
            return
        if width is not None:
            for joint in self.finger_joints:
                p.setJointMotorControl2(
                    self.robot, joint, p.POSITION_CONTROL,
                    targetPosition=min(0.04, width), force=80,
                    maxVelocity=0.05)
            for _ in range(60):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])
            return
        for target in [0.01, 0.02, 0.03, 0.04]:
            for joint in self.finger_joints:
                p.setJointMotorControl2(
                    self.robot, joint,
                    p.POSITION_CONTROL,
                    targetPosition = target,
                    force          = 80,
                    maxVelocity    = 0.05
                )
            for _ in range(25):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])
        print("  [GRIPPER] Open (4cm)")

    def close_gripper(self):
        """Force-controlled close - stops when object felt"""
        if not self.finger_joints:
            return
        prev_torque = 0.0
        for target in [0.038, 0.032, 0.026, 0.020,
                       0.016, 0.013, 0.010, 0.008]:
            for joint in self.finger_joints:
                p.setJointMotorControl2(
                    self.robot, joint,
                    p.POSITION_CONTROL,
                    targetPosition = target,
                    force          = 200,
                    maxVelocity    = 0.02
                )
            for _ in range(35):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])
            torques = [abs(p.getJointState(self.robot, j)[3])
                       for j in self.finger_joints]
            curr = sum(torques)
            if curr > 0.8 and curr > prev_torque * 1.5:
                print(f"  [GRIPPER] Contact at gap={target:.3f}m "
                      f"torque={curr:.2f}")
                break
            prev_torque = curr
        print("  [GRIPPER] Closed")

    def vacuum_attach(self, name):
        obj_id  = self.objects[name]["id"]
        obj_pos = self.get_object_position(name)
        ee_pos  = self.get_end_effector_position()
        offset_z = obj_pos[2] - ee_pos[2] if obj_pos and ee_pos else -0.04
        constraint = p.createConstraint(
            parentBodyUniqueId = self.robot,
            parentLinkIndex    = self.end_effector,
            childBodyUniqueId  = obj_id,
            childLinkIndex     = -1,
            jointType          = p.JOINT_FIXED,
            jointAxis          = [0, 0, 0],
            parentFramePosition = [0, 0, 0],
            childFramePosition  = [0, 0, -offset_z]
        )
        p.changeConstraint(constraint, maxForce=200)
        return constraint

    def vacuum_release(self):
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

    # ----------------------------------------------------------
    # Scene management
    # ----------------------------------------------------------

    def add_object(self, name, x, y, color, z=0.05, half_extent=0.025,
                   grasp_depth=None, vision=True, friction=1.5):
        col    = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_extent] * 3
        )
        vis    = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_extent] * 3,
            rgbaColor=color
        )
        # FIX: Heavier blocks = less drift after release
        obj_id = p.createMultiBody(
            baseMass                 = 0.5,   # was 0.1kg, now 0.5kg
            baseCollisionShapeIndex  = col,
            baseVisualShapeIndex     = vis,
            basePosition             = [x, y, z]
        )

        # FIX: High friction on objects - grip table surface firmly
        p.changeDynamics(
            obj_id, -1,
            lateralFriction  = friction,
            spinningFriction = 0.05,
            rollingFriction  = 0.05
        )

        self.objects[name] = {
            "id": obj_id, "color": color, "position": [x, y, z],
            "half_extent": half_extent,
            "vision": vision,
            "grasp_depth": (self.GRASP_DEPTH if grasp_depth is None
                            else grasp_depth)
        }
        self.logger.info(f"Added object '{name}' at ({x}, {y}, {z})")
        self._sync_obstacles()
        return obj_id

    def add_obstacle(self, name, x, y, size,
                     color=[0.5, 0.5, 0.5, 0.9], z=0.0):
        col      = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis      = p.createVisualShape(
            p.GEOM_BOX, halfExtents=size, rgbaColor=color
        )
        center_z = z + size[2]
        obj_id   = p.createMultiBody(
            baseMass                = 0,
            baseCollisionShapeIndex = col,
            baseVisualShapeIndex    = vis,
            basePosition            = [x, y, center_z]
        )
        self.obstacles[name] = {
            "id": obj_id,
            "position": [x, y, center_z],
            "size": size
        }
        self.rrt.add_obstacle(
            center=(x, y, center_z), size=size, name=name
        )
        self.logger.info(f"Added obstacle '{name}' at ({x}, {y}, {center_z})")
        return obj_id

    def remove_obstacle(self, name):
        if name in self.obstacles:
            p.removeBody(self.obstacles[name]["id"])
            del self.obstacles[name]
            self.rrt.obstacles = [
                o for o in self.rrt.obstacles if o["name"] != name
            ]
            self.logger.info(f"Removed obstacle '{name}'")
        else:
            print(f"  Obstacle '{name}' not found")

    def list_obstacles(self):
        if not self.obstacles:
            print("\n  No obstacles in scene.")
            return
        print("\n  Current obstacles:")
        for name, data in self.obstacles.items():
            pos = data["position"]
            print(f"    {name}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

    def _sync_obstacles(self, ignore_object=None):
        objects_for_rrt = {
            name: self.get_object_position(name)
            for name in self.objects
            if name != ignore_object
        }
        self.rrt.update_obstacles_from_objects(objects_for_rrt)

    def get_object_position(self, name):
        if name not in self.objects:
            return None
        pos, _ = p.getBasePositionAndOrientation(self.objects[name]["id"])
        return pos

    def get_all_object_positions(self):
        return {
            name: self.get_object_position(name)
            for name in self.objects
        }

    # ----------------------------------------------------------
    # Camera
    # ----------------------------------------------------------

    def scan_scene(self):
        self.logger.info("Scanning scene...")
        return self.camera.detect_objects(self.objects, "overhead")

    def get_end_effector_position(self):
        return p.getLinkState(self.robot, self.end_effector)[0]

    # ----------------------------------------------------------
    # Core motion
    # ----------------------------------------------------------

    def move_to_position(self, x, y, z, use_rrt=True, ignore_name=None):
        target  = (x, y, z)
        current = self.get_end_effector_position()
        self.logger.log_movement(
            current, target, "rrt" if use_rrt else "direct"
        )

        if use_rrt and self.motion_config["planner"] == "rrt":
            path, status, blocking = self.rrt.plan(
                current, target, ignore_name=ignore_name
            )
            if status == "blocked" or path is None:
                self._report_blocked(target, blocking)
                return False
            waypoints = path
        else:
            waypoints = [np.array(current), np.array(target)]

        # The route, for whoever is reading the log.
        shown = waypoints if len(waypoints) <= 6 else (
            list(waypoints[:3]) + list(waypoints[-2:]))
        print(f"  [PATH] {len(waypoints)} waypoints: " + " -> ".join(
            f"({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f})" for w in shown)
            + (" (middle ones not shown)" if len(waypoints) > 6 else ""))

        if self.motion_config["smooth_trajectory"] and len(waypoints) > 1:
            try:
                trajectory = self.trajectory.generate_smooth_trajectory(
                    waypoints,
                    num_points=self.motion_config["trajectory_points"]
                )
                positions, velocities, timestamps = (
                    self.trajectory.apply_trapezoidal_profile(trajectory)
                )
                est_time = self.trajectory.get_execution_time(waypoints)
                print(f"  [TRAJ] Estimated execution: {est_time:.1f}s, "
                      f"{len(trajectory)} points")
            except Exception:
                trajectory = waypoints
                velocities = None
        else:
            trajectory = waypoints
            velocities = None

        self._execute_trajectory(trajectory, velocities)
        return True

    def _down_orientation(self, finger_axis=None):
        """
        Gripper pointing straight down with the fingers square to
        the cell axes, at the quarter turn nearest the current yaw.
        finger_axis "x" or "y" fixes which way the fingers open (for
        nests with a neighbour on one side); the nearest half turn
        is used.
        """
        current = p.getLinkState(self.robot, self.end_effector)[1]
        yaw     = p.getEulerFromQuaternion(current)[2]
        if isinstance(finger_axis, tuple):
            # (direction, period): fingers open along this direction
            # (radians from x), for a part the camera found turned at
            # an angle. A square part can be taken at any quarter
            # turn, so the nearest one to the present yaw is used.
            angle, period = finger_axis
            base = self._finger_yaw_for_y() + (angle - math.pi / 2)
            yaw  = base + round((yaw - base) / period) * period
        elif isinstance(finger_axis, (int, float)):
            base = self._finger_yaw_for_y() + (finger_axis - math.pi / 2)
            yaw  = base + round((yaw - base) / math.pi) * math.pi
        elif finger_axis in ("x", "y"):
            base = self._finger_yaw_for_y()
            if finger_axis == "x":
                base += math.pi / 2
            yaw  = base + round((yaw - base) / math.pi) * math.pi
        else:
            yaw  = round(yaw / (math.pi / 2)) * (math.pi / 2)
        return p.getQuaternionFromEuler([math.pi, 0.0, yaw])

    def _square_up(self, x, y, z, finger_axis, speed=0.7):
        """
        Bring the tool vertical above (x, y, z) with the fingers
        opening the way asked, and check that it really got there.
        The same finger line can be reached with the wrist turned
        half a turn either way, and the arm can hold only some of
        those from a given pose, so each is tried and measured.
        Returns (orientation, (dx, dy) position trim), or (None, ...)
        when no wrist angle holds.
        """
        first = self._down_orientation(finger_axis)
        yaw0 = p.getEulerFromQuaternion(first)[2]
        if isinstance(finger_axis, tuple):
            want, period = finger_axis
        elif finger_axis == "x":
            want, period = 0.0, math.pi
        elif finger_axis == "y":
            want, period = math.pi / 2, math.pi
        elif isinstance(finger_axis, (int, float)):
            want, period = float(finger_axis), math.pi
        else:
            want, period = None, math.pi / 2
        turns = (0, 1, -1) if period > 3.0 else (0, 1, -1, 2, -2)

        def finger_error():
            if want is None or len(self.finger_joints) < 2:
                return 0.0
            a = p.getLinkState(self.robot, self.finger_joints[0])[0]
            b = p.getLinkState(self.robot, self.finger_joints[1])[0]
            e = (math.atan2(b[1] - a[1], b[0] - a[0]) - want) % math.pi
            return min(e, math.pi - e)

        for turn in turns:
            orn = p.getQuaternionFromEuler(
                [math.pi, 0.0, yaw0 + turn * period])
            ox = oy = 0.0
            last, stuck, err, fe = None, 0, 1.0, 1.0
            for _ in range(12):
                self._direct_move(x + ox, y + oy, z, speed_factor=speed,
                                  orn=orn)
                ee = self.get_end_effector_position()
                err = math.hypot(x - ee[0], y - ee[1])
                fe = finger_error()
                if err < 0.0015 and fe < 0.05:
                    break
                # A wrist joint on its stop stops making progress:
                # give this angle up and try the other way round.
                key = (round(err, 3), round(fe, 2))
                stuck = stuck + 1 if key == last else 0
                last = key
                if stuck >= 2:
                    break
                if err < 0.02:
                    ox = float(np.clip(ox + (x - ee[0]), -0.06, 0.06))
                    oy = float(np.clip(oy + (y - ee[1]), -0.06, 0.06))
            if err > 0.004 or fe > 0.09:
                continue
            m = p.getMatrixFromQuaternion(
                p.getLinkState(self.robot, self.end_effector)[1])
            if -m[8] < 0.9986:                  # more than 3 deg off plumb
                continue
            if turn:
                print("  [WRIST] Turned the other way round to reach "
                      "this angle")
            return orn, (ox, oy)
        return None, (0.0, 0.0)

    def _finger_yaw_for_y(self):
        """
        Tool yaw at which the fingers open along the cell's y axis,
        worked out from where the two finger links actually are.
        """
        if self._finger_yaw is None:
            a = p.getLinkState(self.robot, self.finger_joints[0])[0]
            b = p.getLinkState(self.robot, self.finger_joints[1])[0]
            world = math.atan2(b[1] - a[1], b[0] - a[0])
            tool = p.getEulerFromQuaternion(
                p.getLinkState(self.robot, self.end_effector)[1])[2]
            # fingers lie along `world` when the tool yaw is `tool`
            self._finger_yaw = (tool - world) + math.pi / 2
        return self._finger_yaw

    def _ik_with_orientation(self, point, orn):
        """Joint-limit-aware IK for a full tool pose."""
        if self._ik_limits is None:
            lower, upper, ranges, movable = [], [], [], []
            for j in range(p.getNumJoints(self.robot)):
                info = p.getJointInfo(self.robot, j)
                if info[2] == p.JOINT_FIXED:
                    continue
                movable.append(j)
                lower.append(info[8])
                upper.append(info[9])
                ranges.append(info[9] - info[8])
            self._ik_limits = (lower, upper, ranges, movable)
        lower, upper, ranges, movable = self._ik_limits
        rest = [p.getJointState(self.robot, j)[0] for j in movable]
        return p.calculateInverseKinematics(
            self.robot, self.end_effector, point, orn,
            lowerLimits=lower, upperLimits=upper,
            jointRanges=ranges, restPoses=rest,
            maxNumIterations=200, residualThreshold=0.0005
        )

    def _direct_move(self, x, y, z, speed_factor=1.0, orn=None):
        target       = np.array([x, y, z])
        current      = np.array(self.get_end_effector_position())
        n            = self.rrt.trajectory_points
        steps_per_pt = max(1, self.sim_config["steps_per_move"] // n)

        for i in range(n + 1):
            t     = i / n
            point = current + t * (target - current)
            if orn is None:
                joint_angles = p.calculateInverseKinematics(
                    self.robot, self.end_effector, point,
                    maxNumIterations=100, residualThreshold=0.001
                )
            else:
                # Tool orientation held: used for every vertical
                # approach so the gripper meets the part square.
                joint_angles = self._ik_with_orientation(point, orn)
            for j, joint_idx in enumerate(self.arm_joints):
                p.setJointMotorControl2(
                    self.robot, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[j],
                    force=self.robot_config["max_force"],
                    maxVelocity=(self.robot_config["max_velocity"]
                                 * speed_factor)
                )
            for _ in range(steps_per_pt):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])

    def _execute_trajectory(self, trajectory, velocities=None):
        steps           = self.sim_config["steps_per_move"]
        points_per_step = max(1, steps // max(len(trajectory), 1))
        base_velocity   = self.robot_config["max_velocity"]

        for idx, point in enumerate(trajectory):
            point = np.array(point)
            if velocities and idx < len(velocities):
                v_norm   = (velocities[idx] / max(velocities)
                            if max(velocities) > 0 else 1.0)
                velocity = max(0.1, v_norm) * base_velocity
            else:
                velocity = base_velocity

            joint_angles = p.calculateInverseKinematics(
                self.robot, self.end_effector, point,
                maxNumIterations=100, residualThreshold=0.001
            )
            for i, joint_idx in enumerate(self.arm_joints):
                p.setJointMotorControl2(
                    self.robot, joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[i],
                    force=self.robot_config["max_force"],
                    maxVelocity=velocity
                )
            for _ in range(points_per_step):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])

    def _report_blocked(self, target, blocking):
        print(f"\n  [BLOCKED] All strategies failed for target "
              f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        if blocking:
            print(f"  Obstacles:")
            for o in blocking:
                c = o["center"]
                print(f"    - {o['name']} at "
                      f"({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")
        print(f"  Try: 'remove obstacle <name>' to clear the path\n")

    # ----------------------------------------------------------
    # Reachability check
    # ----------------------------------------------------------

    def check_reachability(self, x, y, z=0.65):
        dist = math.sqrt(x**2 + y**2)
        if dist > 0.75:
            return {
                "reachable": False, "severity": "error",
                "distance":  round(dist, 3),
                "message":   f"OUT OF REACH — {dist:.2f}m (max 0.75m)",
                "hint":      "Move object closer to robot base"
            }
        elif dist > 0.65:
            return {
                "reachable": True, "severity": "warning",
                "distance":  round(dist, 3),
                "message":   f"NEAR EDGE — {dist:.2f}m",
                "hint":      "Pick may fail"
            }
        elif dist < 0.25:
            return {
                "reachable": False, "severity": "warning",
                "distance":  round(dist, 3),
                "message":   f"TOO CLOSE — {dist:.2f}m (min 0.25m)",
                "hint":      "Move object further from base"
            }
        else:
            return {
                "reachable": True, "severity": "ok",
                "distance":  round(dist, 3),
                "message":   f"Reachable — {dist:.2f}m",
                "hint":      None
            }

    # ----------------------------------------------------------
    # Pick and place
    # ----------------------------------------------------------


    def _descend_until_contact(self, x, y, start_z,
                                torque_threshold=8.0,
                                step_size=0.004, floor_z=None):
        """
        Descend until torque CHANGE spikes = contact detected.
        Uses relative threshold not absolute - works regardless
        of arm configuration and gravity compensation.
        """
        current_z    = start_z
        # Never go lower than the known grasp height when one is
        # given: contact sensing is the guard, not the target.
        min_z        = 0.08 if floor_z is None else max(0.08, floor_z)
        readings     = []

        print(f"  [DESCENT] From z={start_z:.3f}m seeking contact...")

        # Sample baseline torque before moving
        baseline = sum(abs(p.getJointState(self.robot, j)[3])
                       for j in self.arm_joints)
        print(f"  [DESCENT] Baseline torque={baseline:.1f}")

        # Fast approach to just above expected contact
        approach_z = max(min_z + 0.10, start_z - 0.30)
        orn = self._down_orientation(self._approach_finger_axis)
        self._tool_orn = orn
        self._direct_move(x, y, approach_z, speed_factor=0.15, orn=orn)

        # Closed-loop x/y hold. Position control leaves a residual
        # error that changes with height; left alone it reaches
        # centimetres near the base. Measure it and command against it.
        off_x = off_y = 0.0

        def hold_xy(gain):
            nonlocal off_x, off_y
            ee = self.get_end_effector_position()
            off_x = float(np.clip(off_x + gain * (x - ee[0]), -0.06, 0.06))
            off_y = float(np.clip(off_y + gain * (y - ee[1]), -0.06, 0.06))

        if isinstance(self._approach_finger_axis, tuple):
            # A part the camera found turned at an angle.
            squared, trim = self._square_up(x, y, approach_z,
                                            self._approach_finger_axis)
            if squared is not None:
                orn = squared
                self._tool_orn = orn
                off_x, off_y = trim

        for _ in range(3):
            hold_xy(1.0)
            self._direct_move(x + off_x, y + off_y, approach_z,
                              speed_factor=0.15, orn=orn)

        # A vertical tool is not reachable everywhere (far slots,
        # parts close to the base). If the arm cannot hold position
        # with it, give up the orientation, not the position.
        ee = self.get_end_effector_position()
        # Give the arm time to settle on the pose before deciding
        # it cannot hold a vertical tool there. A part gripped with
        # a tilted tool is tilted when it is put down.
        for _ in range(5):
            if math.hypot(x - ee[0], y - ee[1]) <= 0.002:
                break
            hold_xy(1.0)
            self._direct_move(x + off_x, y + off_y, approach_z,
                              speed_factor=0.15, orn=orn)
            ee = self.get_end_effector_position()

        if math.hypot(x - ee[0], y - ee[1]) > 0.008:
            print(f"  [DESCENT] Vertical tool not reachable here — "
                  f"position-only approach")
            orn = None
            self._tool_orn = None
            off_x = off_y = 0.0
            for _ in range(3):
                hold_xy(1.0)
                self._direct_move(x + off_x, y + off_y, approach_z,
                                  speed_factor=0.15)

        # Re-sample baseline after approach
        baseline = sum(abs(p.getJointState(self.robot, j)[3])
                       for j in self.arm_joints)

        # Slow final descent - detect contact
        current_z = approach_z
        frame_count = 0
        while current_z > min_z:
            current_z = max(min_z, current_z - step_size)
            hold_xy(0.5)
            self._direct_move(x + off_x, y + off_y, current_z,
                              speed_factor=0.02, orn=orn)

            # Drifting sideways on the way down: stop descending and
            # recover position first, without the orientation if the
            # arm cannot hold both.
            ee = self.get_end_effector_position()
            if math.hypot(x - ee[0], y - ee[1]) > 0.008:
                for attempt in range(4):
                    if attempt == 2:
                        orn = None
                        self._tool_orn = None
                        off_x = off_y = 0.0
                    hold_xy(1.0)
                    self._direct_move(x + off_x, y + off_y, current_z,
                                      speed_factor=0.1, orn=orn)
                    ee = self.get_end_effector_position()
                    if math.hypot(x - ee[0], y - ee[1]) <= 0.005:
                        break
                readings.clear()
                baseline = sum(abs(p.getJointState(self.robot, j)[3])
                               for j in self.arm_joints)
                continue

            # Save wrist view every 10 steps
            frame_count += 1
            if frame_count % 10 == 0:
                import os
                os.makedirs("wrist_frames", exist_ok=True)
                self.camera.show_wrist_view(
                    f"wrist_frames/frame_{self._wrist_frame:04d}.png"
                )
                self._wrist_frame += 1

            total = sum(abs(p.getJointState(self.robot, j)[3])
                        for j in self.arm_joints)
            readings.append(total)

            # Contact = torque rises significantly above recent average
            if len(readings) > 5:
                recent_avg = sum(readings[-5:]) / 5
                if total > recent_avg * 1.4 and total > baseline + 3.0:
                    print(f"  [DESCENT] Contact at z={current_z:.3f}m "
                          f"torque={total:.1f} baseline={baseline:.1f}")
                    return current_z

        if floor_z is not None:
            # At the grasp height: close any remaining sideways error
            # before the fingers shut, so the part is gripped centred.
            for _ in range(6):
                ee = self.get_end_effector_position()
                if math.hypot(x - ee[0], y - ee[1]) <= 0.001:
                    break
                hold_xy(1.0)
                self._direct_move(x + off_x, y + off_y, current_z,
                                  speed_factor=0.05, orn=orn)

        print(f"  [DESCENT] Reached min z={current_z:.3f}m")
        return current_z

    def _check_grasp_success(self, name, pre_z, orn=None):
        """Lift slightly and confirm the part came up with the gripper."""
        ee = self.get_end_effector_position()
        self._direct_move(ee[0], ee[1], ee[2]+0.05, speed_factor=0.05,
                          orn=orn if orn is not None else self._tool_orn)
        for _ in range(40):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])
        ee_rise = self.get_end_effector_position()[2] - ee[2]
        obj = self.get_object_position(name)
        part_rise = (obj[2] - pre_z) if obj else 0.0
        # Judge the part against how far the gripper actually rose,
        # not against the commanded lift.
        if ee_rise > 0.01 and part_rise > 0.6 * ee_rise:
            print(f"  [GRASP] Object lifted z={obj[2]:.3f}m")
            return True
        print(f"  [GRASP] Object did not lift - grasp failed")
        return False

    def _complete_grasp(self, name):
        """
        Attach, check and lift whatever the fingers have closed
        on. Shared by pick_object (by name) and pick_at (by a
        measured position).
        """
        pos = self.get_object_position(name)
        # Hold whatever wrist pose the grasp was made with for the
        # whole lift. Where the arm could not keep the tool vertical,
        # self._tool_orn is None and letting the wrist find its own
        # way up swings the part by a centimetre.
        hold_orn = self._tool_orn or p.getLinkState(
            self.robot, self.end_effector)[1]
        # Attach the part exactly where it sits in the gripper.
        # The constraint frame on the part is the gripper link pose
        # expressed in the part's own frame, so the part keeps its
        # grasp-time position and orientation and is not pulled
        # sideways or twisted when the constraint engages.
        obj_id   = self.objects[name]["id"]
        obj_pos2 = self.get_object_position(name)

        # A grasp only counts if the part is between the fingers.
        ee_pos2 = self.get_end_effector_position()
        miss = math.hypot(obj_pos2[0] - ee_pos2[0],
                          obj_pos2[1] - ee_pos2[1])
        if abs(obj_pos2[2] - ee_pos2[2]) > self.GRASP_CAPTURE_HEIGHT:
            miss = max(miss, abs(obj_pos2[2] - ee_pos2[2]))
        if miss > self.GRASP_CAPTURE_RADIUS:
            print(f"  [PICK] ❌ Part is {miss*100:.1f}cm from the gripper "
                  f"centre — not gripped")
            self.open_gripper()
            self._direct_move(ee_pos2[0], ee_pos2[1], ee_pos2[2] + 0.25,
                              speed_factor=0.3)
            self._sync_obstacles()
            return {
                "status": "failed", "action": "pick", "target": name,
                "detail": f"grasp failed - '{name}' is {miss*100:.1f}cm "
                          f"from the gripper centre"
            }

        link     = p.getLinkState(self.robot, self.end_effector)
        part_pos, part_orn = p.getBasePositionAndOrientation(obj_id)
        inv_pos, inv_orn   = p.invertTransform(part_pos, part_orn)
        child_pos, child_orn = p.multiplyTransforms(
            inv_pos, inv_orn, link[0], link[1]
        )

        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId     = self.robot,
            parentLinkIndex        = self.end_effector,
            childBodyUniqueId      = obj_id,
            childLinkIndex         = -1,
            jointType              = p.JOINT_FIXED,
            jointAxis              = [0, 0, 0],
            parentFramePosition    = [0, 0, 0],
            childFramePosition     = child_pos,
            parentFrameOrientation = [0, 0, 0, 1],
            childFrameOrientation  = child_orn
        )
        # Parts that are force-fitted must not slip in the grip when
        # they meet a rim: the insertion judges depth from the gripper.
        grip_n = 50 if self.objects[name].get("vision", True) else 200
        p.changeConstraint(self.grasp_constraint, maxForce=grip_n)
        # Width of what is in the gripper, from the finger positions.
        self.last_grip_width = sum(
            p.getJointState(self.robot, j)[0] for j in self.finger_joints)
        self.grasped_object         = name
        self.rrt.carrying_clearance = 0.04

        # Verify grasp by lifting slightly
        pre_z   = obj_pos2[2]
        grasped = self._check_grasp_success(name, pre_z, orn=hold_orn)
        if not grasped:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint   = None
            self.grasped_object     = None
            self.rrt.carrying_clearance = 0
            self.open_gripper()
            return {
                "status": "failed", "action": "pick",
                "target": name,
                "detail": "grasp failed - object did not lift"
            }

        print(f"  [PICK] Lifting '{name}'...")
        # Rise in small steps with the sideways position corrected:
        # a fast lift drags the part within the grip.
        ee_up = self.get_end_effector_position()
        for rise in (0.01, 0.025, 0.05, 0.09, 0.15):
            for _ in range(2):
                self._direct_move(ee_up[0], ee_up[1], ee_up[2] + rise,
                                  speed_factor=0.05, orn=hold_orn)
                now = self.get_end_effector_position()
                if math.hypot(ee_up[0] - now[0], ee_up[1] - now[1]) < 0.0008:
                    break
        # Hold the tool square on the way up too: letting the wrist
        # turn here swings the part in the grip by a centimetre.
        self._direct_move(ee_up[0], ee_up[1], pos[2]+0.40,
                          speed_factor=0.15, orn=hold_orn)
        self._sync_obstacles()

        lifted = self.get_object_position(name)
        if lifted and lifted[2] < pos[2] + 0.15:
            print(f"  [PICK] ⚠️  Grasp failed — object did not lift")
            if self.grasp_constraint:
                p.removeConstraint(self.grasp_constraint)
                self.grasp_constraint = None
            self.grasped_object         = None
            self.rrt.carrying_clearance = 0
            return {
                "status": "failed", "action": "pick", "target": name,
                "detail": "grasp failed — object did not lift"
            }

        self.logger.success(f"Picked '{name}'!")
        return {
            "status": "success", "action": "pick", "target": name,
            "position": list(pos),
            "detail": (f"picked {name} from "
                       f"({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f})")
        }

    def pick_at(self, x, y, centre_z, finger_axis=None, open_to=None,
                grasp_depth=None, label="part"):
        """
        Pick whatever is at a measured position. No part name is used:
        the position comes from the camera, and which body ends up in
        the gripper is decided by what the fingers actually close on.

        x, y      measured centre of the part
        centre_z  height of the part's centre (its size is known from
                  the product being run, so this follows from the
                  measured top face)
        Returns a result dict with "gripped": the body's name, or a
        failure if the fingers closed on nothing.
        """
        if self.grasped_object:
            return {"status": "failed", "action": "pick", "target": label,
                    "detail": f"gripper is not empty "
                              f"(holding '{self.grasped_object}')"}

        self._sync_obstacles()
        self._approach_finger_axis = finger_axis
        self.open_gripper(open_to)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        print(f"\n  [PICK] Approaching {label} at "
              f"({x:.3f}, {y:.3f}) from the camera...")
        if not self.move_to_position(x, y, centre_z + 0.40, use_rrt=True):
            self._sync_obstacles()
            return {"status": "failed", "action": "pick", "target": label,
                    "detail": f"cannot reach above ({x:.3f}, {y:.3f})"}

        self._direct_move(x, y, self.get_end_effector_position()[2],
                          speed_factor=0.3)
        print(f"  [PICK] Descending to contact...")
        depth = self.GRASP_DEPTH if grasp_depth is None else grasp_depth
        self._descend_until_contact(
            x, y, start_z=self.get_end_effector_position()[2],
            torque_threshold=8.0, step_size=0.004, floor_z=centre_z - depth)

        print(f"  [PICK] Gripping...")
        self.close_gripper()
        for _ in range(60):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])
        # How high the gripper was when it closed: with the height of
        # what the part stood on, this says where the part's foot is.
        self.last_grasp_z = self.get_end_effector_position()[2]

        name = self._body_in_gripper()
        if name is None:
            print(f"  [PICK] ❌ Nothing in the gripper")
            self.open_gripper()
            ee = self.get_end_effector_position()
            self._direct_move(ee[0], ee[1], ee[2] + 0.25, speed_factor=0.3)
            self._sync_obstacles()
            return {"status": "failed", "action": "pick", "target": label,
                    "detail": f"the fingers closed on nothing at "
                              f"({x:.3f}, {y:.3f})"}

        result = self._complete_grasp(name)
        if isinstance(result, dict) and result.get("status") == "success":
            result["gripped"] = name
        return result

    def _body_in_gripper(self):
        """
        Which body the fingers have closed on, from contact alone.
        A part counts only if both fingers are touching it.
        """
        touching = []
        for joint in self.finger_joints:
            bodies = set()
            for contact in p.getContactPoints(bodyA=self.robot,
                                              linkIndexA=joint):
                if contact[2] != self.robot:
                    bodies.add(contact[2])
            touching.append(bodies)
        if not touching:
            return None
        both = set.intersection(*touching) if len(touching) > 1 \
            else touching[0]
        if not both:
            return None
        ee = self.get_end_effector_position()
        best, best_distance = None, 1e9
        for body in both:
            name = next((n for n, data in self.objects.items()
                         if data["id"] == body), None)
            if name is None:
                continue
            pos, _ = p.getBasePositionAndOrientation(body)
            distance = math.hypot(pos[0] - ee[0], pos[1] - ee[1])
            if distance < best_distance:
                best, best_distance = name, distance
        if best is None or best_distance > self.GRASP_CAPTURE_RADIUS * 2:
            return None
        return best

    def report_position(self, name):
        """
        Where the cell believes the part is, as a sensor would report
        it: the true position plus a measurement error. The error is
        fixed per part until it is picked, so repeated reads agree
        with each other, as a real sensor's would.
        """
        true = self.get_object_position(name)
        if not true or not self.perception_noise:
            return true
        error = self._perception_error.get(name)
        if error is None:
            error = (self.noise_rng.gauss(0, self.perception_noise),
                     self.noise_rng.gauss(0, self.perception_noise))
            self._perception_error[name] = error
        return (true[0] + error[0], true[1] + error[1], true[2])

    def pick_object(self, name):
        self._perception_error.pop(name, None)   # take a new measurement
        self.last_grip_width = None
        pos = self.report_position(name)
        if not pos:
            self.logger.error(f"Object '{name}' not found")
            return {
                "status": "failed", "action": "pick",
                "target": name,
                "detail": f"object '{name}' not found in scene"
            }

        if self.grasped_object:
            return {
                "status": "failed", "action": "pick", "target": name,
                "detail": f"gripper is not empty "
                          f"(holding '{self.grasped_object}')"
            }

        self.logger.info(f"Picking '{name}' at {pos}")
        self._sync_obstacles(ignore_object=name)

        # Per-part hints: which way the fingers open, and how far
        # (parts with close neighbours, or sitting in a fixture).
        hints = self.pick_hints.get(name, {})
        self._approach_finger_axis = hints.get("finger_axis")
        self.open_gripper(hints.get("open_to"))
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        print(f"\n  [PICK] Approaching '{name}'...")
        success = self.move_to_position(
            pos[0], pos[1], pos[2] + 0.40,
            use_rrt=True, ignore_name=name
        )
        if not success:
            self._sync_obstacles()
            return {
                "status": "failed", "action": "pick", "target": name,
                "detail": f"cannot reach above '{name}' — RRT failed"
            }

        if self.objects[name].get("vision", True):
            print(f"  [PICK] Above target - visual servoing...")
            aligned_x, aligned_y = self.servo.servo_to_object(self, name)
        else:
            # Parts the wrist camera is not set up to recognise are
            # picked from their reported position.
            aligned_x, aligned_y = pos[0], pos[1]
            self._direct_move(aligned_x, aligned_y,
                              self.get_end_effector_position()[2],
                              speed_factor=0.3)

        # If servoing gave up (part outside the camera's reliable
        # range), the gripper may still be centimetres off. Descend
        # on the part's reported position rather than wherever the
        # arm happens to be.
        if math.hypot(aligned_x - pos[0], aligned_y - pos[1]) > 0.01:
            print(f"  [PICK] Servo did not align — "
                  f"using reported part position")
            aligned_x, aligned_y = pos[0], pos[1]
            self._direct_move(aligned_x, aligned_y,
                              self.get_end_effector_position()[2],
                              speed_factor=0.3)

        # Physics-driven descent - no hardcoded height
        print(f"  [PICK] Descending to contact...")
        ee_now    = self.get_end_effector_position()
        contact_z = self._descend_until_contact(
            aligned_x, aligned_y,
            start_z          = ee_now[2],
            torque_threshold = 8.0,
            step_size        = 0.004,
            floor_z          = pos[2] - self.objects[name].get(
                "grasp_depth", self.GRASP_DEPTH)
        )

        # Force-controlled grip
        print(f"  [PICK] Gripping...")
        self.close_gripper()
        for _ in range(60):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        return self._complete_grasp(name)

    def insert_part(self, name, pocket, on_step=None):
        """Seat the held part in a pocket by force-guided insertion."""
        from motion.insertion import ForceGuidedInsertion
        if self._insertion is None:
            self._insertion = ForceGuidedInsertion(self)

        # Start from where earlier parts showed this fixture really is.
        bias = self.pocket_bias if self.use_pocket_bias else (0.0, 0.0)
        target = dict(pocket, x=pocket["x"] + bias[0],
                      y=pocket["y"] + bias[1])
        result = self._insertion.insert(name, target,
                                        on_step=on_step or self.on_force,
                                        use_force=self.use_force_sensing)
        if (result.get("status") == "success"
                and "search_offset_xy" in result):
            dx, dy = result["search_offset_xy"]
            self.pocket_bias_samples.append((bias[0] + dx, bias[1] + dy))
            n = len(self.pocket_bias_samples)
            self.pocket_bias = (
                sum(s[0] for s in self.pocket_bias_samples) / n,
                sum(s[1] for s in self.pocket_bias_samples) / n)
        result["start_offset_mm"] = round(
            1000 * math.hypot(bias[0], bias[1]), 1)
        self._sync_obstacles()
        return result

    def place_by_touch(self, x, y, touch_z, finger_axis=None, open_to=None,
                       on_step=None, label="surface", strict=False,
                       step_m=0.0005):
        """
        Put the held part down by feel. touch_z is the gripper height
        at which the camera expects the part to meet the surface; the
        arm stops short of it and creeps down until the force sensor
        reports contact, then lets go. The surface is felt, not assumed.
        """
        if not self.grasped_object:
            return {"status": "failed", "action": "place", "target": label,
                    "detail": "not holding anything"}
        from motion.insertion import ForceGuidedInsertion
        TOUCH_N = 1.0      # lighter than an insertion: nothing to seat
        if self._insertion is None:
            self._insertion = ForceGuidedInsertion(self)
        ins, name = self._insertion, self.grasped_object
        on_step = on_step or self.on_force

        ee = self.get_end_effector_position()
        safe_z = max(0.42, touch_z + 0.22)
        hold = self._tool_orn or p.getLinkState(
            self.robot, self.end_effector)[1]
        print(f"  [PLACE] Lifting clear to z={safe_z:.3f}")
        self._direct_move(ee[0], ee[1], safe_z, speed_factor=0.4, orn=hold)
        print(f"  [PLACE] Travelling to ({x:.3f}, {y:.3f})")
        for _ in range(8):
            self._direct_move(x, y, safe_z, speed_factor=0.3)
            ee = self.get_end_effector_position()
            if math.hypot(x - ee[0], y - ee[1]) < 0.004:
                break
        orn, _ = self._square_up(x, y, safe_z, finger_axis)
        if orn is None and not strict:
            other = {"x": "y", "y": "x"}.get(finger_axis)
            print("  [PLACE] Wrist cannot hold that angle here, trying "
                  "a quarter turn")
            orn, _ = self._square_up(x, y, safe_z, other)
        if orn is None:
            return {"status": "failed", "action": "place", "target": label,
                    "detail": f"the arm cannot hold the tool square above "
                              f"({x:.3f}, {y:.3f})"}
        self._tool_orn = orn
        for _ in range(4):
            ee = ins._move(x, y, safe_z, orn, speed=0.15)
            if math.hypot(x - ee[0], y - ee[1]) < 0.001:
                break

        z = touch_z + 0.025
        print(f"  [PLACE] Down to z={z:.3f}, then feeling for the surface")
        for _ in range(2):
            ins._move(x, y, z, orn, speed=0.1)
        force, floor = 0.0, touch_z - 0.012
        while z > floor:
            # Half-millimetre steps where contact is expected: the
            # arm is stiff, and whatever it lands on may be loose.
            z -= 0.002 if z > touch_z + 0.008 else step_m
            ins._move(x, y, z, orn)
            # Read the sensor through the whole settling time and keep
            # the peak: first contact can show for only an instant.
            force = 0.0
            for _ in range(12):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])
                force = max(force, ins.contact_force(name))
            if on_step:
                on_step(z, force)
            if force > TOUCH_N:
                break
        felt = force > TOUCH_N
        ins._release(name, open_to)
        self._retreat_straight_up(orn)
        self._sync_obstacles()
        return {"status": "success", "action": "place", "target": label,
                "placed": name, "felt_contact": felt,
                "contact_z": round(z, 4), "expected_z": round(touch_z, 4),
                "force_n": round(force, 1),
                "detail": (f"touched down at {force:.1f} N, "
                           f"{1000 * (z - touch_z):+.1f} mm from where the "
                           f"camera expected" if felt else
                           "no contact felt; released at the lower limit")}

    def _retreat_straight_up(self, orn, clear_z=0.45):
        """
        Leave a placed part without dragging it. The first millimetres
        are taken in small steps with the sideways position corrected
        before each one: a finger that drifts into the part here
        carries it back up out of its seat.
        """
        ee_up = self.get_end_effector_position()
        off_x = off_y = 0.0
        for rise in (0.002, 0.004, 0.006, 0.009, 0.012, 0.016, 0.02,
                     0.03, 0.04, 0.055, 0.07):
            for _ in range(2):
                self._direct_move(ee_up[0] + off_x, ee_up[1] + off_y,
                                  ee_up[2] + rise,
                                  speed_factor=0.08, orn=orn)
                now = self.get_end_effector_position()
                error = math.hypot(ee_up[0] - now[0], ee_up[1] - now[1])
                off_x += ee_up[0] - now[0]
                off_y += ee_up[1] - now[1]
                if error < 0.0007:
                    break
        self._direct_move(ee_up[0], ee_up[1], clear_z, speed_factor=0.4,
                          orn=orn)

    def place_object(self, x, y, stack_on=None):
        if not self.grasped_object:
            return {
                "status": "failed", "action": "place",
                "target": f"({x},{y})",
                "detail": "not holding anything"
            }

        placed = self.grasped_object
        self.logger.info(f"Placing '{placed}' at ({x},{y})")

        if x < 0.58:
            x, y = self._safe_place_position(x, y, ignore_object=stack_on)

        # Rise first
        current   = self.get_end_effector_position()
        safe_rise = max(current[2] + 0.20, 0.55)
        self._direct_move(
            current[0], current[1], safe_rise, speed_factor=0.6
        )

        # Transport to above target
        print(f"  [TRANSPORT] Moving to above target...")
        current_pos = self.get_end_effector_position()
        path, status, _ = self.rrt.plan_transport(
            current_pos, (x, y, safe_rise)
        )
        if status == "blocked" or path is None:
            self.move_to_position(x, y, safe_rise, use_rrt=True)
        else:
            if (self.motion_config["smooth_trajectory"]
                    and len(path) > 1):
                try:
                    traj = self.trajectory.generate_smooth_trajectory(path)
                    _, vels, _ = self.trajectory.apply_trapezoidal_profile(
                        traj
                    )
                    self._execute_trajectory(traj, vels)
                except Exception:
                    self._execute_trajectory(path)
            else:
                self._execute_trajectory(path)

        # Place height
        if stack_on:
            stack_pos = self.get_object_position(stack_on)
            place_z   = (stack_pos[2] + 0.05) if stack_pos else 0.18
            print(f"  [PLACE] Stacking on '{stack_on}' "
                  f"— descend to z={place_z:.3f}")
        else:
            # FIX: Lower place height = less drop = less bounce
            place_z = 0.10

        # Hover exactly above target - arm settles
        self._direct_move(x, y, place_z + 0.15, speed_factor=0.15)
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Physics-driven descent - feel the surface
        ee_now = self.get_end_effector_position()
        if stack_on and stack_pos:
            # Stacking leaves no room for error: square the tool,
            # measure where the part sits in the gripper, aim the
            # part (not the gripper) at the base, and stop just above
            # the base instead of pressing down until something gives.
            orn = self._down_orientation()
            for _ in range(3):
                self._direct_move(x, y, ee_now[2], speed_factor=0.15,
                                  orn=orn)
            ee_now  = self.get_end_effector_position()
            held    = self.get_object_position(placed)
            base_h  = self.objects[stack_on].get("half_extent", 0.025)
            part_h  = self.objects[placed].get("half_extent", 0.025)
            aim_x   = x - (held[0] - ee_now[0])
            aim_y   = y - (held[1] - ee_now[1])
            floor_z = (stack_pos[2] + base_h + part_h
                       + (ee_now[2] - held[2]) + 0.002)
            self._descend_until_contact(
                aim_x, aim_y,
                start_z=ee_now[2],
                torque_threshold=6.0,
                step_size=0.003,
                floor_z=floor_z
            )
        else:
            self._descend_until_contact(
                x, y,
                start_z=ee_now[2],
                torque_threshold=6.0,
                step_size=0.003
            )

        # Settle on surface
        for _ in range(60):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Gradual release then open fingers
        if self.grasp_constraint is not None:
            for force in [40, 20, 5, 0]:
                p.changeConstraint(
                    self.grasp_constraint, maxForce=force
                )
                for _ in range(8):
                    p.stepSimulation()
                    time.sleep(self.sim_config["timestep"])
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None
        self.open_gripper()

        self.grasped_object         = None
        self.rrt.carrying_clearance = 0.0

        # FIX: Fewer settle steps after release (was 150, now 50)
        # Less time for object to slide on table
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Rise away cleanly - straight up first, then move
        released = self.get_object_position(placed)
        if stack_on:
            # Straight up and slowly at first: any sideways motion
            # here drags the part off the one below it.
            self._retreat_straight_up(self._tool_orn)
        else:
            self._direct_move(x, y, 0.45, speed_factor=0.6,
                              orn=self._tool_orn)

        # Release confirmation: the part must stay where it was put
        # down, not come back up with the gripper.
        final = self.get_object_position(placed)
        if released and final and final[2] > released[2] + 0.05:
            print(f"  [PLACE] ⚠️  '{placed}' came up with the gripper "
                  f"— setting it down again")
            ee = self.get_end_effector_position()
            self._direct_move(ee[0], ee[1],
                              ee[2] - (final[2] - released[2]) + 0.005,
                              speed_factor=0.1, orn=self._tool_orn)
            self.open_gripper()
            for _ in range(120):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])
            self._direct_move(x, y, 0.45, speed_factor=0.1,
                              orn=self._tool_orn)
            final = self.get_object_position(placed)
            if final and final[2] > released[2] + 0.05:
                self.grasped_object = placed
                self._sync_obstacles()
                return {
                    "status": "failed", "action": "place",
                    "target": placed,
                    "detail": f"'{placed}' did not release from the "
                              f"gripper — human intervention required"
                }
        self._sync_obstacles()

        # Verify
        if final:
            drift = math.sqrt(
                (final[0]-x)**2 + (final[1]-y)**2
            )
            if drift > 0.08:
                print(f"  [PLACE] ⚠️  Drift {drift*100:.1f}cm from target")
            else:
                print(f"  [PLACE] ✅ {placed} at "
                      f"({final[0]:.3f},{final[1]:.3f})")

        self.logger.success(f"Placed '{placed}' at ({x:.2f},{y:.2f})")
        return {
            "status": "success", "action": "place",
            "target": placed,
            "position": [x, y],
            "verified_position": list(final) if final else None,
            "detail": (f"placed {placed} at ({x:.2f},{y:.2f})"
                       + (f" on {stack_on}" if stack_on else ""))
        }

    def _safe_place_position(self, x, y, min_clearance=0.12,
                              ignore_object=None):
        pos = np.array([x, y, 0.05])
        for obs in self.rrt.obstacles:
            if ignore_object and (
                obs["name"] == f"obj_{ignore_object}"
                or obs["name"] == ignore_object
            ):
                continue
            if obs["name"].startswith("obj_"):
                continue
            obs_xy   = obs["center"][:2]
            to_obs   = pos[:2] - obs_xy
            dist     = np.linalg.norm(to_obs)
            min_dist = np.max(obs["size"][:2]) + min_clearance
            if dist < min_dist:
                direction = (to_obs / dist if dist > 0.001
                             else np.array([1.0, 0.0]))
                nudge  = direction * (min_dist - dist + 0.03)
                new_xy = pos[:2] + nudge
                print(f"  [PLACE] Nudged: "
                      f"({x:.2f},{y:.2f}) -> "
                      f"({new_xy[0]:.2f},{new_xy[1]:.2f})")
                x, y = float(new_xy[0]), float(new_xy[1])
                pos  = np.array([x, y, 0.05])
        return x, y

    def poll_camera_buttons(self):
        """
        Call this in main loop to check camera toggle buttons.
        Switches PyBullet viewport when button is clicked.
        """
        if not self.sim_config["gui"]:
            return
        if not hasattr(self, '_btn_overhead'):
            return

        buttons = {
            "overhead": self._btn_overhead,
            "front":    self._btn_front,
            "top":      self._btn_top,
            "side":     self._btn_side,
            "close":    self._btn_close,
            "wide":     self._btn_wide,
        }

        presets = {
            "overhead": {"distance": 1.2, "yaw": 0,
                         "pitch": -89, "target": [0.4, 0, 0]},
            "front":    {"distance": 1.5, "yaw": 0,
                         "pitch": -20, "target": [0.3, 0, 0.2]},
            "top":      {"distance": 1.0, "yaw": 0,
                         "pitch": -89, "target": [0.5, 0, 0]},
            "side":     {"distance": 1.5, "yaw": 90,
                         "pitch": -20, "target": [0.3, 0, 0.2]},
            "close":    {"distance": 0.5, "yaw": 45,
                         "pitch": -30, "target": [0.45, 0, 0.15]},
            "wide":     {"distance": 2.5, "yaw": 45,
                         "pitch": -30, "target": [0.2, 0, 0.1]},
        }

        for name, btn_id in buttons.items():
            try:
                val = p.readUserDebugParameter(btn_id)
                prev = self._prev_btn_vals.get(name, 0)
                if val != prev:
                    self._prev_btn_vals[name] = val
                    cam = presets[name]
                    p.resetDebugVisualizerCamera(
                        cameraDistance       = cam["distance"],
                        cameraYaw            = cam["yaw"],
                        cameraPitch          = cam["pitch"],
                        cameraTargetPosition = cam["target"]
                    )
                    print(f"  [CAMERA] Switched to {name} view")
            except Exception:
                pass

        # Wrist button - saves current wrist frame
        try:
            wrist_val  = p.readUserDebugParameter(self._btn_wrist)
            wrist_prev = self._prev_btn_vals.get("wrist", 0)
            if wrist_val != wrist_prev:
                self._prev_btn_vals["wrist"] = wrist_val
                if hasattr(self.camera, 'show_wrist_view'):
                    import os
                    os.makedirs("wrist_frames", exist_ok=True)
                    frame = self.camera.show_wrist_view(
                        f"wrist_frames/manual_{wrist_val:.0f}.png"
                    )
                    # Move camera close to gripper
                    ee = self.get_end_effector_position()
                    p.resetDebugVisualizerCamera(
                        cameraDistance       = 0.3,
                        cameraYaw            = 45,
                        cameraPitch          = -20,
                        cameraTargetPosition = list(ee)
                    )
                    print(f"  [CAMERA] Wrist view - camera at gripper")
        except Exception:
            pass

    def home(self):
        self.logger.info("Returning to home")
        self.move_to_position(0.0, 0.0, 0.6, use_rrt=True)
        self.reset_to_home()

    def set_camera(self, preset="default"):
        presets = {
            "default": {"distance": 1.8, "yaw": 45,
                        "pitch": -25, "target": [0.2, 0, 0.2]},
            "front":   {"distance": 1.5, "yaw": 0,
                        "pitch": -20, "target": [0.3, 0, 0.2]},
            "top":     {"distance": 1.5, "yaw": 0,
                        "pitch": -89, "target": [0.3, 0, 0]},
            "side":    {"distance": 1.5, "yaw": 90,
                        "pitch": -20, "target": [0.3, 0, 0.2]},
            "close":   {"distance": 0.8, "yaw": 45,
                        "pitch": -20, "target": [0.35, 0, 0.1]},
            "wide":    {"distance": 2.5, "yaw": 45,
                        "pitch": -30, "target": [0.2, 0, 0.1]},
        }
        cam = presets.get(preset, presets["default"])
        p.resetDebugVisualizerCamera(
            cameraDistance      = cam["distance"],
            cameraYaw           = cam["yaw"],
            cameraPitch         = cam["pitch"],
            cameraTargetPosition = cam["target"]
        )
        print(f"  [VIEW] Camera: {preset}")

    def step(self):
        p.stepSimulation()

    def disconnect(self):
        self.logger.info("Disconnecting")
        p.disconnect()