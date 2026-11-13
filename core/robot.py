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
        self._insertion    = None
        self.on_force      = None     # optional callback(z, force_n)
        self.use_force_sensing = True # False: comparison runs only
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

    def open_gripper(self):
        """Open fingers fully - 4cm gap, clearly visible"""
        if not self.finger_joints:
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
                   grasp_depth=None):
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
            lateralFriction  = 1.5,
            spinningFriction = 0.05,
            rollingFriction  = 0.05
        )

        self.objects[name] = {
            "id": obj_id, "color": color, "position": [x, y, z],
            "half_extent": half_extent,
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

    def _down_orientation(self):
        """
        Gripper pointing straight down with the fingers square to
        the cell axes, at the quarter turn nearest the current yaw.
        """
        current = p.getLinkState(self.robot, self.end_effector)[1]
        yaw     = p.getEulerFromQuaternion(current)[2]
        yaw     = round(yaw / (math.pi / 2)) * (math.pi / 2)
        return p.getQuaternionFromEuler([math.pi, 0.0, yaw])

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
        orn = self._down_orientation()
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

        print(f"  [DESCENT] Reached min z={current_z:.3f}m")
        return current_z

    def _check_grasp_success(self, name, pre_z):
        """Lift slightly and confirm the part came up with the gripper."""
        ee = self.get_end_effector_position()
        self._direct_move(ee[0], ee[1], ee[2]+0.05, speed_factor=0.1,
                          orn=self._tool_orn)
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

    def pick_object(self, name):
        pos = self.get_object_position(name)
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

        self.open_gripper()
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

        print(f"  [PICK] Above target - visual servoing...")
        aligned_x, aligned_y = self.servo.servo_to_object(self, name)

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
        p.changeConstraint(self.grasp_constraint, maxForce=50)
        self.grasped_object         = name
        self.rrt.carrying_clearance = 0.04

        # Verify grasp by lifting slightly
        pre_z   = obj_pos2[2]
        grasped = self._check_grasp_success(name, pre_z)
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
        self._direct_move(pos[0], pos[1], pos[2]+0.15, speed_factor=0.08,
                          orn=self._tool_orn)
        self._direct_move(pos[0], pos[1], pos[2]+0.40, speed_factor=0.3)
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

    def insert_part(self, name, pocket, on_step=None):
        """Seat the held part in a pocket by force-guided insertion."""
        from motion.insertion import ForceGuidedInsertion
        if self._insertion is None:
            self._insertion = ForceGuidedInsertion(self)
        result = self._insertion.insert(name, pocket,
                                        on_step=on_step or self.on_force,
                                        use_force=self.use_force_sensing)
        self._sync_obstacles()
        return result

    def _retreat_straight_up(self, orn, clear_z=0.45):
        """Leave a placed part without dragging it sideways."""
        ee_up = self.get_end_effector_position()
        off_x = off_y = 0.0
        for i in range(1, 8):
            self._direct_move(ee_up[0] + off_x, ee_up[1] + off_y,
                              ee_up[2] + 0.01 * i,
                              speed_factor=0.08, orn=orn)
            now = self.get_end_effector_position()
            off_x += 0.5 * (ee_up[0] - now[0])
            off_y += 0.5 * (ee_up[1] - now[1])
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