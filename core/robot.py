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


class FrankaPandaRobot:
    def __init__(self, config, logger):
        self.config        = config
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

        self.objects          = {}
        self.obstacles        = {}
        self.grasp_constraint = None
        self.grasped_object   = None

        # FIX: Correct slot positions - all different, within safe reach
        # Slots at x=0.60m max - even with 15cm drift stays within 0.75m
        self.conveyor_slots = {
            1: (0.60,  0.25),   # slot 1 - right
            2: (0.60,  0.00),   # slot 2 - centre
            3: (0.60, -0.25),   # slot 3 - left
        }
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
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
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
        if not self.finger_joints:
            return
        for joint in self.finger_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                p.POSITION_CONTROL,
                targetPosition=0.04, force=10
            )

    def close_gripper(self):
        if not self.finger_joints:
            return
        for joint in self.finger_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                p.POSITION_CONTROL,
                targetPosition=0.035, force=5
            )
        for _ in range(30):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

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

    def add_object(self, name, x, y, color, z=0.05):
        col    = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04]
        )
        vis    = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04],
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
            "id": obj_id, "color": color, "position": [x, y, z]
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

    def _direct_move(self, x, y, z, speed_factor=1.0):
        target       = np.array([x, y, z])
        current      = np.array(self.get_end_effector_position())
        n            = self.rrt.trajectory_points
        steps_per_pt = max(1, self.sim_config["steps_per_move"] // n)

        for i in range(n + 1):
            t     = i / n
            point = current + t * (target - current)
            joint_angles = p.calculateInverseKinematics(
                self.robot, self.end_effector, point,
                maxNumIterations=100, residualThreshold=0.001
            )
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

    def pick_object(self, name):
        pos = self.get_object_position(name)
        if not pos:
            self.logger.error(f"Object '{name}' not found")
            return {
                "status": "failed", "action": "pick",
                "target": name,
                "detail": f"object '{name}' not found in scene"
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

        print(f"  [PICK] Descending...")
        self._direct_move(
            aligned_x, aligned_y,
            pos[2] + 0.02, speed_factor=0.3
        )

        self.close_gripper()
        obj_id = self.objects[name]["id"]
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId  = self.robot,
            parentLinkIndex     = self.end_effector,
            childBodyUniqueId   = obj_id,
            childLinkIndex      = -1,
            jointType           = p.JOINT_FIXED,
            jointAxis           = [0, 0, 0],
            parentFramePosition = [0, 0, 0.05],
            childFramePosition  = [0, 0, 0]
        )
        self.grasped_object         = name
        self.rrt.carrying_clearance = 0.04

        print(f"  [PICK] Lifting '{name}'...")
        self._direct_move(
            pos[0], pos[1], pos[2] + 0.40, speed_factor=0.7
        )
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

        # FIX: Pre-place hover - arm settles exactly above slot centre
        # Move to exact XY while still high
        self._direct_move(x, y, place_z + 0.15, speed_factor=0.2)

        # Pause - let arm settle at exact position before descent
        for _ in range(60):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Fast approach
        self._direct_move(x, y, place_z + 0.05, speed_factor=0.3)

        # FIX: Very slow final descent - straight down, no sideways motion
        self._direct_move(x, y, place_z, speed_factor=0.03)

        # Settle before release
        for _ in range(80):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # FIX: Gradual constraint release - reduce force slowly
        # Prevents sudden impulse that causes drift
        if self.grasp_constraint is not None:
            for force in [150, 100, 50, 20, 0]:
                p.changeConstraint(
                    self.grasp_constraint, maxForce=force
                )
                for _ in range(10):
                    p.stepSimulation()
                    time.sleep(self.sim_config["timestep"])
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

        self.grasped_object         = None
        self.rrt.carrying_clearance = 0.0

        # FIX: Fewer settle steps after release (was 150, now 50)
        # Less time for object to slide on table
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Rise away cleanly - straight up first, then move
        self._direct_move(x, y, 0.45, speed_factor=0.6)
        self._sync_obstacles()

        # Verify
        final = self.get_object_position(placed)
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