# =============================================================
#   ROBOT CONTROLLER - Franka Panda
#   Clean rewrite with proper None path handling
#   Multi-strategy RRT + direct descent for picking
# =============================================================

import pybullet as p
import pybullet_data
import time
import numpy as np

from motion.rrt_planner import RRTPlanner
from motion.trajectory import TrajectoryPlanner
from perception.camera import CameraSystem


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

        self.camera = CameraSystem(self.camera_config, logger)
        self.camera.setup_overhead_camera()
        self.camera.setup_side_camera()

        self.objects          = {}
        self.obstacles        = {}
        self.grasp_constraint = None
        self.grasped_object   = None

        self.rrt.set_workspace_limits(
            self.robot_config["x"],
            self.robot_config["y"],
            self.robot_config["z"]
        )
        # Give RRT a reference to robot for Strategy 6 joint reconfiguration
        self.rrt.robot = self

        self.reset_to_home()
        self.logger.success("Franka Panda robot ready")

    # ----------------------------------------------------------
    # Simulation setup
    # ----------------------------------------------------------

    def _init_simulation(self):
        if self.sim_config["gui"]:
            self.physics_client = p.connect(p.GUI)

            # 3D camera view - angled to see robot clearly
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=45,
                cameraPitch=-25,
                cameraTargetPosition=[0.2, 0, 0.2]
            )

            # Clean up GUI - remove unused panels
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)

            # Add coordinate axes visual
            p.addUserDebugLine([0,0,0], [0.3,0,0], [1,0,0], 2)  # X red
            p.addUserDebugLine([0,0,0], [0,0.3,0], [0,1,0], 2)  # Y green
            p.addUserDebugLine([0,0,0], [0,0,0.3], [0,0,1], 2)  # Z blue

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
                targetPosition=0.01, force=10
            )
        for _ in range(200):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

    # ----------------------------------------------------------
    # Scene management
    # ----------------------------------------------------------

    def add_object(self, name, x, y, color, z=0.05):
        col    = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04])
        vis    = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04], rgbaColor=color)
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z]
        )
        self.objects[name] = {"id": obj_id, "color": color, "position": [x, y, z]}
        self.logger.info(f"Added object '{name}' at ({x}, {y}, {z})")
        self._sync_obstacles()
        return obj_id

    def add_obstacle(self, name, x, y, size, color=[0.5, 0.5, 0.5, 0.9], z=0.0):
        col       = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis       = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=color)
        center_z  = z + size[2]
        obj_id    = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, center_z]
        )
        self.obstacles[name] = {"id": obj_id, "position": [x, y, center_z], "size": size}
        self.rrt.add_obstacle(center=(x, y, center_z), size=size, name=name)
        self.logger.info(f"Added obstacle '{name}' at ({x}, {y}, {center_z})")
        return obj_id

    def remove_obstacle(self, name):
        if name in self.obstacles:
            p.removeBody(self.obstacles[name]["id"])
            del self.obstacles[name]
            self.rrt.obstacles = [o for o in self.rrt.obstacles if o["name"] != name]
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
        """Sync graspable objects to RRT as obstacles"""
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
        return {name: self.get_object_position(name) for name in self.objects}

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
        """
        Move end effector to position.
        Tries all 5 strategies if RRT is enabled.
        Returns True on success, False if all strategies fail.
        """
        target  = (x, y, z)
        current = self.get_end_effector_position()
        self.logger.log_movement(current, target, "rrt" if use_rrt else "direct")

        if use_rrt and self.motion_config["planner"] == "rrt":
            path, status, blocking = self.rrt.plan(
                current, target, ignore_name=ignore_name
            )

            # Safety check - path must not be None
            if status == "blocked" or path is None:
                self._report_blocked(target, blocking)
                return False

            waypoints = path

        else:
            # Direct movement - no collision checking
            waypoints = [np.array(current), np.array(target)]

        # Generate TOPP-RA optimised trajectory
        if self.motion_config["smooth_trajectory"] and len(waypoints) > 1:
            try:
                trajectory = self.trajectory.generate_smooth_trajectory(
                    waypoints,
                    num_points=self.motion_config["trajectory_points"]
                )
                # Get velocity profile
                positions, velocities, timestamps = (
                    self.trajectory.apply_trapezoidal_profile(trajectory)
                )
                est_time = self.trajectory.get_execution_time(waypoints)
                print(f"  [TRAJ] Estimated execution: {est_time:.1f}s, "
                      f"{len(trajectory)} points")
            except Exception as e:
                trajectory = waypoints
                velocities = None
        else:
            trajectory = waypoints
            velocities = None

        self._execute_trajectory(trajectory, velocities)
        return True

    def _direct_move(self, x, y, z, speed_factor=1.0):
        """
        Move directly to position using IK only.
        No collision checking. Used for final descent/ascent.
        """
        target  = np.array([x, y, z])
        current = np.array(self.get_end_effector_position())
        n       = self.rrt.trajectory_points
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
                    maxVelocity=self.robot_config["max_velocity"] * speed_factor
                )

            for _ in range(steps_per_pt):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])

    def _execute_trajectory(self, trajectory, velocities=None):
        """
        Execute trajectory with optional velocity profile.
        If velocities provided (from TOPP-RA), uses them for
        realistic speed control - slow at start/end, fast in middle.
        """
        steps           = self.sim_config["steps_per_move"]
        points_per_step = max(1, steps // max(len(trajectory), 1))
        base_velocity   = self.robot_config["max_velocity"]

        for idx, point in enumerate(trajectory):
            point = np.array(point)

            # Use TOPP-RA velocity if available
            if velocities and idx < len(velocities):
                # Scale max velocity by normalised profile (0.1 to 1.0)
                v_norm   = velocities[idx] / max(velocities) if max(velocities) > 0 else 1.0
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
        print(f"\n  [BLOCKED] All 5 strategies failed for target "
              f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        if blocking:
            print(f"  Obstacles:")
            for o in blocking:
                c = o["center"]
                print(f"    - {o['name']} at ({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})")
        print(f"  Try: 'remove obstacle <name>' to clear the path\n")

    # ----------------------------------------------------------
    # Pick and place
    # ----------------------------------------------------------

    def pick_object(self, name):
        pos = self.get_object_position(name)
        if not pos:
            self.logger.error(f"Object '{name}' not found")
            return False

        self.logger.info(f"Picking '{name}' at {pos}")

        # Remove target from collision checking
        self._sync_obstacles(ignore_object=name)

        # Open gripper
        self.open_gripper()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Move above object using multi-strategy RRT
        print(f"\n  [PICK] Approaching '{name}'...")
        success = self.move_to_position(
            pos[0], pos[1], pos[2] + 0.40,
            use_rrt=True,
            ignore_name=name
        )

        if not success:
            print(f"  [PICK] Cannot reach above '{name}' - all strategies failed")
            self._sync_obstacles()
            return False

        print(f"  [PICK] Above target - descending...")

        # Descend straight down - go low enough to properly grasp
        self._direct_move(pos[0], pos[1], pos[2] + 0.02, speed_factor=0.3)

        # Close gripper and attach
        self.close_gripper()
        obj_id = self.objects[name]["id"]
        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot,
            parentLinkIndex=self.end_effector,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.05],
            childFramePosition=[0, 0, 0]
        )
        self.grasped_object = name

        # Inflate collision buffer while carrying object
        # Object adds ~8cm to gripper profile
        self.rrt.carrying_clearance = 0.04
        print(f"  [CARRY] Collision buffer: +4cm for walls during transport")

        # Lift straight up - direct IK
        print(f"  [PICK] Lifting '{name}'...")
        self._direct_move(pos[0], pos[1], pos[2] + 0.4, speed_factor=0.7)

        # Restore full obstacle set with carrying clearance
        self._sync_obstacles()

        self.logger.log_grasp(name, True, pos)
        self.logger.success(f"Picked '{name}'!")
        return True

    def place_object(self, x, y, stack_on=None):
        """
        Place held object at x, y.
        stack_on: name of object we are stacking on (skip nudging for it)
        """
        if not self.grasped_object:
            self.logger.warning("Not holding anything")
            return False

        self.logger.info(f"Placing '{self.grasped_object}' at ({x}, {y})")

        # Only nudge away from walls/obstacles - NOT the stack target
        x, y = self._safe_place_position(x, y, ignore_object=stack_on)

        # First rise straight up - direct IK, well above obstacles
        current   = self.get_end_effector_position()
        # Get safe height accounting for obstacles between here and target
        blocking  = self.rrt._get_obstacles_in_corridor(
            np.array([current[0], current[1], 0]),
            np.array([x, y, 0]),
            self.rrt.obstacles
        ) if hasattr(self.rrt, "_get_obstacles_in_corridor") else []

        safe_rise = self.rrt._safe_height(blocking) if blocking             else max(current[2] + 0.20, 0.55)
        # Always rise at least 0.55m when carrying - well above all walls
        safe_rise = max(safe_rise, 0.55)

        print(f"  [PLACE] Rising to {safe_rise:.2f}m first...")
        self._direct_move(current[0], current[1], safe_rise, speed_factor=0.6)

        # Use transport-aware planning - finds max clearance path
        print(f"  [TRANSPORT] Planning carry path with max obstacle clearance...")
        current_pos = self.get_end_effector_position()
        path, status, blocking = self.rrt.plan_transport(
            current_pos, (x, y, safe_rise)
        )

        if status == "blocked" or path is None:
            # Fall back to standard RRT
            success = self.move_to_position(x, y, safe_rise, use_rrt=True)
            if not success:
                return False
        else:
            # Execute transport path directly
            if self.motion_config["smooth_trajectory"] and len(path) > 1:
                try:
                    trajectory = self.trajectory.generate_smooth_trajectory(path)
                    positions, velocities, _ = self.trajectory.apply_trapezoidal_profile(
                        trajectory
                    )
                    self._execute_trajectory(trajectory, velocities)
                except Exception:
                    self._execute_trajectory(path)
            else:
                self._execute_trajectory(path)

        # Descend - if stacking on object, go higher
        if stack_on:
            stack_pos = self.get_object_position(stack_on)
            descend_z = (stack_pos[2] + 0.09) if stack_pos else 0.13
            print(f"  [PLACE] Stacking on '{stack_on}' at z={descend_z:.2f}m")
        else:
            descend_z = 0.13
        self._direct_move(x, y, descend_z, speed_factor=0.4)

        # Release
        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

        self.open_gripper()
        placed = self.grasped_object
        self.grasped_object = None

        # Reset carrying clearance
        self.rrt.carrying_clearance = 0.0

        # Lift directly
        self._direct_move(x, y, 0.4, speed_factor=0.7)
        self._sync_obstacles()

        self.logger.success(f"Placed '{placed}' at ({x}, {y})")
        return True

    def _safe_place_position(self, x, y, min_clearance=0.12,
                              ignore_object=None):
        """
        Nudge place position away from walls only.
        ignore_object: skip nudging for stack target object.
        """
        pos = np.array([x, y, 0.05])

        for obs in self.rrt.obstacles:
            # Skip the object we are intentionally placing on
            if ignore_object and (
                obs["name"] == f"obj_{ignore_object}" or
                obs["name"] == ignore_object
            ):
                continue

            # Only nudge for walls - not other graspable objects
            if obs["name"].startswith("obj_"):
                continue

            obs_xy   = obs["center"][:2]
            to_obs   = pos[:2] - obs_xy
            dist     = np.linalg.norm(to_obs)
            min_dist = np.max(obs["size"][:2]) + min_clearance

            if dist < min_dist:
                if dist < 0.001:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = to_obs / dist

                nudge  = direction * (min_dist - dist + 0.03)
                new_xy = pos[:2] + nudge

                print(f"  [PLACE] Nudged away from {obs['name']}: "
                      f"({x:.2f},{y:.2f}) -> ({new_xy[0]:.2f},{new_xy[1]:.2f})")
                x, y = float(new_xy[0]), float(new_xy[1])
                pos  = np.array([x, y, 0.05])

        return x, y

    def home(self):
        self.logger.info("Returning to home")
        self.move_to_position(0.0, 0.0, 0.6, use_rrt=True)
        self.reset_to_home()

    def set_camera(self, preset="default"):
        """Switch camera angle presets"""
        presets = {
            "default": {
                "distance": 1.8, "yaw": 45,
                "pitch": -25, "target": [0.2, 0, 0.2]
            },
            "front": {
                "distance": 1.5, "yaw": 0,
                "pitch": -20, "target": [0.3, 0, 0.2]
            },
            "top": {
                "distance": 1.5, "yaw": 0,
                "pitch": -89, "target": [0.3, 0, 0]
            },
            "side": {
                "distance": 1.5, "yaw": 90,
                "pitch": -20, "target": [0.3, 0, 0.2]
            },
            "close": {
                "distance": 0.8, "yaw": 45,
                "pitch": -20, "target": [0.35, 0, 0.1]
            },
            "wide": {
                "distance": 2.5, "yaw": 45,
                "pitch": -30, "target": [0.2, 0, 0.1]
            }
        }
        cam = presets.get(preset, presets["default"])
        p.resetDebugVisualizerCamera(
            cameraDistance=cam["distance"],
            cameraYaw=cam["yaw"],
            cameraPitch=cam["pitch"],
            cameraTargetPosition=cam["target"]
        )
        print(f"  [VIEW] Camera: {preset}")

    def step(self):
        p.stepSimulation()

    def disconnect(self):
        self.logger.info("Disconnecting")
        p.disconnect()