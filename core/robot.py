# =============================================================
#   ROBOT CONTROLLER - Franka Panda
#   RRT collision avoidance + obstacle detection + reporting
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
        self.config       = config
        self.logger       = logger
        self.robot_config = config["robot_limits"]
        self.sim_config   = config["sim"]
        self.motion_config = config["motion"]
        self.camera_config = config["camera"]

        self._init_simulation()

        self.rrt        = RRTPlanner(self.motion_config, logger)
        self.trajectory = TrajectoryPlanner(self.motion_config, logger)

        self.camera = CameraSystem(self.camera_config, logger)
        self.camera.setup_overhead_camera()
        self.camera.setup_side_camera()

        self.objects   = {}
        self.obstacles = {}

        self.grasp_constraint = None
        self.grasped_object   = None

        self.rrt.set_workspace_limits(
            self.robot_config["x"],
            self.robot_config["y"],
            self.robot_config["z"]
        )

        self.reset_to_home()
        self.logger.success("Franka Panda robot ready")

    # ----------------------------------------------------------
    # Simulation setup
    # ----------------------------------------------------------

    def _init_simulation(self):
        if self.sim_config["gui"]:
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.3]
            )
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

        self.num_joints   = p.getNumJoints(self.robot)
        self.arm_joints   = self.robot_config["arm_joints"]
        self.finger_joints = self.robot_config["finger_joints"]
        self.end_effector = self.robot_config["end_effector_index"]
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
                targetPosition=0.04,
                force=10
            )

    def close_gripper(self):
        if not self.finger_joints:
            return
        for joint in self.finger_joints:
            p.setJointMotorControl2(
                self.robot, joint,
                p.POSITION_CONTROL,
                targetPosition=0.01,
                force=10
            )
        for _ in range(200):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

    # ----------------------------------------------------------
    # Scene objects
    # ----------------------------------------------------------

    def add_object(self, name, x, y, color, z=0.05):
        """Add a graspable coloured block"""
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04]
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.04], rgbaColor=color
        )
        obj_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z]
        )
        self.objects[name] = {
            "id": obj_id, "color": color, "position": [x, y, z]
        }
        self.logger.info(f"Added object '{name}' at ({x}, {y}, {z})")
        self._update_rrt_obstacles()
        return obj_id

    def add_obstacle(self, name, x, y, size, color=[0.5, 0.5, 0.5, 0.9], z=0.0):
        """
        Add a static obstacle (wall, block, etc).
        size = [half_width, half_depth, half_height]
        """
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=size, rgbaColor=color
        )
        center_z = z + size[2]
        obj_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, center_z]
        )
        self.obstacles[name] = {
            "id":       obj_id,
            "position": [x, y, center_z],
            "size":     size
        }
        # Register with RRT immediately
        self.rrt.add_obstacle(
            center=(x, y, center_z),
            size=size,
            name=name
        )
        self.logger.info(
            f"Added obstacle '{name}' at ({x}, {y}, {center_z}) "
            f"size={size}"
        )
        return obj_id

    def remove_obstacle(self, name):
        """Remove an obstacle from scene and RRT"""
        if name in self.obstacles:
            p.removeBody(self.obstacles[name]["id"])
            del self.obstacles[name]
            # Rebuild RRT obstacles without this one
            self.rrt.obstacles = [
                o for o in self.rrt.obstacles if o["name"] != name
            ]
            self.logger.info(f"Removed obstacle '{name}'")
        else:
            self.logger.warning(f"Obstacle '{name}' not found")

    def list_obstacles(self):
        """Print all current obstacles"""
        if not self.obstacles:
            print("\n  No obstacles in scene.")
            return
        print("\n  Current obstacles:")
        for name, data in self.obstacles.items():
            pos = data["position"]
            size = data["size"]
            print(f"    {name}: pos=({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) "
                  f"size={size}")

    def _update_rrt_obstacles(self, ignore_object=None):
        """Sync graspable objects into RRT as obstacles"""
        objects_for_rrt = {
            name: self.get_object_position(name)
            for name in self.objects
            if name != ignore_object
        }
        self.rrt.update_obstacles_from_objects(objects_for_rrt)

    def get_object_position(self, name):
        if name not in self.objects:
            return None
        position, _ = p.getBasePositionAndOrientation(
            self.objects[name]["id"]
        )
        return position

    def get_all_object_positions(self):
        return {name: self.get_object_position(name) for name in self.objects}

    # ----------------------------------------------------------
    # Camera
    # ----------------------------------------------------------

    def scan_scene(self):
        """Scan scene and return detected object positions"""
        self.logger.info("Scanning scene...")
        return self.camera.detect_objects(self.objects, "overhead")

    def get_end_effector_position(self):
        return p.getLinkState(self.robot, self.end_effector)[0]

    # ----------------------------------------------------------
    # Motion
    # ----------------------------------------------------------

    def move_to_position(self, x, y, z, use_rrt=True):
        """
        Move to position with RRT collision avoidance.
        If blocked, prints obstacle report and returns False.
        """
        target  = (x, y, z)
        current = self.get_end_effector_position()
        self.logger.log_movement(current, target,
                                  "rrt" if use_rrt else "direct")

        if use_rrt and self.motion_config["planner"] == "rrt":
            path, status, blocking = self.rrt.plan(current, target)

            if status == "blocked":
                self._report_blocked(target, blocking)
                return False

            waypoints = path
        else:
            waypoints = [np.array(current), np.array(target)]

        if self.motion_config["smooth_trajectory"]:
            trajectory = self.trajectory.generate_smooth_trajectory(waypoints)
        else:
            trajectory = waypoints

        self._execute_trajectory(trajectory)
        return True

    def _report_blocked(self, target, blocking):
        """Print a clear obstacle report when path is blocked"""
        obs_details = []
        for o in blocking:
            c = o["center"]
            obs_details.append(
                f"{o['name']} at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"
            )

        print(
            f"\n  [BLOCKED] Cannot reach target "
            f"({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})"
        )
        if obs_details:
            print(f"  Obstacles blocking the path:")
            for detail in obs_details:
                print(f"    - {detail}")
        else:
            print(f"  Path is outside workspace or no valid path found.")
        print(f"  Type 'obstacles' to list all obstacles.")
        print(f"  Type 'remove obstacle <name>' to remove one.\n")

    def _execute_trajectory(self, trajectory):
        steps           = self.sim_config["steps_per_move"]
        points_per_step = max(1, steps // max(len(trajectory), 1))

        for point in trajectory:
            point = np.array(point)
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
                    maxVelocity=self.robot_config["max_velocity"]
                )
            for _ in range(points_per_step):
                p.stepSimulation()
                time.sleep(self.sim_config["timestep"])

    # ----------------------------------------------------------
    # Pick and place
    # ----------------------------------------------------------

    def pick_object(self, name):
        pos = self.get_object_position(name)
        if not pos:
            self.logger.error(f"Object '{name}' not found")
            return False

        self.logger.info(f"Picking '{name}' at {pos}")
        self._update_rrt_obstacles(ignore_object=name)

        self.open_gripper()
        for _ in range(50):
            p.stepSimulation()
            time.sleep(self.sim_config["timestep"])

        # Move above with RRT
        success = self.move_to_position(
            pos[0], pos[1], pos[2] + 0.3, use_rrt=True
        )
        if not success:
            self.logger.warning(f"Cannot reach above '{name}' — path blocked")
            self._update_rrt_obstacles()
            return False

        # Descend straight down
        self.move_to_position(pos[0], pos[1], pos[2] + 0.06, use_rrt=False)

        # Close and attach
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

        # Lift
        self.move_to_position(pos[0], pos[1], pos[2] + 0.4, use_rrt=False)
        self._update_rrt_obstacles()

        self.logger.log_grasp(name, True, pos)
        self.logger.success(f"Picked '{name}'!")
        return True

    def place_object(self, x, y):
        if not self.grasped_object:
            self.logger.warning("Not holding anything")
            return False

        self.logger.info(f"Placing '{self.grasped_object}' at ({x}, {y})")

        success = self.move_to_position(x, y, 0.4, use_rrt=True)
        if not success:
            return False

        self.move_to_position(x, y, 0.13, use_rrt=False)

        if self.grasp_constraint is not None:
            p.removeConstraint(self.grasp_constraint)
            self.grasp_constraint = None

        self.open_gripper()
        placed = self.grasped_object
        self.grasped_object = None

        self.move_to_position(x, y, 0.4, use_rrt=False)
        self._update_rrt_obstacles()

        self.logger.success(f"Placed '{placed}' at ({x}, {y})")
        return True

    def home(self):
        self.logger.info("Returning to home")
        self.move_to_position(0.0, 0.0, 0.6, use_rrt=True)
        self.reset_to_home()

    def step(self):
        p.stepSimulation()

    def disconnect(self):
        self.logger.info("Disconnecting")
        p.disconnect()