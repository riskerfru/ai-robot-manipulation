# =============================================================
#   JOINT CONTROLLER - Direct Joint Control
#   Control individual joints of Franka Panda by name or number
#   Supports natural language like "rotate joint 3 by 45 degrees"
# =============================================================

import pybullet as p
import time
import numpy as np


class JointController:
    def __init__(self, robot_instance, logger):
        self.robot  = robot_instance
        self.logger = logger

        # Franka Panda joint names and their indices
        self.joint_map = {
            # By number
            "1": 0, "2": 1, "3": 2, "4": 3,
            "5": 4, "6": 5, "7": 6,
            # By name
            "joint1": 0, "joint2": 1, "joint3": 2, "joint4": 3,
            "joint5": 4, "joint6": 5, "joint7": 6,
            "j1": 0, "j2": 1, "j3": 2, "j4": 3,
            "j5": 4, "j6": 5, "j7": 6,
            # By description
            "base":        0,
            "shoulder":    1,
            "upper_arm":   2,
            "elbow":       3,
            "forearm":     4,
            "wrist":       5,
            "end":         6,
        }

        # Joint limits for Franka Panda (radians)
        self.joint_limits = {
            0: (-2.8973, 2.8973),   # Base rotation
            1: (-1.7628, 1.7628),   # Shoulder
            2: (-2.8973, 2.8973),   # Upper arm
            3: (-3.0718, -0.0698),  # Elbow
            4: (-2.8973, 2.8973),   # Forearm
            5: (-0.0175, 3.7525),   # Wrist
            6: (-2.8973, 2.8973),   # End rotation
        }

        # Joint descriptions for display
        self.joint_descriptions = {
            0: "Base Rotation",
            1: "Shoulder",
            2: "Upper Arm Twist",
            3: "Elbow",
            4: "Forearm Twist",
            5: "Wrist",
            6: "End Effector Rotation"
        }

        self.logger.info("Joint Controller initialized")

    # ----------------------------------------------------------
    # Core joint control
    # ----------------------------------------------------------

    def get_joint_angle(self, joint_idx):
        """Get current angle of a joint in degrees"""
        state = p.getJointState(self.robot.robot, joint_idx)
        return np.degrees(state[0])

    def get_all_joint_angles(self):
        """Get all current joint angles"""
        angles = {}
        for i in self.robot.arm_joints:
            angles[i] = self.get_joint_angle(i)
        return angles

    def set_joint_angle(self, joint_idx, angle_deg, speed=0.5):
        """
        Move a single joint to a target angle in degrees.
        speed: 0.1 (slow) to 1.0 (fast)
        """
        # Validate joint index
        if joint_idx not in self.robot.arm_joints:
            print(f"  Invalid joint index: {joint_idx}")
            return False

        angle_rad = np.radians(angle_deg)

        # Check joint limits
        limits = self.joint_limits.get(joint_idx, (-3.14, 3.14))
        if not (limits[0] <= angle_rad <= limits[1]):
            limit_deg_min = np.degrees(limits[0])
            limit_deg_max = np.degrees(limits[1])
            print(
                f"\n  [LIMIT] Joint {joint_idx+1} ({self.joint_descriptions[joint_idx]}) "
                f"cannot go to {angle_deg:.1f} degrees."
                f"\n  Allowed range: {limit_deg_min:.1f} to {limit_deg_max:.1f} degrees\n"
            )
            # Clamp to nearest limit
            angle_rad = np.clip(angle_rad, limits[0], limits[1])
            angle_deg = np.degrees(angle_rad)
            print(f"  Clamping to {angle_deg:.1f} degrees instead.")

        current_deg = self.get_joint_angle(joint_idx)
        self.logger.info(
            f"Joint {joint_idx+1} ({self.joint_descriptions[joint_idx]}): "
            f"{current_deg:.1f} -> {angle_deg:.1f} degrees"
        )

        # Move joint
        p.setJointMotorControl2(
            self.robot.robot,
            joint_idx,
            p.POSITION_CONTROL,
            targetPosition=angle_rad,
            force=self.robot.robot_config["max_force"],
            maxVelocity=speed * self.robot.robot_config["max_velocity"]
        )

        # Wait for movement to complete
        for _ in range(300):
            p.stepSimulation()
            time.sleep(self.robot.sim_config["timestep"])
            current = p.getJointState(self.robot.robot, joint_idx)[0]
            if abs(current - angle_rad) < 0.01:
                break

        final_deg = self.get_joint_angle(joint_idx)
        self.logger.success(
            f"Joint {joint_idx+1} reached {final_deg:.1f} degrees"
        )
        return True

    def rotate_joint_by(self, joint_idx, delta_deg, speed=0.5):
        """Rotate a joint by a relative amount in degrees"""
        current_deg = self.get_joint_angle(joint_idx)
        target_deg  = current_deg + delta_deg
        print(
            f"\n  Rotating Joint {joint_idx+1} "
            f"({self.joint_descriptions[joint_idx]}) "
            f"by {delta_deg:+.1f} degrees"
            f"\n  {current_deg:.1f} -> {target_deg:.1f} degrees"
        )
        return self.set_joint_angle(joint_idx, target_deg, speed)

    # ----------------------------------------------------------
    # Natural language parsing
    # ----------------------------------------------------------

    def parse_and_execute(self, command):
        """
        Parse natural language joint commands and execute them.

        Examples:
          "move joint 3 to 45 degrees"
          "rotate joint 2 by -30 degrees"
          "set elbow to 90"
          "move wrist by 45"
          "show joints"
          "reset joints"
        """
        cmd = command.lower().strip()

        # Show joint status
        if any(word in cmd for word in ["show", "status", "list", "angles"]):
            self.show_joint_status()
            return True

        # Reset all joints to home
        if any(word in cmd for word in ["reset", "home", "zero"]):
            self.reset_to_home()
            return True

        # Parse joint identifier
        joint_idx = self._parse_joint(cmd)
        if joint_idx is None:
            print(
                "\n  Could not identify joint. Try:"
                "\n  'move joint 3 to 45 degrees'"
                "\n  'rotate elbow by 30 degrees'"
                "\n  'set wrist to -45'"
            )
            return False

        # Parse angle value
        angle = self._parse_angle(cmd)
        if angle is None:
            print("\n  Could not parse angle. Include a number like '45' or '-30'")
            return False

        # Determine if absolute or relative
        if any(word in cmd for word in ["by", "rotate", "turn", "spin"]):
            return self.rotate_joint_by(joint_idx, angle)
        else:
            return self.set_joint_angle(joint_idx, angle)

    def _parse_joint(self, cmd):
        """Extract joint index from command string"""
        # Try joint number patterns: "joint 3", "j3", "joint3"
        import re

        # Pattern: joint followed by number
        match = re.search(r'joint\s*(\d)', cmd)
        if match:
            key = match.group(1)
            return self.joint_map.get(key)

        # Pattern: j followed by number
        match = re.search(r'\bj(\d)\b', cmd)
        if match:
            key = f"j{match.group(1)}"
            return self.joint_map.get(key)

        # Try named joints
        for name, idx in self.joint_map.items():
            if name in cmd and not name.isdigit():
                return idx

        return None

    def _parse_angle(self, cmd):
        """Extract angle value from command string"""
        import re

        # Look for number (possibly negative) followed by optional "degrees" or "deg"
        match = re.search(r'(-?\d+\.?\d*)\s*(?:degrees?|deg)?', cmd)
        if match:
            # Skip if this is just the joint number (1-7)
            val = float(match.group(1))
            if 1 <= val <= 7 and val == int(val):
                # Check if there's another number
                matches = re.findall(r'-?\d+\.?\d*', cmd)
                numbers = [float(m) for m in matches]
                # Return the first number that isn't a joint number
                for num in numbers:
                    if not (1 <= num <= 7 and num == int(num)):
                        return num
                return None
            return val

        return None

    # ----------------------------------------------------------
    # Display and utility
    # ----------------------------------------------------------

    def show_joint_status(self):
        """Display current status of all joints"""
        print("\n  +---------+----------------------+----------+-------------------+")
        print("  | Joint # | Description          | Current  | Limits            |")
        print("  +---------+----------------------+----------+-------------------+")

        for i in self.robot.arm_joints:
            current   = self.get_joint_angle(i)
            limits    = self.joint_limits.get(i, (-180, 180))
            lim_min   = np.degrees(limits[0])
            lim_max   = np.degrees(limits[1])
            desc      = self.joint_descriptions.get(i, f"Joint {i+1}")
            print(
                f"  | J{i+1:<6} | {desc:<20} | {current:>+6.1f}   | "
                f"{lim_min:>+7.1f} to {lim_max:>+7.1f} |"
            )

        print("  +---------+----------------------+----------+-------------------+\n")

    def reset_to_home(self):
        """Move all joints back to home position"""
        print("\n  Resetting all joints to home position...")
        home = self.robot.robot_config["home_position"]

        for i, joint_idx in enumerate(self.robot.arm_joints):
            p.setJointMotorControl2(
                self.robot.robot,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=home[i],
                force=self.robot.robot_config["max_force"],
                maxVelocity=0.5
            )

        for _ in range(400):
            p.stepSimulation()
            time.sleep(self.robot.sim_config["timestep"])

        print("  All joints at home position.\n")

    def wave(self):
        """Make the robot wave - demo function"""
        print("\n  Robot waving...")
        original = self.get_all_joint_angles()

        self.set_joint_angle(5, 60)
        self.set_joint_angle(6, 45)
        self.set_joint_angle(6, -45)
        self.set_joint_angle(6, 45)
        self.set_joint_angle(6, 0)
        self.set_joint_angle(5, 0)

        print("  Wave complete!\n")

    def print_joint_help(self):
        print("""
  JOINT CONTROL COMMANDS:
  ----------------------------------------
  show joints               show all joint angles
  reset joints              return to home

  move joint <N> to <angle>     set absolute angle
  rotate joint <N> by <angle>   rotate by relative amount
  set <name> to <angle>         use joint name

  Joint numbers: 1-7
  Joint names: base, shoulder, upper_arm,
               elbow, forearm, wrist, end

  Examples:
    move joint 3 to 45
    rotate joint 1 by -30
    set elbow to 90
    rotate wrist by 45
    move joint 6 to -20
  ----------------------------------------
""")