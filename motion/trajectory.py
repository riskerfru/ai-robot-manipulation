# =============================================================
#   TRAJECTORY PLANNER - Smooth Motion Control
#   Converts waypoints into smooth velocity controlled motion
#   Uses trapezoidal velocity profiles (industry standard)
# =============================================================

import numpy as np
import time


class TrajectoryPlanner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        # Motion parameters
        self.max_velocity = config.get("max_velocity", 0.5)       # m/s
        self.max_acceleration = config.get("max_acceleration", 0.3) # m/s²
        self.timestep = 1 / 240  # Match PyBullet timestep

        self.logger.info("Trajectory Planner initialized")

    # ----------------------------------------------------------
    # Trapezoidal velocity profile
    # ----------------------------------------------------------

    def trapezoidal_profile(self, distance, max_vel=None, max_acc=None):
        """
        Generate a trapezoidal velocity profile for a given distance.

        Trapezoidal profile:
        - Accelerate from 0 to max_vel
        - Cruise at max_vel
        - Decelerate from max_vel to 0

        Returns list of (position, velocity) tuples along the path.
        This is the industry standard for smooth robot motion.
        """
        max_vel = max_vel or self.max_velocity
        max_acc = max_acc or self.max_acceleration

        # Time to reach max velocity
        t_acc = max_vel / max_acc

        # Distance covered during acceleration/deceleration
        d_acc = 0.5 * max_acc * t_acc ** 2

        # Check if we have enough distance for full trapezoid
        if 2 * d_acc > distance:
            # Triangle profile — not enough distance to reach max_vel
            t_acc = np.sqrt(distance / max_acc)
            peak_vel = max_acc * t_acc
            t_cruise = 0
        else:
            peak_vel = max_vel
            d_cruise = distance - 2 * d_acc
            t_cruise = d_cruise / max_vel

        total_time = 2 * t_acc + t_cruise

        # Generate position samples
        dt = self.timestep
        times = np.arange(0, total_time, dt)
        positions = []
        velocities = []

        for t in times:
            if t <= t_acc:
                # Acceleration phase
                v = max_acc * t
                s = 0.5 * max_acc * t ** 2
            elif t <= t_acc + t_cruise:
                # Cruise phase
                v = peak_vel
                s = (0.5 * max_acc * t_acc ** 2 +
                     peak_vel * (t - t_acc))
            else:
                # Deceleration phase
                t_dec = t - t_acc - t_cruise
                v = peak_vel - max_acc * t_dec
                s = (0.5 * max_acc * t_acc ** 2 +
                     peak_vel * t_cruise +
                     peak_vel * t_dec - 0.5 * max_acc * t_dec ** 2)

            positions.append(min(s, distance))
            velocities.append(max(v, 0))

        return positions, velocities, total_time

    # ----------------------------------------------------------
    # Path execution
    # ----------------------------------------------------------

    def generate_smooth_trajectory(self, waypoints):
        """
        Convert a list of waypoints into a smooth trajectory.
        Each segment uses a trapezoidal velocity profile.
        Returns list of positions to execute.
        """
        if len(waypoints) < 2:
            return waypoints

        full_trajectory = []

        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i])
            end = np.array(waypoints[i + 1])

            segment = self._generate_segment(start, end)
            full_trajectory.extend(segment)

        self.logger.debug(
            f"Trajectory generated: {len(waypoints)} waypoints → "
            f"{len(full_trajectory)} interpolated points"
        )

        return full_trajectory

    def _generate_segment(self, start, end):
        """Generate smooth points between two waypoints"""
        direction = end - start
        distance = np.linalg.norm(direction)

        if distance < 0.001:
            return [end]

        # Normalize direction
        unit = direction / distance

        # Get position profile
        positions, velocities, total_time = self.trapezoidal_profile(distance)

        # Generate 3D points along segment
        segment_points = []
        for s in positions:
            point = start + unit * s
            segment_points.append(point)

        # Always include exact endpoint
        segment_points.append(end)

        return segment_points

    def interpolate_waypoints(self, waypoints, num_points=50):
        """
        Simple linear interpolation between waypoints.
        Faster alternative to full trapezoidal profile.
        """
        if len(waypoints) < 2:
            return waypoints

        interpolated = []

        for i in range(len(waypoints) - 1):
            start = np.array(waypoints[i])
            end = np.array(waypoints[i + 1])

            segment_points = num_points // (len(waypoints) - 1)

            for j in range(segment_points):
                t = j / segment_points
                # Smooth step interpolation (S-curve)
                t_smooth = t * t * (3 - 2 * t)
                point = start + t_smooth * (end - start)
                interpolated.append(point)

        interpolated.append(np.array(waypoints[-1]))
        return interpolated

    # ----------------------------------------------------------
    # Joint space trajectories
    # ----------------------------------------------------------

    def joint_space_trajectory(self, start_angles, end_angles, duration=2.0):
        """
        Plan smooth joint space trajectory.
        Moves all joints simultaneously for natural motion.
        Uses cubic polynomial interpolation.
        """
        start = np.array(start_angles)
        end = np.array(end_angles)

        times = np.arange(0, duration, self.timestep)
        joint_trajectory = []

        for t in times:
            # Normalized time 0 to 1
            s = t / duration
            # Smooth step (cubic)
            s_smooth = s * s * (3 - 2 * s)
            # Interpolated joint angles
            angles = start + s_smooth * (end - start)
            joint_trajectory.append(angles)

        joint_trajectory.append(end)

        self.logger.debug(
            f"Joint trajectory: {len(joint_trajectory)} points, "
            f"{duration:.1f}s duration"
        )

        return joint_trajectory

    # ----------------------------------------------------------
    # Execution helpers
    # ----------------------------------------------------------

    def execute_trajectory(self, robot, trajectory, steps_per_point=5):
        """
        Execute a trajectory on the robot.
        robot: FrankaPandaRobot instance
        trajectory: list of (x,y,z) positions
        """
        import pybullet as p

        self.logger.debug(f"Executing trajectory: {len(trajectory)} points")

        for i, point in enumerate(trajectory):
            # Calculate IK for this point
            joint_angles = p.calculateInverseKinematics(
                robot.robot,
                robot.end_effector,
                point,
                maxNumIterations=50,
                residualThreshold=0.005
            )

            # Apply joint angles
            for j, joint_idx in enumerate(robot.arm_joints):
                p.setJointMotorControl2(
                    robot.robot,
                    joint_idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_angles[j],
                    force=500,
                    maxVelocity=self.max_velocity * 10
                )

            # Step simulation
            for _ in range(steps_per_point):
                p.stepSimulation()
                time.sleep(self.timestep)

        self.logger.debug("Trajectory execution complete")

    # ----------------------------------------------------------
    # Analysis
    # ----------------------------------------------------------

    def estimate_duration(self, waypoints):
        """Estimate how long a trajectory will take in seconds"""
        total_distance = 0
        for i in range(len(waypoints) - 1):
            d = np.linalg.norm(
                np.array(waypoints[i + 1]) - np.array(waypoints[i])
            )
            total_distance += d

        _, _, duration = self.trapezoidal_profile(total_distance)
        return duration

    def get_path_length(self, waypoints):
        """Calculate total path length in metres"""
        total = 0
        for i in range(len(waypoints) - 1):
            d = np.linalg.norm(
                np.array(waypoints[i + 1]) - np.array(waypoints[i])
            )
            total += d
        return total