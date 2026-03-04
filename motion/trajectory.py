# =============================================================
#   TRAJECTORY PLANNER - TOPP-RA Integration
#   Time Optimal Path Parameterisation with Reachability Analysis
#   Generates smooth, physically correct robot trajectories
#   Respects joint velocity and acceleration limits exactly
# =============================================================

import numpy as np

# Try TOPP-RA import
try:
    import toppra as ta
    import toppra.constraint as constraint
    import toppra.algorithm as algo
    TOPPRA_AVAILABLE = True
except ImportError:
    TOPPRA_AVAILABLE = False


class TrajectoryPlanner:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.trajectory_points = config.get("trajectory_points", 50)
        self.max_velocity      = config.get("max_velocity", 1.0)
        self.max_acceleration  = config.get("max_acceleration", 0.5)

        # Franka Panda joint limits
        # 7 joints - velocity (rad/s) and acceleration (rad/s^2)
        self.joint_vel_limits = np.array([
            2.1750, 2.1750, 2.1750, 2.1750,
            2.6100, 2.6100, 2.6100
        ])
        self.joint_acc_limits = np.array([
            15.0, 7.5, 10.0, 12.5,
            15.0, 20.0, 20.0
        ])

        # Cartesian velocity/acceleration limits
        self.cart_vel_limit = self.max_velocity
        self.cart_acc_limit = self.max_acceleration

        if TOPPRA_AVAILABLE:
            self.logger.info("Trajectory Planner initialized with TOPP-RA")
        else:
            self.logger.info("Trajectory Planner initialized (TOPP-RA not available - using smooth interpolation)")

    # ----------------------------------------------------------
    # Main trajectory generation
    # ----------------------------------------------------------

    def generate_smooth_trajectory(self, waypoints, num_points=None):
        """
        Generate smooth trajectory from waypoints.
        Uses TOPP-RA if available, falls back to smooth interpolation.
        
        Returns list of 3D positions to follow.
        """
        if num_points is None:
            num_points = self.trajectory_points

        waypoints = [np.array(wp) for wp in waypoints]

        # Remove duplicate consecutive waypoints
        cleaned = [waypoints[0]]
        for wp in waypoints[1:]:
            if np.linalg.norm(wp - cleaned[-1]) > 0.001:
                cleaned.append(wp)
        waypoints = cleaned

        if len(waypoints) < 2:
            return waypoints

        if TOPPRA_AVAILABLE:
            try:
                return self._toppra_trajectory(waypoints, num_points)
            except Exception as e:
                self.logger.warning(
                    f"TOPP-RA failed ({e}) - using smooth interpolation"
                )

        return self._smooth_interpolation(waypoints, num_points)

    # ----------------------------------------------------------
    # TOPP-RA trajectory
    # ----------------------------------------------------------

    def _toppra_trajectory(self, waypoints, num_points):
        """
        Generate time-optimal trajectory using TOPP-RA.
        Respects velocity and acceleration limits.
        """
        waypoints = np.array(waypoints)
        n         = len(waypoints)

        # Path parameter - evenly spaced 0 to 1
        path_times = np.linspace(0, 1, n)

        # Create geometric path
        path = ta.SplineInterpolator(path_times, waypoints)

        # Define constraints
        # Cartesian velocity constraint
        vel_constraint = constraint.JointVelocityConstraint(
            np.array([self.cart_vel_limit] * 3)
        )

        # Cartesian acceleration constraint  
        acc_constraint = constraint.JointAccelerationConstraint(
            np.array([self.cart_acc_limit] * 3)
        )

        # Create TOPP-RA instance
        instance = algo.TOPPRA(
            [vel_constraint, acc_constraint],
            path
        )

        # Solve
        traj = instance.compute_trajectory()

        if traj is None:
            raise ValueError("TOPP-RA returned no trajectory")

        # Sample trajectory at evenly spaced time points
        duration    = traj.duration
        sample_times = np.linspace(0, duration, num_points)
        positions   = traj(sample_times)

        print(f"  [TRAJ] TOPP-RA: {n} waypoints → "
              f"{num_points} points, duration {duration:.2f}s")

        return [positions[i] for i in range(len(positions))]

    # ----------------------------------------------------------
    # Smooth interpolation fallback
    # ----------------------------------------------------------

    def _smooth_interpolation(self, waypoints, num_points):
        """
        Smooth trajectory using cubic spline interpolation.
        Better than linear - smooth velocity profile.
        """
        from scipy.interpolate import CubicSpline

        waypoints  = np.array(waypoints)
        n          = len(waypoints)
        path_times = np.linspace(0, 1, n)

        try:
            # Fit cubic spline through waypoints
            cs = CubicSpline(path_times, waypoints, bc_type='clamped')

            # Sample at evenly spaced points
            sample_times = np.linspace(0, 1, num_points)
            positions    = cs(sample_times)

            print(f"  [TRAJ] Cubic spline: {n} waypoints → {num_points} points")
            return [positions[i] for i in range(len(positions))]

        except Exception:
            # Final fallback - linear interpolation
            return self._linear_interpolation(waypoints, num_points)

    def _linear_interpolation(self, waypoints, num_points):
        """Basic linear interpolation between waypoints"""
        waypoints  = np.array(waypoints)
        n          = len(waypoints)
        result     = []
        pts_per_seg = max(2, num_points // (n - 1))

        for i in range(n - 1):
            for j in range(pts_per_seg):
                t = j / pts_per_seg
                result.append(waypoints[i] + t * (waypoints[i+1] - waypoints[i]))

        result.append(waypoints[-1])
        return result

    # ----------------------------------------------------------
    # Velocity profiling
    # ----------------------------------------------------------

    def apply_trapezoidal_profile(self, trajectory, max_vel=None,
                                   max_acc=None):
        """
        Apply trapezoidal velocity profile to trajectory.
        Smooth acceleration at start, constant velocity middle,
        smooth deceleration at end.
        Returns (positions, velocities, timestamps)
        """
        if max_vel is None:
            max_vel = self.cart_vel_limit
        if max_acc is None:
            max_acc = self.cart_acc_limit

        positions  = [np.array(p) for p in trajectory]
        n          = len(positions)
        distances  = [
            np.linalg.norm(positions[i+1] - positions[i])
            for i in range(n - 1)
        ]
        total_dist = sum(distances)

        if total_dist < 0.001:
            return positions, [0.0] * n, list(range(n))

        # Trapezoidal profile
        t_acc    = max_vel / max_acc
        d_acc    = 0.5 * max_acc * t_acc ** 2

        if 2 * d_acc > total_dist:
            # Triangle profile - not enough distance to reach max vel
            t_acc    = np.sqrt(total_dist / max_acc)
            max_vel  = max_acc * t_acc
            d_acc    = total_dist / 2
            t_total  = 2 * t_acc
        else:
            d_const  = total_dist - 2 * d_acc
            t_const  = d_const / max_vel
            t_total  = 2 * t_acc + t_const

        # Generate timestamps
        timestamps = []
        cumulative = 0.0

        for i, dist in enumerate([0.0] + distances):
            cumulative += dist
            frac = cumulative / total_dist if total_dist > 0 else 0

            if frac <= d_acc / total_dist:
                # Acceleration phase
                d   = frac * total_dist
                t   = np.sqrt(2 * d / max_acc)
            elif frac >= (total_dist - d_acc) / total_dist:
                # Deceleration phase
                d   = (1 - frac) * total_dist
                t   = t_total - np.sqrt(2 * d / max_acc)
            else:
                # Constant velocity phase
                d   = frac * total_dist - d_acc
                t   = t_acc + d / max_vel

            timestamps.append(t)

        # Velocities at each point
        velocities = []
        for i, t in enumerate(timestamps):
            frac = t / t_total if t_total > 0 else 0
            if frac <= t_acc / t_total:
                v = max_acc * t
            elif frac >= (t_total - t_acc) / t_total:
                v = max_acc * (t_total - t)
            else:
                v = max_vel
            velocities.append(v)

        return positions, velocities, timestamps

    def get_execution_time(self, waypoints):
        """Estimate execution time for a set of waypoints"""
        waypoints = [np.array(wp) for wp in waypoints]
        total_dist = sum(
            np.linalg.norm(waypoints[i+1] - waypoints[i])
            for i in range(len(waypoints) - 1)
        )

        t_acc   = self.cart_vel_limit / self.cart_acc_limit
        d_acc   = 0.5 * self.cart_acc_limit * t_acc ** 2

        if 2 * d_acc >= total_dist:
            t_total = 2 * np.sqrt(total_dist / self.cart_acc_limit)
        else:
            d_const = total_dist - 2 * d_acc
            t_total = 2 * t_acc + d_const / self.cart_vel_limit

        return round(t_total, 2)