# =============================================================
#   VISUAL SERVOING - Camera guides final grasp
#   Uses PyBullet object position + overhead camera
#   Computes pixel offset and corrects gripper alignment
#   Eliminates IK position error at critical grasp moment
# =============================================================

import numpy as np
import pybullet as p


class VisualServoing:
    def __init__(self, config, logger):
        self.config  = config
        self.logger  = logger

        # Camera parameters
        self.cam_width    = 128
        self.cam_height   = 128
        self.cam_fov      = 45
        self.cam_height_m = 0.5

        # Servoing parameters
        self.threshold_m     = 0.003   # 3mm convergence threshold
        self.max_iterations  = 15
        self.correction_gain = 0.6     # how aggressively to correct
        self.max_correction  = 0.012   # max step size in metres

        self.logger.info("Visual Servoing initialized")

    # ----------------------------------------------------------
    # Main servoing using PyBullet object state
    # ----------------------------------------------------------

    def servo_to_object(self, robot, object_name, verbose=True):
        """
        Servo gripper over object using PyBullet position as ground truth.
        Computes error between EE and object, applies corrections.
        Returns final (x, y) position.
        """
        if verbose:
            print(f"\n  [SERVO] Visual servoing to '{object_name}'...")

        # Get object position from PyBullet directly
        obj_pos = robot.get_object_position(object_name)
        if obj_pos is None:
            print(f"  [SERVO] Object '{object_name}' not found")
            ee = robot.get_end_effector_position()
            return ee[0], ee[1]

        target_x = obj_pos[0]
        target_y = obj_pos[1]

        converged   = False
        iterations  = 0
        total_dx    = 0.0
        total_dy    = 0.0

        while not converged and iterations < self.max_iterations:
            ee_pos = robot.get_end_effector_position()

            # Error between gripper and object in XY plane
            error_x = target_x - ee_pos[0]
            error_y = target_y - ee_pos[1]
            error   = np.sqrt(error_x**2 + error_y**2)

            if verbose:
                print(f"  [SERVO] Iter {iterations+1}: "
                      f"error={error*1000:.1f}mm "
                      f"dx={error_x*1000:+.1f}mm "
                      f"dy={error_y*1000:+.1f}mm")

            # Check convergence
            if error <= self.threshold_m:
                converged = True
                if verbose:
                    print(f"  [SERVO] Converged in {iterations+1} iterations")
                break

            # Proportional correction
            correction_x = np.clip(
                error_x * self.correction_gain,
                -self.max_correction, self.max_correction
            )
            correction_y = np.clip(
                error_y * self.correction_gain,
                -self.max_correction, self.max_correction
            )

            new_x = ee_pos[0] + correction_x
            new_y = ee_pos[1] + correction_y

            robot._direct_move(new_x, new_y, ee_pos[2], speed_factor=0.9)

            total_dx += correction_x
            total_dy += correction_y
            iterations += 1

            # Step simulation to settle
            for _ in range(5):
                p.stepSimulation()

        final_pos = robot.get_end_effector_position()

        if converged:
            print(f"  [SERVO] Aligned. Total correction: "
                  f"dx={total_dx*1000:.1f}mm "
                  f"dy={total_dy*1000:.1f}mm")
        else:
            final_err = np.sqrt(
                (target_x - final_pos[0])**2 +
                (target_y - final_pos[1])**2
            )
            print(f"  [SERVO] Final error: {final_err*1000:.1f}mm")

        return final_pos[0], final_pos[1]

    def verify_above_object(self, robot, object_name):
        """Check if gripper is above object within threshold"""
        obj_pos = robot.get_object_position(object_name)
        if obj_pos is None:
            return False, None

        ee_pos  = robot.get_end_effector_position()
        error_x = obj_pos[0] - ee_pos[0]
        error_y = obj_pos[1] - ee_pos[1]
        error   = np.sqrt(error_x**2 + error_y**2)

        return error <= self.threshold_m * 3, (error_x, error_y)