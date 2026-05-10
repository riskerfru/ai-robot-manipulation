# =============================================================
#   VISUAL SERVOING - Camera guides final grasp
#   Uses PyBullet overhead camera to detect object centre
#   Sends correction commands until gripper is aligned
#   Eliminates IK position error at critical grasp moment
# =============================================================

import numpy as np
import pybullet as p


class VisualServoing:
    def __init__(self, config, logger):
        self.config  = config
        self.logger  = logger

        # Camera parameters
        self.cam_width    = 320
        self.cam_height   = 320
        self.cam_fov      = 60
        self.cam_height_m = 0.5   # Camera mounted 0.5m above end effector

        # Servoing parameters
        self.pixel_threshold  = 10     # pixels — converged when offset below this
        self.max_iterations   = 8      # max correction iterations
        self.max_correction   = 0.008  # max 8mm per step — prevents overshoot

        # Table surface height — objects sit at this z + half-height
        # Updated from 0.0 (floor) to 0.1 (table surface)
        self.surface_z = 0.10

        self.logger.info("Visual Servoing initialized")

    # ----------------------------------------------------------
    # Main servoing method
    # ----------------------------------------------------------

    def servo_to_object(self, robot, object_name, verbose=True):
        """
        Visually servo end effector over target object.
        Assumes robot is already approximately above object.
        Returns final corrected (x, y) position.

        Fast path: if object detected and offset small on first
        frame, skip iterations and return immediately.
        """
        if verbose:
            print(f"\n  [SERVO] Visual servoing to '{object_name}'...")

        converged  = False
        iterations = 0
        total_dx   = 0.0
        total_dy   = 0.0

        # metres per pixel at this camera height and FOV
        metres_per_pixel = (
            2 * self.cam_height_m
            * np.tan(np.radians(self.cam_fov / 2))
            / self.cam_width
        )

        # First frame — check if object is close enough to servo reliably
        # If offset > 25px on first frame, camera angle is bad → use known pos
        ee_pos     = robot.get_end_effector_position()
        frame      = self._capture_overhead(ee_pos)
        obj_pixel  = self._detect_object(frame, object_name)

        if obj_pixel is not None:
            cx = self.cam_width  // 2
            cy = self.cam_height // 2
            init_dx = obj_pixel[0] - cx
            init_dy = obj_pixel[1] - cy
            if abs(init_dx) > 25 or abs(init_dy) > 25:
                if verbose:
                    print(f"  [SERVO] Large initial offset "
                          f"({init_dx:+d}px,{init_dy:+d}px) — "
                          f"using known position (avoids oscillation)")
                final_pos = robot.get_end_effector_position()
                return final_pos[0], final_pos[1]

        while not converged and iterations < self.max_iterations:
            ee_pos = robot.get_end_effector_position()

            # Capture overhead camera frame
            frame = self._capture_overhead(ee_pos)

            # Detect object colour blob
            obj_pixel = self._detect_object(frame, object_name)

            if obj_pixel is None:
                if verbose:
                    print(f"  [SERVO] '{object_name}' not visible — "
                          f"using known position")
                break

            # Pixel offset from image centre
            cx    = self.cam_width  // 2
            cy    = self.cam_height // 2
            dx_px = obj_pixel[0] - cx
            dy_px = obj_pixel[1] - cy

            if verbose:
                print(f"  [SERVO] Iter {iterations+1}: "
                      f"dx={dx_px:+d}px dy={dy_px:+d}px")

            # Check convergence
            if abs(dx_px) <= self.pixel_threshold and \
               abs(dy_px) <= self.pixel_threshold:
                converged = True
                if verbose:
                    print(f"  [SERVO] Converged in {iterations+1} iterations")
                break

            # Convert pixel offset to metres
            # Apply damping factor (0.3) to prevent overshoot
            # Without damping: 90px × 0.0018m = 16cm correction (way too much)
            # With damping:    90px × 0.0018m × 0.3 = 5mm correction (controlled)
            DAMPING = 0.3

            correction_x = float(np.clip(
                dx_px * metres_per_pixel * DAMPING,
                -self.max_correction, self.max_correction
            ))
            correction_y = float(np.clip(
                -dy_px * metres_per_pixel * DAMPING,
                -self.max_correction, self.max_correction
            ))

            new_x = ee_pos[0] + correction_x
            new_y = ee_pos[1] + correction_y

            robot._direct_move(new_x, new_y, ee_pos[2], speed_factor=0.5)

            total_dx   += correction_x
            total_dy   += correction_y
            iterations += 1

            for _ in range(5):   # reduced from 10
                p.stepSimulation()

        final_pos = robot.get_end_effector_position()

        if converged:
            print(f"  [SERVO] Aligned — "
                  f"correction: {total_dx*1000:.1f}mm, {total_dy*1000:.1f}mm")
        elif iterations == 0:
            pass   # not visible on first frame, silent fallback
        else:
            print(f"  [SERVO] Proceeding after {iterations} iterations")

        return final_pos[0], final_pos[1]

    # ----------------------------------------------------------
    # Camera capture — targets table surface not floor
    # ----------------------------------------------------------

    def _capture_overhead(self, ee_pos):
        """
        Capture downward camera view from above end effector.
        Target z = surface_z (table top) not 0 (floor).
        """
        cam_pos = [
            ee_pos[0],
            ee_pos[1],
            ee_pos[2] + self.cam_height_m
        ]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=cam_pos,
            cameraTargetPosition=[ee_pos[0], ee_pos[1], self.surface_z],
            cameraUpVector=[0, 1, 0]
        )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self.cam_fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=2.0
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.cam_height, self.cam_width, 4
        )[:, :, :3]

        return rgb_array

    # ----------------------------------------------------------
    # Object detection by colour
    # ----------------------------------------------------------

    def _detect_object(self, frame, object_name):
        """
        Detect object in frame by colour.
        Returns (pixel_x, pixel_y) of object centre or None.
        """
        colour_ranges = {
            "red":   {"low": [140, 0,   0],   "high": [255, 100, 100]},
            "blue":  {"low": [0,   0,   140],  "high": [100, 100, 255]},
            "green": {"low": [0,   80,  0],    "high": [100, 220, 100]},
        }

        name = object_name.lower()
        if name not in colour_ranges:
            return None

        cr   = colour_ranges[name]
        low  = np.array(cr["low"],  dtype=np.uint8)
        high = np.array(cr["high"], dtype=np.uint8)

        mask = np.all((frame >= low) & (frame <= high), axis=2)

        if not np.any(mask):
            return None

        ys, xs = np.where(mask)
        if len(xs) < 5:   # reduced from 10 — more sensitive
            return None

        return (int(np.mean(xs)), int(np.mean(ys)))

    # ----------------------------------------------------------
    # Verify grasp position visually
    # ----------------------------------------------------------

    def verify_above_object(self, robot, object_name):
        """Quick check — is gripper above the object?"""
        ee_pos     = robot.get_end_effector_position()
        frame      = self._capture_overhead(ee_pos)
        obj_pixel  = self._detect_object(frame, object_name)

        if obj_pixel is None:
            return False, None

        cx   = self.cam_width  // 2
        cy   = self.cam_height // 2
        dx   = obj_pixel[0] - cx
        dy   = obj_pixel[1] - cy
        dist = np.sqrt(dx**2 + dy**2)

        return dist <= self.pixel_threshold * 3, (dx, dy)