# =============================================================
#   CAMERA SYSTEM - Advanced Perception
#   Handles RGB, Depth, and Segmentation cameras
#   Returns accurate positions directly from simulation
# =============================================================

import numpy as np
import time


class CameraSystem:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.width = config["width"]
        self.height = config["height"]
        self.fov = config["fov"]
        self.near = config["near"]
        self.far = config["far"]

        self.cameras = {}

        self.last_rgb = None
        self.last_depth = None
        self.last_seg = None

        self.logger.info("Camera System initialized")

    def setup_overhead_camera(self):
        """Setup overhead camera looking straight down"""
        import pybullet as p

        eye    = self.config["position"]
        target = self.config["target"]

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=eye,
            cameraTargetPosition=target,
            cameraUpVector=[0, 1, 0]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.width / self.height,
            nearVal=self.near,
            farVal=self.far
        )
        self.cameras["overhead"] = {
            "view": view_matrix,
            "projection": projection_matrix,
            "position": eye,
            "target": target
        }
        self.logger.info(f"Overhead camera setup at {eye}")

    def setup_side_camera(self, position=[1.5, 0, 1.0], target=[0, 0, 0.3]):
        """Setup a side view camera"""
        import pybullet as p

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=position,
            cameraTargetPosition=target,
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.width / self.height,
            nearVal=0.1,
            farVal=10.0
        )
        self.cameras["side"] = {
            "view": view_matrix,
            "projection": projection_matrix,
            "position": position,
            "target": target
        }
        self.logger.info(f"Side camera setup at {position}")

    def capture(self, camera_name="overhead"):
        """Capture RGB, Depth, Segmentation from named camera"""
        import pybullet as p

        if camera_name not in self.cameras:
            self.logger.warning(f"Camera '{camera_name}' not found")
            return None, None, None

        cam = self.cameras[camera_name]

        _, _, rgb, depth_raw, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=cam["view"],
            projectionMatrix=cam["projection"],
            renderer=p.ER_TINY_RENDERER
        )

        rgb_array = np.array(rgb, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )[:, :, :3]

        depth_array = np.array(depth_raw).reshape(self.height, self.width)
        seg_array   = np.array(seg).reshape(self.height, self.width)

        self.last_rgb   = rgb_array
        self.last_depth = depth_array
        self.last_seg   = seg_array

        return rgb_array, depth_array, seg_array

    def detect_objects(self, objects_dict, camera_name="overhead"):
        """
        Detect objects in scene.
        Uses segmentation to confirm visibility,
        then returns accurate positions directly from simulation.
        """
        import pybullet as p

        if camera_name not in self.cameras:
            self.logger.warning(f"Camera '{camera_name}' not found")
            return {}

        cam = self.cameras[camera_name]

        _, _, _, _, seg = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=cam["view"],
            projectionMatrix=cam["projection"],
            renderer=p.ER_TINY_RENDERER
        )

        seg_array = np.array(seg).reshape(self.height, self.width)
        detected  = {}

        for name, obj_data in objects_dict.items():
            obj_id     = obj_data["id"]
            obj_pixels = np.where(seg_array == obj_id)

            if len(obj_pixels[0]) > 0:
                # Get accurate real position from simulation
                position, _ = p.getBasePositionAndOrientation(obj_id)
                detected[name] = position
                self.logger.debug(
                    f"Detected {name} at "
                    f"({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
                )

        return detected

    def get_scene_image(self, camera_name="overhead"):
        """Get current RGB image"""
        rgb, _, _ = self.capture(camera_name)
        return rgb

    def save_frame(self, filename, camera_name="overhead"):
        """Save current camera frame to file"""
        try:
            from PIL import Image
            rgb, _, _ = self.capture(camera_name)
            if rgb is not None:
                img = Image.fromarray(rgb)
                img.save(filename)
                self.logger.info(f"Frame saved: {filename}")
        except ImportError:
            self.logger.warning("PIL not installed. Run: pip install Pillow")

    def get_camera_info(self):
        return {
            "resolution": f"{self.width}x{self.height}",
            "fov":        self.fov,
            "cameras":    list(self.cameras.keys()),
            "near":       self.near,
            "far":        self.far
        }