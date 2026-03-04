# =============================================================
#   CLAUDE VISION - Send PyBullet frames to Claude
#   Claude sees the scene directly and reasons visually
#   No object detection needed - Claude identifies everything
# =============================================================

import numpy as np
import base64
import anthropic
import pybullet as p


class ClaudeVision:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.client = anthropic.Anthropic(api_key=config["api_key"])
        self.model  = config["model"]
        self.logger.info("Claude Vision initialized")

    # ----------------------------------------------------------
    # Capture frame from PyBullet
    # ----------------------------------------------------------

    def capture_frame(self, robot_id, width=640, height=480,
                       camera="overhead"):
        """Capture RGB frame from PyBullet simulation"""

        if camera == "overhead":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, 0, 1.8],
                cameraTargetPosition=[0.3, 0, 0],
                cameraUpVector=[0, 1, 0]
            )
        elif camera == "front":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[1.2, 0, 0.8],
                cameraTargetPosition=[0.3, 0, 0.2],
                cameraUpVector=[0, 0, 1]
            )
        elif camera == "side":
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0, 1.2, 0.8],
                cameraTargetPosition=[0.3, 0, 0.2],
                cameraUpVector=[0, 0, 1]
            )
        else:
            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[0.8, 0.8, 1.0],
                cameraTargetPosition=[0.3, 0, 0.2],
                cameraUpVector=[0, 0, 1]
            )

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=width/height,
            nearVal=0.1, farVal=10.0
        )

        _, _, rgb, _, _ = p.getCameraImage(
            width=width, height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_TINY_RENDERER
        )

        # Convert to PNG bytes
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha

        return self._array_to_png_base64(rgb_array)

    def _array_to_png_base64(self, rgb_array):
        """Convert numpy RGB array to base64 PNG"""
        import io
        try:
            from PIL import Image
            img    = Image.fromarray(rgb_array.astype(np.uint8))
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except ImportError:
            # Fallback - simple PPM format encoded as base64
            h, w   = rgb_array.shape[:2]
            header = f"P6\n{w} {h}\n255\n".encode()
            data   = rgb_array.astype(np.uint8).tobytes()
            return base64.b64encode(header + data).decode('utf-8')

    # ----------------------------------------------------------
    # Ask Claude about the scene
    # ----------------------------------------------------------

    def analyse_scene(self, robot_id, question=None):
        """
        Capture scene and ask Claude to analyse it.
        Returns Claude's visual understanding of the scene.
        """
        if question is None:
            question = (
                "Describe what you see in this robot simulation. "
                "Identify: all coloured objects and their approximate positions, "
                "any obstacles or walls, the robot arm position, "
                "and what task could be performed."
            )

        print(f"\n  [VISION] Capturing scene...")

        # Capture from multiple angles
        overhead = self.capture_frame(robot_id, camera="overhead")
        front    = self.capture_frame(robot_id, camera="front")

        print(f"  [VISION] Sending frames to Claude...")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "These are two views of a robot simulation (overhead and front view):"
                    },
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       overhead
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type":       "base64",
                            "media_type": "image/png",
                            "data":       front
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }]
        )

        result = response.content[0].text
        print(f"  [VISION] Claude sees: {result[:200]}...")
        self.logger.info(f"Vision analysis complete")
        return result

    def identify_objects(self, robot_id):
        """Ask Claude to identify all objects in scene"""
        return self.analyse_scene(
            robot_id,
            question=(
                "Look at this robot workspace. "
                "List every coloured object you can see with: "
                "1. Colour "
                "2. Approximate position (left/right/centre, near/far) "
                "3. Any obstacles blocking access to it. "
                "Be concise - one line per object."
            )
        )

    def suggest_next_action(self, robot_id, current_task=None):
        """Ask Claude to suggest what to do next based on visual"""
        task_context = f"Current task: {current_task}" if current_task \
            else "No specific task assigned."

        return self.analyse_scene(
            robot_id,
            question=(
                f"{task_context} "
                "Based on what you see, what should the robot do next? "
                "Consider obstacle positions and object locations. "
                "Give a specific, actionable suggestion."
            )
        )

    def check_grasp_success(self, robot_id, object_name):
        """Visually verify if object was successfully grasped"""
        return self.analyse_scene(
            robot_id,
            question=(
                f"Is the robot gripper currently holding the {object_name} object? "
                "Look carefully at the gripper area. "
                "Answer: YES or NO, then briefly explain what you see."
            )
        )