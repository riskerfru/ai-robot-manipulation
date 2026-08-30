# =============================================================
#   CLIP DETECTOR - Real AI Vision
#   Uses OpenAI CLIP model to understand scene from images
#   No hardcoded colors - genuinely understands what it sees
#   This is how modern robot perception actually works
# =============================================================

import numpy as np
import time


class CLIPDetector:
    def __init__(self, logger):
        self.logger = logger
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        self.loaded = False

        # Cache for performance
        self.text_feature_cache = {}

    # ----------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------

    def load(self):
        """
        Load CLIP model.
        First run will download ~350MB model automatically.
        """
        try:
            import torch
            import clip

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(
                f"Loading CLIP model on {self.device}..."
            )

            self.model, self.preprocess = clip.load(
                "ViT-B/32",
                device=self.device
            )
            self.loaded = True
            self.logger.success(
                f"CLIP loaded successfully on {self.device}"
            )
            return True

        except ImportError:
            self.logger.warning(
                "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
            )
            return False
        except Exception as e:
            self.logger.error(f"CLIP load failed: {e}")
            return False

    def is_loaded(self):
        return self.loaded

    # ----------------------------------------------------------
    # Core CLIP detection
    # ----------------------------------------------------------

    def classify_image(self, image_array, candidate_labels):
        """
        Use CLIP to classify what is in an image.
        image_array: numpy RGB array (H x W x 3)
        candidate_labels: list of text descriptions to match against
        Returns: dict of label -> confidence score
        """
        if not self.loaded:
            self.logger.warning("CLIP not loaded")
            return {}

        try:
            import torch
            import clip
            from PIL import Image

            # Convert numpy array to PIL image
            pil_image = Image.fromarray(image_array.astype(np.uint8))

            # Preprocess image
            image_tensor = self.preprocess(pil_image).unsqueeze(0).to(
                self.device
            )

            # Encode text labels
            text_inputs = clip.tokenize(candidate_labels).to(self.device)

            with torch.no_grad():
                # Get image and text features
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_inputs)

                # Normalise features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Compute similarity scores
                similarity = (100.0 * image_features @ text_features.T)
                scores = similarity.softmax(dim=-1).cpu().numpy()[0]

            return {
                label: float(score)
                for label, score in zip(candidate_labels, scores)
            }

        except Exception as e:
            self.logger.error(f"CLIP classification error: {e}")
            return {}

    def detect_objects_in_scene(self, image_array, object_names):
        """
        Detect which objects are present in the scene.
        Returns: list of detected object names with confidence
        """
        if not self.loaded:
            return []

        # Build descriptive labels for each object
        labels = [f"a {name} colored block" for name in object_names]
        labels.append("empty workspace with no blocks")

        scores = self.classify_image(image_array, labels)

        detected = []
        for name, label in zip(object_names, labels):
            confidence = scores.get(label, 0)
            if confidence > 0.1:  # Confidence threshold
                detected.append({
                    "name": name,
                    "confidence": confidence
                })
                self.logger.debug(
                    f"CLIP detected {name} with {confidence:.2%} confidence"
                )

        # Sort by confidence
        detected.sort(key=lambda x: x["confidence"], reverse=True)
        return detected

    def find_object_region(self, image_array, object_description, grid_size=4):
        """
        Find which region of the image contains a described object.
        Divides image into grid and scores each region.
        Returns: (row, col) of best matching region
        """
        if not self.loaded:
            return None

        h, w = image_array.shape[:2]
        cell_h = h // grid_size
        cell_w = w // grid_size

        best_score = 0
        best_region = (0, 0)

        for row in range(grid_size):
            for col in range(grid_size):
                # Extract region
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                region = image_array[y1:y2, x1:x2]

                # Score with CLIP
                scores = self.classify_image(
                    region,
                    [object_description, "empty floor"]
                )
                score = scores.get(object_description, 0)

                if score > best_score:
                    best_score = score
                    best_region = (row, col)

        return best_region, best_score

    # ----------------------------------------------------------
    # Scene understanding
    # ----------------------------------------------------------

    def describe_scene(self, image_array):
        """
        Generate a natural language description of the scene.
        Used for logging and debugging.
        """
        if not self.loaded:
            return "CLIP not loaded"

        scene_descriptions = [
            "a robot workspace with colored blocks",
            "blocks arranged in a line",
            "blocks stacked on top of each other",
            "blocks scattered randomly",
            "an empty robot workspace",
            "blocks sorted by color",
            "a single block in the workspace"
        ]

        scores = self.classify_image(image_array, scene_descriptions)
        best_description = max(scores, key=scores.get)
        confidence = scores[best_description]

        self.logger.info(
            f"Scene: '{best_description}' ({confidence:.2%})"
        )
        return best_description

    def answer_question(self, image_array, question):
        """
        Answer a visual question about the scene using CLIP.
        Example: "Is the red block to the left of the blue block?"
        Returns: most likely answer with confidence
        """
        if not self.loaded:
            return "CLIP not loaded"

        # Frame question as competing answers
        yes_label = f"yes, {question}"
        no_label = f"no, {question}"

        scores = self.classify_image(image_array, [yes_label, no_label])

        yes_score = scores.get(yes_label, 0)
        no_score = scores.get(no_label, 0)

        if yes_score > no_score:
            return f"Yes ({yes_score:.2%} confident)"
        else:
            return f"No ({no_score:.2%} confident)"

    # ----------------------------------------------------------
    # Integration with camera system
    # ----------------------------------------------------------

    def detect_from_camera(self, camera_system, objects_dict,
                            camera_name="overhead"):
        """
        Full pipeline: capture image → CLIP detection → return positions.
        Combines CLIP understanding with depth-based localisation.
        """
        if not self.loaded:
            self.logger.warning(
                "CLIP not loaded, falling back to segmentation detection"
            )
            return camera_system.detect_objects(
                objects_dict, camera_name
            )

        # Capture scene
        rgb, depth, seg = camera_system.capture(camera_name)
        if rgb is None:
            return {}

        # Use CLIP to understand scene
        object_names = list(objects_dict.keys())
        detected_by_clip = self.detect_objects_in_scene(rgb, object_names)

        self.logger.info(
            f"CLIP detected {len(detected_by_clip)} objects in scene"
        )

        # For each CLIP-detected object, get accurate position
        # using segmentation mask + depth
        positions = {}
        detected_names = {d["name"] for d in detected_by_clip}

        for name, obj_data in objects_dict.items():
            if name not in detected_names:
                continue

            obj_id = obj_data["id"]
            seg_array = np.array(seg)
            obj_pixels = np.where(seg_array == obj_id)

            if len(obj_pixels[0]) > 0:
                cy = int(np.mean(obj_pixels[0]))
                cx = int(np.mean(obj_pixels[1]))
                d = depth[cy, cx] if depth is not None else 1.5

                world_pos = camera_system.pixel_to_world_3d(
                    cx, cy, d, camera_name
                )
                positions[name] = world_pos

        # Describe the scene
        self.describe_scene(rgb)

        return positions

    # ----------------------------------------------------------
    # Installation helper
    # ----------------------------------------------------------

    @staticmethod
    def install_instructions():
        return """
To install CLIP:
1. pip install torch torchvision
2. pip install git+https://github.com/openai/CLIP.git

First run will download ~350MB model automatically.
GPU (CUDA) recommended but CPU works fine for this project.
        """