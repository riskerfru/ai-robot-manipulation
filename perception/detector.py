# =============================================================
#   OVERHEAD DETECTOR - finds parts in a camera image
#
#   Nothing here reads a position out of the simulator. It takes
#   the overhead camera's RGB image, finds coloured regions in the
#   pixels, and works out where each one is on the table from the
#   camera's own geometry: a pixel defines a ray, and the ray is
#   intersected with the plane the part's top face lies on.
#
#   That is what a calibrated overhead camera does on a real cell,
#   so the position carries real error: pixel size, the colour
#   threshold, the blob centroid and the assumed part height all
#   contribute. The cell is given these positions and nothing else.
# =============================================================

import numpy as np
import pybullet as p


class Detection:
    """One part found in the image."""

    hidden = False

    def __init__(self, product, x, y, pixels, width_m, height_m):
        self.product = product
        self.x = x
        self.y = y
        self.pixels = pixels          # blob size, a confidence measure
        self.width_m = width_m        # apparent size on the table
        self.height_m = height_m

    @property
    def position(self):
        return (self.x, self.y)

    def __repr__(self):
        return (f"<{self.product}{' (hidden)' if self.hidden else ''} at "
                f"({self.x:.3f}, {self.y:.3f}), {self.pixels}px, "
                f"{1000 * self.width_m:.0f}x{1000 * self.height_m:.0f} mm>")


class OverheadDetector:
    """
    Finds parts by colour in the overhead image.

    colours: product name -> RGB triple (0-1), as the scene defines
    them. A part is matched to the nearest product colour, so the
    products in one scene have to be distinguishable by colour.
    """

    MIN_PIXELS = 40            # smaller blobs are noise
    MAX_COLOUR_DIST = 0.22     # how close a pixel must be to a colour

    def __init__(self, camera, camera_name="overhead", min_pixels=None):
        self.camera = camera
        self.camera_name = camera_name
        self.min_pixels = self.MIN_PIXELS if min_pixels is None else min_pixels
        self.last_image = None
        self.last_labels = None

    # ---- camera geometry ------------------------------------------
    def _shape(self):
        cam = self.camera.cameras[self.camera_name]
        return (cam.get("height", self.camera.height),
                cam.get("width", self.camera.width))

    def _matrices(self):
        cam = self.camera.cameras[self.camera_name]
        view = np.array(cam["view"], dtype=np.float64).reshape(4, 4, order="F")
        proj = np.array(cam["projection"],
                        dtype=np.float64).reshape(4, 4, order="F")
        return np.linalg.inv(proj @ view)

    def _ray(self, inv, u, v, width, height):
        """World-space ray through the centre of pixel (u, v)."""
        x = 2.0 * (u + 0.5) / width - 1.0
        y = 1.0 - 2.0 * (v + 0.5) / height
        near = inv @ np.array([x, y, -1.0, 1.0])
        far = inv @ np.array([x, y, 1.0, 1.0])
        near = near[:3] / near[3]
        far = far[:3] / far[3]
        return near, far - near

    def pixel_to_plane(self, u, v, plane_z, inv=None, shape=None):
        """Where pixel (u, v) meets the horizontal plane z = plane_z."""
        inv = self._matrices() if inv is None else inv
        height, width = shape or self._shape()
        origin, direction = self._ray(inv, u, v, width, height)
        if abs(direction[2]) < 1e-9:
            return None
        t = (plane_z - origin[2]) / direction[2]
        if t < 0:
            return None
        point = origin + t * direction
        return float(point[0]), float(point[1])

    # ---- image ----------------------------------------------------
    def capture(self):
        cam = self.camera.cameras[self.camera_name]
        height, width = self._shape()
        _, _, rgb, _, _ = p.getCameraImage(
            width=width, height=height,
            viewMatrix=cam["view"], projectionMatrix=cam["projection"],
            renderer=p.ER_TINY_RENDERER)
        image = (np.reshape(rgb, (height, width, 4))
                 [:, :, :3].astype(np.float64) / 255.0)
        self.last_image = image
        return image

    @staticmethod
    def _label(mask):
        """
        Label connected regions in a boolean mask (4-connected), with
        a two-pass union-find. No SciPy dependency.
        """
        height, width = mask.shape
        labels = np.zeros((height, width), dtype=np.int32)
        parent = [0]

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

        nxt = 1
        for row in range(height):
            for col in range(width):
                if not mask[row, col]:
                    continue
                up = labels[row - 1, col] if row else 0
                left = labels[row, col - 1] if col else 0
                if up and left:
                    labels[row, col] = min(up, left)
                    union(up, left)
                elif up or left:
                    labels[row, col] = up or left
                else:
                    labels[row, col] = nxt
                    parent.append(nxt)
                    nxt += 1
        for row in range(height):
            for col in range(width):
                if labels[row, col]:
                    labels[row, col] = find(labels[row, col])
        return labels

    # ---- detection ------------------------------------------------
    def detect(self, colours, part_height_m, area=None, image=None,
               expect_size_m=None, size_tol_m=0.008, min_fill=0.55,
               min_size_m=0.02, max_size_m=0.09):
        """
        Find parts of the given colours.

        colours: {product: (r, g, b)} in 0-1
        part_height_m: height of the part's top face above the table
                       top (the plane the blob centroid is mapped on)
        expect_size_m: the size the part should appear, when known.
                       Blobs that differ by more than size_tol_m are
                       a different product size and are left out;
                       blobs with fewer pixels than min_fill of the
                       expected area, or clearly not square, are
                       reported as partly hidden instead of measured.
        area: ((x_min, x_max), (y_min, y_max)) to keep, or None
        Returns (parts, hidden): both lists of Detection, largest
        blob first. A hidden part has no reliable position.
        """
        from cell import layout as L
        image = self.capture() if image is None else image
        height, width, _ = image.shape
        inv = self._matrices()
        plane_z = L.TABLE_Z + part_height_m

        names = list(colours)
        table = np.array([colours[n][:3] for n in names], dtype=np.float64)
        flat = image.reshape(-1, 3)
        # Nearest scene colour for every pixel, and how near it is.
        distance = np.linalg.norm(flat[:, None, :] - table[None, :, :], axis=2)
        nearest = np.argmin(distance, axis=1)
        closeness = np.min(distance, axis=1)
        nearest = nearest.reshape(height, width)
        close = (closeness <= self.MAX_COLOUR_DIST).reshape(height, width)

        found, hidden = [], []
        for index, product in enumerate(names):
            mask = close & (nearest == index)
            if not mask.any():
                continue
            labels = self._label(mask)
            self.last_labels = labels
            for label in np.unique(labels):
                if label == 0:
                    continue
                rows, cols = np.where(labels == label)
                if len(rows) < self.min_pixels:
                    continue
                centre = self.pixel_to_plane(cols.mean(), rows.mean(),
                                             plane_z, inv, (height, width))
                if centre is None:
                    continue
                # Apparent size, from the blob's extent on the plane.
                low = self.pixel_to_plane(cols.min(), rows.min(), plane_z,
                                          inv, (height, width))
                high = self.pixel_to_plane(cols.max(), rows.max(), plane_z,
                                           inv, (height, width))
                size_x = abs(high[0] - low[0]) if low and high else 0.0
                size_y = abs(high[1] - low[1]) if low and high else 0.0
                if not (min_size_m <= max(size_x, size_y) <= max_size_m):
                    continue
                if area:
                    (x_min, x_max), (y_min, y_max) = area
                    if not (x_min <= centre[0] <= x_max
                            and y_min <= centre[1] <= y_max):
                        continue
                part = Detection(product, centre[0], centre[1],
                                 int(len(rows)), size_x, size_y)
                if expect_size_m:
                    big = max(size_x, size_y)
                    small = min(size_x, size_y)
                    if abs(big - expect_size_m) > size_tol_m:
                        continue            # another product size
                    # A part seen whole is square and fills its box.
                    # Anything less is behind something.
                    area_px = len(rows)
                    box_px = self._expected_pixels(expect_size_m,
                                                   plane_z, inv,
                                                   (height, width))
                    if (small < expect_size_m - size_tol_m
                            or (box_px and area_px < min_fill * box_px)):
                        part.hidden = True
                        hidden.append(part)
                        continue
                found.append(part)
        found.sort(key=lambda d: -d.pixels)
        hidden.sort(key=lambda d: -d.pixels)
        return found, hidden

    def _expected_pixels(self, size_m, plane_z, inv, shape):
        """How many pixels a part of this size covers at the centre
        of the image, used to tell a whole part from a hidden one."""
        height, width = shape
        centre = self.pixel_to_plane(width / 2, height / 2, plane_z, inv,
                                     shape)
        edge = self.pixel_to_plane(width / 2 + 10, height / 2, plane_z, inv,
                                   shape)
        if not centre or not edge:
            return None
        metres_per_pixel = abs(edge[0] - centre[0]) / 10.0
        if metres_per_pixel <= 0:
            return None
        return (size_m / metres_per_pixel) ** 2
