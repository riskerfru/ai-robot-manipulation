# =============================================================
#   PACKING CELL - mixed-case packing with format changeover
#
#   An FMCG end-of-line cell in miniature. Cartons of tea wait on
#   the infeed in two formats (standard and travel) and several
#   flavours. A shelf-ready tray with locating nests sits on the
#   conveyor. An order says which flavours go in the case; the
#   robot seats each carton in a nest by force-guided insertion.
#
#   Everything the robot needs for a format comes from the SKU
#   master below: carton size, where the cartons are, the tray
#   pattern, nest clearance, grasp height. Changing format is a
#   data change, not a programming change.
# =============================================================

import pybullet as p

from cell import layout as L

TRAY_CENTRE   = L.CONVEYOR_SLOTS[2]
BASE_Z        = L.TABLE_Z + 0.016      # top of the conveyor fixture plate
PLATE_T       = 0.004
NEST_DEPTH    = 0.008
CLEARANCE     = 0.002                  # per side
WALL          = 0.010
FINGER_LIFT   = 0.0265                 # gripper point above carton base
RELEASE_GAP   = 0.009                  # finger opening beyond the carton
SEATED_TOL    = 0.0015                 # carton base within this of the nest floor
CARTON_FRICTION = 0.5                  # board on steel; lets the fingers
                                       # square a turned carton as they close

TRAY_COLOUR    = [0.72, 0.58, 0.40, 1.0]     # board
STATION_COLOUR = [0.78, 0.80, 0.83, 1.0]

# ---- scene ------------------------------------------------------
# A scene describes what the line is packing: the products, the
# pack formats (carton size and tray pattern) and the stock on the
# infeed. Everything else is derived from it: infeed positions,
# tray geometry, grasp height, the planner's facts.
#
# The limits below are what this gripper and arm have been tested
# with. A scene outside them is refused; it is not attempted.
SIZE_MM_RANGE = (35, 55)       # carton edge (cartons are cubes)
MAX_ROWS, MAX_COLS = 3, 2      # tray pattern
INFEED_X = (0.33, 0.44)
INFEED_SLOTS = [(x, y) for y in (0.36, 0.25, 0.14, -0.14, -0.25, -0.36)
                for x in INFEED_X]

DEFAULT_SCENE = {
    "name": "Tea cartons",
    "item": "carton",
    "products": {
        "lemon": [0.96, 0.80, 0.16],
        "mint":  [0.24, 0.70, 0.46],
        "berry": [0.62, 0.26, 0.60],
    },
    "formats": {
        "standard": {"size_mm": 50, "rows": 2, "cols": 2,
                     "stock": {"lemon": 2, "mint": 2, "berry": 2}},
        "travel":   {"size_mm": 40, "rows": 3, "cols": 2,
                     "stock": {"lemon": 2, "mint": 2, "berry": 2}},
    },
    # One carton arrives bulged (format, product, which one from 1).
    "damaged": [["standard", "mint", 1]],
}

PALETTE = [[0.96, 0.80, 0.16], [0.24, 0.70, 0.46], [0.62, 0.26, 0.60],
           [0.93, 0.50, 0.12], [0.20, 0.62, 0.78], [0.85, 0.36, 0.52]]

SCENE = {}              # name, item, damaged; filled by load_scene
FLAVOUR_COLOURS = {}    # product -> rgba
FORMATS = {}            # format -> carton, tray, stock positions


class SceneError(Exception):
    """The scene cannot be built in this cell. The message says why."""


def _name_ok(text):
    import re
    return isinstance(text, str) and re.fullmatch(r"[a-z][a-z0-9]*", text)


def load_scene(scene=None):
    """
    Validate a scene (dict, path to a JSON file, or None for the
    default) and make it the current one. Raises SceneError listing
    every problem found.
    """
    import json
    if scene is None:
        scene = DEFAULT_SCENE
    elif isinstance(scene, str):
        try:
            with open(scene) as f:
                scene = json.load(f)
        except (OSError, ValueError) as exc:
            raise SceneError(f"cannot read scene file: {exc}")
    if not isinstance(scene, dict):
        raise SceneError("a scene must be a JSON object")

    problems = []
    products = scene.get("products")
    if isinstance(products, list):
        products = {name: None for name in products}
    if not isinstance(products, dict) or not products:
        problems.append("'products' must list at least one product")
        products = {}
    for name in products:
        if not _name_ok(name):
            problems.append(f"product name '{name}' must be one lowercase "
                            f"word (letters and digits)")

    formats = scene.get("formats")
    if not isinstance(formats, dict) or not formats:
        problems.append("'formats' must define at least one format")
        formats = {}
    built = {}
    slot = 0
    for fmt, f in formats.items():
        if not _name_ok(fmt):
            problems.append(f"format name '{fmt}' must be one lowercase word")
            continue
        if not isinstance(f, dict):
            problems.append(f"format '{fmt}' must be an object")
            continue
        size, rows, cols = f.get("size_mm"), f.get("rows"), f.get("cols")
        if not isinstance(size, (int, float)) or not (
                SIZE_MM_RANGE[0] <= size <= SIZE_MM_RANGE[1]):
            problems.append(f"format '{fmt}': size_mm must be between "
                            f"{SIZE_MM_RANGE[0]} and {SIZE_MM_RANGE[1]} "
                            f"(the gripper's tested range)")
            continue
        if not (isinstance(rows, int) and 1 <= rows <= MAX_ROWS
                and isinstance(cols, int) and 1 <= cols <= MAX_COLS):
            problems.append(f"format '{fmt}': tray pattern must be 1 to "
                            f"{MAX_ROWS} rows by 1 to {MAX_COLS} columns")
            continue
        stock = f.get("stock")
        if not isinstance(stock, dict) or not stock:
            problems.append(f"format '{fmt}': 'stock' must give a count "
                            f"per product")
            continue
        positions = {}
        for product, count in stock.items():
            if product not in products:
                problems.append(f"format '{fmt}': unknown product "
                                f"'{product}'")
                continue
            if (not isinstance(count, int) or isinstance(count, bool)
                    or count < 0):
                problems.append(f"format '{fmt}': stock of {product} must "
                                f"be a whole number")
                continue
            positions[product] = INFEED_SLOTS[slot:slot + count]
            slot += count
        carton = size / 1000.0
        built[fmt] = {
            "description": f"{fmt} {scene.get('item', 'carton')}, "
                           f"{size:g} mm",
            "carton": carton,
            # Rows run along the conveyor, where the fingers open, so
            # the row pitch leaves room for them.
            "tray": {"rows": rows, "cols": cols,
                     "row_pitch": max(0.085, carton + 0.040)},
            "stock": positions,
        }
    if slot > len(INFEED_SLOTS):
        problems.append(f"{slot} cartons in stock, but the infeed holds "
                        f"{len(INFEED_SLOTS)}")

    damaged = []
    for entry in scene.get("damaged", []):
        try:
            fmt, product, which = entry
            count = len(built[fmt]["stock"][product])
            if not (isinstance(which, int) and 1 <= which <= count):
                raise ValueError
            damaged.append((fmt, product, which - 1))
        except (ValueError, KeyError, TypeError):
            problems.append(f"'damaged' entry {entry} does not name a "
                            f"carton in stock (format, product, number)")
    extra = scene.get("damaged_random", 0)
    if not isinstance(extra, int) or extra < 0:
        problems.append("'damaged_random' must be a whole number")
        extra = 0

    if problems:
        raise SceneError("; ".join(problems))

    FORMATS.clear()
    FORMATS.update(built)
    FLAVOUR_COLOURS.clear()
    for i, (name, colour) in enumerate(products.items()):
        rgb = list(colour[:3]) if colour else PALETTE[i % len(PALETTE)]
        FLAVOUR_COLOURS[name] = rgb + [1.0]
    letters = [f[0] for f in built]
    SCENE.clear()
    SCENE.update({
        "name": str(scene.get("name", "Packing scene")),
        "item": str(scene.get("item", "carton")),
        "damaged": damaged,
        "damaged_random": extra,
        "codes": {f: (f[0] if letters.count(f[0]) == 1
                      else chr(ord("a") + i))
                  for i, f in enumerate(built)},
    })
    return SCENE


def describe_scene():
    """A few lines for the operator: what can be ordered."""
    lines = [f"Scene: {SCENE['name']}"]
    for fmt, f in FORMATS.items():
        stock = ", ".join(f"{len(v)} {k}" for k, v in f["stock"].items())
        lines.append(f"  {fmt}: {f['carton'] * 1000:g} mm, tray of "
                     f"{nests_per_tray(fmt)} "
                     f"({f['tray']['rows']} x {f['tray']['cols']}), "
                     f"stock: {stock}")
    return lines


def nests_per_tray(fmt):
    t = FORMATS[fmt]["tray"]
    return t["rows"] * t["cols"]


def unit_name(fmt, flavour, index):
    return f"{flavour}_{SCENE['codes'][fmt]}{index + 1}"


def nest_offsets(fmt):
    """Nest centres relative to the tray centre, numbered from 1."""
    f = FORMATS[fmt]
    t = f["tray"]
    col_pitch = f["carton"] + 2 * CLEARANCE + WALL + 0.002
    out = {}
    n = 1
    for r in range(t["rows"]):
        for c in range(t["cols"]):
            dx = (c - (t["cols"] - 1) / 2) * col_pitch
            dy = ((t["rows"] - 1) / 2 - r) * t["row_pitch"]
            out[n] = (dx, dy)
            n += 1
    return out


def tray_half_extents(fmt):
    f = FORMATS[fmt]
    half = f["carton"] / 2 + CLEARANCE
    offs = nest_offsets(fmt).values()
    hx = max(abs(o[0]) for o in offs) + half + WALL
    hy = max(abs(o[1]) for o in offs) + half + WALL
    return hx, hy


def nest_pocket(fmt, nest, tray_xy=TRAY_CENTRE):
    """Nominal pocket for the insertion routine."""
    f = FORMATS[fmt]
    dx, dy = nest_offsets(fmt)[nest]
    half = f["carton"] / 2
    return {
        "name": f"nest {nest}",
        "x": tray_xy[0] + dx, "y": tray_xy[1] + dy,
        "floor_z": BASE_Z + PLATE_T,
        "depth": NEST_DEPTH,
        "half": half,
        "clearance": CLEARANCE,
        "finger_axis": "y",
        "release_open": half + RELEASE_GAP,
        "seated_tol": SEATED_TOL,
        "tamp": True,
        # The carton's height in the gripper is known to well under a
        # millimetre here, so "in the nest" can be judged tightly. A
        # carton resting on a wall edge is not accepted as down.
        "depth_tol": 0.0012,
    }


def _tray_boxes(fmt):
    """Plate plus the wall pieces that leave the nests open."""
    f = FORMATS[fmt]
    half = f["carton"] / 2 + CLEARANCE
    hx, hy = tray_half_extents(fmt)
    boxes = [((hx, hy, PLATE_T / 2), (0.0, 0.0, PLATE_T / 2))]
    wz, wh = PLATE_T + NEST_DEPTH / 2, NEST_DEPTH / 2
    offs = nest_offsets(fmt)
    xs = sorted({round(o[0], 6) for o in offs.values()})
    ys = sorted({round(o[1], 6) for o in offs.values()})

    # Full-width bars between and outside the rows
    y_edges = [-hy] + [e for y in ys for e in (y - half, y + half)] + [hy]
    for lo, hi in zip(y_edges[0::2], y_edges[1::2]):
        boxes.append(((hx, (hi - lo) / 2, wh), (0.0, (lo + hi) / 2, wz)))
    # Within each row, blocks between and outside the nests
    x_edges = [-hx] + [e for x in xs for e in (x - half, x + half)] + [hx]
    for y in ys:
        for lo, hi in zip(x_edges[0::2], x_edges[1::2]):
            boxes.append((((hi - lo) / 2, half, wh), ((lo + hi) / 2, y, wz)))
    return boxes


def add_tray(fmt, xy=TRAY_CENTRE):
    boxes = _tray_boxes(fmt)
    kw = dict(shapeTypes=[p.GEOM_BOX] * len(boxes),
              halfExtents=[b[0] for b in boxes])
    col = p.createCollisionShapeArray(
        collisionFramePositions=[b[1] for b in boxes], **kw)
    vis = p.createVisualShapeArray(
        visualFramePositions=[b[1] for b in boxes],
        rgbaColors=[TRAY_COLOUR] * len(boxes), **kw)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[xy[0], xy[1], BASE_Z])
    p.changeDynamics(body, -1, lateralFriction=0.4)
    return body


# ---- real-line conditions ----------------------------------------
# What a line does that a clean simulation does not. All of it is
# injected by this file from a seeded random generator; none of it
# is told to the controller.
LINE_NOISE = {
    "scatter_m": 0.005,        # carton off its infeed mark, +/- per axis
    "yaw_deg": 2.0,            # carton turned on the infeed, +/-
    "sensor_sigma_m": 0.0012,  # error of the reported carton position
    "bulge_m": 0.005,          # extra width of a damaged carton
    "tray_offset_m": 0.0025,   # tray off its mark, +/- per axis
}
WIDTH_TOLERANCE = 0.0025       # gripper width check, +/- metres
CAMERA_SIZE_TOLERANCE = 0.0035 # camera size check, +/- metres
REJECT_BIN = [(0.36, 0.0), (0.46, 0.0)]
REJECT_COLOUR = [0.86, 0.42, 0.42, 1.0]


def add_reject_bin():
    for x, y in REJECT_BIN:
        pad = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.045, 0.045, 0.002],
                                  rgbaColor=REJECT_COLOUR)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=pad,
                          basePosition=[x, y, L.TABLE_Z + 0.002])


def add_stock(robot, noise=None, rng=None):
    """Put every carton of every format on its infeed position."""
    import math
    noise = noise or {}
    damaged = set()
    if noise:
        damaged = set(SCENE["damaged"])
        units = [(fmt, product, i) for fmt, f in FORMATS.items()
                 for product, places in f["stock"].items()
                 for i in range(len(places))]
        spare = [u for u in units if u not in damaged]
        for _ in range(min(SCENE["damaged_random"], len(spare))):
            damaged.add(spare.pop(rng.randrange(len(spare))))
    for fmt, f in FORMATS.items():
        for flavour, places in f["stock"].items():
            for i, (x, y) in enumerate(places):
                half = f["carton"] / 2
                if (fmt, flavour, i) in damaged:
                    half += noise.get("bulge_m", 0.0) / 2
                pad = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.045, 0.045, 0.002],
                    rgbaColor=STATION_COLOUR)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=pad,
                                  basePosition=[x, y, L.TABLE_Z + 0.002])
                name = unit_name(fmt, flavour, i)
                yaw = 0.0
                if rng is not None:
                    s = noise.get("scatter_m", 0.0)
                    x += rng.uniform(-s, s)
                    y += rng.uniform(-s, s)
                    a = math.radians(noise.get("yaw_deg", 0.0))
                    yaw = rng.uniform(-a, a)
                robot.add_object(name, x=x, y=y,
                                 color=FLAVOUR_COLOURS[flavour],
                                 z=L.TABLE_Z + half + 0.004,
                                 half_extent=half,
                                 grasp_depth=half - FINGER_LIFT,
                                 vision=False, friction=CARTON_FRICTION)
                if yaw:
                    body = robot.objects[name]["id"]
                    pos, _ = p.getBasePositionAndOrientation(body)
                    p.resetBasePositionAndOrientation(
                        body, pos, p.getQuaternionFromEuler([0, 0, yaw]))
                robot.stations[name] = (x, y)
                # Fingers open along the conveyor everywhere, so a carton
                # is never turned between pick and nest, and only far
                # enough to clear the carton (neighbours are close).
                robot.pick_hints[name] = {"finger_axis": "y",
                                          "open_to": half + 0.015,
                                          "station_open": half + 0.015}


class PackingLine:
    """
    State of the line: which format is set up, which tray is on the
    conveyor, what stock is left, which nests are filled.
    """

    def __init__(self, robot, fmt=None, tray_offset=None,
                 on_step=None, noise=None, seed=0):
        import random
        fmt = fmt or next(iter(FORMATS))
        self.seed = seed
        self.trays_loaded = 0
        self.robot = robot
        self.on_step = on_step or (lambda: None)
        self.noise = noise
        self.rejected = []                  # carton names in the reject bin
        self.on_reject = None               # callback(dict), for the audit
        self.detector = None                # set up on the first scan
        self.detected = []                  # cartons of the current format
        self.by_format = {}                 # format -> cartons seen
        self.hidden = []                    # seen but partly hidden
        self.taken = []                     # detections already assigned
        self.format = None
        self.tray_body = None
        self.tray_offset = (0.0, 0.0)
        self.nests = {}                     # nest -> record (see pack_nest)
        self.available = {
            fmt_name: {flavour: [unit_name(fmt_name, flavour, i)
                                 for i in range(len(places))]
                       for flavour, places in f["stock"].items()}
            for fmt_name, f in FORMATS.items()}
        if noise:
            rng = random.Random(seed)
            robot.noise_rng = random.Random(seed + 1)
            robot.perception_noise = noise.get("sensor_sigma_m", 0.0)
            robot.part_gate = self.width_check
            add_reject_bin()
            add_stock(robot, noise, rng)
        else:
            add_stock(robot)
        self.load_tray(fmt, tray_offset)

    # ---- looking at the cell ---------------------------------------
    SCAN_PARK = (0.10, 0.45, 0.70)      # arm clear of the camera

    def _ensure_detector(self):
        if self.detector is None:
            import config as cfg
            from perception.detector import OverheadDetector
            self.robot.camera.setup_overhead_camera(cfg.CELL_CAMERA, "cell")
            self.detector = OverheadDetector(self.robot.camera, "cell")
        return self.detector

    def scan(self):
        """
        Look at the cell and find the cartons of the current format.
        The arm is parked clear of the camera first, because it hides
        the infeed from where it usually sits. Positions come from
        the image only.
        """
        detector = self._ensure_detector()
        self.robot.move_to_position(*self.SCAN_PARK, use_rrt=False)
        colours = {name: rgba[:3] for name, rgba in FLAVOUR_COLOURS.items()}
        area = ((INFEED_X[0] - 0.06, INFEED_X[1] + 0.06), (-0.45, 0.45))
        image = detector.capture()
        # One image, read once per carton size: the apparent size is
        # what separates the formats, so each pass keeps only the
        # cartons of that size.
        self.by_format = {}
        self.hidden = []
        for fmt, f in FORMATS.items():
            size = f["carton"]
            found, hidden = detector.detect(
                colours, size, expect_size_m=size, area=area, image=image)
            self.by_format[fmt] = found
            self.hidden.extend(hidden)
        self.detected = self.by_format.get(self.format, [])
        self.taken = []
        return self.detected

    def stock_levels(self, fmt=None):
        """How many cartons of each product the camera can see."""
        fmt = fmt or self.format
        counts = {name: 0 for name in FLAVOUR_COLOURS}
        for d in self.by_format.get(fmt, []):
            if d not in self.taken:
                counts[d.product] = counts.get(d.product, 0) + 1
        return counts

    def take(self, product):
        """Reserve the nearest unused carton of a product, or None."""
        for d in self.detected:
            if d.product == product and d not in self.taken:
                self.taken.append(d)
                return d
        return None

    def pick_spec(self, detection):
        """
        What the robot needs to pick at a measured position. The
        carton's size comes from the image too: a carton that is not
        the nominal size has to be gripped at its own height, or the
        fingers meet it too low and push it aside.
        """
        nominal = FORMATS[self.format]["carton"]
        seen = max(detection.width_m, detection.height_m)
        size = min(max(seen, nominal - 0.008), nominal + 0.008)
        return {"x": detection.x, "y": detection.y,
                "centre_z": L.TABLE_Z + size / 2,
                "finger_axis": "y",
                "open_to": size / 2 + 0.015,
                "grasp_depth": size / 2 - FINGER_LIFT,
                "seen_mm": round(seen * 1000, 1)}

    # ---- trays ---------------------------------------------------
    def true_tray_centre(self):
        return (TRAY_CENTRE[0] + self.tray_offset[0],
                TRAY_CENTRE[1] + self.tray_offset[1])

    def next_tray_offset(self):
        """How far the next tray stops from its mark (not told to
        the controller). Seeded, so a run can be repeated."""
        import random
        limit = (self.noise or LINE_NOISE)["tray_offset_m"]
        rng = random.Random(self.seed * 7919 + self.trays_loaded)
        return (rng.uniform(-limit, limit), rng.uniform(-limit, limit))

    def load_tray(self, fmt, tray_offset=None):
        """A fresh tray arrives. It is never exactly on its mark."""
        if tray_offset is None:
            tray_offset = self.next_tray_offset()
        self.trays_loaded += 1
        self.format = fmt
        self.tray_offset = tray_offset
        self.tray_body = add_tray(fmt, self.true_tray_centre())
        self.nests = {}
        self.robot.pockets = {}
        self.robot.pocket_bias = (0.0, 0.0)
        self.robot.pocket_bias_samples = []

    def eject_tray(self, travel=0.62, steps=90):
        """The conveyor indexes the finished tray out of the cell."""
        if self.tray_body is None:
            return
        bodies = [self.tray_body] + [
            self.robot.objects[r["gripped"]]["id"]
            for r in self.nests.values()
            if r.get("gripped") in self.robot.objects]
        for _ in range(steps):
            for body in bodies:
                pos, orn = p.getBasePositionAndOrientation(body)
                p.resetBasePositionAndOrientation(
                    body, [pos[0], pos[1] - travel / steps, pos[2]], orn)
                p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
            self.on_step()
        for record in self.nests.values():
            name = record.get("gripped")
            if name and name in self.robot.objects:
                p.removeBody(self.robot.objects.pop(name)["id"])
                self.robot.stations.pop(name, None)
        p.removeBody(self.tray_body)
        self.tray_body = None
        self.nests = {}
        self.robot._sync_obstacles()

    # ---- orders --------------------------------------------------
    def subtasks_for(self, items):
        """
        Turn validated order lines into pick-and-seat steps, one per
        carton. Each step carries where the camera saw that carton
        and which nest it goes in. No carton is named: the robot is
        told a position, and what it grips is whatever it grips.
        """
        steps = []
        nest = 1
        for product, qty in items:
            for _ in range(qty):
                detection = self.take(product)
                if detection is None:
                    break           # validated against the scan already
                self.nests[nest] = {"nest": nest, "product": product,
                                    "detection": detection,
                                    "gripped": None}
                steps.append({
                    "action": "insert",
                    "product": product,
                    "nest": nest,
                    "pick": self.pick_spec(detection),
                    "pocket": nest_pocket(self.format, nest),
                    "label": f"{product} {SCENE['item']} into nest {nest}",
                })
                nest += 1
        return steps

    def record_gripped(self, nest, name):
        """Note which body ended up in the nest, for the tray check."""
        if nest in self.nests:
            self.nests[nest]["gripped"] = name

    # ---- quality gate at the pick ----------------------------------
    def width_check(self, name, nest=None):
        """
        Called with a carton in the gripper. Compares the width the
        fingers closed to with the SKU master. A carton outside the
        tolerance goes to the reject bin and, if the camera saw
        another of the same product, that one takes its nest.
        Returns None if the carton is good, or a dict describing the
        rejection ("substitute" is a pick spec, or None).
        """
        width = self.robot.last_grip_width
        if width is None or nest is None or nest not in self.nests:
            return None
        nominal = FORMATS[self.format]["carton"]
        record = self.nests[nest]
        detection = record["detection"]
        seen = max(detection.width_m, detection.height_m)
        # Two measurements, taken different ways: the camera's view of
        # the carton on the infeed and the gripper's own opening. A
        # carton is only rejected when both say it is out of size, so
        # one odd reading does not throw away a good carton.
        if (abs(width - nominal) <= WIDTH_TOLERANCE
                or abs(seen - nominal) <= CAMERA_SIZE_TOLERANCE):
            return None

        product = record["product"]
        replacement = self.take(product)       # another from the scan
        if replacement is not None:
            record["detection"] = replacement
            record["gripped"] = None
        else:
            del self.nests[nest]
        bin_xy = REJECT_BIN[len(self.rejected) % len(REJECT_BIN)]
        self.rejected.append(name)
        return {"part": name, "flavour": product, "nest": nest,
                "width_mm": round(width * 1000, 1),
                "seen_mm": round(seen * 1000, 1),
                "nominal_mm": round(nominal * 1000, 1),
                "tolerance_mm": WIDTH_TOLERANCE * 1000,
                "reject_to": bin_xy,
                "substitute": (self.pick_spec(replacement)
                               if replacement is not None else None)}

    def check_tray(self):
        """Ground truth from the simulator: is each carton seated?"""
        import math
        out = []
        for nest, record in sorted(self.nests.items()):
            name = record.get("gripped")
            pocket = nest_pocket(self.format, nest, self.true_tray_centre())
            if not name:
                out.append({"nest": nest, "carton": None,
                            "flavour": record["product"], "seated": False,
                            "gap_mm": 0.0, "side_mm": 0.0,
                            "detail": "nest left empty"})
                continue
            pos = self.robot.get_object_position(name)
            gap = pos[2] - pocket["half"] - pocket["floor_z"]
            side = max(abs(pos[0] - pocket["x"]), abs(pos[1] - pocket["y"]))
            seated = (abs(gap) <= SEATED_TOL
                      and side <= pocket["clearance"] + 0.001)
            out.append({"nest": nest, "carton": name,
                        "flavour": record["product"], "seated": seated,
                        "gap_mm": round(gap * 1000, 1),
                        "side_mm": round(side * 1000, 1)})
        return out


load_scene()      # the default scene, until another is loaded
