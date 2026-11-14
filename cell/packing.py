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

TRAY_COLOUR    = [0.72, 0.58, 0.40, 1.0]     # board
STATION_COLOUR = [0.78, 0.80, 0.83, 1.0]

FLAVOUR_COLOURS = {
    "lemon": [0.96, 0.80, 0.16, 1.0],
    "mint":  [0.24, 0.70, 0.46, 1.0],
    "berry": [0.62, 0.26, 0.60, 1.0],
}

# ---- SKU master ------------------------------------------------
# rows run along the conveyor (y), columns across it (x). The row
# pitch is wide because the gripper fingers open along y.
FORMATS = {
    "standard": {
        "description": "standard carton, 50 mm",
        "carton": 0.050,
        "tray": {"rows": 2, "cols": 2, "row_pitch": 0.090},
        "stock": {            # flavour -> infeed positions (x, y)
            "lemon": [(0.33, 0.36), (0.44, 0.36)],
            "mint":  [(0.33, 0.25), (0.44, 0.25)],
            "berry": [(0.33, 0.14), (0.44, 0.14)],
        },
    },
    "travel": {
        "description": "travel carton, 40 mm",
        "carton": 0.040,
        "tray": {"rows": 3, "cols": 2, "row_pitch": 0.085},
        "stock": {
            "lemon": [(0.33, -0.14), (0.44, -0.14)],
            "mint":  [(0.33, -0.25), (0.44, -0.25)],
            "berry": [(0.33, -0.36), (0.44, -0.36)],
        },
    },
}


def nests_per_tray(fmt):
    t = FORMATS[fmt]["tray"]
    return t["rows"] * t["cols"]


def unit_name(fmt, flavour, index):
    return f"{flavour}_{'s' if fmt == 'standard' else 't'}{index + 1}"


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


def add_stock(robot):
    """Put every carton of every format on its infeed position."""
    for fmt, f in FORMATS.items():
        half = f["carton"] / 2
        for flavour, places in f["stock"].items():
            for i, (x, y) in enumerate(places):
                pad = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=[0.045, 0.045, 0.002],
                    rgbaColor=STATION_COLOUR)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=pad,
                                  basePosition=[x, y, L.TABLE_Z + 0.002])
                name = unit_name(fmt, flavour, i)
                robot.add_object(name, x=x, y=y,
                                 color=FLAVOUR_COLOURS[flavour],
                                 z=L.TABLE_Z + half + 0.004,
                                 half_extent=half,
                                 grasp_depth=half - FINGER_LIFT,
                                 vision=False)
                robot.stations[name] = (x, y)


class PackingLine:
    """
    State of the line: which format is set up, which tray is on the
    conveyor, what stock is left, which nests are filled.
    """

    def __init__(self, robot, fmt="standard", tray_offset=(0.0, 0.0),
                 on_step=None):
        self.robot = robot
        self.on_step = on_step or (lambda: None)
        self.format = None
        self.tray_body = None
        self.tray_offset = (0.0, 0.0)
        self.nests = {}                     # nest -> carton name
        self.available = {
            fmt_name: {flavour: [unit_name(fmt_name, flavour, i)
                                 for i in range(len(places))]
                       for flavour, places in f["stock"].items()}
            for fmt_name, f in FORMATS.items()}
        add_stock(robot)
        self.load_tray(fmt, tray_offset)

    # ---- trays ---------------------------------------------------
    def true_tray_centre(self):
        return (TRAY_CENTRE[0] + self.tray_offset[0],
                TRAY_CENTRE[1] + self.tray_offset[1])

    def load_tray(self, fmt, tray_offset=(0.0, 0.0)):
        """A fresh tray arrives. It is never exactly on its mark."""
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
        bodies = [self.tray_body] + [self.robot.objects[n]["id"]
                                     for n in self.nests.values()]
        for _ in range(steps):
            for body in bodies:
                pos, orn = p.getBasePositionAndOrientation(body)
                p.resetBasePositionAndOrientation(
                    body, [pos[0], pos[1] - travel / steps, pos[2]], orn)
                p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
            self.on_step()
        for name in self.nests.values():
            p.removeBody(self.robot.objects.pop(name)["id"])
            self.robot.stations.pop(name, None)
        p.removeBody(self.tray_body)
        self.tray_body = None
        self.nests = {}
        self.robot._sync_obstacles()

    # ---- orders --------------------------------------------------
    def stock_levels(self, fmt=None):
        fmt = fmt or self.format
        return {fl: len(units) for fl, units in self.available[fmt].items()}

    def subtasks_for(self, items):
        """
        Turn validated order lines into pick-and-seat steps. Cartons
        of one flavour go side by side; nests fill in number order.
        """
        fmt = self.format
        steps = []
        nest = 1
        for flavour, qty in items:
            for _ in range(qty):
                name = self.available[fmt][flavour].pop(0)
                self.nests[nest] = name
                self.robot.pockets[name] = nest_pocket(fmt, nest)
                steps.append({"action": "insert", "object": name,
                              "label": f"{flavour} carton into nest {nest}"})
                nest += 1
        return steps

    def check_tray(self):
        """Ground truth from the simulator: is each carton seated?"""
        import math
        out = []
        for nest, name in sorted(self.nests.items()):
            pocket = nest_pocket(self.format, nest, self.true_tray_centre())
            pos = self.robot.get_object_position(name)
            gap = pos[2] - pocket["half"] - pocket["floor_z"]
            side = max(abs(pos[0] - pocket["x"]), abs(pos[1] - pocket["y"]))
            seated = abs(gap) <= 0.002 and side <= pocket["clearance"] + 0.001
            out.append({"nest": nest, "carton": name,
                        "flavour": name.split("_")[0], "seated": seated,
                        "gap_mm": round(gap * 1000, 1),
                        "side_mm": round(side * 1000, 1)})
        return out
