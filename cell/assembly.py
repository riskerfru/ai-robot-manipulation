# =============================================================
#   ASSEMBLY CELL - a housing with pockets and three supply stations
#
#   The housing arrives on the conveyor and is fixed. Each
#   component is picked from its own station and seated in its
#   pocket. The pockets leave about a millimetre per side, so a
#   component that is not lined up lands on the rim: seating it
#   takes the force-guided insertion in motion/insertion.py.
# =============================================================

import pybullet as p

from cell import layout as L

HOUSING_CENTRE = L.CONVEYOR_SLOTS[2]
HOUSING_HALF   = (0.055, 0.150)          # x, y
BASE_Z         = L.TABLE_Z + 0.016       # top of the conveyor fixture plate
PLATE_T        = 0.004                   # housing floor thickness
POCKET_DEPTH   = 0.008
CLEARANCE      = 0.0015                  # per side
FINGER_LIFT    = 0.0265                  # gripper point above the part's base

HOUSING_COLOUR = [0.30, 0.33, 0.38, 1.0]
STATION_COLOUR = [0.78, 0.80, 0.83, 1.0]

# name -> size (cube edge), colour, station centre, pocket offset along y
COMPONENTS = {
    "motor":   {"size": 0.050, "colour": [0.93, 0.50, 0.12, 1.0],
                "station": (0.40,  0.30), "pocket_dy":  0.100},
    "bearing": {"size": 0.040, "colour": [0.20, 0.62, 0.78, 1.0],
                "station": (0.40,  0.00), "pocket_dy":  0.000},
    "sensor":  {"size": 0.035, "colour": [0.55, 0.36, 0.78, 1.0],
                "station": (0.40, -0.30), "pocket_dy": -0.100},
}


def pocket_floor_z():
    return BASE_Z + PLATE_T


def pocket_for(name, housing_xy=HOUSING_CENTRE):
    """Nominal pocket of a component: centre, floor height, clearance."""
    c = COMPONENTS[name]
    return {
        "name": name,
        "x": housing_xy[0],
        "y": housing_xy[1] + c["pocket_dy"],
        "floor_z": pocket_floor_z(),
        "depth": POCKET_DEPTH,
        "half": c["size"] / 2,
        "clearance": CLEARANCE,
        # Same finger direction as at the supply stations, so the part
        # is not turned on the way (a square part turned by two degrees
        # no longer fits a 1.5 mm clearance). The fingers then open
        # towards the neighbouring pockets, so only a little.
        "finger_axis": "y",
        "release_open": c["size"] / 2 + 0.009,
    }


def _housing_boxes():
    """Box list (half extents, centre) for the plate and pocket walls."""
    hx, hy = HOUSING_HALF
    boxes = [((hx, hy, PLATE_T / 2), (0.0, 0.0, PLATE_T / 2))]
    wz = PLATE_T + POCKET_DEPTH / 2
    wh = POCKET_DEPTH / 2
    edges = [-hy]
    for c in sorted(COMPONENTS.values(), key=lambda v: v["pocket_dy"]):
        half = c["size"] / 2 + CLEARANCE
        y0, y1 = c["pocket_dy"] - half, c["pocket_dy"] + half
        # cross bar from the previous edge up to this pocket
        boxes.append(((hx, (y0 - edges[-1]) / 2, wh),
                      (0.0, (y0 + edges[-1]) / 2, wz)))
        # the two blocks either side of the pocket in x
        side = (hx - half) / 2
        for sign in (-1, 1):
            boxes.append(((side, half, wh),
                          (sign * (half + side), c["pocket_dy"], wz)))
        edges.append(y1)
    boxes.append(((hx, (hy - edges[-1]) / 2, wh),
                  (0.0, (hy + edges[-1]) / 2, wz)))
    return boxes


def add_housing(xy=HOUSING_CENTRE):
    boxes = _housing_boxes()
    kw = dict(shapeTypes=[p.GEOM_BOX] * len(boxes),
              halfExtents=[b[0] for b in boxes])
    col = p.createCollisionShapeArray(
        collisionFramePositions=[b[1] for b in boxes], **kw)
    vis = p.createVisualShapeArray(
        visualFramePositions=[b[1] for b in boxes],
        rgbaColors=[HOUSING_COLOUR] * len(boxes), **kw)
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[xy[0], xy[1], BASE_Z])
    p.changeDynamics(body, -1, lateralFriction=0.4)
    return body


def add_components(robot, positions=None):
    """Put one of each component on its station."""
    for name, c in COMPONENTS.items():
        x, y = (positions or {}).get(name, c["station"])
        half = c["size"] / 2
        pad = p.createVisualShape(p.GEOM_BOX,
                                  halfExtents=[0.06, 0.06, 0.002],
                                  rgbaColor=STATION_COLOUR)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=pad,
                          basePosition=[c["station"][0], c["station"][1],
                                        L.TABLE_Z + 0.002])
        robot.add_object(name, x=x, y=y, color=c["colour"],
                         z=L.TABLE_Z + half + 0.004, half_extent=half,
                         grasp_depth=half - FINGER_LIFT,
                         vision=False)
