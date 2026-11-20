# =============================================================
#   SHOP FLOOR - four stations along a lane
#
#   The robot drives a lane in front of a row of stations, the way
#   a mobile manipulator tends a line of machines:
#
#       STORAGE   raw blanks waiting to be processed
#       MACHINE   one fixture; runs a cycle once the robot has left
#       INSPECT   a measuring pad and a red reject bin
#       DISPATCH  an outgoing tray on a short conveyor
#
#   A pedestrian walkway crosses the lane between MACHINE and
#   INSPECT; the base must slow right down through it. A pallet has
#   been left in the lane between STORAGE and MACHINE, so the route
#   has to go round it.
#
#   World frame: x points from the lane towards the stations, y runs
#   along the lane. The arm stands on the floor inside the chassis,
#   so station tables are at the same height as the fixed cell's
#   table and all of the arm's tuning still applies.
# =============================================================

import math
import random

import pybullet as p

from cell import layout as L

TABLE_Z = L.TABLE_Z                 # station table top, as the fixed cell
TABLE_X = (0.27, 0.77)              # every station table spans this in x
TABLE_HALF_Y = 0.30

LANE_X = -0.40                      # base x while travelling
DOCK_X = 0.0                        # base x when docked at a station

# Robot footprint about its base, for route planning (yaw 0).
FOOT_FRONT, FOOT_BACK, FOOT_SIDE = 0.22, 0.36, 0.27

STATIONS = {
    "STORAGE":  -1.95,
    "MACHINE":  -0.65,
    "INSPECT":   0.65,
    "DISPATCH":  1.95,
}

WALKWAY_Y = (-0.17, 0.17)           # crosses the lane between MACHINE/INSPECT
WALKWAY_X = (-1.60, 0.24)
PALLET = ((-0.62, -0.16), (-1.42, -1.12))     # x range, y range
FLOOR_X = (-1.55, 0.05)            # where the base centre may go
FLOOR_Y = (-2.40, 2.40)

NOMINAL_PART_MM = 44.0
PART_TOLERANCE_MM = 3.0

# Equipment on each station, at nominal places in the station's own
# frame (x from the dock, y from the station centre line). The robot
# is told these; the camera finds where they really are.
EQUIPMENT = {
    "MACHINE":  [("fixture", (0.52, 0.00), 0.14, (0.30, 0.32, 0.36))],
    "INSPECT":  [("pad",     (0.52, -0.17), 0.14, (0.15, 0.16, 0.18)),
                 ("bin",     (0.52, 0.12), 0.22, (0.78, 0.16, 0.16))],
    "DISPATCH": [("tray",    (0.52, 0.00), 0.22, (0.55, 0.36, 0.20))],
}

RAW_RGB = (0.85, 0.48, 0.20)        # copper-coloured blanks
MACHINED_RGB = (0.62, 0.66, 0.70)   # bright steel once machined

WALL_T, PLATE_T, WALL_H = 0.006, 0.005, 0.020


# ---- bodies -----------------------------------------------------
def _box(half, rgba, pos, collide=True, mass=0.0, yaw=0.0):
    col = (p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
           if collide else -1)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    return p.createMultiBody(
        baseMass=mass, baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis, basePosition=pos,
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]))


def add_open_box(centre, base_z, inner, rgba):
    """A plate with four low walls: a fixture, pad, bin or tray."""
    half = inner / 2
    parts = [([half + WALL_T, half + WALL_T, PLATE_T / 2],
              [0, 0, PLATE_T / 2])]
    wz = PLATE_T + WALL_H / 2
    for sx, sy, hx, hy in ((1, 0, WALL_T / 2, half + WALL_T),
                           (-1, 0, WALL_T / 2, half + WALL_T),
                           (0, 1, half, WALL_T / 2),
                           (0, -1, half, WALL_T / 2)):
        parts.append(([hx, hy, WALL_H / 2],
                      [sx * (half + WALL_T / 2), sy * (half + WALL_T / 2),
                       wz]))
    n = len(parts)
    col = p.createCollisionShapeArray(
        shapeTypes=[p.GEOM_BOX] * n,
        halfExtents=[h for h, _ in parts],
        collisionFramePositions=[o for _, o in parts])
    vis = p.createVisualShapeArray(
        shapeTypes=[p.GEOM_BOX] * n,
        halfExtents=[h for h, _ in parts],
        rgbaColors=[list(rgba) + [1.0]] * n,
        visualFramePositions=[o for _, o in parts])
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[centre[0], centre[1], base_z])
    p.changeDynamics(body, -1, lateralFriction=0.8)
    return body


def add_part(robot, name, size, rgb, x, y, yaw):
    half = size / 2
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half] * 3)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half] * 3,
                              rgbaColor=list(rgb) + [1.0])
    body = p.createMultiBody(
        baseMass=0.3, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
        basePosition=[x, y, TABLE_Z + half + 0.001],
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]))
    p.changeDynamics(body, -1, lateralFriction=0.8, spinningFriction=0.05,
                     rollingFriction=0.05)
    robot.objects[name] = {"id": body, "color": list(rgb) + [1.0],
                           "position": [x, y, TABLE_Z + half],
                           "half_extent": half, "vision": False,
                           "grasp_depth": 0.0}
    return body


def station_rect(name):
    """The station's table top, world frame: ((x0, x1), (y0, y1))."""
    y = STATIONS[name]
    return (TABLE_X, (y - TABLE_HALF_Y, y + TABLE_HALF_Y))


def lamp_position(name):
    return (TABLE_X[1] + 0.10, STATIONS[name] + 0.22, 0.55)


# ---- the floor ----------------------------------------------------
def build_floor(robot, seed=0, parts=4, bad_part=True, gui=False):
    """Lay out the floor. Returns a dict of body ids and the rng."""
    rng = random.Random(seed)
    built = {"tables": {}, "equipment": {}, "lamps": {}, "rng": rng,
             "machine_body": None}
    p.changeVisualShape(robot.plane, -1, rgbaColor=[0.74, 0.75, 0.76, 1])

    # Lane, walkway and pallet.
    _box([(FLOOR_X[1] - FLOOR_X[0]) / 2 + 0.25, (FLOOR_Y[1] - FLOOR_Y[0]) / 2
          + 0.30, 0.0005], [0.60, 0.62, 0.64, 1],
         [(FLOOR_X[0] + FLOOR_X[1]) / 2 - 0.05, 0, 0.0005], collide=False)
    stripes = int((WALKWAY_X[1] - WALKWAY_X[0]) / 0.12)
    for i in range(stripes):
        x = WALKWAY_X[0] + 0.06 + i * 0.12
        _box([0.03, (WALKWAY_Y[1] - WALKWAY_Y[0]) / 2, 0.001],
             [0.95, 0.80, 0.10, 1] if i % 2 == 0 else [0.12, 0.12, 0.12, 1],
             [x, (WALKWAY_Y[0] + WALKWAY_Y[1]) / 2, 0.001], collide=False)
    (px0, px1), (py0, py1) = PALLET
    built["pallet"] = _box([(px1 - px0) / 2, (py1 - py0) / 2, 0.075],
                           [0.62, 0.46, 0.26, 1],
                           [(px0 + px1) / 2, (py0 + py1) / 2, 0.075])

    for name, sy in STATIONS.items():
        cx = (TABLE_X[0] + TABLE_X[1]) / 2
        built["tables"][name] = _box(
            [(TABLE_X[1] - TABLE_X[0]) / 2, TABLE_HALF_Y, TABLE_Z / 2],
            [0.82, 0.82, 0.84, 1], [cx, sy, TABLE_Z / 2])
        # Edge strip so stations read apart in the picture.
        _box([0.008, TABLE_HALF_Y, 0.004], [0.15, 0.35, 0.65, 1],
             [TABLE_X[0] + 0.008, sy, TABLE_Z + 0.004], collide=False)
        # Status lamp on a post behind the table.
        lx, ly, lz = lamp_position(name)
        _box([0.012, 0.012, lz / 2], [0.25, 0.25, 0.25, 1],
             [lx, ly, lz / 2], collide=False)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.04,
                                  rgbaColor=[0.2, 0.8, 0.3, 1])
        built["lamps"][name] = p.createMultiBody(
            baseVisualShapeIndex=vis, basePosition=[lx, ly, lz + 0.04])
        if gui:
            p.addUserDebugText(name, [lx, ly - 0.30, lz + 0.12],
                               textColorRGB=[0, 0, 0], textSize=1.4)
        for role, (ox, oy), inner, rgb in EQUIPMENT.get(name, []):
            jitter = (rng.uniform(-0.003, 0.003), rng.uniform(-0.003, 0.003))
            centre = (DOCK_X + ox + jitter[0], sy + oy + jitter[1])
            built["equipment"][(name, role)] = add_open_box(
                centre, TABLE_Z, inner, rgb)

    # The machine itself, behind its table.
    built["machine_body"] = _box([0.15, 0.26, 0.30], [0.35, 0.42, 0.50, 1],
                                 [TABLE_X[1] + 0.20, STATIONS["MACHINE"],
                                  0.30])
    # Inspection camera gantry, for the look of it.
    _box([0.01, 0.01, 0.35], [0.2, 0.2, 0.2, 1],
         [TABLE_X[1] + 0.03, STATIONS["INSPECT"] - 0.28, 0.35],
         collide=False)

    # Blanks on the storage table, nominal size, one out of tolerance.
    sy = STATIONS["STORAGE"]
    slots = [(0.44, -0.18), (0.44, 0.02), (0.44, 0.20),
             (0.62, -0.12), (0.62, 0.10)]
    rng.shuffle(slots)
    bad = rng.randrange(parts) if bad_part else -1
    for i in range(parts):
        ox, oy = slots[i]
        size = (50.0 if i == bad else
                NOMINAL_PART_MM + rng.uniform(-0.6, 0.6)) / 1000
        add_part(robot, f"blank_{i}", size, RAW_RGB,
                 DOCK_X + ox + rng.uniform(-0.012, 0.012),
                 sy + oy + rng.uniform(-0.012, 0.012),
                 rng.uniform(-0.6, 0.6))
    built["bad_index"] = bad
    for _ in range(120):
        robot.step()
    return built


def set_lamp(built, station, rgb):
    p.changeVisualShape(built["lamps"][station], -1,
                        rgbaColor=list(rgb) + [1.0])


def set_colour(body, rgb):
    p.changeVisualShape(body, -1, rgbaColor=list(rgb) + [1.0])


# ---- routes ---------------------------------------------------------
def _blocked_cells():
    """Obstacles inflated by the robot footprint (Minkowski, yaw 0)."""
    boxes = []
    for name in STATIONS:
        (x0, x1), (y0, y1) = station_rect(name)
        boxes.append((x0, x1, y0, y1))
    boxes.append((TABLE_X[1], TABLE_X[1] + 0.40,
                  STATIONS["MACHINE"] - 0.30, STATIONS["MACHINE"] + 0.30))
    (px0, px1), (py0, py1) = PALLET
    boxes.append((px0, px1, py0, py1))
    return [(x0 - FOOT_FRONT, x1 + FOOT_BACK, y0 - FOOT_SIDE,
             y1 + FOOT_SIDE) for x0, x1, y0, y1 in boxes]


def free(x, y, boxes=None):
    boxes = boxes if boxes is not None else _blocked_cells()
    if not (FLOOR_X[0] <= x <= FLOOR_X[1] + 0.005
            and FLOOR_Y[0] <= y <= FLOOR_Y[1]):
        return False
    return not any(x0 < x < x1 and y0 < y < y1
                   for x0, x1, y0, y1 in boxes)


def plan_route(start, goal, cell=0.05):
    """
    A* over a grid of the floor, then shortcut by line of sight.
    Returns a list of (x, y) waypoints from start to goal, or None.
    """
    import heapq
    boxes = _blocked_cells()

    def key(pt):
        return (round((pt[0] - FLOOR_X[0]) / cell),
                round((pt[1] - FLOOR_Y[0]) / cell))

    def centre(k):
        return (FLOOR_X[0] + k[0] * cell, FLOOR_Y[0] + k[1] * cell)

    s, g = key(start), key(goal)
    if not free(*goal, boxes):
        return None
    open_set = [(0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    steps = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
             if dx or dy]
    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == g:
            break
        for dx, dy in steps:
            nxt = (cur[0] + dx, cur[1] + dy)
            cx, cy = centre(nxt)
            if nxt != g and not free(cx, cy, boxes):
                continue
            step = math.hypot(dx, dy) * cell
            new = cost[cur] + step
            if new < cost.get(nxt, 1e9):
                cost[nxt] = new
                came[nxt] = cur
                h = math.hypot(nxt[0] - g[0], nxt[1] - g[1]) * cell
                heapq.heappush(open_set, (new + h, nxt))
    if g not in came:
        return None
    path, k = [], g
    while k is not None:
        path.append(centre(k))
        k = came[k]
    path.reverse()
    path[0], path[-1] = tuple(start), tuple(goal)

    def clear(a, b):
        n = max(2, int(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.02))
        return all(free(a[0] + (b[0] - a[0]) * i / n,
                        a[1] + (b[1] - a[1]) * i / n, boxes)
                   for i in range(1, n))

    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not clear(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def in_walkway(x, y):
    return (WALKWAY_X[0] <= x <= WALKWAY_X[1]
            and WALKWAY_Y[0] - FOOT_SIDE <= y <= WALKWAY_Y[1] + FOOT_SIDE)
