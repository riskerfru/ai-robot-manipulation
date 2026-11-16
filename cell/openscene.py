# =============================================================
#   OPEN SCENE - a table of unknown things
#
#   Shapes, colours, sizes and positions are drawn at random. The
#   robot's side of the code is told none of it: no names, no
#   colour table, no list of what is where. It has a camera.
#
#   Everything the planner knows about a thing is what the camera
#   measured: where it is, its colour as three numbers, how long,
#   how wide, how tall, how round. What to call it is up to the
#   language model, which is shown the same picture.
# =============================================================

import math
import random

import numpy as np
import pybullet as p

from cell import layout as L

TABLE_RGBA = [0.80, 0.80, 0.82, 1.0]
BELT_RGBA = [0.16, 0.16, 0.17, 1.0]
BELT_T = 0.003
CONVEYOR_POINT = (L.SLOT_X, 0.0)        # where the belt presents a tray
ZONE_X = (0.29, 0.43)                   # where loose things start out
ZONE_Y = (-0.17, 0.36)
MIN_SPACING = 0.125

# Paint the scene builder draws from. The robot never sees this list.
_PAINT = [(0.85, 0.12, 0.12), (0.95, 0.50, 0.10), (0.96, 0.82, 0.14),
          (0.20, 0.68, 0.30), (0.15, 0.40, 0.85), (0.55, 0.25, 0.70),
          (0.95, 0.55, 0.70), (0.96, 0.96, 0.96), (0.10, 0.75, 0.80)]
_SHAPES = ("cube", "cylinder", "bar")

WALL_T = 0.006
PLATE_T = 0.005
WALL_H = 0.020
INNER = 0.22                            # inside width of a container
SPOT_PITCH = 0.10
FINGER_LIFT = 0.0265                    # gripper height above a part's foot
MAX_GRIP = 0.062
MIN_GRIP = 0.020
MIN_HEIGHT = 0.028
MAX_STACK_M = 0.115                     # top of a stack above the table


# ---- building the physical scene --------------------------------
def _box(half, rgba, pos, mass=0.0, yaw=0.0, friction=None):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    body = p.createMultiBody(
        baseMass=mass, baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis, basePosition=pos,
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]))
    if friction is not None:
        p.changeDynamics(body, -1, lateralFriction=friction,
                         spinningFriction=0.05, rollingFriction=0.05)
    return body


def add_container(centre, base_z, rgba):
    """An open box: a plate with four low walls. One fixed body."""
    half = INNER / 2
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
        rgbaColors=[rgba] * n,
        visualFramePositions=[o for _, o in parts])
    body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                             baseVisualShapeIndex=vis,
                             basePosition=[centre[0], centre[1], base_z])
    p.changeDynamics(body, -1, lateralFriction=0.8)
    return body


def add_loose_thing(robot, name, shape, rgb, x, y, yaw, rng):
    """One loose part. Registered with the robot only as a body."""
    rgba = list(rgb) + [1.0]
    height = rng.uniform(0.036, 0.050)
    if shape == "cylinder":
        radius = rng.uniform(0.018, 0.025)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius,
                                     height=height)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius,
                                  length=height, rgbaColor=rgba)
        half = radius
    else:
        if shape == "cube":
            a = b = rng.uniform(0.036, 0.050) / 2
        else:
            a, b = rng.uniform(0.064, 0.078) / 2, rng.uniform(0.030, 0.040) / 2
        ext = [a, b, height / 2]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ext)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ext,
                                  rgbaColor=rgba)
        half = max(a, b)
    body = p.createMultiBody(
        baseMass=0.3, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
        basePosition=[x, y, L.TABLE_Z + height / 2 + 0.001],
        baseOrientation=p.getQuaternionFromEuler([0, 0, yaw]))
    p.changeDynamics(body, -1, lateralFriction=0.8,
                     spinningFriction=0.05, rollingFriction=0.05)
    robot.objects[name] = {"id": body, "color": rgba,
                           "position": [x, y, L.TABLE_Z + height / 2],
                           "half_extent": half, "vision": False,
                           "grasp_depth": 0.0}
    return body


def build_open_scene(robot, seed=0, count=6, gui=False):
    """Table, belt, two containers and a random scatter of things."""
    rng = random.Random(seed)
    hx, hy, hz = L.TABLE_HALF
    table = _box([hx, hy, hz], TABLE_RGBA,
                 [L.TABLE_CENTRE[0], L.TABLE_CENTRE[1], hz])
    belt = _box([0.125, 0.47, BELT_T / 2], BELT_RGBA,
                [L.SLOT_X, 0.0, L.TABLE_Z + BELT_T / 2])
    # A grippy table top and smoother parts: a part pressed onto
    # another must not be able to drag it across the table.
    p.changeDynamics(table, -1, lateralFriction=1.3)
    robot.cell_bodies = {"table": table, "conveyor": belt, "slots": {}}
    robot.conveyor_slots = {}
    robot.slot_occupants = {}

    scene = {"rng": rng, "tray_rgba": [0.55, 0.36, 0.20, 1.0],
             "bin_rgba": [0.27, 0.30, 0.34, 1.0], "loose": 0}
    scene["tray"] = add_container(CONVEYOR_POINT, L.TABLE_Z + BELT_T,
                                  scene["tray_rgba"])
    bin_at = (0.38 + rng.uniform(-0.006, 0.006),
              -0.375 + rng.uniform(-0.004, 0.004))
    scene["bin"] = add_container(bin_at, L.TABLE_Z, scene["bin_rgba"])

    paints = _PAINT[:]
    rng.shuffle(paints)
    placed = []
    for index in range(count):
        for _ in range(400):
            x = rng.uniform(*ZONE_X)
            y = rng.uniform(*ZONE_Y)
            if all(math.hypot(x - a, y - b) >= MIN_SPACING
                   for a, b in placed):
                break
        else:
            break
        placed.append((x, y))
        shape = rng.choice(_SHAPES)
        add_loose_thing(robot, f"body_{index}", shape,
                        paints[index % len(paints)], x, y,
                        rng.uniform(-math.pi / 2, math.pi / 2), rng)
        scene["loose"] += 1
    for _ in range(120):
        robot.step()
    robot._sync_obstacles()
    return scene


# ---- what the robot's side knows: only what the camera measured ----
class Thing:
    """Something the camera found. Numbers only; no name."""

    def __init__(self, ident, blob):
        self.id = ident
        self.on = None          # id of the thing it stands on
        self.under = None       # id of the thing standing on it
        self.inside = None      # id of the container it is in
        self.held = False
        self.hold = FINGER_LIFT
        self.take(blob)

    def take(self, blob):
        for key in ("container", "x", "y", "rgb", "long", "short", "angle",
                    "top", "ground", "height", "roundness", "fill"):
            setattr(self, key, getattr(blob, key))
        if blob.container:
            self.floor = blob.floor
            self.inner_x, self.inner_y = blob.inner_x, blob.inner_y

    # geometry the code needs to act; none of it is a name
    @property
    def is_round(self):
        return self.roundness > 0.88

    @property
    def is_rect(self):
        return self.fill > 0.93 and not self.is_round

    @property
    def grip(self):
        return self.short if self.is_rect else (self.long + self.short) / 2


class OpenCell:
    SCAN_PARK = (0.10, 0.45, 0.70)      # arm clear of the camera

    def __init__(self, robot, scene, gui=False):
        self.robot = robot
        self.scene = scene
        self.gui = gui
        self.finder = None
        self.things = {}
        self.table_z = L.TABLE_Z
        self.next_id = 1
        self.scans = 0
        self._labels = []

    # ---- looking ---------------------------------------------------
    def _ensure_finder(self):
        if self.finder is None:
            import config as cfg
            from perception.detector import ObjectFinder
            self.robot.camera.setup_overhead_camera(cfg.CELL_CAMERA, "cell")
            self.finder = ObjectFinder(self.robot.camera, "cell")
        return self.finder

    def scan(self):
        """Park the arm, take one picture, measure what is there."""
        finder = self._ensure_finder()
        if not self.robot.grasped_object:
            self.robot.move_to_position(*self.SCAN_PARK, use_rrt=False)
        blobs, table_z = finder.find()
        if table_z is not None:
            self.table_z = table_z
        self.scans += 1
        old, new, free = self.things, {}, list(blobs)

        def colour_gap(thing, blob):
            return float(np.abs(np.asarray(thing.rgb) - blob.rgb).max())

        def claim(thing, reach, gap, same_size=False):
            best, best_d = None, reach
            for blob in free:
                if blob.container != thing.container:
                    continue
                if colour_gap(thing, blob) > gap:
                    continue
                if same_size and (abs(blob.long - thing.long) > 0.006
                                  or abs(blob.short - thing.short) > 0.006):
                    continue
                d = math.hypot(blob.x - thing.x, blob.y - thing.y)
                if d < best_d:
                    best, best_d = blob, d
            if best is not None:
                free.remove(best)
                on = thing.on
                thing.take(best)
                if on is not None and on in old:
                    thing.ground = old[on].top
                    thing.height = thing.top - thing.ground
                new[thing.id] = thing
            return best is not None

        # What is in the gripper, or hidden under something, is not in
        # the picture; it is carried over as it was.
        for thing in old.values():
            if thing.held or (thing.under is not None
                              and thing.under in old):
                new[thing.id] = thing
        rest = [t for t in old.values() if t.id not in new]
        rest.sort(key=lambda t: t.on is None)       # stacked ones first
        for thing in rest:
            claim(thing, 0.05 if thing.container else 0.03, 0.15)
        # The edge of a covered thing can show round what covers it.
        for thing in new.values():
            if thing.under is not None:
                free[:] = [b for b in free
                           if math.hypot(b.x - thing.x, b.y - thing.y) > 0.04
                           or colour_gap(thing, b) > 0.15]
        for thing in rest:
            if thing.id not in new:
                claim(thing, 2.0, 0.08, same_size=True)   # it was moved
        for blob in sorted(free, key=lambda b: (not b.container,
                                                round(b.x, 2), b.y)):
            new[self.next_id] = Thing(self.next_id, blob)
            self.next_id += 1

        self.things = new
        for thing in new.values():
            if thing.on is not None and thing.on not in new:
                thing.on = None
            if thing.under is not None and (
                    thing.under not in new or new[thing.under].on != thing.id):
                thing.under = None
            thing.inside = None
            if thing.container or thing.held:
                continue
            for box in new.values():
                if (box.container and box.inner_x[0] < thing.x < box.inner_x[1]
                        and box.inner_y[0] < thing.y < box.inner_y[1]):
                    thing.inside = box.id
        self._draw_labels()
        return self.listing()

    def listing(self):
        return sorted(self.things.values(), key=lambda t: t.id)

    def _draw_labels(self):
        if not self.gui:
            return
        for item in self._labels:
            p.removeUserDebugItem(item)
        self._labels = [
            p.addUserDebugText(str(t.id), [t.x, t.y, t.top + 0.035],
                               textColorRGB=[0, 0, 0], textSize=1.6)
            for t in self.things.values() if not t.held]

    def held(self):
        return next((t for t in self.things.values() if t.held), None)

    def conveyor_container(self):
        for t in self.things.values():
            if t.container and math.hypot(t.x - CONVEYOR_POINT[0],
                                          t.y - CONVEYOR_POINT[1]) < 0.06:
                return t
        return None

    # ---- can it be done? (geometry only) ---------------------------
    def why_not_grip(self, thing):
        if thing.container:
            return "it is too big for the gripper"
        if thing.under is not None:
            return f"thing {thing.under} is standing on it"
        if thing.grip > MAX_GRIP:
            return (f"it is {1000 * thing.grip:.0f} mm across, wider than "
                    f"the gripper opens")
        if thing.grip < MIN_GRIP or thing.height < MIN_HEIGHT:
            return "it is too small or too flat to grip"
        if self.finger_axis(thing) is None:
            return ("there is no room for the fingers: it is hard up "
                    "against a wall or another thing")
        return None

    def why_not_stack(self, held, base):
        if base.container:
            return None
        if base.under is not None:
            return f"thing {base.under} is already on thing {base.id}"
        if base.is_round or held.is_round:
            return ("round things are not stacked: they tip and roll "
                    "when pressed together")
        if min(base.long, base.short) < 0.034:
            return f"the top of thing {base.id} is too small to stand on"
        if held.grip > base.long + 0.012:
            return f"thing {held.id} is wider than the top of thing {base.id}"
        if base.top - self.table_z + held.height > MAX_STACK_M:
            return "that stack would be too tall to be stable"
        return None

    def spots_in(self, box, taken=()):
        """Free places inside a container, from its measured inside."""
        cx = (box.inner_x[0] + box.inner_x[1]) / 2
        cy = (box.inner_y[0] + box.inner_y[1]) / 2
        nx = max(1, int((box.inner_x[1] - box.inner_x[0] - 0.01)
                        // SPOT_PITCH))
        ny = max(1, int((box.inner_y[1] - box.inner_y[0] - 0.01)
                        // SPOT_PITCH))
        busy = [(t.x, t.y) for t in self.things.values()
                if not t.container and not t.held and t.inside == box.id]
        busy += list(taken)
        spots = []
        for i in range(nx):
            for j in range(ny):
                x = cx + (i - (nx - 1) / 2) * SPOT_PITCH
                y = cy + (j - (ny - 1) / 2) * SPOT_PITCH
                if all(math.hypot(x - a, y - b) > 0.055 for a, b in busy):
                    spots.append((x, y))
        spots.sort(key=lambda s: (-round(s[0], 2), s[1]))
        return spots

    def table_spot(self, taken=()):
        """A clear place on the open table."""
        busy = [(t.x, t.y) for t in self.things.values()
                if not t.container and not t.held] + list(taken)
        boxes = [t for t in self.things.values() if t.container]
        best, best_d = None, 1e9
        mx, my = sum(ZONE_X) / 2, sum(ZONE_Y) / 2
        x = ZONE_X[0]
        while x <= ZONE_X[1] + 1e-9:
            y = ZONE_Y[0]
            while y <= ZONE_Y[1] + 1e-9:
                clear = all(math.hypot(x - a, y - b) >= 0.10
                            for a, b in busy)
                clear = clear and not any(
                    b.inner_x[0] - 0.07 < x < b.inner_x[1] + 0.07
                    and b.inner_y[0] - 0.07 < y < b.inner_y[1] + 0.07
                    for b in boxes)
                d = math.hypot(x - mx, y - my)
                if clear and d < best_d:
                    best, best_d = (x, y), d
                y += 0.02
            x += 0.02
        return best

    FINGER_T = 0.014        # finger thickness outside the opening

    def finger_axis(self, thing):
        """
        Which way the fingers should open to take this thing: across
        its narrow side if it has one, and never into a wall or a
        neighbour. Returns (direction, period) or None if there is no
        clear way in.
        """
        if thing.is_round:
            options = [math.pi / 2, 0.0, math.pi / 4, 3 * math.pi / 4]
        elif thing.long - thing.short < 0.008:
            options = [thing.angle + math.pi / 2, thing.angle]
        else:
            options = [thing.angle + math.pi / 2]
        reach = min(0.04, thing.grip / 2 + 0.014) + self.FINGER_T
        boxes = [t for t in self.things.values() if t.container]
        others = [t for t in self.things.values()
                  if not t.container and not t.held and t.id != thing.id
                  and t.id != thing.on]
        for angle in options:
            clear = True
            for sign in (1, -1):
                fx = thing.x + sign * reach * math.cos(angle)
                fy = thing.y + sign * reach * math.sin(angle)
                for box in boxes:
                    in_x = box.inner_x[0] < fx < box.inner_x[1]
                    in_y = box.inner_y[0] < fy < box.inner_y[1]
                    if thing.inside == box.id:
                        clear = clear and in_x and in_y
                    else:
                        near_x = box.inner_x[0] - 0.012 < fx < box.inner_x[1] + 0.012
                        near_y = box.inner_y[0] - 0.012 < fy < box.inner_y[1] + 0.012
                        clear = clear and not (near_x and near_y)
                for other in others:
                    if abs(other.top - thing.top) > 0.03 and other.top < thing.ground + 0.012:
                        continue            # well below the fingers
                    if math.hypot(fx - other.x, fy - other.y) < (
                            math.hypot(other.long, other.short) / 2 + 0.006):
                        clear = False
            if clear:
                return (angle, math.pi)
        return None

    # ---- acting ------------------------------------------------------
    def pick_args(self, thing):
        centre_z = (thing.top + thing.ground) / 2
        return dict(
            x=thing.x, y=thing.y, centre_z=centre_z,
            finger_axis=self.finger_axis(thing),
            open_to=min(0.04, thing.grip / 2 + 0.014),
            grasp_depth=centre_z - (thing.ground + FINGER_LIFT),
            label=f"thing {thing.id}")

    def note_picked(self, thing):
        thing.held = True
        # Where the part's foot is below the gripper, from the height
        # the gripper closed at and the height the part stood on.
        grasp_z = getattr(self.robot, "last_grasp_z", None)
        thing.hold = (grasp_z - thing.ground if grasp_z is not None
                      else FINGER_LIFT)
        if not 0.010 < thing.hold < 0.045:
            thing.hold = FINGER_LIFT
        if thing.on is not None and thing.on in self.things:
            self.things[thing.on].under = None
        thing.on = thing.inside = None

    def note_placed(self, thing, x, y, surface_z, on=None, inside=None):
        thing.held = False
        thing.x, thing.y = x, y
        thing.ground = surface_z
        thing.top = surface_z + thing.height
        thing.on, thing.inside = on, inside
        if on is not None:
            self.things[on].under = thing.id

    def dispatch(self, travel=0.62, steps=90):
        """The belt takes the container at the loading point away,
        with whatever is in it, and brings an empty one."""
        box = self.conveyor_container()
        if box is None:
            return 0
        tray = self.scene["tray"]
        low, high = p.getAABB(tray)
        riders = [n for n, d in self.robot.objects.items()
                  if low[0] < p.getBasePositionAndOrientation(d["id"])[0][0]
                  < high[0]
                  and low[1] < p.getBasePositionAndOrientation(d["id"])[0][1]
                  < high[1]]
        bodies = [tray] + [self.robot.objects[n]["id"] for n in riders]
        for _ in range(steps):
            for body in bodies:
                pos, orn = p.getBasePositionAndOrientation(body)
                p.resetBasePositionAndOrientation(
                    body, [pos[0], pos[1] - travel / steps, pos[2]], orn)
                p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            self.robot.step()
        for name in riders:
            p.removeBody(self.robot.objects.pop(name)["id"])
        p.removeBody(tray)
        self.scene["tray"] = add_container(
            CONVEYOR_POINT, L.TABLE_Z + BELT_T, self.scene["tray_rgba"])
        for _ in range(40):
            self.robot.step()
        gone = [t.id for t in self.things.values() if t.inside == box.id]
        for ident in gone:
            del self.things[ident]
        self.robot._sync_obstacles()
        return len(gone)
