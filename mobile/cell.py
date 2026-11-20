# =============================================================
#   MOBILE CELL - stations, parts and the process route
#
#   Raw blanks start in STORAGE. Every part must go
#
#       STORAGE -> MACHINE -> INSPECT -> DISPATCH   (passed)
#                                     -> reject bin (failed)
#
#   The robot only sees the station it is docked at. On arrival the
#   mast camera measures that station afresh, which is what makes
#   the parking error harmless: targets come from the picture, not
#   from where the base thinks it stopped.
#
#   The stations run themselves:
#   - MACHINE starts a cycle once a raw part sits in its fixture and
#     the robot has left. The fixture's seated sensor (simulated
#     from the part's true position) refuses a part more than 5 mm
#     off centre - a machine will not cut a part it cannot clamp.
#   - INSPECT measures a part placed on its pad with the mast camera
#     and passes it if it is within tolerance of nominal size.
#   - DISPATCH sends its tray out on a conveyor and brings an empty.
# =============================================================

import math

import numpy as np
import pybullet as p

import config as cfg
from cell import openscene as O
from mobile import floor as F
from mobile.base import compose

MACHINE_CYCLE_S = 15.0
SEAT_TOL_M = 0.005

RAW, MACHINING, MACHINED, PASS, FAIL = ("RAW", "MACHINING", "MACHINED",
                                        "PASS", "FAIL")
DISPATCHED, SCRAPPED = "DISPATCHED", "SCRAPPED"

# What each piece of equipment will take, by part state. This is the
# process route; the guardian enforces it before the arm moves.
ACCEPTS = {"fixture": {RAW}, "pad": {MACHINED}, "bin": {FAIL},
           "tray": {PASS}}

LAMP = {"idle": (0.20, 0.80, 0.30), "busy": (0.98, 0.70, 0.10),
        "done": (0.20, 0.45, 0.95), "fault": (0.90, 0.15, 0.15),
        "locked": (0.55, 0.20, 0.70)}


class _StepHub:
    """Calls registered functions on every physics step, wherever the
    step comes from - the base driving or the arm moving."""
    installed = None

    def __init__(self):
        self.callbacks = []
        self.steps = 0
        self._original = p.stepSimulation

        def stepped(*a, **k):
            out = self._original(*a, **k)
            self.steps += 1
            for fn in list(self.callbacks):
                fn()
            return out

        p.stepSimulation = stepped

    @classmethod
    def get(cls):
        if cls.installed is None:
            cls.installed = cls()
        return cls.installed


class MobileCell(O.OpenCell):

    def __init__(self, robot, supervised, base, built, gui=False,
                 use_vision=True, log=None):
        super().__init__(robot, built, gui=gui)
        self.supervised = supervised
        self.base = base
        self.built = built
        self.use_vision = use_vision
        self.log = log or (lambda *a: None)
        self.state = {}                 # thing id -> process state
        self.measured_mm = {}           # thing id -> inspected size
        self.locked = set()             # stations under lockout
        self.machine = {"status": "idle", "part": None, "until": None,
                        "fault": None}
        self.hub = _StepHub.get()
        self.hub.callbacks[:] = [self._tick]     # one cell per process
        self.placements = []            # error records, for the numbers
        self.dispatched = 0
        for st in F.STATIONS:
            F.set_lamp(built, st, LAMP["idle"])

    # ---- time --------------------------------------------------------
    @property
    def now(self):
        return self.hub.steps * cfg.SIM_TIMESTEP

    def _tick(self):
        m = self.machine
        if m["status"] == "busy" and self.now >= m["until"]:
            ident = m["part"]
            m["status"], m["until"] = "done", None
            self.state[ident] = MACHINED
            thing = self.things.get(ident)
            body = self._body_of(thing) if thing else None
            if body is not None:
                F.set_colour(body, F.MACHINED_RGB)
            if thing is not None:
                thing.rgb = np.array(F.MACHINED_RGB)
            F.set_lamp(self.built, "MACHINE", LAMP["done"])
            self.log("MACHINE", f"cycle complete - part {ident} machined")
        if (m["status"] == "idle" and m["part"] is not None
                and self.base.station != "MACHINE"
                and self.state.get(m["part"]) == RAW):
            off = self._seat_offset(m["part"])
            if off is not None and off > SEAT_TOL_M:
                m["status"], m["fault"] = "fault", (
                    f"part {m['part']} not seated: {off * 1000:.1f} mm off "
                    f"fixture centre (limit {SEAT_TOL_M * 1000:.0f} mm)")
                F.set_lamp(self.built, "MACHINE", LAMP["fault"])
                self.log("MACHINE", f"REFUSED TO START - {m['fault']}")
                return
            m["status"], m["until"] = "busy", self.now + MACHINE_CYCLE_S
            self.state[m["part"]] = MACHINING
            F.set_lamp(self.built, "MACHINE", LAMP["busy"])
            self.log("MACHINE", f"robot clear of the station - cycle "
                                f"started on part {m['part']} "
                                f"({MACHINE_CYCLE_S:.0f} s)")

    # ---- bodies and truth (used only by station sensors) ------------
    def _body_of(self, thing):
        best, best_d = None, 0.03
        for d in self.robot.objects.values():
            (x, y, _), _ = p.getBasePositionAndOrientation(d["id"])
            dist = math.hypot(x - thing.x, y - thing.y)
            if dist < best_d:
                best, best_d = d["id"], dist
        return best

    def _seat_offset(self, ident):
        thing = self.things.get(ident)
        fixture = self.built["equipment"].get(("MACHINE", "fixture"))
        body = self._body_of(thing) if thing else None
        if body is None or fixture is None:
            return None
        (px, py, _), _ = p.getBasePositionAndOrientation(body)
        (fx, fy, _), _ = p.getBasePositionAndOrientation(fixture)
        return math.hypot(px - fx, py - fy)

    # ---- looking -------------------------------------------------------
    def _ensure_finder(self):
        from perception.detector import ObjectFinder
        self.robot.camera.setup_overhead_camera(self.base.camera_config(),
                                                "mast")
        if self.finder is None:
            self.finder = ObjectFinder(self.robot.camera, "mast")
        return self.finder

    def scan(self):
        """
        Park the arm clear of the mast camera, take one picture of the
        station the base is docked at, and update what is known
        about that station only.
        """
        station = self.base.station
        if station is None:
            return self.listing()
        self.base.park_for_scan()
        finder = self._ensure_finder()
        (x0, x1), (y0, y1) = F.station_rect(station)
        blobs, table_z = finder.find(area=((x0 - 0.02, x1 + 0.02),
                                           (y0 - 0.02, y1 + 0.02)))
        if table_z is not None:
            self.table_z = table_z
        self.scans += 1
        here = {i: t for i, t in self.things.items()
                if getattr(t, "station", None) == station and not t.held}
        free = list(blobs)

        def colour_gap(thing, blob):
            return float(np.abs(np.asarray(thing.rgb) - blob.rgb).max())

        claimed = {}
        for thing in sorted(here.values(), key=lambda t: not t.container):
            best, best_d = None, 0.05 if thing.container else 0.035
            for blob in free:
                if blob.container != thing.container:
                    continue
                if colour_gap(thing, blob) > 0.18:
                    continue
                d = math.hypot(blob.x - thing.x, blob.y - thing.y)
                if d < best_d:
                    best, best_d = blob, d
            if best is not None:
                free.remove(best)
                if thing.container:
                    thing.take(best)
                else:
                    # A part does not change size. Inside a fixture its
                    # walls hide part of it, so a fresh view can make it
                    # look smaller: keep the size measured in the open.
                    keep = {k: getattr(thing, k) for k in
                            ("long", "short", "height", "roundness", "fill")}
                    shrunk = (best.short < keep["short"] - 0.004
                              or best.long < keep["long"] - 0.004)
                    old_angle = thing.angle
                    thing.take(best)
                    for k, v in keep.items():
                        setattr(thing, k, v)
                    thing.ground = thing.top - thing.height
                    if shrunk:
                        thing.angle = old_angle
                claimed[thing.id] = thing
        gone = [i for i in here if i not in claimed]
        for i in gone:
            del self.things[i]
        for blob in sorted(free, key=lambda b: (not b.container,
                                                round(b.x, 2), b.y)):
            thing = O.Thing(self.next_id, blob)
            thing.station = station
            thing.role = None
            self.things[self.next_id] = thing
            if not blob.container and station == "STORAGE":
                self.state[self.next_id] = RAW
            self.next_id += 1
        # Equipment: match containers to the station's known fittings.
        boxes = [t for t in self.things.values()
                 if t.container and t.station == station]
        for role, (ox, oy), _, _ in F.EQUIPMENT.get(station, []):
            nx, ny = compose((F.DOCK_X, F.STATIONS[station], 0.0), ox, oy)
            near = min(boxes, key=lambda b: math.hypot(b.x - nx, b.y - ny),
                       default=None)
            if near is not None and math.hypot(near.x - nx,
                                               near.y - ny) < 0.06:
                near.role = role
        for thing in self.things.values():
            if thing.station != station or thing.container or thing.held:
                continue
            thing.inside = None
            for box in boxes:
                if (box.inner_x[0] < thing.x < box.inner_x[1]
                        and box.inner_y[0] < thing.y < box.inner_y[1]):
                    thing.inside = box.id
        self._draw_labels()
        return self.listing()

    def at_station(self, station=None):
        station = station or self.base.station
        return [t for t in self.listing()
                if getattr(t, "station", None) == station and not t.held]

    def equipment(self, role, station=None):
        station = station or self.base.station
        for t in self.at_station(station):
            if t.container and getattr(t, "role", None) == role:
                return t
        return None

    def parts_in(self, role, station=None):
        box = self.equipment(role, station)
        if box is None:
            return []
        return [t for t in self.at_station(station)
                if not t.container and t.inside == box.id]

    # ---- acting ---------------------------------------------------------
    def table_spot(self, taken=()):
        """A clear place on the current station's table."""
        station = self.base.station
        (x0, x1), (y0, y1) = F.station_rect(station)
        busy = [(t.x, t.y) for t in self.at_station(station)
                if not t.container] + list(taken)
        boxes = [t for t in self.at_station(station) if t.container]
        best, best_d = None, 1e9
        mx, my = 0.45, F.STATIONS[station]
        x = x0 + 0.08
        while x <= x1 - 0.08:
            y = y0 + 0.07
            while y <= y1 - 0.07:
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

    def pick(self, thing):
        self.base.tuck()            # start every approach from home posture
        args = self.pick_args(thing)
        result = self.supervised.pick_at(**args)
        if result.get("status") == "success":
            self.note_picked(thing)
            thing.station = None
            if self.machine["part"] == thing.id:
                self.machine.update(part=None, status="idle", fault=None)
                F.set_lamp(self.built, "MACHINE", LAMP["idle"])
        return result

    def place(self, role):
        """
        Put the held part into a piece of equipment at this station
        (or on the table, role "table"). Uses the camera's measurement
        of where the equipment is, or - with use_vision off - its
        nominal map position seen through the base's dead reckoning.
        """
        thing = self.held()
        station = self.base.station
        if thing is None:
            return {"status": "failed", "detail": "nothing in the gripper"}
        long_one = thing.is_rect and thing.long > thing.short + 0.012
        if role == "table":
            spot = self.table_spot()
            if spot is None:
                return {"status": "failed", "detail": "no clear spot"}
            surface, inside, truth = self.table_z, None, None
        else:
            box = self.equipment(role)
            if box is None:
                return {"status": "failed",
                        "detail": f"no {role} found at {station}"}
            spots = self.spots_in(box)
            if not spots:
                return {"status": "failed", "detail": f"{role} is full"}
            spot, surface, inside = spots[0], box.floor, box.id
            body = self.built["equipment"].get((station, role))
            truth = (p.getBasePositionAndOrientation(body)[0][:2]
                     if body is not None else None)
            if not self.use_vision:
                # Where the map says this fitting is, aimed at through
                # the base's belief about its own pose.
                ox, oy = next((o for r, o, _, _ in F.EQUIPMENT[station]
                               if r == role))
                nx, ny = compose((F.DOCK_X, F.STATIONS[station], 0.0),
                                 ox, oy)
                if len(spots) > 1 or (box.inner_x[1] - box.inner_x[0]) > 0.16:
                    nx += spot[0] - (box.inner_x[0] + box.inner_x[1]) / 2
                    ny += spot[1] - (box.inner_y[0] + box.inner_y[1]) / 2
                spot = self.base.dead_reckoned(nx, ny)
        touch = surface + thing.hold
        self.base.tuck()
        result = self.supervised.place_by_touch(
            spot[0], spot[1], touch,
            finger_axis="x" if long_one else "y", strict=long_one,
            open_to=min(0.04, thing.grip / 2 + 0.012),
            label=f"{role} at {station}")
        if result.get("status") != "success":
            return result
        self.note_placed(thing, spot[0], spot[1], surface, inside=inside)
        thing.station = station
        body = self._body_of(thing)
        if truth is not None and body is not None and role != "table":
            (px, py, _), _ = p.getBasePositionAndOrientation(body)
            err = math.hypot(px - truth[0], py - truth[1])
            if role in ("fixture", "pad"):
                self.placements.append({"role": role, "station": station,
                                        "error_mm": err * 1000,
                                        "vision": self.use_vision})
            if role in ("fixture", "pad"):
                result["placement_error_mm"] = round(err * 1000, 1)
        if role == "fixture":
            self.machine.update(part=thing.id, status="idle", fault=None)
        if role == "bin":
            self.state[thing.id] = SCRAPPED
        return result

    def inspect(self):
        """
        Measure the part on the pad with a fresh mast-camera picture.
        The measurement is only read here; what the arm knows about
        the part (where to grip it) is left as it was.
        """
        parts = self.parts_in("pad", "INSPECT")
        if not parts:
            return None
        part = parts[0]
        long_m, short_m = part.long, part.short
        if self.base.station == "INSPECT" and self.use_vision:
            self.base.park_for_scan()
            finder = self._ensure_finder()
            (x0, x1), (y0, y1) = F.station_rect("INSPECT")
            blobs, _ = finder.find(area=((x0, x1), (y0, y1)))
            near = [b for b in blobs if not b.container
                    and math.hypot(b.x - part.x, b.y - part.y) < 0.04]
            if near:
                b = min(near, key=lambda b: math.hypot(b.x - part.x,
                                                       b.y - part.y))
                long_m, short_m = b.long, b.short
        # The longest side seen from above. A part that settles slightly
        # tilted shows one side foreshortened; the other stays true.
        size = long_m * 1000
        ok = abs(size - F.NOMINAL_PART_MM) <= F.PART_TOLERANCE_MM
        self.state[part.id] = PASS if ok else FAIL
        self.measured_mm[part.id] = round(size, 1)
        F.set_lamp(self.built, "INSPECT", LAMP["idle" if ok else "fault"])
        return {"part": part.id, "size_mm": round(size, 1),
                "dims_mm": [round(long_m * 1000, 1),
                            round(short_m * 1000, 1)],
                "result": PASS if ok else FAIL}

    def wait_for_machine(self, timeout_s=30.0):
        start = self.now
        while self.machine["status"] == "busy" and self.now - start < timeout_s:
            self.base._tick()
        return self.machine["status"]

    def dispatch(self, steps=120, travel=0.50):
        """The tray leaves on its conveyor; an empty one arrives."""
        key = ("DISPATCH", "tray")
        tray = self.built["equipment"][key]
        box = self.equipment("tray", "DISPATCH")
        riders = [t for t in self.parts_in("tray", "DISPATCH")]
        bodies = [tray] + [b for b in (self._body_of(t) for t in riders)
                           if b is not None]
        for _ in range(steps):
            for body in bodies:
                pos, orn = p.getBasePositionAndOrientation(body)
                p.resetBasePositionAndOrientation(
                    body, [pos[0] + travel / steps, pos[1], pos[2]], orn)
                p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            self.base._tick()
        for body in bodies[1:]:
            for name, d in list(self.robot.objects.items()):
                if d["id"] == body:
                    del self.robot.objects[name]
            p.removeBody(body)
        p.removeBody(tray)
        _, (ox, oy), inner, rgb = F.EQUIPMENT["DISPATCH"][0]
        self.built["equipment"][key] = F.add_open_box(
            compose((F.DOCK_X, F.STATIONS["DISPATCH"], 0.0), ox, oy),
            F.TABLE_Z, inner, rgb)
        for t in riders:
            self.state[t.id] = DISPATCHED
            self.things.pop(t.id, None)
        if box is not None:
            self.things.pop(box.id, None)
        self.dispatched += len(riders)
        for _ in range(40):
            self.base._tick()
        self.robot._sync_obstacles()
        return len(riders)

    def lockout(self, station, on=True):
        if on:
            self.locked.add(station)
            F.set_lamp(self.built, station, LAMP["locked"])
        else:
            self.locked.discard(station)
            F.set_lamp(self.built, station, LAMP["idle"])

    def _draw_labels(self):
        if not self.gui:
            return
        for item in self._labels:
            p.removeUserDebugItem(item)
        self._labels = [
            p.addUserDebugText(str(t.id), [t.x, t.y, t.top + 0.035],
                               textColorRGB=[0, 0, 0], textSize=1.4)
            for t in self.things.values() if not t.held]


def build_mobile_cell(logger=None, gui=False, seed=0, parts=4,
                      bad_part=True, use_vision=True, log=None,
                      docking_error_m=0.015, follow_camera=None):
    """Robot, floor, safety gate, soft PLC, base and cell, wired up."""
    import random
    from types import SimpleNamespace

    from audit.trail import AuditTrail
    from cell.factory import default_limits
    from hardware.sim_backend import make_sim_robot
    from mobile.base import MobileBase
    from plc.bus import InMemoryBus
    from plc.interlocks import PLCInterlocks
    from plc.soft_plc import SoftPLC
    from safety.supervised_robot import SupervisedRobot
    from safety.supervisor import SafetySupervisor
    from utils.logger import NullLogger

    robot = make_sim_robot(logger or NullLogger(), gui=gui)
    built = F.build_floor(robot, seed=seed, parts=parts, bad_part=bad_part,
                          gui=gui)
    bus = InMemoryBus()
    soft_plc = SoftPLC(bus)
    soft_plc.start()
    interlocks = PLCInterlocks(bus)
    interlocks.wait_until_alive()
    safety = SafetySupervisor(default_limits(), interlock_source=interlocks)
    audit = AuditTrail(None)
    supervised = SupervisedRobot(robot, safety, audit)
    base = MobileBase(robot, supervised, random.Random(seed + 101),
                      docking_error_m=docking_error_m,
                      follow_camera=gui if follow_camera is None
                      else follow_camera)
    base.set_pose(F.DOCK_X, F.STATIONS["STORAGE"], 0.0)
    base.station = "STORAGE"
    base.believed_pose = base.true_pose
    base.stow()
    cell = MobileCell(robot, supervised, base, built, gui=gui,
                      use_vision=use_vision, log=log)
    return SimpleNamespace(robot=robot, built=built, safety=safety,
                           audit=audit, supervised=supervised, base=base,
                           cell=cell, bus=bus, soft_plc=soft_plc,
                           interlocks=interlocks)
