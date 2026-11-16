"""Shared test doubles. No PyBullet, no network, no API key."""


class FakeLogger:
    def __getattr__(self, name):
        return lambda *a, **k: None


class FakeCellRobot:
    """A robot that always succeeds; parts land where they are placed
    plus an optional per-call offset (to model placement drift)."""

    def __init__(self, parts=None, slots=None, place_offsets=None):
        self.parts = dict(parts or {"red": (0.40, 0.0, 0.125),
                                    "blue": (0.40, 0.30, 0.125),
                                    "green": (0.40, -0.30, 0.125)})
        self.objects = {n: {} for n in self.parts}
        self.obstacles = {}
        self.conveyor_slots = dict(slots or {1: (0.63, 0.30),
                                             2: (0.63, 0.0),
                                             3: (0.63, -0.30)})
        self.slot_occupants = {}
        self.grasped_object = None
        self.place_offsets = list(place_offsets or [])
        self.calls = []
        self.tcp = (0.30, 0.0, 0.50)
        self.stopped = False

    def get_end_effector_position(self):
        return self.tcp

    def get_object_position(self, name):
        return self.parts.get(name)

    def get_all_object_positions(self):
        return dict(self.parts)

    def check_reachability(self, x, y, z=0.65):
        import math
        d = math.hypot(x, y)
        ok = 0.25 <= d <= 0.75
        return {"reachable": ok, "severity": "ok" if ok else "error",
                "distance": round(d, 3), "message": "", "hint": None}

    def move_to_position(self, x, y, z, use_rrt=True, ignore_name=None):
        self.calls.append(("move", x, y, z))
        self.tcp = (x, y, z)
        return True

    def pick_object(self, name):
        self.calls.append(("pick", name))
        self.grasped_object = name
        return {"status": "success", "action": "pick", "target": name,
                "detail": f"picked {name}"}

    def place_object(self, x, y, stack_on=None):
        self.calls.append(("place", x, y))
        held = self.grasped_object
        dx, dy = self.place_offsets.pop(0) if self.place_offsets else (0, 0)
        self.parts[held] = (x + dx, y + dy, 0.14)
        self.grasped_object = None
        self.tcp = (x, y, 0.30)
        return {"status": "success", "action": "place", "target": held,
                "detail": f"placed {held}"}

    def home(self):
        self.calls.append(("home",))
        return True

    def stop(self):
        self.stopped = True


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt
