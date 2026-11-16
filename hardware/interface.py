# =============================================================
#   ROBOT INTERFACE - the contract every backend implements
#
#   The agent, the placement verifier and the safety supervisor
#   only ever talk to this interface. Swapping PyBullet for a
#   real UR5e changes the backend, not the code above it.
# =============================================================

from abc import ABC, abstractmethod


class RobotInterface(ABC):
    """
    Minimal set of operations the cell software needs from a robot.

    Result dictionaries follow one shape everywhere:
        {"status": "success" | "failed", "action": str,
         "target": str, "detail": str, ...}

    Attributes every backend exposes:
        objects         {name: {...}}   parts known to the cell
        grasped_object  name or None
        conveyor_slots  {slot_number: (x, y)}
        slot_occupants  {slot_number: part_name}
    """

    objects = {}
    grasped_object = None
    conveyor_slots = {}
    slot_occupants = {}

    # ---- state ------------------------------------------------
    @abstractmethod
    def get_end_effector_position(self):
        """TCP position (x, y, z) in the cell frame, metres."""

    @abstractmethod
    def get_object_position(self, name):
        """Measured part position (x, y, z) or None if not found."""

    @abstractmethod
    def get_all_object_positions(self):
        """{name: (x, y, z) or None} for every known part."""

    @abstractmethod
    def check_reachability(self, x, y, z=0.65):
        """{"reachable": bool, "severity": str, "distance": float,
            "message": str, "hint": str or None}"""

    # ---- motion -----------------------------------------------
    @abstractmethod
    def move_to_position(self, x, y, z, use_rrt=True, ignore_name=None):
        """Move the TCP. Returns True on success."""

    @abstractmethod
    def pick_object(self, name):
        """Pick a named part. Returns a result dictionary."""

    @abstractmethod
    def place_object(self, x, y, stack_on=None):
        """Place the held part at (x, y). Returns a result dictionary."""

    @abstractmethod
    def home(self):
        """Return to the home pose."""

    # ---- stop -------------------------------------------------
    def stop(self):
        """Stop motion as fast as the backend allows. Optional."""

    def disconnect(self):
        """Release the connection. Optional."""


REQUIRED_METHODS = (
    "get_end_effector_position", "get_object_position",
    "get_all_object_positions", "check_reachability",
    "move_to_position", "pick_object", "place_object", "home",
)


def conforms(robot):
    """True if an object provides every method the cell software calls."""
    return all(callable(getattr(robot, m, None)) for m in REQUIRED_METHODS)
