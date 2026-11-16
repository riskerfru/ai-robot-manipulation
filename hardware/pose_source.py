# =============================================================
#   POSE SOURCES - where part positions come from on hardware
#
#   In simulation the physics engine reports where every part
#   is. A real cell has to measure it. A pose source that only
#   knows nominal positions (fixtures) cannot detect a misplaced
#   part, and says so through `measures = False`; the placement
#   verifier then refuses to report a verified placement.
# =============================================================

from abc import ABC, abstractmethod


class PoseSource(ABC):
    measures = False          # True only if positions are sensed

    @abstractmethod
    def names(self):
        """Names of the parts this source knows."""

    @abstractmethod
    def get(self, name):
        """(x, y, z) in the cell frame, or None."""

    def notify_placed(self, name, x, y, z):
        """Called after the robot releases a part."""

    def notify_picked(self, name):
        """Called after the robot grips a part."""


class FixturePoseSource(PoseSource):
    """
    Parts sit in fixtures at surveyed positions. Positions are
    nominal: nothing here senses where a part really is.
    """
    measures = False

    def __init__(self, nests, part_z):
        self._poses = {n: (x, y, part_z) for n, (x, y) in nests.items()}

    def names(self):
        return list(self._poses)

    def get(self, name):
        return self._poses.get(name)

    def notify_placed(self, name, x, y, z):
        self._poses[name] = (x, y, z)


class CallbackPoseSource(PoseSource):
    """
    Adapter for a measuring system (2D/3D camera, slot sensors).
    `locate(name)` must return a measured (x, y, z) in the cell
    frame or None when the part is not seen.
    """
    measures = True

    def __init__(self, part_names, locate):
        self._names = list(part_names)
        self._locate = locate

    def names(self):
        return list(self._names)

    def get(self, name):
        return self._locate(name)
