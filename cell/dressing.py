# =============================================================
#   CELL DRESSING - how the simulated cell looks on camera
#
#   Visual only. Nothing here has a collision shape or changes
#   the physics, the layout or the part colours the vision code
#   relies on. Used by record_demo.py.
# =============================================================

import math
import pybullet as p

from cell import layout as L

ALUMINIUM = [0.66, 0.68, 0.71, 1.0]
STEEL     = [0.34, 0.36, 0.40, 1.0]
BELT      = [0.10, 0.11, 0.12, 1.0]
FIXTURE   = [0.86, 0.88, 0.90, 1.0]
FLOOR     = [0.80, 0.81, 0.83, 1.0]
SAFETY    = [0.98, 0.78, 0.10, 1.0]

LAMP_ON  = {"red":   [0.95, 0.42, 0.42, 1.0],   # kept outside the
            "amber": [1.00, 0.72, 0.20, 1.0],   # vision colour ranges
            "green": [0.40, 0.90, 0.55, 1.0]}
LAMP_OFF = {"red":   [0.40, 0.33, 0.33, 1.0],
            "amber": [0.42, 0.39, 0.32, 1.0],
            "green": [0.32, 0.40, 0.35, 1.0]}

GATE_Y = -0.62
GATE_X = (0.02, 0.82)
GATE_Z = 0.32


def _visual(shape, pos, rgba, orn=(0, 0, 0, 1), **kw):
    vis = p.createVisualShape(shape, rgbaColor=rgba, **kw)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                             basePosition=pos, baseOrientation=orn)


class CellDressing:

    def __init__(self, robot):
        bodies = robot.cell_bodies
        p.changeVisualShape(robot.plane, -1, rgbaColor=FLOOR,
                            textureUniqueId=-1)
        p.changeVisualShape(bodies["table"], -1, rgbaColor=ALUMINIUM)
        p.changeVisualShape(bodies["conveyor"], -1, rgbaColor=BELT)
        for body in bodies["slots"].values():
            p.changeVisualShape(body, -1, rgbaColor=FIXTURE)

        # Conveyor side rails
        for dx in (-0.09, 0.09):
            _visual(p.GEOM_BOX, [L.SLOT_X + dx, 0.0, L.TABLE_Z + 0.012],
                    STEEL, halfExtents=[0.008, 0.46, 0.012])

        # Table legs / frame
        hx, hy, _ = L.TABLE_HALF
        cx, cy = L.TABLE_CENTRE
        _visual(p.GEOM_BOX, [cx, cy, 0.004], STEEL,
                halfExtents=[hx + 0.02, hy + 0.02, 0.004])

        # Floor marking around the cell
        for (px, py, sx, sy) in ((cx, GATE_Y - 0.10, hx + 0.30, 0.02),
                                 (cx, -GATE_Y + 0.10, hx + 0.30, 0.02),
                                 (cx + hx + 0.28, 0.0, 0.02, 0.74),
                                 (cx - hx - 0.28, 0.0, 0.02, 0.74)):
            _visual(p.GEOM_BOX, [px, py, 0.002], SAFETY,
                    halfExtents=[sx, sy, 0.002])

        # Guard gate on the operator side: two posts and a bar
        for gx in GATE_X:
            _visual(p.GEOM_CYLINDER, [gx, GATE_Y, GATE_Z / 2], STEEL,
                    radius=0.015, length=GATE_Z)
        self._gate_len = GATE_X[1] - GATE_X[0]
        self._gate = _visual(p.GEOM_BOX, [0, 0, 0], SAFETY,
                             halfExtents=[self._gate_len / 2, 0.012, 0.012])
        self.set_guard(True)

        # Stack light behind the robot
        base = [0.84, 0.60]
        _visual(p.GEOM_CYLINDER, [base[0], base[1], 0.25], STEEL,
                radius=0.012, length=0.50)
        self._lamps = {}
        for i, name in enumerate(("green", "amber", "red")):
            self._lamps[name] = _visual(
                p.GEOM_CYLINDER, [base[0], base[1], 0.53 + 0.065 * i],
                LAMP_OFF[name], radius=0.038, length=0.06)
        self.set_light("green")

    def set_guard(self, closed):
        """Bar across the opening when closed, swung upright when open."""
        if closed:
            pos = [(GATE_X[0] + GATE_X[1]) / 2, GATE_Y, GATE_Z]
            orn = [0, 0, 0, 1]
        else:
            pos = [GATE_X[0], GATE_Y, GATE_Z + self._gate_len / 2]
            orn = p.getQuaternionFromEuler([0, math.pi / 2, 0])
        p.resetBasePositionAndOrientation(self._gate, pos, orn)

    def set_light(self, lit):
        for name, body in self._lamps.items():
            colour = LAMP_ON[name] if name == lit else LAMP_OFF[name]
            p.changeVisualShape(body, -1, rgbaColor=colour)
