# =============================================================
#   CELL SCENE - builds the simulated cell from cell/layout.py
#   Used by main.py (GUI), run_cell.py and eval_physics.py so
#   every entry point runs in exactly the same cell.
# =============================================================

import math
import pybullet as p

from cell import layout as L


def sample_part_positions(rng):
    """Random, non-overlapping part positions inside the infeed region."""
    placed = {}
    for name in L.PART_NESTS:
        for _ in range(200):
            x = round(rng.uniform(*L.INFEED_X), 3)
            y = round(rng.uniform(*L.INFEED_Y), 3)
            if all(math.hypot(x - q[0], y - q[1]) >= L.INFEED_MIN_SPACING
                   for q in placed.values()):
                placed[name] = (x, y)
                break
        else:
            placed[name] = L.PART_NESTS[name]
    return placed


def build_cell(robot, part_positions=None, gui=False, settle_steps=100):
    """
    Create table, conveyor, slot markers and parts.

    part_positions: {name: (x, y)}, None for the nominal nests, or {}
    for a cell with no dispatch parts (the assembly cell).
    Returns the slot dictionary that was installed on the robot.
    """
    hx, hy, hz = L.TABLE_HALF
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                                    rgbaColor=[0.55, 0.35, 0.15, 1.0])
    table_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[L.TABLE_CENTRE[0], L.TABLE_CENTRE[1], hz])

    positions = L.PART_NESTS if part_positions is None else part_positions
    for name, (x, y) in positions.items():
        robot.add_object(name, x=x, y=y,
                         color=L.PART_COLOURS[name], z=L.PART_Z)

    conv_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.08, 0.44, 0.005])
    conv_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.08, 0.44, 0.005],
        rgbaColor=[0.15, 0.15, 0.15, 1.0])
    conveyor_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=conv_col,
        baseVisualShapeIndex=conv_vis,
        basePosition=[L.SLOT_X, 0.0, L.TABLE_Z + 0.005])

    w = L.SLOT_HALF_W
    slot_ids = {}
    for slot_num, (sx, sy) in L.CONVEYOR_SLOTS.items():
        sc = p.createCollisionShape(p.GEOM_BOX, halfExtents=[w, w, 0.004])
        sv = p.createVisualShape(p.GEOM_BOX, halfExtents=[w, w, 0.004],
                                 rgbaColor=[0.85, 0.85, 0.85, 1.0])
        slot_ids[slot_num] = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=sc,
            baseVisualShapeIndex=sv,
            basePosition=[sx, sy, L.TABLE_Z + 0.012])
        if gui:
            p.addUserDebugText(text=str(slot_num),
                               textPosition=[sx, sy, L.TABLE_Z + 0.08],
                               textColorRGB=[1, 1, 1], textSize=1.5)
            p.addUserDebugLine([sx - w, sy - w, L.TABLE_Z + 0.013],
                               [sx + w, sy - w, L.TABLE_Z + 0.013],
                               [0.9, 0.9, 0.9], 2)

    robot.conveyor_slots = dict(L.CONVEYOR_SLOTS)
    robot.slot_occupants = {}
    robot.cell_bodies = {"table": table_id, "conveyor": conveyor_id,
                         "slots": slot_ids}

    for _ in range(settle_steps):
        robot.step()
    return robot.conveyor_slots


def refill_infeed(robot, part_positions=None, settle_steps=60):
    """
    Simulate the conveyor indexing away and the infeed presenting
    new parts: move every part back to an infeed position.
    """
    positions = part_positions or L.PART_NESTS
    for name, (x, y) in positions.items():
        body = robot.objects[name]["id"]
        p.resetBasePositionAndOrientation(body, [x, y, L.PART_Z],
                                          [0, 0, 0, 1])
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
    robot.slot_occupants = {}
    for _ in range(settle_steps):
        robot.step()
