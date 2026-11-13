# =============================================================
#   FORCE-GUIDED INSERTION
#
#   Seats a held part in a pocket that leaves about a millimetre
#   per side. Position alone is not enough: the part's position in
#   the gripper and the housing's position are each known only to
#   a few millimetres. So the arm lowers the part until it feels
#   contact, checks from its own height whether the part is in the
#   pocket or resting on the rim, and if it is on the rim, searches
#   in a spiral under light contact until the part drops in.
#
#   The force signal is the contact force between the held part
#   and whatever it touches other than the gripper, which is what
#   a wrist force/torque sensor reports after gravity compensation.
#   In simulation it is read from the physics engine.
# =============================================================

import math
import time

import pybullet as p

CONTACT_N      = 2.0      # contact detected above this
PRESS_LIMIT_N  = 40.0     # never push harder than this
STEP_M         = 0.001    # descent step
SEARCH_STEP_M  = 0.001    # spiral pitch
SEARCH_MAX_M   = 0.007    # largest search radius
SEATED_TOL_M   = 0.002    # part base within this of the pocket floor


class ForceGuidedInsertion:

    def __init__(self, robot):
        self.robot = robot
        self.trace = []            # (ee_z, force_N) per step

    # ---- sensing -------------------------------------------------
    def contact_force(self, part):
        """Vertical contact force on the held part, gripper excluded (N)."""
        body = self.robot.objects[part]["id"]
        fz = 0.0
        for c in p.getContactPoints(bodyA=body):
            first = c[1] == body
            other = c[2] if first else c[1]
            if other in (self.robot.robot, body):
                continue
            # Normal force plus friction: a part wedged between the
            # pocket walls is held up by friction alone. The normal
            # reported by the engine points from the second body to
            # the first.
            sign = 1.0 if first else -1.0
            fz += sign * (c[9] * c[7][2] + c[10] * c[11][2]
                          + c[12] * c[13][2])
        return max(0.0, fz)

    def _settle(self, steps=12):
        for _ in range(steps):
            p.stepSimulation()
            time.sleep(self.robot.sim_config["timestep"])

    def _move(self, x, y, z, orn, speed=0.05):
        """Small closed-loop move: returns the gripper position reached."""
        r = self.robot
        ox = oy = 0.0
        for _ in range(3):
            r._direct_move(x + ox, y + oy, z, speed_factor=speed, orn=orn)
            ee = r.get_end_effector_position()
            if math.hypot(x - ee[0], y - ee[1]) < 0.0004:
                break
            ox += x - ee[0]
            oy += y - ee[1]
        return r.get_end_effector_position()

    # ---- the insertion -------------------------------------------
    def insert(self, part, pocket, on_step=None, use_force=True):
        """
        pocket: dict with x, y, floor_z, depth, half, clearance (nominal).
        use_force=False is the comparison case: go to the nominal
        position and let go, with no search and no seating check.
        Returns a dict: status, detail, seated_error_mm, offset_mm,
        peak_force_n, searched, search_offset_mm.
        """
        r = self.robot
        if r.grasped_object != part:
            return {"status": "failed", "action": "insert", "target": part,
                    "detail": f"not holding '{part}'"}

        self.trace = []
        grip_above_base = pocket["half"] - r.objects[part]["grasp_depth"]
        seat_z = pocket["floor_z"] + grip_above_base   # gripper z, seated
        rim_z  = seat_z + pocket["depth"]
        x, y   = pocket["x"], pocket["y"]

        # Above the pocket, tool vertical and square to the housing.
        r._direct_move(x, y, rim_z + 0.06, speed_factor=0.3)
        orn = r._down_orientation()
        r._tool_orn = orn
        for _ in range(4):
            self._move(x, y, rim_z + 0.03, orn, speed=0.15)

        peak = 0.0
        searched = False
        offset = (0.0, 0.0)

        def read(z):
            nonlocal peak
            self._settle()
            f = self.contact_force(part)
            peak = max(peak, f)
            self.trace.append((z, f))
            if on_step:
                on_step(z, f)
            return f

        def lower_until_contact(cx, cy, z_from, z_to):
            """Step down; return (z reached, force)."""
            z = z_from
            f = 0.0
            while z > z_to:
                if z > rim_z + 0.003:
                    step = STEP_M
                elif z > rim_z - 0.001 or z <= seat_z + 0.002:
                    step = STEP_M / 4          # creep onto rim and floor
                else:
                    step = STEP_M / 2
                z = max(z_to, z - step)
                self._move(cx, cy, z, orn)
                f = read(z)
                if f > CONTACT_N:
                    break
            return z, f

        # Keep going a little past the nominal depth: the part's
        # height in the gripper is only known to about 2 mm, and the
        # floor has to be felt, not assumed.
        floor_cmd = seat_z - 0.004

        def seated(z_now):
            """Deep enough that the part must be on the pocket floor."""
            return z_now <= seat_z + 0.0025

        z, f = lower_until_contact(x, y, rim_z + 0.03, floor_cmd)

        if not use_force:
            self._release(part)
            r._retreat_straight_up(orn)
            return {
                "status": "success", "action": "insert", "target": part,
                "peak_force_n": round(peak, 1), "searched": False,
                "unverified": True,
                "detail": f"'{part}' released at the nominal position "
                          f"(not checked)",
            }

        if not seated(z):
            # Resting on the rim, or caught on an edge: spiral search
            # under light contact until the part goes all the way down.
            searched = True
            found = False
            radius = SEARCH_STEP_M
            while radius <= SEARCH_MAX_M and not found:
                points = max(6, int(2 * math.pi * radius / SEARCH_STEP_M))
                for i in range(points):
                    a = 2 * math.pi * i / points
                    cx = x + radius * math.cos(a)
                    cy = y + radius * math.sin(a)
                    self._move(x + offset[0], y + offset[1],
                               rim_z + 0.0015, orn)
                    self._move(cx, cy, rim_z + 0.0015, orn)
                    offset = (cx - x, cy - y)
                    zz, ff = lower_until_contact(cx, cy, rim_z + 0.0015,
                                                 floor_cmd)
                    if seated(zz):
                        z, f = zz, ff
                        found = True
                        break
                radius += SEARCH_STEP_M
            if not found:
                self._move(x + offset[0], y + offset[1], rim_z + 0.03, orn,
                           speed=0.15)
                return {
                    "status": "failed", "action": "insert", "target": part,
                    "detail": f"'{part}' did not seat within "
                              f"{SEARCH_MAX_M * 1000:.0f} mm of nominal",
                    "peak_force_n": round(peak, 1), "searched": True,
                    "search_offset_mm": round(1000 * math.hypot(*offset), 1),
                }

        self._release(part)
        r._retreat_straight_up(orn)

        pos = r.get_object_position(part)
        base_err = (pos[2] - pocket["half"]) - pocket["floor_z"]
        seated = abs(base_err) <= SEATED_TOL_M
        result = {
            "status": "success" if seated else "failed",
            "action": "insert", "target": part,
            "seated_error_mm": round(base_err * 1000, 1),
            "peak_force_n": round(peak, 1),
            "searched": searched,
            "search_offset_mm": round(1000 * math.hypot(*offset), 1),
            "detail": (f"'{part}' seated ({base_err * 1000:+.1f} mm), "
                       f"peak {peak:.0f} N"
                       + (f", found by search {1000 * math.hypot(*offset):.0f} mm"
                          f" from nominal" if searched else "")
                       if seated else
                       f"'{part}' not seated: base {base_err * 1000:+.1f} mm "
                       f"above the pocket floor"),
        }
        return result

    def _release(self, part):
        r = self.robot
        if r.grasp_constraint is not None:
            p.removeConstraint(r.grasp_constraint)
            r.grasp_constraint = None
        r.open_gripper()
        self._settle(60)
        r.grasped_object = None
