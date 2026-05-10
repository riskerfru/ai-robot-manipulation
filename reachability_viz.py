# =============================================================
#   REACHABILITY VISUALISER
#   Draws coloured zone rings directly in PyBullet viewer.
#
#   Three zones:
#     GREEN  (0.25m - 0.65m) — safe picking zone
#     ORANGE (0.65m - 0.75m) — edge of reach, may fail
#     RED    (0.75m+)        — out of reach
#     GREY   (0.0m - 0.25m)  — too close, arm folds
#
#   Also marks each object with a vertical coloured line
#   showing its individual reachability status.
#
#   Usage:
#     from debug.reachability_viz import ReachabilityVisualiser
#     viz = ReachabilityVisualiser(robot)
#     viz.draw_zones()           # draw once on startup
#     viz.update_objects()       # call after each action
#     viz.clear()                # remove all lines
# =============================================================

import math
import pybullet as p


class ReachabilityVisualiser:
    """
    Draws reachability zones and per-object status in PyBullet viewer.

    Zone colours:
      Green  = safe reach (0.25 - 0.65m)
      Orange = warning zone (0.65 - 0.75m)
      Red    = out of reach (> 0.75m)
      Grey   = too close (< 0.25m)

    Object markers:
      Green pillar  = object reachable
      Orange pillar = object at edge of reach
      Red pillar    = object out of reach
    """

    # Zone radii (metres from robot base)
    INNER_DEAD   = 0.25   # too close — arm folds
    SAFE_MAX     = 0.65   # safe picking limit
    WARN_MAX     = 0.75   # warning — may fail
    # Beyond WARN_MAX = out of reach

    # Zone colours [R, G, B] 0-1
    COL_SAFE    = [0.0, 0.9, 0.0]   # green
    COL_WARN    = [1.0, 0.6, 0.0]   # orange
    COL_DANGER  = [1.0, 0.0, 0.0]   # red
    COL_DEAD    = [0.5, 0.5, 0.5]   # grey (too close)
    COL_TABLE   = [0.4, 0.3, 0.1]   # brown (table outline)

    # Line settings
    ZONE_HEIGHT  = 0.11   # just above table surface (z=0.10)
    N_SEGMENTS   = 60     # segments per circle (smoother = more)
    LINE_WIDTH   = 2

    def __init__(self, robot):
        self.robot       = robot
        self._zone_ids   = []    # PyBullet line IDs for zones
        self._object_ids = {}    # PyBullet line IDs per object name
        self._table_ids  = []    # PyBullet line IDs for table outline

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def draw_zones(self):
        """
        Draw all reachability zone rings.
        Call once after simulation starts.
        """
        self.clear_zones()

        # Dead zone (too close) — grey dashed inner circle
        self._zone_ids += self._draw_circle(
            radius=self.INNER_DEAD,
            z=self.ZONE_HEIGHT,
            color=self.COL_DEAD,
            dashed=True
        )

        # Safe zone outer boundary — green solid
        self._zone_ids += self._draw_circle(
            radius=self.SAFE_MAX,
            z=self.ZONE_HEIGHT,
            color=self.COL_SAFE,
            dashed=False
        )

        # Warning zone outer boundary — orange solid
        self._zone_ids += self._draw_circle(
            radius=self.WARN_MAX,
            z=self.ZONE_HEIGHT,
            color=self.COL_WARN,
            dashed=False
        )

        # Fill warning ring with orange dashes
        self._zone_ids += self._draw_ring_fill(
            r_inner=self.SAFE_MAX,
            r_outer=self.WARN_MAX,
            z=self.ZONE_HEIGHT,
            color=self.COL_WARN
        )

        # Labels as short radial lines at 0°, 90°, 180°, 270°
        self._zone_ids += self._draw_label_ticks()

        print(f"  [REACH] Zones drawn:")
        print(f"    Grey  ring: < {self.INNER_DEAD}m  (too close)")
        print(f"    Green ring: {self.INNER_DEAD}m - {self.SAFE_MAX}m (safe)")
        print(f"    Orange ring: {self.SAFE_MAX}m - {self.WARN_MAX}m (edge)")
        print(f"    Red beyond: > {self.WARN_MAX}m (out of reach)")

    def draw_table_outline(self, cx=0.35, cy=0.0, w=0.35, d=0.40, z=0.101):
        """
        Draw a rectangle showing the table boundary.
        Matches the table dimensions in main.py.
        """
        self.clear_table()
        corners = [
            [cx - w, cy - d, z],
            [cx + w, cy - d, z],
            [cx + w, cy + d, z],
            [cx - w, cy + d, z],
        ]
        for i in range(4):
            a = corners[i]
            b = corners[(i + 1) % 4]
            lid = p.addUserDebugLine(a, b, self.COL_TABLE,
                                     lineWidth=self.LINE_WIDTH)
            self._table_ids.append(lid)

    def update_objects(self):
        """
        Draw a vertical coloured pillar above each object
        showing its reachability status.
        Call after every pick/place action.
        """
        self.clear_objects()

        for name in self.robot.objects:
            pos = self.robot.get_object_position(name)
            if pos is None:
                continue

            dist = math.sqrt(pos[0]**2 + pos[1]**2)

            # Pick colour based on distance
            if dist < self.INNER_DEAD:
                color = self.COL_DEAD
                label = "TOO CLOSE"
            elif dist <= self.SAFE_MAX:
                color = self.COL_SAFE
                label = "OK"
            elif dist <= self.WARN_MAX:
                color = self.COL_WARN
                label = "EDGE"
            else:
                color = self.COL_DANGER
                label = "OUT OF REACH"

            # Draw vertical pillar above object
            base  = [pos[0], pos[1], pos[2] + 0.04]
            top   = [pos[0], pos[1], pos[2] + 0.25]
            lid = p.addUserDebugLine(base, top, color,
                                     lineWidth=3)
            self._object_ids[name] = lid

            # Small cross at top of pillar
            cross_size = 0.015
            for dx, dy in [(cross_size, 0), (-cross_size, 0),
                           (0, cross_size), (0, -cross_size)]:
                lid2 = p.addUserDebugLine(
                    [pos[0], pos[1], top[2]],
                    [pos[0] + dx, pos[1] + dy, top[2]],
                    color, lineWidth=2
                )
                if name not in self._object_ids:
                    self._object_ids[name] = []
                # Store as list if multiple lines per object
                if isinstance(self._object_ids.get(name), int):
                    self._object_ids[name] = [self._object_ids[name], lid2]
                else:
                    self._object_ids[name].append(lid2)

    # ----------------------------------------------------------
    # Clear methods
    # ----------------------------------------------------------

    def clear_zones(self):
        for lid in self._zone_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._zone_ids = []

    def clear_objects(self):
        for name, lids in self._object_ids.items():
            if isinstance(lids, int):
                lids = [lids]
            for lid in lids:
                try:
                    p.removeUserDebugItem(lid)
                except Exception:
                    pass
        self._object_ids = {}

    def clear_table(self):
        for lid in self._table_ids:
            try:
                p.removeUserDebugItem(lid)
            except Exception:
                pass
        self._table_ids = []

    def clear(self):
        self.clear_zones()
        self.clear_objects()
        self.clear_table()

    # ----------------------------------------------------------
    # Drawing helpers
    # ----------------------------------------------------------

    def _draw_circle(self, radius, z, color, dashed=False):
        """Draw a circle of given radius at height z."""
        ids  = []
        step = 2 if dashed else 1   # skip segments for dashed effect

        for i in range(0, self.N_SEGMENTS, step):
            a1 = 2 * math.pi * i       / self.N_SEGMENTS
            a2 = 2 * math.pi * (i + 1) / self.N_SEGMENTS

            p1 = [radius * math.cos(a1), radius * math.sin(a1), z]
            p2 = [radius * math.cos(a2), radius * math.sin(a2), z]

            lid = p.addUserDebugLine(p1, p2, color,
                                     lineWidth=self.LINE_WIDTH)
            ids.append(lid)

        return ids

    def _draw_ring_fill(self, r_inner, r_outer, z, color, n_spokes=24):
        """Fill a ring with radial spoke lines."""
        ids = []
        for i in range(n_spokes):
            angle = 2 * math.pi * i / n_spokes
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            p1 = [r_inner * cos_a, r_inner * sin_a, z]
            p2 = [r_outer * cos_a, r_outer * sin_a, z]
            lid = p.addUserDebugLine(p1, p2, color, lineWidth=1)
            ids.append(lid)
        return ids

    def _draw_label_ticks(self):
        """Draw short tick marks at cardinal points on each zone ring."""
        ids      = []
        tick_len = 0.03
        z        = self.ZONE_HEIGHT

        for radius, color in [
            (self.INNER_DEAD, self.COL_DEAD),
            (self.SAFE_MAX,   self.COL_SAFE),
            (self.WARN_MAX,   self.COL_WARN),
        ]:
            for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                p1 = [radius * cos_a,             radius * sin_a,             z]
                p2 = [(radius + tick_len) * cos_a, (radius + tick_len) * sin_a, z]
                lid = p.addUserDebugLine(p1, p2, color, lineWidth=3)
                ids.append(lid)

        return ids
