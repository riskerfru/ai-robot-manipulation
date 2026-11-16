#!/usr/bin/env python3
"""Verifier logic check: 12 fixed cases against a stub robot.

No physics, no GUI, no API key, under 2 seconds. The stub robot reports a
scripted part position and lands every retry exactly on target, so this
checks the verifier's decisions (accept, retry, fail honestly), not how well
placements work. For measured placement performance in the real simulation
run eval_physics.py.

Baseline  = no verification: if the part is on the table, call it placed.
Verifier  = PlacementVerifier (retry, calibration, honest failure).
"""

from __future__ import annotations

from place_verifier import PlacementVerifier


class FakeRobot:
    def __init__(self, pos, reachable=True, distance=0.4):
        self.pos = pos
        self.reachable = reachable
        self.distance = distance

    def get_object_position(self, name):
        return self.pos

    def check_reachability(self, x, y):
        return {"reachable": self.reachable, "distance": self.distance}

    def pick_object(self, name):
        return {"status": "success"}

    def place_object(self, x, y):
        self.pos = (x, y, 0.16)
        return {"status": "success"}

    def get_end_effector_position(self):
        return (0.3, 0.0, 0.5)

    def move_to_position(self, x, y, z, use_rrt=False):
        return bool(self.reachable)


# 12 fixed cases. Include easy, warn-band, retry, missing, and two hard cases.
CASES = [
    ("accurate_s1",   (0.60,  0.25, 0.16), 0.60,  0.25, True,  0.40),
    ("accurate_s2",   (0.60,  0.00, 0.16), 0.60,  0.00, True,  0.40),
    ("accurate_s3",   (0.60, -0.25, 0.16), 0.60, -0.25, True,  0.40),
    ("drift_3cm",     (0.60,  0.28, 0.16), 0.60,  0.25, True,  0.40),
    ("drift_6cm",     (0.60,  0.31, 0.16), 0.60,  0.25, True,  0.40),
    ("drift_7cm",     (0.63,  0.31, 0.16), 0.60,  0.25, True,  0.40),
    ("just_under_8",  (0.60,  0.329, 0.16), 0.60, 0.25, True,  0.40),
    ("drift_9cm",     (0.60,  0.34, 0.16), 0.60,  0.25, True,  0.40),
    ("drift_15cm",    (0.60,  0.40, 0.16), 0.60,  0.25, True,  0.40),
    ("missing",       None,                 0.60,  0.25, True,  0.40),
    ("off_table",     (0.60,  0.25, 0.01), 0.60,  0.25, True,  0.40),  # hard
    ("unreachable",   (0.95,  0.95, 0.16), 0.60,  0.25, False, 1.30), # hard
]


def baseline_ok(pos) -> bool:
    """Manual / scripted place: on-table and visible ⇒ success."""
    return pos is not None and pos[2] >= 0.05


def advanced_ok(pos, reachable, out) -> bool:
    """Operator-honest: recover when possible, fail when the block is gone or unreachable."""
    if pos is None or (pos[2] < 0.05) or not reachable:
        return out["status"] == "failed"
    return out["status"] in ("success", "recovered")


def main() -> None:
    base_pass = 0
    adv_pass = 0
    rows = []
    print("case           no-verify verifier  detail")
    for name, pos, tx, ty, reachable, dist in CASES:
        b = baseline_ok(pos)
        v = PlacementVerifier(FakeRobot(pos, reachable=reachable, distance=dist))
        out = v.verify_and_fix(name, tx, ty, slot_num=2)
        a = advanced_ok(pos, reachable, out)
        base_pass += int(b)
        adv_pass += int(a)
        detail = f"{out['status']}, {out['drift_cm']:.1f}cm"
        print(f"{name:14s}  {int(b)}         {int(a)}         {detail}")
        rows.append((name, b, a, detail))

    n = len(CASES)
    print()
    print(f"Counted as placed with no verification: {base_pass}/{n} "
          f"(three of them are 9, 15 and 78 cm off target)")
    print(f"Verifier decision correct: {adv_pass}/{n}")
    print("Stub robot, scripted positions. See eval_physics.py for physics.")


if __name__ == "__main__":
    main()
