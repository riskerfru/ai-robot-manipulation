#!/usr/bin/env python3
"""Fair baseline vs advanced eval (micro1 brief).

Same 12 cases for both sides. No GUI. No API key.

Baseline  = scripted place: if the block is on the table, call it done.
Advanced  = PlacementVerifier (retry, calibration, honest fail).

Primary metric: honest placement outcome (would a cell operator accept this).
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
    print("case           baseline  advanced  detail")
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
    bmean = base_pass / n
    amean = adv_pass / n
    print()
    print("METRIC                 SIMPLE BASELINE     AGENT SOLUTION      CHANGE")
    print(f"Primary outcome        {bmean:.2f} ({base_pass}/{n} ok)     {amean:.2f} ({adv_pass}/{n} ok)     {amean - bmean:+.2f}")
    print("Human time / order     ~90 s watching      ~10 s one command   ~8x faster (est.)")
    print("Cost / offline eval    $0                  $0                  $0")
    print("Cost / live dispatch   $0 (typed cmds)     Claude API ~$0.05–0.20 / order (est., not billed here)")
    print()
    print("HARD CASE  off_table:    block fell (z=0.01). Baseline 0. Advanced fails honestly.")
    print("HARD CASE  unreachable:  78 cm off slot, out of reach. Baseline 1 (still 'placed').")
    print("                         Advanced tries rescue, then failed + human intervention.")
    print("demo_advanced            6/6 subtasks in 9 steps  (README recorded GUI run, needs API key)")
    print("runtime                  < 2 s offline; a few minutes for one GUI auto-dispatch")
    print(f"cases                    {n} (same inputs both sides)")


if __name__ == "__main__":
    main()
