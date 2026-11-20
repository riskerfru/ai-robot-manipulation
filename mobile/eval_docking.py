# =============================================================
#   Does the mast camera matter? Measure it.
#
#   The base parks up to 15 mm / 1.5 deg off its mark each time.
#   For each trial the robot carries a blank from STORAGE, docks at
#   MACHINE and loads the fixture, then the error between the part
#   and the fixture centre is measured from the simulator's ground
#   truth. Two ways of aiming:
#
#     vision         - the mast camera re-measures the fixture after
#                      docking and the arm aims at what it sees
#     dead reckoning - the arm aims at the fixture's map position,
#                      trusting that the base stopped where it meant to
#
#   The machine only clamps a part within 5 mm of centre.
#
#       python -m mobile.eval_docking --trials 10
# =============================================================

import argparse
import contextlib
import io
import statistics
import sys

from mobile.cell import SEAT_TOL_M, build_mobile_cell


def trials(use_vision, n, seed):
    with contextlib.redirect_stdout(io.StringIO()):
        w = build_mobile_cell(seed=seed, parts=1, bad_part=False,
                              use_vision=use_vision)
        c, b = w.cell, w.base
        c.scan()
        part = next(t for t in c.at_station() if not t.container)
        c.pick(part)
        errors, parking = [], []
        for _ in range(n):
            b.stow()
            out = b.go_to_station("MACHINE")
            parking.append(out["parking_error_mm"])
            c.scan()
            r = c.place("fixture")
            if r.get("status") != "success":
                print("   place failed:", r.get("detail"), file=sys.stderr)
                errors.append(None)
                break
            errors.append(r["placement_error_mm"])
            c.machine.update(part=None)         # no cycle in this test
            c.pick(c.parts_in("fixture")[0])
            b.stow()
            b.go_to_station("STORAGE")
        w.robot.disconnect()
    return errors, parking


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    tol = SEAT_TOL_M * 1000
    print(f"{args.trials} dockings at MACHINE, fixture accepts <= "
          f"{tol:.0f} mm off centre\n")
    for name, vision in (("vision", True), ("dead reckoning", False)):
        errs, parking = trials(vision, args.trials, args.seed)
        ok = [e for e in errs if e is not None] or [float("nan")]
        seated = sum(e <= tol for e in ok)
        print(f"{name:15} parking error mean {statistics.mean(parking):5.1f}"
              f" mm | placement error mean {statistics.mean(ok):5.1f} mm, "
              f"max {max(ok):5.1f} mm | seated {seated}/{len(errs)}")


if __name__ == "__main__":
    main()
