#!/usr/bin/env python3
"""
Inspection gauge study: how well the mast camera measures a part.

Reference cubes of known size are set on the inspection pad at ten
angles and random positions, measured, and the extent-based size is
fitted with a correction

    size = extent + a + b * (|cos t| + |sin t| - 1)

on some sizes and checked on the others (sizes the fit never saw).
The constants go into mobile/cell.py GAUGE.

    python -m mobile.eval_inspection            camera with noise
    python -m mobile.eval_inspection --ideal    perfect pictures

Simulation only.
"""

import argparse
import contextlib
import io
import math
import random

import numpy as np
import pybullet as p

TRAIN = (41.0, 43.0, 45.0, 46.6, 48.0)
TEST = (42.0, 44.0, 46.0, 47.0, 50.0)
YAWS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.3, -0.6)


def collect(noise, frames=2, seed=5):
    from mobile import floor as F
    from mobile.agent import execute
    from mobile.cell import build_mobile_cell
    with contextlib.redirect_stdout(io.StringIO()):
        w = build_mobile_cell(gui=False, seed=1, parts=3, camera_noise=noise)
        cell = w.cell
        cell.scan()
        execute({"action": "goto", "station": "INSPECT"}, cell,
                lambda *a: None)
        cell.base.park_for_scan()
    finder = cell._ensure_finder()
    pad = cell.built["equipment"][("INSPECT", "pad")]
    lo, hi = p.getAABB(pad)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    (x0, x1), (y0, y1) = F.station_rect("INSPECT")
    rnd, rows, misses = random.Random(seed), [], 0
    for size in sorted(TRAIN + TEST):
        for yaw in YAWS:
            dx, dy = rnd.uniform(-0.006, 0.006), rnd.uniform(-0.006, 0.006)
            with contextlib.redirect_stdout(io.StringIO()):
                F.add_part(w.robot, "probe", size / 1000, F.RAW_RGB,
                           cx + dx, cy + dy, yaw)
            body = w.robot.objects["probe"]["id"]
            p.resetBasePositionAndOrientation(
                body, [cx + dx, cy + dy, hi[2] + size / 2000 + 0.002],
                p.getQuaternionFromEuler([0, 0, yaw]))
            for _ in range(200):
                p.stepSimulation()
            for _ in range(frames):
                blobs, _ = finder.find(area=((x0, x1), (y0, y1)))
                parts = [b for b in blobs if not b.container]
                if not parts:
                    misses += 1
                    continue
                b = min(parts, key=lambda b: math.hypot(b.x - cx, b.y - cy))
                rows.append((size, yaw, b.long * 1000, b.angle))
            p.removeBody(body)
            del w.robot.objects["probe"]
    return np.array(rows), misses


def fit(rows):
    true, ext, ang = rows[:, 0], rows[:, 2], rows[:, 3]
    t = ang % (math.pi / 2)
    feat = np.abs(np.cos(t)) + np.abs(np.sin(t)) - 1
    X = np.stack([np.ones_like(ext), feat], 1)
    train = np.isin(true, TRAIN)
    coef, *_ = np.linalg.lstsq(X[train], (true - ext)[train], rcond=None)
    err = ext + X @ coef - true
    raw = ext - true
    return coef, err[~train], raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ideal", action="store_true")
    args = ap.parse_args()
    rows, misses = collect(noise=not args.ideal)
    coef, err, raw = fit(rows)
    print(f"camera: {'ideal' if args.ideal else 'with noise'}; "
          f"{len(rows)} measurements, {misses} frames with no part found")
    print(f"before correction: mean error {raw.mean():+.2f} mm, "
          f"spread {raw.std():.2f} mm")
    print(f"correction a = {coef[0]:.3f} mm, b = {coef[1]:.3f} mm")
    print(f"on sizes not used in the fit: mean error {err.mean():+.2f} mm, "
          f"spread (1 sigma) {err.std():.2f} mm, worst {np.abs(err).max():.2f} mm")


if __name__ == "__main__":
    main()
