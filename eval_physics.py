#!/usr/bin/env python3
# =============================================================
#   PHYSICS EVALUATION
#
#   Runs the real PyBullet cell headless, many times, and scores
#   every placement against ground truth from the physics engine.
#   No language model is involved: orders run as recipes, so this
#   measures the execution, verification and safety layers that
#   every order (typed, recipe or language-model) passes through.
#
#   Arms (what happens after the gripper opens)
#     open_loop      report success, never measure
#     verify_only    measure; out of tolerance = honest failure
#     verify_retry   measure; re-pick and re-place up to 2 times
#     retry_calib    as verify_retry, plus learned per-slot offset
#
#   Conditions
#     nominal        randomised part positions, no disturbance
#     disturbed      30 % of releases: part knocked 3-15 cm in a
#                    random direction (slip, vibration, contact)
#     biased         every release lands 6 cm off in +y with 1 cm
#                    noise (tool-centre-point calibration error)
#
#   The disturbances are injected by this script after the
#   gripper opens. They are a test input, not an emergent result.
#
#   Usage:  python eval_physics.py --orders 40 --workers 2
# =============================================================

import argparse
import contextlib
import io
import json
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

TOL = 0.02                      # placement tolerance, metres
ARMS = ["open_loop", "verify_only", "verify_retry", "retry_calib"]
CONDITIONS = ["nominal", "disturbed", "biased"]
RECIPE_SLOT = {"red": 1, "blue": 2, "green": 3}


class OpenLoopVerifier:
    """Stands in for 'no verification': believes every placement."""
    calibration, drift_history = {}, {}

    def verify_and_fix(self, name, tx, ty, slot_num=None, attempt=0,
                       commanded=None):
        return {"status": "success", "drift_cm": 0.0,
                "detail": "not measured", "retried": False}

    def load_calibration(self, path=None): pass
    def save_calibration(self, path=None): pass
    def get_calibration_report(self): pass


def make_verifier(arm, robot):
    from place_verifier import PlacementVerifier
    if arm == "open_loop":
        return OpenLoopVerifier()
    common = dict(drift_warn=0.6 * TOL, drift_retry=TOL,
                  calibration_path=None)
    if arm == "verify_only":
        return PlacementVerifier(robot, max_retries=0,
                                 use_calibration=False, **common)
    if arm == "verify_retry":
        return PlacementVerifier(robot, max_retries=2,
                                 use_calibration=False, **common)
    return PlacementVerifier(robot, max_retries=2,
                             use_calibration=True, **common)


def install_disturbance(robot, condition, rng):
    """Wrap the backend's place_object with the condition's disturbance."""
    import pybullet as p
    original = robot.place_object
    counter = {"places": 0}

    def shift(name, dx, dy):
        body = robot.objects[name]["id"]
        pos, orn = p.getBasePositionAndOrientation(body)
        p.resetBasePositionAndOrientation(
            body, [pos[0] + dx, pos[1] + dy, pos[2] + 0.005], orn)
        p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
        for _ in range(60):
            robot.step()

    def place(x, y, stack_on=None):
        held = robot.grasped_object
        counter["places"] += 1
        out = original(x, y, stack_on=stack_on)
        ok = out.get("status") == "success" if isinstance(out, dict) else out
        if ok and held:
            if condition == "disturbed" and rng.random() < 0.30:
                r, a = rng.uniform(0.03, 0.15), rng.uniform(0, 2 * math.pi)
                shift(held, r * math.cos(a), r * math.sin(a))
            elif condition == "biased":
                shift(held, rng.gauss(0, 0.01), 0.06 + rng.gauss(0, 0.01))
        return out

    robot.place_object = place
    return counter


def run_order(arm, condition, seed, carry):
    """One order of three parts. Returns per-part records."""
    import numpy as np
    import pybullet as p
    from cell.factory import build_sim_cell
    from plc.recipes import RECIPES
    from utils.logger import NullLogger

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    cell = build_sim_cell(NullLogger(), seed=seed, randomize=True,
                          with_plc=False)
    try:
        counter = install_disturbance(cell.robot, condition,
                                      random.Random(seed * 7919 + 1))
        verifier = make_verifier(arm, cell.supervised)
        if arm == "retry_calib":
            verifier.drift_history = carry.setdefault("history", {})
            verifier.calibration = carry.setdefault("calibration", {})
        cell.agent.verifier = verifier
        result = cell.agent.run("dispatch",
                                subtasks=list(RECIPES[1]["subtasks"]))
        reported = set(cell.robot.slot_occupants.values())
        records = []
        for part, slot in RECIPE_SLOT.items():
            sx, sy = cell.robot.conveyor_slots[slot]
            pos = cell.robot.get_object_position(part)
            drift = math.hypot(pos[0] - sx, pos[1] - sy)
            on_table = pos[2] > 0.05
            records.append({
                "part": part,
                "reported_ok": part in reported,
                "in_spec": bool(on_table and drift <= TOL),
                "drift_cm": round(100 * drift, 2),
            })
        return {"arm": arm, "condition": condition, "seed": seed,
                "order_success": bool(result["success"]),
                "places": counter["places"],
                "safety_denied": cell.safety.denied_count,
                "parts": records}
    finally:
        p.disconnect()


def worker(job):
    arm, condition, seeds = job
    carry, out = {}, []
    sink = io.StringIO()
    for seed in seeds:
        sink.seek(0); sink.truncate()
        try:
            with contextlib.redirect_stdout(sink):
                out.append(run_order(arm, condition, seed, carry))
        except Exception as exc:
            out.append({"arm": arm, "condition": condition, "seed": seed,
                        "error": repr(exc)})
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    centre = (ph + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise(orders):
    table = {}
    for cond in CONDITIONS:
        for arm in ARMS:
            rows = [o for o in orders if o.get("arm") == arm
                    and o.get("condition") == cond and "error" not in o]
            parts = [pt for o in rows for pt in o["parts"]]
            n = len(parts)
            if n == 0:
                continue
            good = sum(pt["reported_ok"] and pt["in_spec"] for pt in parts)
            escaped = sum(pt["reported_ok"] and not pt["in_spec"] for pt in parts)
            caught = sum(not pt["reported_ok"] and not pt["in_spec"] for pt in parts)
            alarm = sum(not pt["reported_ok"] and pt["in_spec"] for pt in parts)
            table[(cond, arm)] = {
                "orders": len(rows), "parts": n,
                "good": good, "escaped": escaped,
                "caught": caught, "false_alarm": alarm,
                "good_ci": wilson(good, n), "escaped_ci": wilson(escaped, n),
                "places_per_part": sum(o["places"] for o in rows) / n,
                "mean_drift_cm": sum(pt["drift_cm"] for pt in parts) / n,
                "errors": sum(1 for o in orders if o.get("arm") == arm
                              and o.get("condition") == cond and "error" in o),
            }
    return table


def render(table, orders_per_cell):
    lines = [f"Placement tolerance: {100 * TOL:.0f} cm. "
             f"{orders_per_cell} orders x 3 parts per cell. "
             f"95% Wilson intervals in brackets.", ""]
    for cond in CONDITIONS:
        lines += [f"### {cond}", "",
                  "| Arm | Good parts | Escaped defects | Caught | "
                  "False alarms | Places / part | Mean drift |",
                  "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            r = table.get((cond, arm))
            if not r:
                continue
            n = r["parts"]
            g, e = r["good_ci"], r["escaped_ci"]
            lines.append(
                f"| {arm} | {r['good']}/{n} = {100 * r['good'] / n:.0f}% "
                f"[{100 * g[0]:.0f}-{100 * g[1]:.0f}] | "
                f"{r['escaped']}/{n} = {100 * r['escaped'] / n:.0f}% "
                f"[{100 * e[0]:.0f}-{100 * e[1]:.0f}] | "
                f"{r['caught']} | {r['false_alarm']} | "
                f"{r['places_per_part']:.2f} | {r['mean_drift_cm']:.1f} cm |")
        lines.append("")
    lines += [
        "Good parts: reported placed and truly within tolerance.",
        "Escaped defects: reported placed but out of tolerance "
        "(the part that jams the line downstream).",
        "Caught: out of tolerance and reported as a failure.",
        "False alarms: within tolerance but reported as a failure.",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, default=40)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default="eval_out")
    ap.add_argument("--seed", type=int, default=20261112)
    args = ap.parse_args()

    seeds = [args.seed + i for i in range(args.orders)]
    jobs = [(arm, cond, seeds) for cond in CONDITIONS for arm in ARMS]
    t0 = time.time()
    orders = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, chunk in enumerate(pool.map(worker, jobs)):
            orders.extend(chunk)
            print(f"  [{i + 1}/{len(jobs)}] {jobs[i][1]:10s} {jobs[i][0]:13s} "
                  f"done  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "physics_eval_raw.json"), "w") as f:
        json.dump({"tolerance_m": TOL, "seeds": seeds, "orders": orders}, f)
    report = render(summarise(orders), args.orders)
    errors = [o for o in orders if "error" in o]
    if errors:
        report += f"\n\n{len(errors)} order(s) raised and were excluded: " \
                  f"{sorted({e['error'] for e in errors})}"
    with open(os.path.join(args.out, "physics_eval.md"), "w") as f:
        f.write(report + "\n")
    print("\n" + report)
    print(f"\nruntime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
