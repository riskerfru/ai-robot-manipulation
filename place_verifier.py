# =============================================================
#   PLACEMENT VERIFIER
#
#   After every place, measure where the part landed and decide:
#     in tolerance            -> success
#     out of tolerance        -> pick it up and place it again
#     retries exhausted       -> failed (never reported as placed)
#     part lost / off table   -> failed
#     cell cannot measure     -> failed, "unverified"
#
#   A repeatable offset per slot is learned and corrected
#   (calibration). A one-off disturbance is not treated as an
#   offset: the correction is applied only when recent errors
#   agree with each other.
#
#   Usage:
#     verifier = PlacementVerifier(robot)
#     result = verifier.verify_and_fix(part, target_x, target_y,
#                                      slot_num=2)
# =============================================================

import json
import math
import os


class PlacementVerifier:

    # Defaults. All of them are constructor parameters: the
    # tolerance belongs to the part drawing, not to this file.
    DRIFT_WARN = 0.05        # m, accept but warn above this
    DRIFT_RETRY = 0.08       # m, out of tolerance above this
    TABLE_Z_MIN = 0.05       # m, below this the part left the table
    MAX_RETRIES = 2
    MAX_CORRECTION = 0.15    # m, a larger learned offset is a fault
    MIN_SAMPLES = 2          # placements before an offset is trusted
    WINDOW = 5               # placements averaged per slot

    def __init__(self, robot, drift_warn=None, drift_retry=None,
                 max_retries=None, table_z_min=None,
                 use_calibration=True,
                 calibration_path="slot_calibration.json",
                 max_correction=None):
        self.robot = robot
        self.drift_warn = self.DRIFT_WARN if drift_warn is None else drift_warn
        self.drift_retry = (self.DRIFT_RETRY if drift_retry is None
                            else drift_retry)
        self.max_retries = (self.MAX_RETRIES if max_retries is None
                            else max_retries)
        self.table_z_min = (self.TABLE_Z_MIN if table_z_min is None
                            else table_z_min)
        self.max_correction = (self.MAX_CORRECTION if max_correction is None
                               else max_correction)
        if self.drift_warn > self.drift_retry:
            raise ValueError("drift_warn must not exceed drift_retry")
        self.use_calibration = use_calibration
        self.calibration_path = calibration_path
        self.drift_history = {}   # slot -> [(dx, dy)] commanded - measured
        self.calibration = {}     # slot -> (dx, dy) correction to add

    # ---- main entry ---------------------------------------------
    def verify_and_fix(self, object_name, target_x, target_y,
                       slot_num=None, attempt=0, commanded=None):
        """
        Measure the placement and correct it if needed.

        `commanded` is the position the robot was actually sent to
        when it differs from the target (a calibrated place).

        Returns a dict with:
          status    "success" | "failed"
          drift_cm  measured distance from the target
          detail    text for the operator and the audit trail
          retried   True if the part was picked up and placed again
          terminal  True if retries were exhausted (failed only)
          unverified True if the cell could not measure (failed only)
        """
        retried = attempt > 0

        if not getattr(self.robot, "can_measure_parts", True):
            return {
                "status": "failed", "drift_cm": 0.0, "retried": retried,
                "unverified": True,
                "detail": f"{object_name} placement could not be measured "
                          f"(no part sensing in this cell) — not verified",
            }

        actual = self.robot.get_object_position(object_name)
        if not actual:
            return {
                "status": "failed", "drift_cm": 0.0, "retried": retried,
                "detail": f"{object_name} not found after place",
            }

        if actual[2] < self.table_z_min:
            return {
                "status": "failed", "drift_cm": 999.0, "retried": retried,
                "detail": f"{object_name} fell off table "
                          f"(z={actual[2]:.3f}m)",
            }

        drift = math.hypot(actual[0] - target_x, actual[1] - target_y)
        drift_cm = drift * 100
        print(f"  [VERIFY] {object_name} "
              f"target=({target_x:.2f},{target_y:.2f}) "
              f"actual=({actual[0]:.2f},{actual[1]:.2f}) "
              f"drift={drift_cm:.1f}cm")

        if slot_num is not None:
            cx, cy = commanded or (target_x, target_y)
            self._record_drift(slot_num, cx, cy, actual[0], actual[1])

        if drift < self.drift_warn:
            print(f"  [VERIFY] ✅ Placement accurate ({drift_cm:.1f}cm drift)")
            return {
                "status": "success", "drift_cm": drift_cm, "retried": retried,
                "detail": f"placed accurately ({drift_cm:.1f}cm drift)",
            }

        if drift < self.drift_retry:
            print(f"  [VERIFY] ⚠️  Drift {drift_cm:.1f}cm — "
                  f"acceptable but not ideal")
            return {
                "status": "success", "drift_cm": drift_cm, "retried": retried,
                "detail": f"placed with {drift_cm:.1f}cm drift (acceptable)",
            }

        # Out of tolerance from here on.
        print(f"  [VERIFY] ❌ Drift {drift_cm:.1f}cm — out of tolerance")

        if attempt >= self.max_retries:
            print(f"  [VERIFY] ❌ Retries exhausted ({self.max_retries}) — "
                  f"reporting failure at {drift_cm:.1f}cm drift")
            return {
                "status": "failed", "drift_cm": drift_cm, "retried": retried,
                "terminal": True,
                "detail": f"{object_name} still {drift_cm:.1f}cm off target "
                          f"after {attempt} retries — "
                          f"human intervention required",
            }

        reach = self.robot.check_reachability(actual[0], actual[1])
        if not reach["reachable"]:
            print(f"  [VERIFY] ❌ Part is out of reach "
                  f"({reach['distance']}m) — attempting rescue")
            if not self._move_above(actual):
                return {
                    "status": "failed", "drift_cm": drift_cm,
                    "retried": retried,
                    "detail": f"{object_name} drifted {drift_cm:.1f}cm to "
                              f"({actual[0]:.2f},{actual[1]:.2f}) — "
                              f"out of reach ({reach['distance']}m) — "
                              f"human intervention required",
                }
        else:
            print(f"  [VERIFY] Part reachable — picking up for retry "
                  f"(attempt {attempt + 1}/{self.max_retries})")

        if not self._ok(self.robot.pick_object(object_name)):
            return {
                "status": "failed", "drift_cm": drift_cm, "retried": True,
                "detail": f"could not pick up {object_name} for retry",
            }

        place_x, place_y = self.corrected_target(target_x, target_y, slot_num)
        print(f"  [VERIFY] Retrying at ({place_x:.2f}, {place_y:.2f})")

        if not self._ok(self.robot.place_object(place_x, place_y)):
            return {
                "status": "failed", "drift_cm": drift_cm, "retried": True,
                "detail": "retry placement failed",
            }

        # The retry is measured like any other placement.
        return self.verify_and_fix(object_name, target_x, target_y,
                                   slot_num=slot_num, attempt=attempt + 1,
                                   commanded=(place_x, place_y))

    # ---- helpers ------------------------------------------------
    @staticmethod
    def _ok(result):
        if isinstance(result, dict):
            return result.get("status") == "success"
        return bool(result)

    def _move_above(self, actual_pos):
        """Try to get over a part that the reach check rejected."""
        try:
            return bool(self.robot.move_to_position(
                actual_pos[0], actual_pos[1], actual_pos[2] + 0.40,
                use_rrt=True))
        except Exception as exc:
            print(f"  [VERIFY] Rescue failed: {exc}")
            return False

    # ---- calibration --------------------------------------------
    def _record_drift(self, slot_num, commanded_x, commanded_y,
                      actual_x, actual_y):
        """Record where a part landed relative to where it was sent."""
        history = self.drift_history.setdefault(slot_num, [])
        history.append((commanded_x - actual_x, commanded_y - actual_y))
        del history[:-self.WINDOW]

        if len(history) < self.MIN_SAMPLES:
            return
        n = len(history)
        mx = sum(d[0] for d in history) / n
        my = sum(d[1] for d in history) / n
        spread = math.sqrt(sum((d[0] - mx) ** 2 + (d[1] - my) ** 2
                               for d in history) / n)
        size = math.hypot(mx, my)

        # Trust the offset only if the errors agree with each other.
        # Scattered errors are disturbances, not a miscalibrated slot.
        if spread < size <= self.max_correction:
            self.calibration[slot_num] = (mx, my)
            print(f"  [CALIBRATE] Slot {slot_num} offset learned: "
                  f"dx={mx * 100:.1f}cm dy={my * 100:.1f}cm")
        else:
            self.calibration.pop(slot_num, None)

    def corrected_target(self, target_x, target_y, slot_num):
        """Where to send the robot so the part lands on the target."""
        if (self.use_calibration and slot_num is not None
                and slot_num in self.calibration):
            dx, dy = self.calibration[slot_num]
            print(f"  [CALIBRATE] Applying correction: "
                  f"({target_x:.2f},{target_y:.2f}) → "
                  f"({target_x + dx:.2f},{target_y + dy:.2f})")
            return target_x + dx, target_y + dy
        return target_x, target_y

    _apply_calibration = corrected_target

    def get_calibration_report(self):
        if not self.calibration:
            print("  [CALIBRATE] No calibration data yet")
            return
        print("\n  [CALIBRATE] Learned slot offsets:")
        for slot_num, (dx, dy) in self.calibration.items():
            print(f"    Slot {slot_num}: "
                  f"dx={dx * 100:+.1f}cm  dy={dy * 100:+.1f}cm")

    def save_calibration(self, path=None):
        path = path or self.calibration_path
        if not path:
            return
        with open(path, "w") as f:
            json.dump({str(k): list(v) for k, v in self.calibration.items()},
                      f, indent=2)
        print(f"  [CALIBRATE] Saved to {path}")

    def load_calibration(self, path=None):
        path = path or self.calibration_path
        if not path or not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.calibration = {int(k): tuple(v) for k, v in data.items()}
        print(f"  [CALIBRATE] Loaded from {path}")
        self.get_calibration_report()
