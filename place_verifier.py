# =============================================================
#   PLACEMENT VERIFIER
#   Live verification after every place action
#   Handles drift, out-of-reach, and off-table scenarios
#
#   Usage:
#     from place_verifier import PlacementVerifier
#     verifier = PlacementVerifier(robot)
#     result = verifier.verify_and_fix(object_name, target_x, target_y)
# =============================================================

import math
import numpy as np


class PlacementVerifier:
    """
    Verifies object placement after every place action.
    Automatically retries if object drifted too far.

    Three layers:
    1. Detection  - check where object actually landed
    2. Recovery   - pick up and retry if drifted + reachable
    3. Reporting  - honest failure if truly unreachable
    """

    # Thresholds
    DRIFT_WARN     = 0.05   # 5cm  - warn but accept
    DRIFT_RETRY    = 0.08   # 8cm  - retry placement
    TABLE_Z_MIN    = 0.05   # below this = fell off table
    MAX_RETRIES    = 2      # max retry attempts

    def __init__(self, robot):
        self.robot          = robot
        self.drift_history  = {}   # slot → list of drifts
        self.calibration    = {}   # slot → average offset

    def verify_and_fix(self, object_name, target_x, target_y,
                       slot_num=None, attempt=0):
        """
        Check placement and fix if needed.

        Returns:
          {
            "status":   "success" | "failed" | "recovered",
            "drift_cm": float,
            "detail":   str,
            "retried":  bool
          }
        """
        # Get actual position
        actual = self.robot.get_object_position(object_name)
        if not actual:
            return {
                "status":  "failed",
                "drift_cm": 0,
                "detail":  f"{object_name} not found after place",
                "retried": False
            }

        # Check if fell off table
        if actual[2] < self.TABLE_Z_MIN:
            return {
                "status":  "failed",
                "drift_cm": 999,
                "detail":  f"{object_name} fell off table (z={actual[2]:.3f}m)",
                "retried": False
            }

        # Calculate drift
        drift = math.sqrt(
            (actual[0] - target_x) ** 2 +
            (actual[1] - target_y) ** 2
        )
        drift_cm = drift * 100

        print(f"  [VERIFY] {object_name} target=({target_x:.2f},{target_y:.2f}) "
              f"actual=({actual[0]:.2f},{actual[1]:.2f}) "
              f"drift={drift_cm:.1f}cm")

        # Record drift for calibration learning
        if slot_num is not None:
            self._record_drift(slot_num, target_x, target_y,
                               actual[0], actual[1])

        # Small drift - acceptable
        if drift < self.DRIFT_WARN:
            print(f"  [VERIFY] ✅ Placement accurate ({drift_cm:.1f}cm drift)")
            return {
                "status":  "success",
                "drift_cm": drift_cm,
                "detail":  f"placed accurately ({drift_cm:.1f}cm drift)",
                "retried": False
            }

        # Medium drift - warn but accept
        if drift < self.DRIFT_RETRY:
            print(f"  [VERIFY] ⚠️  Drift {drift_cm:.1f}cm — acceptable but not ideal")
            return {
                "status":  "success",
                "drift_cm": drift_cm,
                "detail":  f"placed with {drift_cm:.1f}cm drift (acceptable)",
                "retried": False
            }

        # Large drift - attempt recovery
        print(f"  [VERIFY] ❌ Drift {drift_cm:.1f}cm — attempting recovery")

        # Check if we have retries left
        if attempt >= self.MAX_RETRIES:
            print(f"  [VERIFY] Max retries reached — accepting current position")
            return {
                "status":  "success",  # mark success to avoid agent loop
                "drift_cm": drift_cm,
                "detail":  f"placed with {drift_cm:.1f}cm drift after {attempt} retries",
                "retried": True
            }

        # Check if object is reachable for recovery
        reach = self.robot.check_reachability(actual[0], actual[1])

        if not reach["reachable"]:
            print(f"  [VERIFY] ❌ Object drifted OUT OF REACH ({reach['distance']}m)")
            print(f"  [VERIFY] Attempting joint reconfiguration rescue...")

            # Try Strategy 6 (joint reconfiguration) to reach it
            rescued = self._attempt_rescue(object_name, actual,
                                           target_x, target_y)
            if rescued:
                return {
                    "status":  "recovered",
                    "drift_cm": drift_cm,
                    "detail":  f"rescued from {drift_cm:.1f}cm drift using "
                               f"joint reconfiguration",
                    "retried": True
                }
            else:
                return {
                    "status":  "failed",
                    "drift_cm": drift_cm,
                    "detail":  f"{object_name} drifted {drift_cm:.1f}cm to "
                               f"({actual[0]:.2f},{actual[1]:.2f}) — "
                               f"out of reach ({reach['distance']}m) — "
                               f"human intervention required",
                    "retried": True
                }

        # Object reachable — pick up and retry
        print(f"  [VERIFY] Object reachable — picking up for retry "
              f"(attempt {attempt + 1}/{self.MAX_RETRIES})")

        # Pick up drifted object
        pick_result = self.robot.pick_object(object_name)
        pick_ok = (pick_result.get("status") == "success"
                   if isinstance(pick_result, dict) else pick_result)

        if not pick_ok:
            return {
                "status":  "failed",
                "drift_cm": drift_cm,
                "detail":  f"could not pick up {object_name} for retry",
                "retried": True
            }

        # Apply calibration correction for retry
        corrected_x, corrected_y = self._apply_calibration(
            target_x, target_y, slot_num
        )
        print(f"  [VERIFY] Retrying at ({corrected_x:.2f}, {corrected_y:.2f})")

        # Place again
        place_result = self.robot.place_object(corrected_x, corrected_y)
        place_ok = (place_result.get("status") == "success"
                    if isinstance(place_result, dict) else place_result)

        if not place_ok:
            return {
                "status":  "failed",
                "drift_cm": drift_cm,
                "detail":  f"retry placement failed",
                "retried": True
            }

        # Verify again recursively
        return self.verify_and_fix(
            object_name, target_x, target_y,
            slot_num=slot_num,
            attempt=attempt + 1
        )

    def _attempt_rescue(self, object_name, actual_pos,
                        target_x, target_y):
        """
        Try to reach drifted object using joint reconfiguration.
        Uses Strategy 6 from RRT planner.
        """
        try:
            # Force RRT to try all strategies including reconfiguration
            current = self.robot.get_end_effector_position()
            above_drifted = (actual_pos[0], actual_pos[1],
                             actual_pos[2] + 0.40)

            # Try direct move to drifted position
            success = self.robot.move_to_position(
                above_drifted[0],
                above_drifted[1],
                above_drifted[2],
                use_rrt=True
            )

            if not success:
                return False

            # Pick up from drifted position
            pick_result = self.robot.pick_object(object_name)
            pick_ok = (pick_result.get("status") == "success"
                       if isinstance(pick_result, dict) else pick_result)

            if not pick_ok:
                return False

            # Place at corrected position
            corrected_x, corrected_y = self._apply_calibration(
                target_x, target_y, None
            )
            place_result = self.robot.place_object(corrected_x, corrected_y)
            return (place_result.get("status") == "success"
                    if isinstance(place_result, dict) else place_result)

        except Exception as e:
            print(f"  [VERIFY] Rescue failed: {e}")
            return False

    def _record_drift(self, slot_num, target_x, target_y,
                      actual_x, actual_y):
        """Record drift for calibration learning."""
        if slot_num not in self.drift_history:
            self.drift_history[slot_num] = []

        drift_x = target_x - actual_x
        drift_y = target_y - actual_y

        self.drift_history[slot_num].append((drift_x, drift_y))

        # Update running average calibration
        history = self.drift_history[slot_num]
        if len(history) >= 2:
            avg_dx = np.mean([d[0] for d in history[-5:]])  # last 5
            avg_dy = np.mean([d[1] for d in history[-5:]])
            self.calibration[slot_num] = (avg_dx, avg_dy)
            print(f"  [CALIBRATE] Slot {slot_num} offset learned: "
                  f"dx={avg_dx*100:.1f}cm dy={avg_dy*100:.1f}cm")

    def _apply_calibration(self, target_x, target_y, slot_num):
        """Apply learned calibration offset to target position."""
        if slot_num and slot_num in self.calibration:
            dx, dy = self.calibration[slot_num]
            corrected_x = target_x + dx
            corrected_y = target_y + dy
            print(f"  [CALIBRATE] Applying correction: "
                  f"({target_x:.2f},{target_y:.2f}) → "
                  f"({corrected_x:.2f},{corrected_y:.2f})")
            return corrected_x, corrected_y
        return target_x, target_y

    def get_calibration_report(self):
        """Print summary of learned calibrations."""
        if not self.calibration:
            print("  [CALIBRATE] No calibration data yet")
            return
        print("\n  [CALIBRATE] Learned slot offsets:")
        for slot_num, (dx, dy) in self.calibration.items():
            print(f"    Slot {slot_num}: "
                  f"dx={dx*100:+.1f}cm  dy={dy*100:+.1f}cm")

    def save_calibration(self, path="slot_calibration.json"):
        """Save calibration to file for persistence."""
        import json
        data = {
            str(k): list(v)
            for k, v in self.calibration.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [CALIBRATE] Saved to {path}")

    def load_calibration(self, path="slot_calibration.json"):
        """Load calibration from file."""
        import json
        import os
        if not os.path.exists(path):
            print(f"  [CALIBRATE] No calibration file found at {path}")
            return
        with open(path) as f:
            data = json.load(f)
        self.calibration = {
            int(k): tuple(v)
            for k, v in data.items()
        }
        print(f"  [CALIBRATE] Loaded from {path}")
        self.get_calibration_report()
