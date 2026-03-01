# =============================================================
#   LOGGER - Professional Logging System
#   Logs every command, position, error and event
#   Saves session for replay and debugging
# =============================================================

import logging
import os
import json
import time
from datetime import datetime


class RobotLogger:
    def __init__(self, config):
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = []

        os.makedirs("logs", exist_ok=True)

        self.logger = logging.getLogger("RobotAI")
        self.logger.setLevel(getattr(logging, config["log_level"]))

        if not self.logger.handlers:
            # Console handler with UTF-8 encoding
            import sys
            console_handler = logging.StreamHandler(
                open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
            )
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)

            if config["enabled"]:
                log_filename = f"logs/session_{self.session_id}.log"
                file_handler = logging.FileHandler(
                    log_filename, encoding='utf-8'
                )
                file_handler.setLevel(logging.DEBUG)
                file_format = logging.Formatter(
                    "%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
                file_handler.setFormatter(file_format)
                self.logger.addHandler(file_handler)

        self.info(f"Session started: {self.session_id}")

    def info(self, message):
        self.logger.info(message)
        self._save_event("INFO", message)

    def debug(self, message):
        self.logger.debug(message)
        self._save_event("DEBUG", message)

    def warning(self, message):
        self.logger.warning(message)
        self._save_event("WARNING", message)

    def error(self, message):
        self.logger.error(message)
        self._save_event("ERROR", message)

    def success(self, message):
        self.logger.info(f"[OK] {message}")
        self._save_event("SUCCESS", message)

    def log_command(self, user_input, ai_response=None):
        self.info(f"USER COMMAND: '{user_input}'")
        if ai_response:
            self.debug(f"AI RESPONSE: {ai_response}")
        self._save_event("COMMAND", {
            "user_input": user_input,
            "ai_response": ai_response,
            "timestamp": time.time()
        })

    def log_position(self, position, label="Position"):
        x, y, z = position
        msg = f"{label}: x={x:.4f}, y={y:.4f}, z={z:.4f}"
        self.debug(msg)
        self._save_event("POSITION", {
            "label": label,
            "x": x, "y": y, "z": z,
            "timestamp": time.time()
        })

    def log_movement(self, from_pos, to_pos, method="direct"):
        self.info(
            f"MOVE [{method}]: "
            f"({from_pos[0]:.3f},{from_pos[1]:.3f},{from_pos[2]:.3f}) -> "
            f"({to_pos[0]:.3f},{to_pos[1]:.3f},{to_pos[2]:.3f})"
        )
        self._save_event("MOVEMENT", {
            "from": list(from_pos),
            "to": list(to_pos),
            "method": method,
            "timestamp": time.time()
        })

    def log_grasp(self, object_name, success, position=None):
        status = "SUCCESS" if success else "FAILED"
        self.info(f"GRASP {status}: {object_name}")
        self._save_event("GRASP", {
            "object": object_name,
            "success": success,
            "position": list(position) if position else None,
            "timestamp": time.time()
        })

    def log_task(self, task_name, steps, status="started"):
        self.info(f"TASK {status.upper()}: '{task_name}' ({len(steps)} steps)")
        for i, step in enumerate(steps):
            self.debug(f"  Step {i+1}: {step}")
        self._save_event("TASK", {
            "task": task_name,
            "steps": steps,
            "status": status,
            "timestamp": time.time()
        })

    def log_error(self, error, context=""):
        self.error(f"ERROR in {context}: {str(error)}")
        self._save_event("ERROR", {
            "error": str(error),
            "context": context,
            "timestamp": time.time()
        })

    def log_detection(self, detected_objects):
        self.info(f"DETECTED {len(detected_objects)} objects:")
        for name, pos in detected_objects.items():
            self.info(f"  {name}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        self._save_event("DETECTION", {
            "objects": {
                name: list(pos) for name, pos in detected_objects.items()
            },
            "timestamp": time.time()
        })

    def log_robot_mode(self, mode, robot_type):
        self.info(f"MODE: {mode.upper()} | ROBOT: {robot_type}")
        self._save_event("CONFIG", {
            "mode": mode,
            "robot_type": robot_type,
            "timestamp": time.time()
        })

    def _save_event(self, event_type, data):
        self.session_log.append({
            "type": event_type,
            "data": data,
            "timestamp": time.time()
        })

    def save_session(self):
        replay_file = f"logs/replay_{self.session_id}.json"
        with open(replay_file, "w", encoding='utf-8') as f:
            json.dump({
                "session_id": self.session_id,
                "events": self.session_log
            }, f, indent=2)
        self.info(f"Session saved: {replay_file}")

    def get_session_summary(self):
        commands  = [e for e in self.session_log if e["type"] == "COMMAND"]
        movements = [e for e in self.session_log if e["type"] == "MOVEMENT"]
        grasps    = [e for e in self.session_log if e["type"] == "GRASP"]
        errors    = [e for e in self.session_log if e["type"] == "ERROR"]

        summary = (
            f"\n========================================"
            f"\n  SESSION SUMMARY: {self.session_id}"
            f"\n========================================"
            f"\n  Commands given:   {len(commands)}"
            f"\n  Movements made:   {len(movements)}"
            f"\n  Grasp attempts:   {len(grasps)}"
            f"\n  Errors:           {len(errors)}"
            f"\n========================================"
        )
        self.info(summary)
        return summary