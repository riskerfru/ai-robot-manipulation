# =============================================================
#   SOFT PLC - stand-in for the cell PLC in simulation and tests
#
#   Plays the PLC side of the handshake on a SignalBus and lets a
#   test (or a demo script) open the guard, press the emergency
#   stop or stall the heartbeat. In a plant this role is taken by
#   the real cell PLC; nothing on the robot side changes.
# =============================================================

import threading
import time


class SoftPLC:

    def __init__(self, bus, scan_s=0.02):
        self.bus = bus
        self.scan_s = scan_s
        self._beat = 0
        self._beat_enabled = True
        self._thread = None
        self._running = False
        for name in ("ESTOP_OK", "GUARD_CLOSED", "MODE_AUTO",
                     "MOTION_PERMIT"):
            self.bus.write(name, True)
        self.bus.write("JOB_START", False)
        self.bus.write("FAULT_RESET", False)
        self.scan()

    # ---- cyclic scan -------------------------------------------
    def scan(self):
        if self._beat_enabled:
            self._beat = (self._beat + 1) & 0xFFFF
            self.bus.write("PLC_HEARTBEAT", self._beat)

    def start(self):
        """Run the scan in a background thread (demo / integration)."""
        self._running = True

        def loop():
            while self._running:
                self.scan()
                time.sleep(self.scan_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    # ---- operator / process actions ----------------------------
    def request_job(self, job_id, recipe):
        self.bus.write("JOB_ID", job_id)
        self.bus.write("RECIPE", recipe)
        self.bus.write("JOB_START", True)

    def acknowledge_done(self):
        self.bus.write("JOB_START", False)

    def reset_fault(self):
        self.bus.write("JOB_START", False)
        self.bus.write("FAULT_RESET", True)

    def release_reset(self):
        self.bus.write("FAULT_RESET", False)

    def open_guard(self):
        self.bus.write("GUARD_CLOSED", False)

    def close_guard(self):
        self.bus.write("GUARD_CLOSED", True)

    def press_estop(self):
        self.bus.write("ESTOP_OK", False)

    def release_estop(self):
        self.bus.write("ESTOP_OK", True)

    def stall_heartbeat(self):
        self._beat_enabled = False

    def resume_heartbeat(self):
        self._beat_enabled = True
