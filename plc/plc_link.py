# =============================================================
#   PLC LINK - the runner's side of the PLC, soft or real
#
#   SoftPlcLink : the Python SoftPLC (tests, offline path). It
#                 raises JOB_START itself when a job is requested
#                 and does not validate recipes or compute
#                 MOTION_PERMIT; the robot side catches both.
#   LivePlcLink : the OpenPLC ladder program over Modbus TCP. A
#                 job request is only a write to RECIPE; the ladder
#                 decides whether JOB_START is raised.
#
#   The live link also plays the plant: in this build the e-stop,
#   guard and mode switch are simulated, so their PLC inputs
#   (ESTOP_OK, GUARD_CLOSED, MODE_AUTO) are written over Modbus the
#   way a test bench would drive them.
#
#   PlcMonitor watches the PLC's outputs and prints the handshake
#   as it happens: [PLC] JOB_START raised, recipe 3 /
#   [PLC] MOTION_PERMIT dropped: guard open.
# =============================================================

import threading
import time

from plc.recipes import RECIPES

SCAN_S = 0.02                 # task0 interval in plc/ladder


class SoftPlcLink:
    live = False
    can_stall_heartbeat = True

    def __init__(self, soft_plc, bus):
        self.plc = soft_plc
        self.bus = bus

    def start(self):
        pass

    def stop(self):
        self.plc.stop()

    def request_job(self, job_id, recipe):
        self.plc.request_job(job_id, recipe)

    def withdraw(self):
        self.bus.write("JOB_START", False)
        self.bus.write("RECIPE", 0)

    def acknowledge_done(self):
        self.plc.acknowledge_done()

    def reset_fault(self):
        self.plc.reset_fault()

    def release_reset(self):
        self.plc.release_reset()

    def open_guard(self):
        self.plc.open_guard()

    def close_guard(self):
        self.plc.close_guard()

    def press_estop(self):
        self.plc.press_estop()

    def release_estop(self):
        self.plc.release_estop()

    def stall_heartbeat(self):
        self.plc.stall_heartbeat()

    def resume_heartbeat(self):
        self.plc.resume_heartbeat()


class LivePlcLink:
    live = True
    can_stall_heartbeat = False         # it is a real PLC

    def __init__(self, bus):
        self.bus = bus

    def start(self):
        """Healthy plant, no pending job, any stale fault latch cleared."""
        self.bus.write("RECIPE", 0)
        self.bus.write("ESTOP_OK", True)
        self.bus.write("GUARD_CLOSED", True)
        self.bus.write("MODE_AUTO", True)
        self.reset_fault()
        self.release_reset()

    def stop(self):
        self.bus.write("RECIPE", 0)
        self.bus.write("FAULT_RESET", False)

    def request_job(self, job_id, recipe):
        # JOB_ID belongs to the PLC; only the recipe is requested.
        self.bus.write("RECIPE", int(recipe) & 0xFFFF)

    def withdraw(self):
        # A refused request must not start later on its own, e.g. the
        # moment the guard closes.
        self.bus.write("RECIPE", 0)

    def acknowledge_done(self):
        pass                             # the ladder drops JOB_START on DONE

    def reset_fault(self):
        self.bus.write("FAULT_RESET", True)
        time.sleep(4 * SCAN_S)           # held long enough for a scan to see it

    def release_reset(self):
        self.bus.write("FAULT_RESET", False)
        time.sleep(2 * SCAN_S)

    def open_guard(self):
        self.bus.write("GUARD_CLOSED", False)

    def close_guard(self):
        self.bus.write("GUARD_CLOSED", True)

    def press_estop(self):
        self.bus.write("ESTOP_OK", False)

    def release_estop(self):
        self.bus.write("ESTOP_OK", True)

    def stall_heartbeat(self):
        raise RuntimeError("cannot stall a real PLC's heartbeat from here")

    def resume_heartbeat(self):
        pass


def permit_reason(v):
    """Why MOTION_PERMIT is low, from the PLC's own inputs."""
    causes = []
    if not v["ESTOP_OK"]:
        causes.append("e-stop pressed")
    if not v["GUARD_CLOSED"]:
        causes.append("guard open")
    if not v["MODE_AUTO"]:
        causes.append("mode not AUTO")
    # Inputs healthy but no permit: the ladder's robot watchdog.
    return ", ".join(causes) or "robot heartbeat lost (PLC watchdog)"


def refusal_reason(bus, recipe):
    """Best explanation for a request the PLC did not start."""
    v = bus.read_many(["ESTOP_OK", "GUARD_CLOSED", "MODE_AUTO",
                       "MOTION_PERMIT", "READY", "FAULT"])
    if recipe not in RECIPES:
        return f"recipe {recipe} is not on the approved list"
    if not v["MOTION_PERMIT"]:
        return f"MOTION_PERMIT is low: {permit_reason(v)}"
    if v["FAULT"] or not v["READY"]:
        return "robot not READY"
    return "cell fault latched in the PLC: FAULT_RESET needed"


class PlcMonitor:
    """Prints PLC output transitions from a background thread."""

    WATCH = ["MOTION_PERMIT", "JOB_START", "RECIPE", "JOB_ID",
             "ESTOP_OK", "GUARD_CLOSED", "MODE_AUTO"]

    def __init__(self, bus, say, period_s=SCAN_S):
        self.bus, self.say, self.period_s = bus, say, period_s
        self._stop = threading.Event()
        self._thread = None
        self._last = None
        self.events = []                 # (time, text) for tests

    def _emit(self, text):
        self.events.append((time.monotonic(), text))
        self.say(f"  [PLC] {text}")

    def _check(self):
        v = self.bus.read_many(self.WATCH)
        last, self._last = self._last, v
        if last is None:
            return
        if v["JOB_START"] and not last["JOB_START"]:
            self._emit(f"JOB_START raised, recipe {v['RECIPE']}, "
                       f"job {v['JOB_ID']}")
        elif last["JOB_START"] and not v["JOB_START"]:
            self._emit("JOB_START dropped")
        if last["MOTION_PERMIT"] and not v["MOTION_PERMIT"]:
            self._emit(f"MOTION_PERMIT dropped: {permit_reason(v)}")
        elif v["MOTION_PERMIT"] and not last["MOTION_PERMIT"]:
            self._emit("MOTION_PERMIT restored")

    def _loop(self):
        lost = False
        while not self._stop.wait(self.period_s):
            try:
                self._check()
                lost = False
            except Exception as e:
                if not lost:
                    self._emit(f"link error: {e}")
                lost = True

    def start(self):
        try:
            self._check()                # baseline
        except Exception:
            pass
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="plc-monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
