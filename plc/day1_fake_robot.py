# =============================================================
#   DAY 1 STEP 4 - a fake robot against the real ladder program
#
#   Plays the robot side of the handshake over plc/bus.py's
#   ModbusBus, and the plant side (e-stop, guard, mode switch,
#   operator reset) the way a test bench would. Every decision is
#   made by the ladder program in OpenPLC; this script only reports
#   robot state and watches what the PLC does with it.
#
#   Scenarios, in order:
#     1. full job: READY -> JOB_START -> BUSY -> DONE -> JOB_START drops
#     2. unapproved recipe: JOB_START is never raised
#     3. guard opened mid-job: MOTION_PERMIT and JOB_START drop,
#        the cell fault latches, and only FAULT_RESET clears it
#
#   Usage: python -m plc.day1_fake_robot [host] [port]
#   Needs the OpenPLC runtime running plc/ladder. Not a pytest test.
# =============================================================

import sys
import threading
import time

from plc.bus import ModbusBus
from plc.signals import FaultCode

SCAN_S = 0.02            # task0 interval in the ladder project
WAIT_S = 3.0             # generous: watchdog is 1 s


class Check:
    def __init__(self):
        self.failed = []

    def __call__(self, ok, what):
        print(f"    {'PASS' if ok else 'FAIL'}  {what}")
        if not ok:
            self.failed.append(what)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def wait_for(bus, name, value, timeout=WAIT_S):
    """Poll until signal == value. Returns seconds taken, or None."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if bus.read(name) == value:
            return time.monotonic() - t0
        time.sleep(0.005)
    return None


def stays(bus, name, value, duration):
    """True if the signal holds `value` for the whole duration."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < duration:
        if bus.read(name) != value:
            return False
        time.sleep(0.01)
    return True


class FakeRobot:
    """The robot-owned half of signals.py, plus a live heartbeat."""

    def __init__(self, bus):
        self.bus = bus
        self._beat = 0
        self._running = True
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self):
        for name in ("READY", "BUSY", "DONE", "FAULT", "DEGRADED"):
            self.bus.write(name, False)
        for name in ("FAULT_CODE", "PARTS_OK", "PARTS_FAILED", "SLOT_MASK"):
            self.bus.write(name, 0)
        self.bus.write("ROBOT_READY", True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)

    def _heartbeat(self):
        while self._running:
            self._beat = (self._beat + 1) & 0xFFFF
            self.bus.write("ROBOT_HEARTBEAT", self._beat)
            time.sleep(0.05)

    def state(self, ready=False, busy=False, done=False, fault=False):
        self.bus.write("READY", ready)
        self.bus.write("BUSY", busy)
        self.bus.write("DONE", done)
        self.bus.write("FAULT", fault)


def plant_healthy(bus):
    bus.write("ESTOP_OK", True)
    bus.write("GUARD_CLOSED", True)
    bus.write("MODE_AUTO", True)


def pulse_reset(bus):
    bus.write("FAULT_RESET", True)
    time.sleep(10 * SCAN_S)
    bus.write("FAULT_RESET", False)
    time.sleep(5 * SCAN_S)


def run_job(bus, robot, check, recipe, label):
    """READY with a recipe loaded; wait for JOB_START and go BUSY."""
    bus.write("RECIPE", recipe)
    robot.state(ready=True)
    t = wait_for(bus, "JOB_START", True)
    check(t is not None, f"{label}: JOB_START raised for recipe {recipe}"
          + (f" ({t * 1000:.0f} ms after READY)" if t is not None else ""))
    if t is None:
        return None
    job_id = bus.read("JOB_ID")
    log(f"[PLC] JOB_START raised, recipe {bus.read('RECIPE')}, job {job_id}")
    bus.write("JOB_ID_ECHO", job_id)
    bus.write("PARTS_OK", 0)
    bus.write("PARTS_FAILED", 0)
    robot.state(busy=True)
    log("[ROBOT] BUSY")
    return job_id


def main(host, port):
    check = Check()
    bus = ModbusBus(host, port=port)
    robot = FakeRobot(bus)

    print(f"\n=== setup ({host}:{port}) ===")
    bus.write("RECIPE", 0)
    bus.write("FAULT_RESET", False)
    plant_healthy(bus)
    robot.start()
    b0 = bus.read("PLC_HEARTBEAT")
    time.sleep(0.2)
    b1 = bus.read("PLC_HEARTBEAT")
    check(b1 != b0, f"PLC_HEARTBEAT moving ({b0} -> {b1})")
    # A fault latched by an earlier run would block every job below.
    pulse_reset(bus)
    t = wait_for(bus, "MOTION_PERMIT", True)
    check(t is not None, "MOTION_PERMIT up with plant healthy and robot alive")

    # ---------------------------------------------------------------
    print("\n=== 1. full job handshake ===")
    job_id = run_job(bus, robot, check, recipe=2, label="job")
    if job_id is not None:
        time.sleep(0.5)                                   # "working"
        bus.write("PARTS_OK", 1)
        bus.write("PARTS_FAILED", 0)
        robot.state(done=True)
        log("[ROBOT] DONE, 1 part ok")
        t = wait_for(bus, "JOB_START", False)
        check(t is not None, "JOB_START dropped on DONE"
              + (f" ({t * 1000:.0f} ms)" if t is not None else ""))
        log("[PLC] JOB_START dropped")
        check(bus.read("JOB_ID") == (job_id + 1) & 0xFFFF,
              f"JOB_ID incremented ({job_id} -> {bus.read('JOB_ID')})")
        check(bus.read("RECIPE") == 0, "RECIPE consumed (back to 0)")
        robot.state(ready=True)
        log("[ROBOT] READY")
        check(stays(bus, "JOB_START", False, 0.5),
              "no second job without a new RECIPE")

    # ---------------------------------------------------------------
    print("\n=== 2. unapproved recipe ===")
    bus.write("RECIPE", 9)
    log("[PLANNER] RECIPE = 9 (not on the approved list)")
    check(stays(bus, "JOB_START", False, 1.0),
          "JOB_START never raised for recipe 9 - robot never asked")
    bus.write("RECIPE", 0)

    # ---------------------------------------------------------------
    print("\n=== 3. guard opened mid-job ===")
    job_id = run_job(bus, robot, check, recipe=1, label="job")
    if job_id is not None:
        time.sleep(0.3)
        bus.write("GUARD_CLOSED", False)
        t_open = time.monotonic()
        log("[PLANT] GUARD_CLOSED = false")
        t = wait_for(bus, "MOTION_PERMIT", False, timeout=1.0)
        check(t is not None, "MOTION_PERMIT dropped"
              + (f" ({t * 1000:.0f} ms after the guard write, "
                 f"scan is {SCAN_S * 1000:.0f} ms)" if t is not None else ""))
        log("[PLC] MOTION_PERMIT dropped: guard open")
        t = wait_for(bus, "JOB_START", False, timeout=1.0)
        check(t is not None, "JOB_START dropped - job stopped by the PLC")

        # The robot sees the permit gone and stops with a safety fault.
        bus.write("FAULT_CODE", FaultCode.SAFETY_STOP)
        robot.state(fault=True)
        log(f"[ROBOT] FAULT, code {FaultCode.SAFETY_STOP} (SAFETY_STOP)")
        time.sleep(0.1)
        check(bus.read("RECIPE") == 0, "RECIPE consumed on FAULT")
        log(f"    (guard open for {time.monotonic() - t_open:.2f} s)")

        print("\n--- the fault latch ---")
        # Reset while the guard is still open must not clear the latch.
        pulse_reset(bus)
        bus.write("GUARD_CLOSED", True)
        log("[PLANT] guard closed again")
        check(wait_for(bus, "MOTION_PERMIT", True) is not None,
              "MOTION_PERMIT back once the guard is closed")
        # Robot comes back READY (as if its own reset went through) and
        # a valid recipe is waiting: only the PLC's latch stands in the way.
        bus.write("FAULT_CODE", 0)
        robot.state(ready=True)
        bus.write("RECIPE", 1)
        check(stays(bus, "JOB_START", False, 1.0),
              "JOB_START held off: latch survived a reset sent while the "
              "guard was open")
        # CELL_FAULTED lives at %MX0.0, outside the Modbus map. Writing
        # the old address (coil 6) must have no effect on the latch.
        bus._client.write_coil(6, False, device_id=1)
        check(stays(bus, "JOB_START", False, 0.5),
              "writing coil 6 (old %QX0.6) does not clear the latch")

        pulse_reset(bus)
        log("[OPERATOR] FAULT_RESET pulsed, guard closed")
        t = wait_for(bus, "JOB_START", True)
        check(t is not None, "latch cleared by FAULT_RESET - JOB_START "
              "raised for the waiting recipe")
        if t is not None:
            # finish this job cleanly so the cell is left idle
            robot.state(busy=True)
            time.sleep(0.2)
            robot.state(done=True)
            wait_for(bus, "JOB_START", False)
            robot.state(ready=True)

    robot.stop()
    bus.write("RECIPE", 0)
    bus.close()

    print()
    if check.failed:
        print(f"{len(check.failed)} check(s) FAILED:")
        for f in check.failed:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed: the ladder program ran the handshake, "
          "refused the bad recipe, and stopped the job when the guard opened.")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 502
    main(host, port)
