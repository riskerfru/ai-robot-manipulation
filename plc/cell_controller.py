# =============================================================
#   CELL CONTROLLER - robot side of the PLC handshake
#
#        PLC                         robot cell controller
#   ------------------------------------------------------------
#                                    READY = 1
#   JOB_ID, RECIPE, JOB_START = 1 ->
#                                    READY = 0, BUSY = 1
#                                    ... runs the job ...
#                                    BUSY = 0, DONE = 1   (or FAULT = 1)
#   JOB_START = 0                 ->
#                                    DONE = 0, READY = 1
#
#   After FAULT the controller stays faulted until the PLC drops
#   JOB_START and pulses FAULT_RESET, and the safety supervisor
#   accepts the reset (i.e. the cause has actually cleared).
#
#   ROBOT_HEARTBEAT keeps counting while a job runs. The agent's
#   run() blocks for the whole job, so a background thread beats
#   for its duration; otherwise the PLC's watchdog (1 s in the
#   ladder program) would see the robot as dead and drop
#   MOTION_PERMIT mid-motion. A runner that leaves the controller
#   unpolled between jobs can keep the beat going with
#   start_heartbeat().
# =============================================================

import threading
from enum import Enum

from audit.trail import NullAudit
from plc.recipes import get_recipe
from plc.signals import FaultCode
from safety.supervisor import Deny, SafetyState


class CellState(str, Enum):
    READY = "READY"
    BUSY  = "BUSY"
    DONE  = "DONE"
    FAULT = "FAULT"


class CellController:

    def __init__(self, bus, agent, safety, audit=None, interlocks=None,
                 heartbeat_period_s=0.1):
        self.bus = bus
        self.agent = agent
        self.safety = safety
        self.audit = audit or NullAudit()
        self.interlocks = interlocks       # PLCInterlocks, for heartbeat
        self.state = CellState.READY
        self.fault_code = FaultCode.NONE
        self.last_result = None
        self.heartbeat_period_s = heartbeat_period_s
        self._beat = 0
        self._beat_lock = threading.Lock()
        self._beater = None                # (thread, stop event) or None
        self._prev_reset = False
        self._publish()

    # ---- heartbeat -----------------------------------------------
    def _beat_once(self):
        with self._beat_lock:
            self._beat = (self._beat + 1) & 0xFFFF
            self.bus.write("ROBOT_HEARTBEAT", self._beat)

    def _beat_loop(self, stop):
        while not stop.wait(self.heartbeat_period_s):
            try:
                self._beat_once()
            except Exception:
                # Fieldbus hiccup: skip this beat. If the link stays
                # down the PLC's watchdog drops the permit, which is
                # the right outcome.
                pass

    def start_heartbeat(self):
        """Beat in the background until stop_heartbeat(). Idempotent."""
        if self._beater is not None:
            return False
        stop = threading.Event()
        thread = threading.Thread(target=self._beat_loop, args=(stop,),
                                  name="robot-heartbeat", daemon=True)
        self._beater = (thread, stop)
        thread.start()
        return True

    def stop_heartbeat(self):
        if self._beater is None:
            return
        thread, stop = self._beater
        self._beater = None
        stop.set()
        thread.join(timeout=2 * self.heartbeat_period_s + 1.0)

    # ---- one scan ------------------------------------------------
    def poll(self):
        """Call cyclically. Returns the state after this scan."""
        if self.interlocks is not None:
            self.interlocks.read()          # keep heartbeat tracking fresh
        start = bool(self.bus.read("JOB_START"))
        reset = bool(self.bus.read("FAULT_RESET"))
        reset_edge = reset and not self._prev_reset
        self._prev_reset = reset

        if self.state is CellState.READY:
            if self.safety.state is not SafetyState.RUN:
                # A stop while idle (guard opened between jobs).
                if reset_edge:
                    self._try_reset()
            elif start:
                self._run_job()

        elif self.state is CellState.DONE:
            if not start:
                self._set_state(CellState.READY)

        elif self.state is CellState.FAULT:
            if reset_edge and not start:
                self._try_reset()

        self._publish()
        return self.state

    # ---- job -----------------------------------------------------
    def _run_job(self):
        # Beat for the length of the job unless a runner already is.
        started_here = self.start_heartbeat()
        try:
            self._run_job_body()
        finally:
            if started_here:
                self.stop_heartbeat()

    def _run_job_body(self):
        job_id = int(self.bus.read("JOB_ID"))
        recipe_no = int(self.bus.read("RECIPE"))
        self.audit.job_id = job_id
        self.bus.write("JOB_ID_ECHO", job_id)
        self.bus.write("PARTS_OK", 0)
        self.bus.write("PARTS_FAILED", 0)
        self._set_state(CellState.BUSY)
        self._publish()

        recipe = get_recipe(recipe_no)
        if recipe is None:
            self._fault(FaultCode.UNKNOWN_RECIPE,
                        f"recipe {recipe_no} is not defined")
            return

        self.audit.log("job_start", job_id=job_id, recipe=recipe_no,
                       name=recipe["name"])
        try:
            result = self.agent.run(recipe["name"],
                                    subtasks=list(recipe["subtasks"]))
        except Exception as exc:
            self._fault(FaultCode.INTERNAL, f"job raised: {exc}")
            return
        self.last_result = result

        placed = sum(1 for t in result.get("completed", [])
                     if "place" in str(t) or "insert" in str(t))
        failed = len(result.get("failed", []))
        self.bus.write("PARTS_OK", placed)
        self.bus.write("PARTS_FAILED", failed)

        if self.safety.state is not SafetyState.RUN:
            trip = self.safety.last_trip
            if trip and trip[0] is Deny.INTERLOCK_STALE:
                self._fault(FaultCode.PLC_HEARTBEAT,
                            "PLC heartbeat or fieldbus lost during the job")
            else:
                self._fault(FaultCode.SAFETY_STOP,
                            trip[1] if trip else "safety stop")
        elif not result.get("success"):
            self._fault(FaultCode.PART_NOT_PLACED,
                        result.get("summary", "job incomplete"))
        else:
            self.audit.log("job_done", job_id=job_id, parts_ok=placed)
            self._set_state(CellState.DONE)

    def _fault(self, code, reason):
        self.fault_code = code
        self.audit.log("job_fault", code=code, reason=reason)
        self._set_state(CellState.FAULT)

    def _try_reset(self):
        decision = self.safety.reset(operator="plc_fault_reset")
        self.audit.log("fault_reset", **decision.as_dict())
        if decision.allowed:
            self.fault_code = FaultCode.NONE
            stop = getattr(self.agent, "e_stop", None)
            if stop is not None:
                stop.reset()
            self._set_state(CellState.READY)

    # ---- outputs -------------------------------------------------
    def _set_state(self, state):
        if state is not self.state:
            self.audit.log("cell_state", previous=self.state.value,
                           state=state.value)
        self.state = state

    def _publish(self):
        healthy = self.safety.state is SafetyState.RUN
        w = self.bus.write
        w("READY", self.state is CellState.READY and healthy)
        w("BUSY",  self.state is CellState.BUSY)
        w("DONE",  self.state is CellState.DONE)
        w("FAULT", self.state is CellState.FAULT)
        w("FAULT_CODE", self.fault_code)
        w("DEGRADED", bool(getattr(self.agent, "degraded", False)))
        w("ROBOT_READY", healthy)
        self._beat_once()
        robot = getattr(self.agent, "robot", None)
        occupants = getattr(robot, "slot_occupants", {}) or {}
        mask = 0
        for slot in occupants:
            mask |= 1 << (int(slot) - 1)
        w("SLOT_MASK", mask)
