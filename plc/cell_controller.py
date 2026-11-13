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
# =============================================================

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

    def __init__(self, bus, agent, safety, audit=None, interlocks=None):
        self.bus = bus
        self.agent = agent
        self.safety = safety
        self.audit = audit or NullAudit()
        self.interlocks = interlocks       # PLCInterlocks, for heartbeat
        self.state = CellState.READY
        self.fault_code = FaultCode.NONE
        self.last_result = None
        self._beat = 0
        self._prev_reset = False
        self._publish()

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
        self._beat = (self._beat + 1) & 0xFFFF
        healthy = self.safety.state is SafetyState.RUN
        w = self.bus.write
        w("READY", self.state is CellState.READY and healthy)
        w("BUSY",  self.state is CellState.BUSY)
        w("DONE",  self.state is CellState.DONE)
        w("FAULT", self.state is CellState.FAULT)
        w("FAULT_CODE", self.fault_code)
        w("DEGRADED", bool(getattr(self.agent, "degraded", False)))
        w("ROBOT_READY", healthy)
        w("ROBOT_HEARTBEAT", self._beat)
        robot = getattr(self.agent, "robot", None)
        occupants = getattr(robot, "slot_occupants", {}) or {}
        mask = 0
        for slot in occupants:
            mask |= 1 << (int(slot) - 1)
        w("SLOT_MASK", mask)
