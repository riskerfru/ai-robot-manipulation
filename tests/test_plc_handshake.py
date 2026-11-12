"""PLC handshake: the PLC is the master, faults need a real reset."""

from audit.trail import AuditTrail
from plc.bus import InMemoryBus
from plc.cell_controller import CellController, CellState
from plc.interlocks import PLCInterlocks
from plc.signals import BY_NAME, SIGNALS, FaultCode
from plc.soft_plc import SoftPLC
from safety.supervisor import MotionCommand, SafetyState, SafetySupervisor
from tests.fakes import FakeClock


class FakeAgent:
    """Stands in for the agent; optionally trips a hazard mid-job."""

    def __init__(self, safety, success=True, during=None):
        self.safety, self.success, self.during = safety, success, during
        self.degraded = False
        self.runs = []

        class Robot:
            slot_occupants = {}
        self.robot = Robot()

    def run(self, order, subtasks=None):
        self.runs.append((order, subtasks))
        if self.during:
            self.during()
        ok = self.safety.authorize(MotionCommand("home")).allowed
        if ok and self.success:
            self.robot.slot_occupants = {1: "red", 2: "blue", 3: "green"}
            return {"success": True, "failed": [], "summary": "done",
                    "completed": ["pick", "place_slot"] * 3}
        return {"success": False, "completed": [],
                "failed": [{"task": "x", "reason": "y"}], "summary": "failed"}


def make_cell(success=True, during=None):
    bus = InMemoryBus()
    plc = SoftPLC(bus)
    clock = FakeClock()
    interlocks = PLCInterlocks(bus, clock=clock)
    interlocks.read(); plc.scan(); interlocks.read()   # heartbeat seen moving
    assert interlocks.plc_alive
    safety = SafetySupervisor(interlock_source=interlocks, clock=clock)
    agent = FakeAgent(safety, success, during)
    audit = AuditTrail()
    ctrl = CellController(bus, agent, safety, audit=audit,
                          interlocks=interlocks)
    return bus, plc, ctrl, safety, agent, audit, clock


def test_signal_map_has_no_address_collisions():
    seen = set()
    for s in SIGNALS:
        key = (s.kind, s.address)
        assert key not in seen, s.name
        seen.add(key)
        assert (s.address < 16) == (s.writer == "plc"), s.name
    assert len(BY_NAME) == len(SIGNALS)


def test_robot_never_starts_without_job_start():
    bus, plc, ctrl, _, agent, _, _ = make_cell()
    for _ in range(5):
        plc.scan()
        assert ctrl.poll() is CellState.READY
    assert agent.runs == [] and bus.read("READY")


def test_full_handshake_cycle():
    bus, plc, ctrl, _, agent, _, _ = make_cell()
    plc.request_job(job_id=501, recipe=1)
    assert ctrl.poll() is CellState.DONE
    assert bus.read("DONE") and not bus.read("READY")
    assert bus.read("JOB_ID_ECHO") == 501
    assert bus.read("PARTS_OK") == 3 and bus.read("SLOT_MASK") == 0b111
    # DONE is held until the PLC drops JOB_START: no double start.
    assert ctrl.poll() is CellState.DONE and len(agent.runs) == 1
    plc.acknowledge_done()
    assert ctrl.poll() is CellState.READY and bus.read("READY")


def test_unknown_recipe_faults():
    bus, plc, ctrl, *_ = make_cell()
    plc.request_job(job_id=1, recipe=99)
    assert ctrl.poll() is CellState.FAULT
    assert bus.read("FAULT_CODE") == FaultCode.UNKNOWN_RECIPE


def test_failed_placement_faults_and_never_reports_done():
    bus, plc, ctrl, *_ = make_cell(success=False)
    plc.request_job(job_id=2, recipe=1)
    assert ctrl.poll() is CellState.FAULT
    assert bus.read("FAULT_CODE") == FaultCode.PART_NOT_PLACED
    assert not bus.read("DONE")


def test_guard_opened_mid_job_faults_and_reset_needs_guard_closed():
    cell = {}
    bus, plc, ctrl, safety, _, audit, _ = make_cell(
        during=lambda: cell["plc"].open_guard())
    cell["plc"] = plc
    plc.request_job(job_id=3, recipe=1)
    assert ctrl.poll() is CellState.FAULT
    assert bus.read("FAULT_CODE") == FaultCode.SAFETY_STOP
    assert safety.state is SafetyState.PROTECTIVE_STOP

    plc.reset_fault()                      # guard still open
    assert ctrl.poll() is CellState.FAULT
    plc.release_reset(); ctrl.poll()

    plc.close_guard(); plc.scan()
    assert ctrl.poll() is CellState.FAULT  # closing alone is not a reset
    plc.reset_fault()
    assert ctrl.poll() is CellState.READY and bus.read("READY")
    assert [e["data"]["allowed"] for e in audit.events("fault_reset")] \
        == [False, True]


def test_reset_is_ignored_while_job_start_is_still_high():
    bus, plc, ctrl, *_ = make_cell(success=False)
    plc.request_job(job_id=4, recipe=1)
    ctrl.poll()
    bus.write("FAULT_RESET", True)         # JOB_START still high
    assert ctrl.poll() is CellState.FAULT


def test_emergency_stop_mid_job():
    cell = {}
    bus, plc, ctrl, safety, *_ = make_cell(
        during=lambda: cell["plc"].press_estop())
    cell["plc"] = plc
    plc.request_job(job_id=5, recipe=1)
    assert ctrl.poll() is CellState.FAULT
    assert safety.state is SafetyState.EMERGENCY_STOP
    assert not bus.read("ROBOT_READY")


def test_silent_plc_denies_motion():
    bus, plc, ctrl, safety, agent, _, clock = make_cell()
    plc.stall_heartbeat()
    clock.advance(2.0)                     # heartbeat frozen for 2 s
    assert not safety.authorize(MotionCommand("home")).allowed
    plc.resume_heartbeat(); plc.scan()
    assert safety.reset("operator").allowed
    assert safety.authorize(MotionCommand("home")).allowed


def test_plc_is_not_trusted_until_its_heartbeat_moves():
    bus = InMemoryBus()
    SoftPLC(bus)                           # one scan, then silent
    interlocks = PLCInterlocks(bus)
    safety = SafetySupervisor(interlock_source=interlocks)
    assert not safety.authorize(MotionCommand("home")).allowed
    assert not interlocks.plc_alive


def test_fieldbus_read_failure_denies_motion():
    class DeadBus(InMemoryBus):
        def read_many(self, names):
            raise IOError("link down")

    interlocks = PLCInterlocks(DeadBus())
    safety = SafetySupervisor(interlock_source=interlocks)
    assert not safety.authorize(MotionCommand("home")).allowed
