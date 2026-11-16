"""
End-to-end in the real physics simulation (headless, ~20 s):
PLC handshake -> recipe -> gated motion -> verified placement.
Skipped automatically where PyBullet is not installed.
"""

import math

import pytest

pytest.importorskip("pybullet")

from audit.trail import verify_records                    # noqa: E402
from cell.factory import build_sim_cell                   # noqa: E402
from plc.cell_controller import CellState                 # noqa: E402
from plc.signals import FaultCode                         # noqa: E402
from safety.supervisor import SafetyState                 # noqa: E402
from utils.logger import NullLogger                       # noqa: E402


@pytest.fixture
def cell():
    import pybullet as p
    c = build_sim_cell(NullLogger())
    yield c
    c.soft_plc.stop()
    p.disconnect()


def test_plc_job_places_three_parts_within_tolerance(cell):
    assert cell.controller.poll() is CellState.READY
    cell.soft_plc.request_job(job_id=1, recipe=1)
    assert cell.controller.poll() is CellState.DONE
    assert cell.bus.read("PARTS_OK") == 3
    for slot, part in cell.robot.slot_occupants.items():
        sx, sy = cell.robot.conveyor_slots[slot]
        x, y, _ = cell.robot.get_object_position(part)
        assert math.hypot(x - sx, y - sy) < 0.05
    assert verify_records(cell.audit.records)[0]
    assert len(cell.audit.events("verification")) == 3
    assert cell.safety.denied_count == 0


def test_guard_opening_mid_job_stops_the_robot(cell):
    backend, original = cell.robot, cell.robot.place_object
    fired = []

    def place(x, y, stack_on=None):
        out = original(x, y, stack_on=stack_on)
        if not fired:
            fired.append(True)
            cell.soft_plc.open_guard()
        return out

    backend.place_object = place
    cell.controller.poll()
    cell.soft_plc.request_job(job_id=2, recipe=1)
    assert cell.controller.poll() is CellState.FAULT
    assert cell.bus.read("FAULT_CODE") == FaultCode.SAFETY_STOP
    assert cell.safety.state is SafetyState.PROTECTIVE_STOP
    # Parts two and three never left the infeed.
    for part in ("blue", "green"):
        assert cell.robot.get_object_position(part)[0] < 0.55
    assert cell.bus.read("PARTS_OK") == 1
