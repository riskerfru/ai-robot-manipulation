"""
Force-guided assembly: geometry and force reading without physics runs,
then one end-to-end job in the simulator (about a minute).
"""

import pytest

from cell import assembly as A


def test_each_pocket_leaves_the_stated_clearance():
    for name, c in A.COMPONENTS.items():
        pocket = A.pocket_for(name)
        assert pocket["half"] == c["size"] / 2
        assert pocket["clearance"] == A.CLEARANCE
        assert pocket["depth"] == A.POCKET_DEPTH


def test_pockets_do_not_overlap_and_stay_inside_the_housing():
    spans = sorted((c["pocket_dy"] - c["size"] / 2 - A.CLEARANCE,
                    c["pocket_dy"] + c["size"] / 2 + A.CLEARANCE)
                   for c in A.COMPONENTS.values())
    assert spans[0][0] > -A.HOUSING_HALF[1]
    assert spans[-1][1] < A.HOUSING_HALF[1]
    for (_, hi), (lo, _) in zip(spans, spans[1:]):
        assert lo - hi > 0.01                 # a real wall between pockets


def test_fingertips_stay_above_the_pocket_rim():
    # The gripper point sits FINGER_LIFT above the part's base and the
    # fingertips reach 14.5 mm below it.
    assert A.FINGER_LIFT - 0.0145 > A.POCKET_DEPTH


def test_nominal_pocket_moves_with_the_housing():
    shifted = A.pocket_for("motor", (A.HOUSING_CENTRE[0] + 0.003,
                                     A.HOUSING_CENTRE[1]))
    assert shifted["x"] - A.pocket_for("motor")["x"] == pytest.approx(0.003)


def test_insert_without_a_pocket_is_refused():
    from tests.fakes import FakeCellRobot
    from agent_loop import WarehouseAgent
    from utils.logger import NullLogger
    robot = FakeCellRobot()
    robot.pockets = {}
    agent = WarehouseAgent(robot, {"api_key": "", "model": "x"},
                           NullLogger(), llm_enabled=False)
    result = agent.run("assemble", subtasks=[{"action": "insert",
                                              "object": "motor"}])
    assert not result["success"]


def test_assembly_job_seats_three_components_in_a_shifted_housing():
    p = pytest.importorskip("pybullet")
    from cell.factory import build_sim_cell
    from plc.cell_controller import CellState
    from utils.logger import NullLogger

    offset = (0.002, -0.0015)       # the controller is not told this
    cell = build_sim_cell(NullLogger(), scene="assembly",
                          housing_offset=offset)
    try:
        assert cell.controller.poll() is CellState.READY
        cell.soft_plc.request_job(job_id=7, recipe=3)
        assert cell.controller.poll() is CellState.DONE
        assert cell.bus.read("PARTS_OK") == 3

        centre = (A.HOUSING_CENTRE[0] + offset[0],
                  A.HOUSING_CENTRE[1] + offset[1])
        for name in A.COMPONENTS:
            pocket = A.pocket_for(name, centre)
            x, y, z = cell.robot.get_object_position(name)
            assert abs(z - pocket["half"] - pocket["floor_z"]) < 0.002
            assert abs(x - pocket["x"]) <= pocket["clearance"] + 0.0005
            assert abs(y - pocket["y"]) <= pocket["clearance"] + 0.0005
        results = [r["data"]["result"]
                   for r in cell.audit.events("insertion")]
        # A component may need a second attempt; none may be skipped.
        assert sum(r["status"] == "success" for r in results) == 3
    finally:
        cell.soft_plc.stop()
        p.disconnect()
