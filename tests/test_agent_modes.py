"""Agent pipeline without the language model: recipes and degraded mode."""

from agent_loop import (WarehouseAgent, deterministic_decision, match_recipe)
from audit.trail import AuditTrail
from place_verifier import PlacementVerifier
from plc.recipes import RECIPES
from safety.supervised_robot import SupervisedRobot
from safety.supervisor import SafetySupervisor
from tests.fakes import FakeCellRobot, FakeLogger

AI = {"api_key": "", "model": "test-model"}


def make_agent(backend, llm_enabled=False, audit=None, **verifier_kw):
    audit = audit or AuditTrail()
    robot = SupervisedRobot(backend, SafetySupervisor(), audit)
    verifier = PlacementVerifier(robot, calibration_path=None, **verifier_kw)
    return WarehouseAgent(robot, AI, FakeLogger(), audit=audit,
                          verifier=verifier, llm_enabled=llm_enabled), audit


def test_recipe_runs_end_to_end_without_a_language_model():
    backend = FakeCellRobot()
    agent, audit = make_agent(backend)
    result = agent.run("dispatch", subtasks=list(RECIPES[1]["subtasks"]))
    assert result["success"] and len(result["completed"]) == 6
    assert result["steps_taken"] == 6
    assert backend.slot_occupants == {1: "red", 2: "blue", 3: "green"}
    assert not result["degraded"]            # recipe mode is not a fault
    assert len(audit.events("verification")) == 3
    assert not audit.events("llm_decision")


def test_missing_api_key_degrades_to_recipes_and_says_so():
    agent, audit = make_agent(FakeCellRobot(), llm_enabled=True)
    assert agent.degraded and "API key" in agent.degraded_reason
    result = agent.run("dispatch all blocks to the conveyor")
    assert result["success"] and result["degraded"]
    assert audit.events("degraded_mode")


def test_free_text_order_with_no_recipe_fails_cleanly_when_model_is_down():
    backend = FakeCellRobot()
    agent, _ = make_agent(backend, llm_enabled=True)
    result = agent.run("build a pyramid out of the blocks")
    assert not result["success"]
    assert backend.calls == []               # nothing moved


def test_language_model_failure_mid_order_falls_back_to_the_recipe_step():
    class DeadClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                raise TimeoutError("model timed out")

    backend = FakeCellRobot()
    agent, audit = make_agent(backend, llm_enabled=True)
    agent.client, agent.degraded = DeadClient(), False
    result = agent.run("x", subtasks=list(RECIPES[2]["subtasks"]))
    assert result["success"] and result["degraded"]
    assert "timed out" in audit.events("degraded_mode")[-1]["data"]["reason"]
    assert backend.slot_occupants == {1: "red"}


def test_out_of_tolerance_placement_is_reported_as_a_failed_job():
    # Every placement lands 12 cm off: retries cannot fix it.
    backend = FakeCellRobot(place_offsets=[(0.0, 0.12)] * 20)
    agent, audit = make_agent(backend, use_calibration=False)
    result = agent.run("x", subtasks=[
        {"action": "pick", "object": "red"},
        {"action": "place_slot", "slot": 2},
    ])
    assert not result["success"]
    assert 2 not in backend.slot_occupants   # slot never marked filled
    last = audit.events("verification")[-1]["data"]["result"]
    assert last["status"] == "failed"


def test_unsafe_coordinates_from_a_decision_are_stopped_by_the_gate():
    backend = FakeCellRobot()
    agent, audit = make_agent(backend)
    result = agent.run("x", subtasks=[
        {"action": "pick", "object": "red"},
        {"action": "place", "x": 1.40, "y": 0.90},
    ])
    assert not result["success"]
    assert not any(c[0] == "place" for c in backend.calls)
    assert any(not e["data"]["allowed"]
               for e in audit.events("safety_decision"))


def test_helpers():
    d = deterministic_decision({"action": "place_slot", "slot": 2})
    assert d["action"] == "place_slot" and d["params"] == {"slot": 2}
    assert match_recipe("please dispatch all blocks") == \
        RECIPES[1]["subtasks"]
    assert match_recipe("juggle") is None
