"""Typed instruction -> recipe number, and the PLC-side plumbing around it."""

import time
from types import SimpleNamespace

import pytest

from plc.bus import InMemoryBus
from plc.plc_link import LivePlcLink, PlcMonitor, permit_reason, refusal_reason
from plc.recipe_planner import (NO_RECIPE, choose_recipe, keyword_recipe,
                                parse_answer)


class FakeClient:
    """Stands in for anthropic.Anthropic; returns a canned answer."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text",
                                                        text=self.answer)])


def test_parse_answer_accepts_json_with_surrounding_text():
    assert parse_answer('Sure. {"recipe": 3, "reason": "assembly"}') == \
        (3, "assembly")


@pytest.mark.parametrize("bad", ["no json", '{"recipe": "3"}',
                                 '{"recipe": true}', '{"recipe": -1}',
                                 '{"recipe": 70000}'])
def test_parse_answer_rejects_anything_but_a_register_sized_integer(bad):
    with pytest.raises(ValueError):
        parse_answer(bad)


def test_model_answer_becomes_the_recipe_number():
    client = FakeClient('{"recipe": 2, "reason": "red only"}')
    r = choose_recipe("ship just the red one", client, "some-model")
    assert (r.recipe, r.source) == (2, "model")
    assert client.calls[0]["model"] == "some-model"
    assert "ship just the red one" in client.calls[0]["messages"][0]["content"]


def test_planner_does_not_validate_the_number_the_plc_does():
    # An unapproved number is passed through untouched: refusing it is
    # the PLC's decision, not the planner's.
    r = choose_recipe("do something odd", FakeClient('{"recipe": 7}'), "m")
    assert r.recipe == 7


def test_keyword_fallback_without_a_client():
    assert keyword_recipe("assemble the unit") == 3
    assert keyword_recipe("dispatch only the red part") == 2
    assert keyword_recipe("dispatch all parts to the conveyor") == 1
    assert keyword_recipe("make me a coffee") == NO_RECIPE
    assert choose_recipe("assemble the unit").source == "keywords"


def plc_image(**overrides):
    bus = InMemoryBus()
    for name in ("ESTOP_OK", "GUARD_CLOSED", "MODE_AUTO", "MOTION_PERMIT",
                 "READY"):
        bus.write(name, True)
    for name, value in overrides.items():
        bus.write(name, value)
    return bus


def test_refusal_reasons_name_the_cause():
    assert "not on the approved list" in refusal_reason(plc_image(), 9)
    assert "guard open" in refusal_reason(
        plc_image(GUARD_CLOSED=False, MOTION_PERMIT=False), 1)
    assert "heartbeat" in refusal_reason(plc_image(MOTION_PERMIT=False), 1)
    assert "FAULT_RESET" in refusal_reason(plc_image(), 1)


def test_permit_reason_lists_every_open_input():
    v = {"ESTOP_OK": False, "GUARD_CLOSED": False, "MODE_AUTO": True}
    assert permit_reason(v) == "e-stop pressed, guard open"


def test_live_link_withdraws_a_refused_request():
    bus = InMemoryBus()
    link = LivePlcLink(bus)
    link.request_job(job_id=1, recipe=3)
    assert bus.read("RECIPE") == 3 and not bus.read("JOB_START")
    link.withdraw()
    assert bus.read("RECIPE") == 0


def test_monitor_logs_the_handshake():
    bus = plc_image()
    said = []
    mon = PlcMonitor(bus, said.append, period_s=0.005)
    mon.start()
    try:
        bus.write("RECIPE", 3)
        bus.write("JOB_START", True)
        time.sleep(0.05)
        bus.write("GUARD_CLOSED", False)
        bus.write("MOTION_PERMIT", False)
        time.sleep(0.05)
    finally:
        mon.stop()
    texts = [t for _, t in mon.events]
    assert "JOB_START raised, recipe 3, job 0" in texts
    assert "MOTION_PERMIT dropped: guard open" in texts
    assert said[0].startswith("  [PLC] ")
