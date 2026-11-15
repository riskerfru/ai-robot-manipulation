"""
Free-form commands: whatever the language model replies with, code
checks it against the cell before the robot moves. The model is
replaced by a fake client, so no key is needed and no robot runs.
"""

import json

import pytest

import commands as C
from cell import packing as P


class FakeLine:
    def __init__(self, fmt="standard", stock=None, nests=None):
        self.format = fmt
        self.nests = nests or {}
        self._stock = stock or {"standard": {"lemon": 2, "mint": 2,
                                             "berry": 2},
                                "travel": {"lemon": 2, "mint": 2,
                                           "berry": 2}}

    def stock_levels(self, fmt=None):
        return dict(self._stock[fmt or self.format])


class FakeRobot:
    def __init__(self, line, holding=None):
        self.packing_line = line
        self.grasped_object = holding


class FakeCell:
    def __init__(self, **kwargs):
        self.robot = FakeRobot(FakeLine(**kwargs))


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Block:
            pass

        class Response:
            pass

        block, response = Block(), Response()
        block.text = self.replies.pop(0)
        response.content = [block]
        return response


def steps_of(*pairs):
    return [dict(step=name, **args) for name, args in pairs]


# ---- the briefing -------------------------------------------------

def test_the_briefing_names_a_colour_for_every_product():
    brief = C.briefing(FakeCell())
    for product in P.FLAVOUR_COLOURS:
        assert brief["products"][product]["colour"]


def test_the_briefing_reports_what_the_camera_sees_not_a_fixed_list():
    cell = FakeCell(stock={"standard": {"lemon": 0, "mint": 1, "berry": 0},
                           "travel": {"lemon": 2, "mint": 2, "berry": 2}})
    brief = C.briefing(cell)
    assert brief["camera_sees_now"]["standard"]["lemon"] == 0
    assert brief["camera_sees_now"]["standard"]["mint"] == 1


def test_the_briefing_lists_only_skills_the_robot_has():
    brief = C.briefing(FakeCell())
    assert set(brief["skills"]) == set(C.SKILLS)
    assert "weld" not in brief["skills"]


# ---- checking a plan ----------------------------------------------

def test_a_sensible_plan_is_accepted():
    plan = steps_of(("insert", {"product": "lemon", "nest": 1}),
                    ("insert", {"product": "mint", "nest": 2}),
                    ("dispatch", {}))
    assert C.validate(plan, FakeCell()) == []


def test_a_skill_the_robot_does_not_have_is_refused():
    problems = C.validate(steps_of(("stack", {})), FakeCell())
    assert "not a skill" in problems[0]


def test_a_product_that_is_not_in_the_scene_is_refused():
    problems = C.validate(steps_of(("pick", {"product": "banana"})),
                          FakeCell())
    assert "not a product" in problems[0]


def test_a_nest_beyond_the_tray_is_refused():
    problems = C.validate(
        steps_of(("insert", {"product": "lemon", "nest": 99})), FakeCell())
    assert "does not exist" in problems[0]


def test_the_same_nest_twice_is_refused():
    problems = C.validate(
        steps_of(("insert", {"product": "lemon", "nest": 1}),
                 ("insert", {"product": "mint", "nest": 1})), FakeCell())
    assert "already full" in problems[0]


def test_a_nest_already_holding_something_is_refused():
    cell = FakeCell(nests={2: {"product": "mint", "gripped": "mint_0"}})
    problems = C.validate(
        steps_of(("insert", {"product": "lemon", "nest": 2})), cell)
    assert "already full" in problems[0]


def test_two_picks_in_a_row_are_refused_one_gripper():
    problems = C.validate(steps_of(("pick", {"product": "lemon"}),
                                   ("pick", {"product": "mint"})),
                          FakeCell())
    assert "already holding" in problems[0]


def test_placing_with_an_empty_gripper_is_refused():
    problems = C.validate(steps_of(("place", {"where": "reject"})),
                          FakeCell())
    assert "gripper is empty" in problems[0]


def test_picking_more_than_the_camera_sees_is_refused():
    cell = FakeCell(stock={"standard": {"lemon": 1, "mint": 0, "berry": 0},
                           "travel": {"lemon": 0, "mint": 0, "berry": 0}})
    problems = C.validate(
        steps_of(("insert", {"product": "lemon", "nest": 1}),
                 ("insert", {"product": "lemon", "nest": 2})), cell)
    assert "cannot see any lemon" in problems[0]


def test_dispatching_while_holding_something_is_refused():
    plan = steps_of(("pick", {"product": "lemon"}), ("dispatch", {}))
    assert "put down" in C.validate(plan, FakeCell())[0]


def test_a_changeover_gives_a_fresh_empty_tray():
    cell = FakeCell(nests={1: {"product": "lemon", "gripped": "lemon_0"}})
    plan = steps_of(("change_format", {"format": "travel"}),
                    ("insert", {"product": "lemon", "nest": 1}))
    assert C.validate(plan, cell) == []


def test_a_format_that_does_not_exist_is_refused():
    problems = C.validate(steps_of(("change_format", {"format": "jumbo"})),
                          FakeCell())
    assert "not a format" in problems[0]


def test_the_bigger_tray_of_the_travel_format_is_allowed():
    cell = FakeCell(fmt="travel")
    nests = P.nests_per_tray("travel")
    plan = steps_of(("insert", {"product": "lemon", "nest": nests}))
    assert C.validate(plan, cell) == []


def test_putting_an_item_back_returns_it_to_stock():
    cell = FakeCell(stock={"standard": {"lemon": 1, "mint": 0, "berry": 0},
                           "travel": {"lemon": 0, "mint": 0, "berry": 0}})
    plan = steps_of(("pick", {"product": "lemon"}),
                    ("place", {"where": "infeed"}),
                    ("pick", {"product": "lemon"}))
    assert C.validate(plan, cell) == []


def test_a_report_needs_something_to_say():
    assert "needs" in C.validate(steps_of(("report", {})), FakeCell())[0]


def test_an_endless_plan_is_refused():
    plan = steps_of(*(("scan", {}),) * (C.MAX_STEPS + 1))
    assert "more than" in C.validate(plan, FakeCell())[0]


def test_an_empty_plan_is_refused():
    assert C.validate([], FakeCell()) != []


# ---- asking the model ----------------------------------------------

def test_a_good_reply_is_used_as_it_is():
    reply = json.dumps({"steps": steps_of(("scan", {})), "note": "looking"})
    steps, note, attempts = C.plan("look", FakeCell(), FakeClient([reply]))
    assert attempts == 1 and note == "looking"


def test_the_briefing_is_sent_with_the_command():
    client = FakeClient([json.dumps({"steps": steps_of(("scan", {}))})])
    C.plan("have a look", FakeCell(), client)
    sent = client.calls[0]["messages"][0]["content"]
    assert "have a look" in sent and "camera_sees_now" in sent


def test_a_reply_wrapped_in_a_code_fence_is_still_read():
    reply = "```json\n" + json.dumps({"steps": steps_of(("scan", {}))}) \
            + "\n```"
    steps, _, _ = C.plan("look", FakeCell(), FakeClient([reply]))
    assert steps[0]["step"] == "scan"


def test_a_bad_plan_is_sent_back_once_with_the_reason():
    bad = json.dumps({"steps": steps_of(
        ("insert", {"product": "lemon", "nest": 99}))})
    good = json.dumps({"steps": steps_of(
        ("insert", {"product": "lemon", "nest": 1}))})
    client = FakeClient([bad, good])
    steps, _, attempts = C.plan("fill it", FakeCell(), client)
    assert attempts == 2
    assert "nest 99 does not exist" in client.calls[1]["messages"][-1][
        "content"]


def test_a_plan_that_stays_wrong_is_refused_not_attempted():
    bad = json.dumps({"steps": steps_of(("fly", {}))})
    with pytest.raises(C.CommandError) as caught:
        C.plan("fly", FakeCell(), FakeClient([bad, bad]))
    assert "not a skill" in str(caught.value)


def test_a_reply_that_is_not_json_is_refused():
    with pytest.raises(C.CommandError):
        C.plan("x", FakeCell(), FakeClient(["sorry, I cannot", "nor this"]))


def test_a_refusal_from_the_model_is_a_report_and_is_accepted():
    reply = json.dumps({"steps": steps_of(
        ("report", {"text": "there is no pyramid skill"}))})
    steps, _, _ = C.plan("build a pyramid", FakeCell(), FakeClient([reply]))
    assert steps[0]["step"] == "report"
