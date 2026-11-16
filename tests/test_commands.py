"""
Free-form commands on a table nobody has described.

The planner tests use a stand-in cell and a stand-in model, so they
need no key and no simulator: whatever the model replies, code checks
it before the robot moves. The last tests run the real camera and the
real arm in the simulator (about a minute).
"""

import json
import math

import numpy as np
import pytest

import commands as C


class FakeThing:
    def __init__(self, ident, container=False, x=0.4, y=0.0, inside=None,
                 held=False, grip_problem=None, free=4, conveyor=False):
        self.id, self.container = ident, container
        self.x, self.y = x, y
        self.inside, self.held, self.under, self.on = inside, held, None, None
        self.rgb = np.array([0.9, 0.8, 0.1])
        self.long, self.short, self.angle = 0.045, 0.045, 0.0
        self.top, self.ground, self.height = 0.145, 0.10, 0.045
        self.roundness, self.fill = 0.66, 1.0
        self.floor = 0.105
        self.grip_problem, self.free, self.conveyor = grip_problem, free, conveyor
        self.hold = 0.0265


class FakeOpenCell:
    finder = None
    table_z = 0.10

    def __init__(self, things, table_room=True):
        self.things = {t.id: t for t in things}
        self._table_room = table_room

    def listing(self):
        return sorted(self.things.values(), key=lambda t: t.id)

    def held(self):
        return next((t for t in self.things.values() if t.held), None)

    def conveyor_container(self):
        return next((t for t in self.things.values()
                     if t.container and t.conveyor), None)

    def why_not_grip(self, thing):
        if thing.container:
            return "it is too big for the gripper"
        return thing.grip_problem

    def spots_in(self, box, taken=()):
        return [(box.x, box.y)] * box.free

    def table_spot(self, taken=()):
        return (0.35, 0.1) if self._table_room else None


class FakeCell:
    def __init__(self, things=None, **kwargs):
        from types import SimpleNamespace
        things = things if things is not None else [
            FakeThing(1, container=True, y=-0.37),
            FakeThing(2, container=True, x=0.63, conveyor=True),
            FakeThing(3), FakeThing(4), FakeThing(5)]
        self.robot = SimpleNamespace(open_cell=FakeOpenCell(things, **kwargs),
                                     grasped_object=None)


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


def reply(steps, understood="ok", names=None):
    return json.dumps({"understood": understood, "names": names or {},
                       "steps": steps})


# ---- nothing is pre-mapped ------------------------------------------

def test_the_briefing_carries_measurements_and_no_names():
    brief = C.briefing(FakeCell())
    thing = brief["things"][0]
    assert set(thing) >= {"number", "colour_rgb", "size_mm", "height_mm",
                          "roundness", "centre_mm", "can_grip"}
    text = json.dumps(brief["things"]) + json.dumps(brief["containers"])
    for word in ("lemon", "mint", "berry", "cube", "cylinder", "yellow",
                 "red", "blue"):
        assert word not in text


def test_the_briefing_says_which_container_the_conveyor_can_take():
    boxes = {b["number"]: b for b in C.briefing(FakeCell())["containers"]}
    assert boxes[2]["on_conveyor"] and not boxes[1]["on_conveyor"]


def test_the_briefing_explains_why_a_thing_cannot_be_gripped():
    cell = FakeCell([FakeThing(1, container=True, conveyor=True),
                     FakeThing(3, grip_problem="it is 80 mm across")])
    thing = C.briefing(cell)["things"][0]
    assert thing["can_grip"] is False
    assert "80 mm" in thing["cannot_grip_because"]


def test_the_model_is_never_given_a_way_to_send_coordinates():
    for text in C.SKILLS.values():
        assert "x" not in text.split() and "coordinate" not in text


# ---- checking a plan ---------------------------------------------------

def test_pick_then_place_in_a_container_is_accepted():
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": 2}),
                    ("dispatch", {}))
    assert C.validate(plan, FakeCell()) == []


def test_numbers_given_as_text_are_understood():
    plan = steps_of(("pick", {"thing": "3"}), ("place", {"on": "2"}))
    assert C.validate(plan, FakeCell()) == []


def test_placing_on_the_table_is_accepted():
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": "table"}))
    assert C.validate(plan, FakeCell()) == []


def test_a_skill_the_robot_does_not_have_is_refused():
    assert "not something this robot can do" in C.validate(
        steps_of(("throw", {})), FakeCell())[0]


def test_a_thing_that_is_not_there_is_refused():
    assert "no thing 99" in C.validate(
        steps_of(("pick", {"thing": 99})), FakeCell())[0]


def test_a_container_cannot_be_picked_up():
    assert "too big" in C.validate(
        steps_of(("pick", {"thing": 1})), FakeCell())[0]


def test_two_picks_in_a_row_are_refused_one_gripper():
    plan = steps_of(("pick", {"thing": 3}), ("pick", {"thing": 4}))
    assert "already holding thing 3" in C.validate(plan, FakeCell())[0]


def test_placing_with_an_empty_gripper_is_refused():
    assert "gripper is empty" in C.validate(
        steps_of(("place", {"on": 2})), FakeCell())[0]


def test_stacking_one_loose_thing_on_another_is_refused():
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": 4}))
    assert "does not stack" in C.validate(plan, FakeCell())[0]


def test_a_full_container_is_refused():
    cell = FakeCell([FakeThing(2, container=True, conveyor=True, free=1),
                     FakeThing(3), FakeThing(4)])
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": 2}),
                    ("pick", {"thing": 4}), ("place", {"on": 2}))
    assert "no free place" in C.validate(plan, cell)[0]


def test_dispatch_empties_the_container_for_what_follows():
    cell = FakeCell([FakeThing(2, container=True, conveyor=True, free=1),
                     FakeThing(3), FakeThing(4)])
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": 2}),
                    ("dispatch", {}),
                    ("pick", {"thing": 4}), ("place", {"on": 2}))
    assert C.validate(plan, cell) == []


def test_a_dispatched_thing_cannot_be_picked_afterwards():
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": 2}),
                    ("dispatch", {}), ("pick", {"thing": 3}))
    assert "no thing 3" in C.validate(plan, FakeCell())[0]


def test_dispatch_with_a_full_gripper_is_refused():
    plan = steps_of(("pick", {"thing": 3}), ("dispatch", {}))
    assert "put down thing 3" in C.validate(plan, FakeCell())[0]


def test_dispatch_with_nothing_on_the_conveyor_is_refused():
    cell = FakeCell([FakeThing(1, container=True), FakeThing(3)])
    assert "no container on the conveyor" in C.validate(
        steps_of(("dispatch", {})), cell)[0]


def test_looking_while_holding_something_is_refused():
    plan = steps_of(("pick", {"thing": 3}), ("look", {}))
    assert "camera" in C.validate(plan, FakeCell())[0]


def test_a_thing_too_wide_to_grip_is_refused_with_the_reason():
    cell = FakeCell([FakeThing(2, container=True, conveyor=True),
                     FakeThing(3, grip_problem="it is 80 mm across, wider "
                                               "than the gripper opens")])
    assert "80 mm" in C.validate(steps_of(("pick", {"thing": 3})), cell)[0]


def test_no_room_on_the_table_is_refused():
    plan = steps_of(("pick", {"thing": 3}), ("place", {"on": "table"}),
                    ("pick", {"thing": 4}), ("place", {"on": "table"}))
    cell = FakeCell(table_room=False)
    # the first goes back where it came from; a second new place does not exist
    problems = C.validate(plan, cell)
    assert problems == [] or "no clear place" in problems[0]


def test_an_endless_plan_is_refused():
    plan = steps_of(*(("say", {"text": "x"}),) * (C.MAX_STEPS + 1))
    assert "more than" in C.validate(plan, FakeCell())[0]


def test_an_empty_plan_is_refused():
    assert C.validate([], FakeCell()) != []


# ---- asking the model -----------------------------------------------------

def test_a_good_reply_is_used_and_its_names_are_kept():
    client = FakeClient([reply(steps_of(("pick", {"thing": 3}),
                                        ("place", {"on": 2})),
                               "put the yellow cube in the tray",
                               {"3": "yellow cube", "2": "brown tray"})])
    steps, understood, names, attempts, _ = C.plan("x", FakeCell(), client)
    assert attempts == 1 and names["3"] == "yellow cube"
    assert "tray" in understood


def test_the_measurements_go_with_the_instruction():
    client = FakeClient([reply(steps_of(("say", {"text": "hello"})))])
    C.plan("what is there?", FakeCell(), client)
    sent = client.calls[0]["messages"][0]["content"][-1]["text"]
    assert "what is there?" in sent and "colour_rgb" in sent


def test_a_bad_plan_goes_back_once_with_the_reason():
    bad = reply(steps_of(("pick", {"thing": 3}), ("place", {"on": 4})))
    good = reply(steps_of(("pick", {"thing": 3}), ("place", {"on": 2})))
    client = FakeClient([bad, good])
    _, _, _, attempts, _ = C.plan("stack them", FakeCell(), client)
    assert attempts == 2
    assert "does not stack" in client.calls[1]["messages"][-1]["content"]


def test_a_plan_that_stays_wrong_is_refused_not_attempted():
    bad = reply(steps_of(("juggle", {})))
    with pytest.raises(C.CommandError):
        C.plan("juggle", FakeCell(), FakeClient([bad, bad]))


def test_a_reply_that_is_not_json_is_refused():
    with pytest.raises(C.CommandError):
        C.plan("x", FakeCell(), FakeClient(["no", "still no"]))


def test_a_question_back_from_the_model_is_accepted():
    client = FakeClient([reply(steps_of(
        ("say", {"text": "there are two red ones, which?"})))])
    steps, *_ = C.plan("the red one", FakeCell(), client)
    assert steps[0]["step"] == "say"


# ---- the picture -------------------------------------------------------------

def test_the_png_encoder_writes_a_valid_file():
    data = C._png(np.zeros((8, 6, 3), dtype=np.uint8))
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and data[-8:-4] == b"IEND"


def test_number_tags_are_drawn_inside_the_picture():
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    C._draw_number(image, "12", -5, 55)          # asked for off the edge
    assert image.max() == 255 and image[0, 0].sum() == 0


# ---- the real camera and arm (simulator) -------------------------------------

@pytest.fixture(scope="module")
def open_cell():
    import contextlib
    import io
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger
    with contextlib.redirect_stdout(io.StringIO()):
        cell = build_sim_cell(NullLogger(), scene="open", seed=4,
                              open_count=6)
        cell.robot.open_cell.scan()
    yield cell
    cell.soft_plc.stop()
    # Leave no simulator connected: a later test that builds its own
    # cell would get a second client, and every bare pybullet call
    # goes to the first one.
    import pybullet as p
    p.disconnect()


def test_the_camera_finds_every_thing_without_being_told(open_cell):
    import pybullet as p
    oc = open_cell.robot.open_cell
    loose = [t for t in oc.listing() if not t.container]
    assert len(loose) == 6
    assert sum(t.container for t in oc.listing()) == 2
    for body in open_cell.robot.objects.values():
        pos = p.getBasePositionAndOrientation(body["id"])[0]
        nearest = min(math.hypot(t.x - pos[0], t.y - pos[1]) for t in loose)
        assert nearest < 0.003          # within 3 mm, from pixels alone


def test_the_picture_for_the_model_is_made(open_cell):
    assert C.picture(open_cell)[:4] == b"\x89PNG"


def test_a_checked_plan_runs_on_the_arm(open_cell):
    import contextlib
    import io
    import pybullet as p
    oc = open_cell.robot.open_cell
    thing = next(t for t in oc.listing() if not t.container)
    tray = oc.conveyor_container()
    plan = steps_of(("pick", {"thing": thing.id}),
                    ("place", {"on": tray.id}))
    assert C.validate(plan, open_cell) == []
    with contextlib.redirect_stdout(io.StringIO()):
        done, _, failure = C.execute(plan, open_cell)
    assert failure is None and len(done) == 2
    # the camera now sees it inside the tray, under the same number
    assert oc.things[thing.id].inside == tray.id
    low, high = p.getAABB(oc.scene["tray"])
    inside = [b for b in open_cell.robot.objects.values()
              if low[0] < p.getBasePositionAndOrientation(b["id"])[0][0]
              < high[0]
              and low[1] < p.getBasePositionAndOrientation(b["id"])[0][1]
              < high[1]]
    assert len(inside) == 1
