"""
Packing orders: the plan is validated by code whatever produced it.
The language model is replaced by a fake client, so no key is needed.
One end-to-end order runs in the simulator (about a minute).
"""

import json

import pytest

import packing_orders as PO
from cell import packing as P


class FakeLine:
    """Stands in for the line: stock is what a scan would have seen."""

    def __init__(self, fmt="standard", stock=None):
        self.format = fmt
        self._stock = stock or {"standard": {"lemon": 2, "mint": 2, "berry": 2},
                                "travel": {"lemon": 2, "mint": 2, "berry": 2}}

    def stock_levels(self, fmt=None):
        return dict(self._stock[fmt or self.format])


class FakeClient:
    """Stands in for the Anthropic client: returns canned replies in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.replies.pop(0)

        class Block:
            pass

        class Response:
            pass

        block, response = Block(), Response()
        block.text = text
        response.content = [block]
        return response


GOOD = {"format": "standard", "note": "ok",
        "items": [{"flavour": "lemon", "quantity": 2},
                  {"flavour": "mint", "quantity": 1},
                  {"flavour": "berry", "quantity": 1}]}


# ---- validation --------------------------------------------------

def test_a_full_tray_from_stock_is_accepted():
    items, problems = PO.validate(
        "standard", [("lemon", 2), ("mint", 1), ("berry", 1)], FakeLine())
    assert problems == []
    assert items == [("lemon", 2), ("mint", 1), ("berry", 1)]


@pytest.mark.parametrize("fmt, items, fragment", [
    ("standard", [("lemon", 2), ("mint", 1)], "holds 4"),
    ("standard", [("lemon", 3), ("mint", 1)], "in stock"),
    ("standard", [("vanilla", 4)], "no 'vanilla'"),
    ("jumbo", [("lemon", 4)], "unknown format"),
    ("standard", [("lemon", 0), ("mint", 4)], "whole number"),
    ("standard", [("lemon", 2.5), ("mint", 2)], "whole number"),
])
def test_unpackable_plans_are_refused_with_a_reason(fmt, items, fragment):
    _, problems = PO.validate(fmt, items, FakeLine())
    assert problems and fragment in "; ".join(problems)


def test_travel_tray_needs_six():
    _, problems = PO.validate(
        "travel", [("lemon", 2), ("mint", 2), ("berry", 2)], FakeLine())
    assert problems == []
    assert P.nests_per_tray("travel") == 6


# ---- offline parser ----------------------------------------------

def test_offline_parser_reads_counts_and_keeps_the_format():
    plan = PO.plan_offline("Pack a mixed case of standard cartons: two "
                           "lemon, one mint and one berry.", FakeLine())
    assert plan.format == "standard"
    assert plan.items == [("lemon", 2), ("mint", 1), ("berry", 1)]


def test_offline_parser_sees_a_changeover():
    plan = PO.plan_offline("Change over to the travel size and pack two "
                           "lemon, two mint and two berry.", FakeLine())
    assert plan.format == "travel"
    assert sum(q for _, q in plan.items) == 6


def test_offline_parser_refuses_what_it_cannot_pack():
    with pytest.raises(PO.PlanError):
        PO.plan_offline("pack four lemon", FakeLine())


# ---- language-model planner (fake client) --------------------------

def test_model_plan_is_used_when_valid():
    client = FakeClient([json.dumps(GOOD)])
    plan = PO.plan_with_llm("two lemon, a mint and a berry", FakeLine(),
                            client, model="m")
    assert plan.source == "language model" and plan.attempts == 1
    assert plan.items == [("lemon", 2), ("mint", 1), ("berry", 1)]
    sent = client.calls[0]["messages"][0]["content"]
    assert "products_in_stock" in sent          # the model saw the facts


def test_model_reply_in_a_code_fence_is_read():
    client = FakeClient(["```json\n" + json.dumps(GOOD) + "\n```"])
    assert PO.plan_with_llm("x", FakeLine(), client, model="m").items


def test_wrong_plan_is_sent_back_once_then_accepted():
    bad = dict(GOOD, items=[{"flavour": "lemon", "quantity": 4}])
    client = FakeClient([json.dumps(bad), json.dumps(GOOD)])
    plan = PO.plan_with_llm("x", FakeLine(), client, model="m")
    assert plan.attempts == 2
    feedback = client.calls[1]["messages"][-1]["content"]
    assert "in stock" in feedback


def test_plan_that_stays_wrong_is_refused():
    bad = dict(GOOD, items=[{"flavour": "lemon", "quantity": 4}])
    client = FakeClient([json.dumps(bad), json.dumps(bad)])
    with pytest.raises(PO.PlanError):
        PO.plan_with_llm("x", FakeLine(), client, model="m")


def test_reply_that_is_not_json_is_refused():
    client = FakeClient(["I think lemon is nice.", "Still chatting."])
    with pytest.raises(PO.PlanError):
        PO.plan_with_llm("x", FakeLine(), client, model="m")


def test_model_can_decline_an_impossible_order():
    client = FakeClient([json.dumps({"error": "only two lemon in stock"})])
    with pytest.raises(PO.PlanError, match="two lemon"):
        PO.plan_with_llm("ten lemon", FakeLine(), client, model="m")


def test_model_cannot_invent_a_format_or_overfill():
    invented = {"format": "jumbo",
                "items": [{"flavour": "lemon", "quantity": 4}]}
    client = FakeClient([json.dumps(invented), json.dumps(invented)])
    with pytest.raises(PO.PlanError, match="unknown format"):
        PO.plan_with_llm("x", FakeLine(), client, model="m")


# ---- tray geometry -------------------------------------------------

@pytest.mark.parametrize("fmt", list(P.FORMATS))
def test_nests_sit_inside_the_tray_with_walls_between(fmt):
    hx, hy = P.tray_half_extents(fmt)
    half = P.FORMATS[fmt]["carton"] / 2 + P.CLEARANCE
    offsets = P.nest_offsets(fmt)
    assert len(offsets) == P.nests_per_tray(fmt)
    for dx, dy in offsets.values():
        assert abs(dx) + half < hx and abs(dy) + half < hy
    points = sorted(offsets.values())
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        gap = max(abs(ax - bx), abs(ay - by)) - 2 * half
        assert gap >= 0.008


def test_fingers_open_along_the_wide_pitch():
    pocket = P.nest_pocket("travel", 1)
    assert pocket["finger_axis"] == "y"
    rows = sorted({round(o[1], 4) for o in P.nest_offsets("travel").values()})
    assert rows[1] - rows[0] >= 0.085


# ---- one order end to end in the simulator ---------------------------

def test_typed_order_packs_a_tray_that_is_off_its_mark():
    p = pytest.importorskip("pybullet")
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger

    cell = build_sim_cell(NullLogger(), scene="packing",
                          housing_offset=(0.002, -0.0015))
    try:
        out = PO.run_order(cell, "pack two lemon, one mint and one berry",
                           PO.plan_offline)
        assert out["result"]["success"]
        assert [c["flavour"] for c in out["check"]] == \
            ["lemon", "lemon", "mint", "berry"]
        assert all(c["seated"] for c in out["check"])
        assert len(cell.audit.events("order_plan")) == 1

        refused = PO.run_order(cell, "pack nine lemon", PO.plan_offline)
        assert "refused" in refused
        assert len(cell.audit.events("order_refused")) == 1
    finally:
        cell.soft_plc.stop()
        p.disconnect()


# ---- the overhead detector -------------------------------------

def test_a_pixel_maps_onto_the_plane_it_is_told_about():
    """The ray through a pixel, met with a known plane, is the whole
    basis of the measured positions. Check it on a camera we build
    here, against points we place ourselves."""
    p = pytest.importorskip("pybullet")
    import numpy as np
    from perception.detector import OverheadDetector
    from cell import layout as L

    client = p.connect(p.DIRECT)
    try:
        view = p.computeViewMatrix(cameraEyePosition=[0.48, 0.0, 1.45],
                                   cameraTargetPosition=[0.48, 0.0, 0.10],
                                   cameraUpVector=[0, 1, 0])
        projection = p.computeProjectionMatrixFOV(42, 1.0, 0.1, 10.0)

        class FakeCamera:
            width = height = 512
            cameras = {"cell": {"view": view, "projection": projection,
                                "width": 512, "height": 512}}

        detector = OverheadDetector(FakeCamera(), "cell")
        plane = L.TABLE_Z + 0.05
        for world in ((0.48, 0.0), (0.33, 0.20), (0.60, -0.15)):
            clip = (np.array(projection).reshape(4, 4, order="F")
                    @ np.array(view).reshape(4, 4, order="F")
                    @ np.array([world[0], world[1], plane, 1.0]))
            ndc = clip[:3] / clip[3]
            u = (ndc[0] + 1) / 2 * 512 - 0.5
            v = (1 - ndc[1]) / 2 * 512 - 0.5
            back = detector.pixel_to_plane(u, v, plane)
            assert back == pytest.approx(world, abs=0.001)
    finally:
        p.disconnect(client)


def test_connected_regions_are_separated():
    from perception.detector import OverheadDetector
    import numpy as np
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:4, 1:4] = True          # one square
    mask[6:9, 6:9] = True          # another, not touching
    labels = OverheadDetector._label(mask)
    assert len({int(v) for v in np.unique(labels)} - {0}) == 2


def test_the_camera_finds_the_cartons_and_the_robot_picks_one():
    """
    End to end on the real thing: scan the cell, check every carton
    is found near where it actually is, then pick at a measured
    position and confirm the gripper caught that carton.
    """
    import math
    p = pytest.importorskip("pybullet")
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger

    cell = build_sim_cell(NullLogger(), scene="packing",
                          line_noise=P.LINE_NOISE, seed=3)
    try:
        line = cell.robot.packing_line
        found = line.scan()
        assert len(found) == 6                      # six standard cartons
        assert sum(line.stock_levels().values()) == 6
        assert sum(line.stock_levels("travel").values()) == 6

        truth = {name: cell.robot.get_object_position(name)
                 for name in cell.robot.objects}
        for d in found:
            nearest = min((math.hypot(pos[0] - d.x, pos[1] - d.y)
                           for name, pos in truth.items()
                           if name.startswith(d.product)), default=9)
            assert nearest < 0.004          # measured within 4 mm

        detection = found[0]
        spec = line.pick_spec(detection)
        result = cell.robot.pick_at(spec["x"], spec["y"], spec["centre_z"],
                                    finger_axis=spec["finger_axis"],
                                    open_to=spec["open_to"],
                                    grasp_depth=spec["grasp_depth"],
                                    label=detection.product)
        assert result["status"] == "success"
        # The gripper reports what it caught; it must be of the
        # product the camera saw at that spot.
        assert result["gripped"].startswith(detection.product)
        held = cell.robot.get_object_position(result["gripped"])
        tool = cell.robot.get_end_effector_position()
        assert math.hypot(held[0] - tool[0], held[1] - tool[1]) < 0.006
    finally:
        cell.soft_plc.stop()
        p.disconnect()


def test_an_order_for_cartons_that_are_not_there_is_refused():
    p = pytest.importorskip("pybullet")
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger

    cell = build_sim_cell(NullLogger(), scene="packing",
                          line_noise=P.LINE_NOISE, seed=5)
    try:
        out = PO.run_order(cell, "pack four lemon", PO.plan_offline)
        assert "refused" in out and "in stock" in out["refused"]
        assert cell.audit.events("scan")        # it looked first
    finally:
        cell.soft_plc.stop()
        p.disconnect()
