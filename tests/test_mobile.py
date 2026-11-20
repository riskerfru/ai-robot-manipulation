"""Mobile manipulator: frames, routing, guardian rules, and one full run."""

import contextlib
import io
import math
from types import SimpleNamespace

import pytest

from mobile import cell as C
from mobile import floor as F
from mobile.base import compose, relative
from mobile.guardian import Guardian
from safety.supervisor import MotionCommand, SafetySupervisor


# ---- frames ------------------------------------------------------------------
def test_compose_and_relative_are_inverse():
    pose = (0.3, -1.9, math.radians(1.2))
    wx, wy = compose(pose, 0.52, 0.04)
    bx, by = relative(pose, wx, wy)
    assert abs(bx - 0.52) < 1e-9 and abs(by - 0.04) < 1e-9


def test_supervisor_checks_targets_in_the_base_frame():
    s = SafetySupervisor()
    far = MotionCommand("pick", (0.40, -1.95, 0.125), payload_kg=0.5)
    assert not s.authorize(far).allowed          # base at origin: out of reach
    s.base_pose = (0.0, -1.95, 0.0)
    assert s.authorize(far).allowed              # base moved: in front of it


# ---- routing -----------------------------------------------------------------
def test_route_goes_round_the_pallet():
    route = F.plan_route((F.LANE_X, F.STATIONS["STORAGE"]),
                         (F.LANE_X, F.STATIONS["MACHINE"]))
    assert route is not None
    (px0, px1), (py0, py1) = F.PALLET
    for (x0, y0), (x1, y1) in zip(route, route[1:]):
        for i in range(21):
            x, y = x0 + (x1 - x0) * i / 20, y0 + (y1 - y0) * i / 20
            # the base centre stays out of the pallet grown by the footprint
            assert not (px0 - 0.2 < x < px1 + 0.2 and py0 < y < py1)


def test_lane_crosses_the_walkway():
    assert F.in_walkway(F.LANE_X, 0.0)
    assert not F.in_walkway(F.LANE_X, F.STATIONS["INSPECT"])


# ---- guardian, against a stand-in cell --------------------------------------
class Part(SimpleNamespace):
    pass


def fake_cell(here="STORAGE", held=None, machine="idle", state=None,
              fittings=(), parts_in=None, locked=()):
    things = {1: Part(id=1, container=False, station="STORAGE", held=False,
                      inside=None),
              2: Part(id=2, container=False, station="INSPECT", held=False,
                      inside=None)}
    if held is not None:
        things[held].held, things[held].station = True, None
    parts_in = parts_in or {}
    return SimpleNamespace(
        base=SimpleNamespace(station=here),
        things=things,
        state=state or {1: C.RAW, 2: C.MACHINED},
        locked=set(locked),
        machine={"status": machine, "part": 9 if machine != "idle" else None},
        held=lambda: things[held] if held is not None else None,
        equipment=lambda role: role if role in fittings else None,
        parts_in=lambda role: parts_in.get(role, []))


def verdict(cell, **action):
    return Guardian(cell).check(action)


def test_guardian_accepts_a_legal_pick():
    assert verdict(fake_cell(), action="pick", part=1)


def test_guardian_refuses_unknown_skill():
    assert verdict(fake_cell(), action="teleport").code == "G01"


def test_guardian_refuses_speeding():
    assert verdict(fake_cell(), action="goto", station="MACHINE",
                   speed=1.5).code == "G10"


def test_guardian_refuses_entering_a_busy_machine():
    v = verdict(fake_cell(machine="busy"), action="goto", station="MACHINE")
    assert v.code == "G04"


def test_guardian_refuses_locked_station():
    v = verdict(fake_cell(locked={"INSPECT"}), action="goto",
                station="INSPECT")
    assert v.code == "G03"


def test_guardian_refuses_picking_a_part_at_another_station():
    assert verdict(fake_cell(), action="pick", part=2).code == "G05"


def test_guardian_refuses_second_pick():
    assert verdict(fake_cell(held=1), action="pick", part=2).code == "G06"


def test_guardian_enforces_the_process_route():
    cell = fake_cell(here="INSPECT", held=1, fittings=("pad",))
    v = verdict(cell, action="place", into="pad")     # RAW onto inspection
    assert v.code == "G08"


def test_guardian_refuses_shipping_a_failed_part():
    cell = fake_cell(here="DISPATCH", held=1, fittings=("tray",),
                     state={1: C.FAIL, 2: C.MACHINED})
    assert verdict(cell, action="place", into="tray").code == "G11"


def test_guardian_refuses_loading_an_occupied_fixture():
    cell = fake_cell(here="MACHINE", held=1, fittings=("fixture",),
                     parts_in={"fixture": [Part(id=7)]})
    assert verdict(cell, action="place", into="fixture").code == "G09"


def test_guardian_lets_a_failed_part_into_the_bin():
    cell = fake_cell(here="INSPECT", held=1, fittings=("bin",),
                     state={1: C.FAIL, 2: C.MACHINED})
    assert verdict(cell, action="place", into="bin")


# ---- the whole thing, in simulation -----------------------------------------
def test_full_run_offline_with_injected_mistakes():
    pytest.importorskip("pybullet")
    from mobile.agent import ScriptedPlanner, run
    from mobile.cell import build_mobile_cell
    with contextlib.redirect_stdout(io.StringIO()):
        w = build_mobile_cell(seed=1)
        w.cell.scan()
        out = run("process all the parts in storage", w.cell,
                  ScriptedPlanner(mistakes=True), lambda *a: None)
        w.robot.disconnect()
    assert out["finished"]
    assert sorted(out["states"].values()) == [C.DISPATCHED] * 3 + [C.SCRAPPED]
    assert {r[1] for r in out["rejected"]} == {"G10", "G04", "G11"}
    # every fixture and pad placement within the 5 mm the machine accepts
    assert w.cell.placements
    assert all(p["error_mm"] <= C.SEAT_TOL_M * 1000
               for p in w.cell.placements)
