"""The demo panel is driven by audit events. No physics, no ffmpeg."""

from PIL import Image

from record_demo import H, SCENE_W, W, Hud


def test_panel_follows_the_audit_trail():
    hud = Hud("t", "s", "f")
    hud.on_audit("job_start", {"job_id": 7, "recipe": 1, "name": "dispatch"})
    hud.on_audit("order_decomposed", {"subtasks": [
        {"action": "pick", "object": "red"},
        {"action": "place_slot", "slot": 1}]})
    hud.on_audit("recipe_decision", {"subtask": {}, "decision": {}})
    assert [t[1] for t in hud.tasks] == ["current", "pending"]
    hud.on_audit("action_result", {"subtask": {},
                                   "result": {"status": "success"}})
    assert [t[1] for t in hud.tasks] == ["done", "pending"]
    hud.on_audit("verification", {"part": "red", "slot": 1, "result": {
        "status": "success", "drift_cm": 0.4, "retried": False}})
    assert "0.4 cm" in hud.measure[0]
    hud.on_audit("safety_trip", {"state": "PROTECTIVE_STOP",
                                 "reason": "guard door open"})
    hud.on_audit("cell_state", {"previous": "BUSY", "state": "FAULT"})
    assert hud.safety_state == "PROTECTIVE_STOP" and hud.cell_state == "FAULT"
    hud.on_audit("fault_reset", {"allowed": False,
                                 "reason": "cannot reset: guard door open"})
    assert hud.safety_state == "PROTECTIVE_STOP"
    hud.on_audit("fault_reset", {"allowed": True, "reason": "ok"})
    assert hud.safety_state == "RUN"


def test_unmeasured_placement_is_labelled_as_such():
    hud = Hud("t", "s", "f")
    hud.measured = False
    hud.on_audit("verification", {"part": "blue", "slot": 2, "result": {
        "status": "success", "drift_cm": 0.0, "retried": False}})
    assert "not measured" in hud.measure[0]


def test_frame_has_the_expected_size():
    hud = Hud("t", "s", "f")
    frame = hud.draw(Image.new("RGB", (SCENE_W, H)))
    assert frame.size == (W, H)


def test_free_text_order_drives_the_cell_state():
    hud = Hud("t", "s", "f")
    hud.on_audit("order_received", {"order": "dispatch all", "recipe": False})
    assert hud.cell_state == "BUSY"
    hud.on_audit("order_result", {"success": True})
    assert hud.cell_state == "DONE"


def test_marks_do_not_depend_on_font_glyphs():
    hud = Hud("t", "s", "f")
    hud.tasks = [["a", "done"], ["b", "current"], ["c", "pending"],
                 ["d", "failed"]]
    hud.truth = [("red", 0.5, True), ("blue", 9.2, False)]
    frame = hud.draw(Image.new("RGB", (SCENE_W, H)))
    colours = {frame.getpixel((x, y)) for x in range(SCENE_W + 24, SCENE_W + 40)
               for y in range(190, 300)}
    from record_demo import BLUE, GREEN, RED
    assert {GREEN, BLUE, RED} <= colours
