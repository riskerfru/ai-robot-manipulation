"""Offline tests for placement thresholds. No PyBullet, no API key."""

from place_verifier import PlacementVerifier


class FakeRobot:
    def __init__(self, pos, reachable=True, distance=0.4):
        self.pos = pos
        self.reachable = reachable
        self.distance = distance
        self.picks = 0
        self.places = 0

    def get_object_position(self, name):
        return self.pos

    def check_reachability(self, x, y):
        return {"reachable": self.reachable, "distance": self.distance}

    def pick_object(self, name):
        self.picks += 1
        return {"status": "success"}

    def place_object(self, x, y):
        self.places += 1
        # Land on target after one retry so the loop terminates.
        self.pos = (x, y, 0.16)
        return {"status": "success"}

    def get_end_effector_position(self):
        return (0.3, 0.0, 0.5)

    def move_to_position(self, x, y, z, use_rrt=False):
        return bool(self.reachable)


def test_accurate_place_is_success():
    v = PlacementVerifier(FakeRobot((0.60, 0.25, 0.16)))
    out = v.verify_and_fix("red", 0.60, 0.25, slot_num=1)
    assert out["status"] == "success"
    assert out["drift_cm"] < 1.0
    assert out["retried"] is False


def test_five_to_eight_cm_is_accepted_with_warning():
    v = PlacementVerifier(FakeRobot((0.60, 0.31, 0.16)))  # 6 cm in y
    out = v.verify_and_fix("red", 0.60, 0.25, slot_num=1)
    assert out["status"] == "success"
    assert 5.0 <= out["drift_cm"] < 8.0


def test_off_table_is_hard_fail():
    v = PlacementVerifier(FakeRobot((0.60, 0.25, 0.01)))
    out = v.verify_and_fix("red", 0.60, 0.25)
    assert out["status"] == "failed"
    assert "fell off table" in out["detail"]


def test_missing_object_is_fail():
    bot = FakeRobot((0.60, 0.25, 0.16))
    bot.pos = None
    v = PlacementVerifier(bot)
    out = v.verify_and_fix("red", 0.60, 0.25)
    assert out["status"] == "failed"


def test_large_drift_retries_when_reachable():
    v = PlacementVerifier(FakeRobot((0.60, 0.40, 0.16)))  # 15 cm
    out = v.verify_and_fix("red", 0.60, 0.25, slot_num=2)
    assert v.robot.picks >= 1
    assert out["status"] in ("success", "recovered")


def test_unreachable_large_drift_fails_honestly():
    v = PlacementVerifier(FakeRobot((0.90, 0.90, 0.16), reachable=False, distance=1.2))
    out = v.verify_and_fix("red", 0.60, 0.25, slot_num=2)
    assert out["status"] == "failed"
    assert "out of reach" in out["detail"]


def test_calibration_average_and_apply():
    v = PlacementVerifier(FakeRobot((0.55, 0.25, 0.16)))
    v._record_drift(2, 0.60, 0.25, 0.55, 0.25)
    v._record_drift(2, 0.60, 0.25, 0.54, 0.25)
    cx, cy = v._apply_calibration(0.60, 0.25, 2)
    assert cx > 0.60
    assert abs(cy - 0.25) < 1e-9
