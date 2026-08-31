import config


def test_default_mode_is_simulation():
    assert config.MODE == "simulation"


def test_panda_limits_are_sane():
    lim = config.ROBOT_LIMITS["franka_panda"]
    assert lim["z"][0] < lim["z"][1]
    assert len(lim["arm_joints"]) == 7
    assert lim["urdf"].endswith("panda.urdf")


def test_api_key_is_not_hardcoded():
    # Key must come from the environment, never the repo.
    assert config.ANTHROPIC_API_KEY == "" or not config.ANTHROPIC_API_KEY.startswith("sk-ant-hardcoded")
