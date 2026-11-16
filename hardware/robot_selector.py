# =============================================================
#   ROBOT SELECTOR - one place that decides which backend runs
# =============================================================

import config as cfg


def make_robot(logger, mode=None, gui=None, pose_source=None, ur_config=None):
    """
    mode "simulation": PyBullet Franka Panda.
    mode "real_robot": UR e-Series over RTDE (needs `ur_rtde`, a
                       surveyed URCellConfig and a pose source).
    """
    mode = mode or cfg.MODE
    if mode == "simulation":
        from hardware.sim_backend import make_sim_robot
        return make_sim_robot(logger, gui=cfg.SIM_GUI if gui is None else gui)

    if mode == "real_robot":
        from hardware.ur_rtde import URCellConfig, URRobot
        if pose_source is None:
            raise ValueError(
                "a real cell needs a pose source: FixturePoseSource for "
                "nominal positions, or a measuring source so placements "
                "can be verified")
        if ur_config is None:
            settings = cfg.REAL_ROBOT_SETTINGS["ur5e"]
            ur_config = URCellConfig(ip=settings["ip"])
        robot = URRobot(ur_config, pose_source, logger=logger)
        if not robot.can_measure_parts and logger is not None:
            logger.warning(
                "Pose source does not measure part positions: placements "
                "cannot be verified and jobs will not report DONE.")
        return robot

    raise ValueError(f"unknown mode {mode!r}")
