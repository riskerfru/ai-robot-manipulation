# =============================================================
#   SIMULATION BACKEND - PyBullet Franka Panda
#   Builds the simulated robot for GUI or headless use.
# =============================================================

import config as cfg
from hardware.interface import RobotInterface


def build_robot_config(gui):
    limits = cfg.ROBOT_LIMITS["franka_panda"]
    return {
        "robot_limits": limits,
        "sim": {
            "gravity":        cfg.SIM_GRAVITY,
            "timestep":       cfg.SIM_TIMESTEP,
            "steps_per_move": cfg.SIM_STEPS_PER_MOVE,
            "gui":            gui,
        },
        "motion": {
            "planner":            cfg.MOTION["planner"],
            "rrt_max_iterations": cfg.MOTION["rrt_max_iterations"],
            "rrt_step_size":      cfg.MOTION["rrt_step_size"],
            "rrt_goal_threshold": cfg.MOTION["rrt_goal_threshold"],
            "smooth_trajectory":  cfg.MOTION["smooth_trajectory"],
            "trajectory_points":  cfg.MOTION["trajectory_points"],
            "max_velocity":       limits["max_velocity"],
            "max_acceleration":   0.3,
        },
        "camera": cfg.CAMERA,
    }


def make_sim_robot(logger, gui=False, realtime=None, save_wrist_frames=None):
    """
    Create the PyBullet robot.

    realtime=False runs physics as fast as the CPU allows (headless
    evaluation). The physics timestep is unchanged; only the
    wall-clock pacing between steps is removed.
    """
    from core.robot import FrankaPandaRobot

    realtime = gui if realtime is None else realtime
    save_wrist_frames = gui if save_wrist_frames is None else save_wrist_frames

    robot = FrankaPandaRobot(build_robot_config(gui), logger)
    if not realtime:
        # sim_config["timestep"] is read after start-up only to pace
        # the loop with time.sleep(); the engine keeps its own value.
        robot.sim_config["timestep"] = 0
    if not save_wrist_frames:
        robot.camera.show_wrist_view = lambda *a, **k: None
    return robot


# The simulated robot already provides the interface methods.
def _register():
    try:
        from core.robot import FrankaPandaRobot
        RobotInterface.register(FrankaPandaRobot)
    except Exception:
        pass


_register()
