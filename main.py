# =============================================================
#   MAIN - AI Powered Robot Control System
#   Full integration: Robot + AI + Vision + Motion + Obstacles
#   + Direct Joint Control
# =============================================================

import sys
import time
import pybullet as p

import config as cfg

from utils.logger import RobotLogger
from core.robot import FrankaPandaRobot
from core.ai_controller import AIController
from core.task_planner import TaskPlanner
from core.joint_controller import JointController
from perception.clip_detector import CLIPDetector


def build_configs():
    robot_config = {
        "robot_limits": cfg.ROBOT_LIMITS[cfg.ROBOT_TYPE],
        "sim": {
            "gravity":        cfg.SIM_GRAVITY,
            "timestep":       cfg.SIM_TIMESTEP,
            "steps_per_move": cfg.SIM_STEPS_PER_MOVE,
            "gui":            cfg.SIM_GUI
        },
        "motion": {
            "planner":            cfg.MOTION["planner"],
            "rrt_max_iterations": cfg.MOTION["rrt_max_iterations"],
            "rrt_step_size":      cfg.MOTION["rrt_step_size"],
            "rrt_goal_threshold": cfg.MOTION["rrt_goal_threshold"],
            "smooth_trajectory":  cfg.MOTION["smooth_trajectory"],
            "trajectory_points":  cfg.MOTION["trajectory_points"],
            "max_velocity":       cfg.ROBOT_LIMITS[cfg.ROBOT_TYPE]["max_velocity"],
            "max_acceleration":   0.3
        },
        "camera": cfg.CAMERA
    }
    ai_config = {
        "api_key":       cfg.ANTHROPIC_API_KEY,
        "model":         cfg.AI["model"],
        "max_tokens":    cfg.AI["max_tokens"],
        "memory_length": cfg.AI["memory_length"]
    }
    planner_config = {
        "api_key":       cfg.ANTHROPIC_API_KEY,
        "model":         cfg.AI["model"],
        "max_tokens":    cfg.AI["max_tokens"],
        "memory_length": cfg.AI["memory_length"]
    }
    return robot_config, ai_config, planner_config


def execute_plan(plan, robot, ai, logger):
    steps     = plan.get("steps", [])
    task_name = plan.get("task_name", plan.get("intent", "task"))
    logger.info(f"Executing: '{task_name}' ({len(steps)} steps)")

    for i, step in enumerate(steps):
        action = step.get("action", "")
        desc   = step.get("description", action)
        logger.info(f"Step {i+1}/{len(steps)}: {desc}")

        try:
            if action == "pick":
                obj     = step.get("object", "")
                success = robot.pick_object(obj)
                if success:
                    ai.update_state(
                        robot_position=robot.get_end_effector_position(),
                        holding=obj
                    )

            elif action == "place":
                x = step.get("x", 0.0)
                y = step.get("y", 0.4)
                robot.place_object(x, y)
                ai.update_state(
                    robot_position=robot.get_end_effector_position(),
                    holding=None
                )

            elif action == "move":
                x = step.get("x", 0.0)
                y = step.get("y", 0.0)
                z = step.get("z", 0.5)
                robot.move_to_position(x, y, z)
                ai.update_state(
                    robot_position=robot.get_end_effector_position()
                )

            elif action == "home":
                robot.home()
                ai.update_state(
                    robot_position=robot.get_end_effector_position(),
                    holding=None
                )

            elif action == "scan":
                detected = robot.scan_scene()
                ai.update_state(detected_objects=detected)

            elif action == "open_gripper":
                robot.open_gripper()

            elif action == "close_gripper":
                robot.close_gripper()

            else:
                logger.warning(f"Unknown action: {action}")

        except Exception as e:
            logger.log_error(e, f"Step {i+1}: {action}")
            print(f"  Error in step: {e}")

        for _ in range(10):
            robot.step()


def handle_special_command(user_input, robot, ai, planner, joints, logger):
    cmd = user_input.lower().strip()

    # ---- Joint control commands ----
    joint_keywords = [
        "joint", "j1", "j2", "j3", "j4", "j5", "j6", "j7",
        "base", "shoulder", "elbow", "forearm", "wrist",
        "upper_arm", "show joints", "reset joints", "wave",
        "joint help"
    ]
    if any(kw in cmd for kw in joint_keywords):
        if cmd == "joint help":
            joints.print_joint_help()
        elif cmd == "wave":
            joints.wave()
        elif any(w in cmd for w in ["show joints", "joint status", "joint angles"]):
            joints.show_joint_status()
        elif "reset joints" in cmd:
            joints.reset_to_home()
        else:
            joints.parse_and_execute(user_input)
        return True

    # ---- Scan ----
    if cmd in ["scan", "scan scene", "look"]:
        detected = robot.scan_scene()
        ai.update_state(detected_objects=detected)
        planner.update_objects(detected)
        return True

    # ---- Stack all ----
    if "stack" in cmd and "all" in cmd:
        detected = robot.scan_scene()
        plan     = planner.plan_stack_all(detected)
        execute_plan(plan, robot, ai, logger)
        planner.add_to_memory(user_input, plan["steps"])
        return True

    # ---- Sort ----
    if cmd.startswith("sort"):
        detected = robot.scan_scene()
        plan     = planner.plan_sort_by_color(detected)
        execute_plan(plan, robot, ai, logger)
        planner.add_to_memory(user_input, plan["steps"])
        return True

    # ---- Home ----
    if cmd in ["home", "reset", "go home"]:
        robot.home()
        return True

    # ---- Obstacles ----
    if cmd in ["obstacles", "list obstacles", "show obstacles"]:
        robot.list_obstacles()
        return True

    if cmd.startswith("remove obstacle") or cmd.startswith("remove obs"):
        parts = cmd.split()
        if len(parts) >= 3:
            robot.remove_obstacle(parts[-1])
        else:
            print("  Usage: remove obstacle <name>")
        return True

    if cmd.startswith("add obstacle"):
        parts = cmd.split()
        try:
            name = parts[2]
            x    = float(parts[3])
            y    = float(parts[4])
            robot.add_obstacle(
                name=name, x=x, y=y,
                size=[0.03, 0.15, 0.15],
                color=[0.6, 0.3, 0.1, 0.9]
            )
            print(f"  Obstacle '{name}' added at ({x}, {y})")
        except (IndexError, ValueError):
            print("  Usage: add obstacle <name> <x> <y>")
        return True

    # ---- Ask ----
    if cmd.startswith("?") or cmd.startswith("ask "):
        question = cmd.lstrip("?").lstrip("ask ").strip()
        answer   = ai.ask(question)
        print(f"\n  AI: {answer}\n")
        return True

    # ---- Suggest ----
    if cmd in ["suggest", "what next"]:
        suggestion = ai.suggest_next_action()
        print(f"\n  AI Suggestion: {suggestion}\n")
        return True

    # ---- Memory ----
    if cmd in ["clear memory", "forget"]:
        ai.clear_memory()
        planner.memory = []
        print("  Memory cleared.")
        return True

    # ---- Status ----
    if cmd in ["status", "state"]:
        pos     = robot.get_end_effector_position()
        holding = robot.grasped_object or "nothing"
        print(f"\n  Position:  ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        print(f"  Holding:   {holding}")
        print(f"  Objects:   {list(robot.objects.keys())}")
        print(f"  Obstacles: {list(robot.obstacles.keys())}\n")
        return True

    # ---- Help ----
    if cmd in ["help", "h"]:
        print_help()
        return True

    return False


def print_help():
    print("""
========================================
  AI ROBOT COMMAND REFERENCE
========================================
  NATURAL LANGUAGE (AI powered):
    pick up red block
    move to position 0.3 0.2 0.5
    go to blue block
    stack all blocks
    sort blocks by color

  JOINT CONTROL:
    show joints               show all joint angles
    reset joints              return to home
    move joint 3 to 45        set joint 3 to 45 degrees
    rotate joint 1 by -30     rotate joint 1 by -30 degrees
    set elbow to 90           use joint name
    rotate wrist by 45        relative rotation
    wave                      robot waves hand
    joint help                full joint command list

  OBSTACLE COMMANDS:
    obstacles                 list all obstacles
    add obstacle <n> <x> <y>  add new wall
    remove obstacle <n>       remove obstacle

  BUILT-IN:
    scan      scan scene        home    go home
    status    robot state       suggest AI suggestion
    ? <q>     ask AI question   forget  clear memory

    help   show this menu
    quit   exit
========================================
""")


def main():
    print("""
========================================
  AI ROBOT CONTROL SYSTEM
  Franka Panda + Claude AI + RRT
  + Direct Joint Control
========================================
""")

    robot_config, ai_config, planner_config = build_configs()

    logger = RobotLogger(cfg.LOGGING)
    logger.log_robot_mode(cfg.MODE, cfg.ROBOT_TYPE)

    print("Initializing systems...")
    robot   = FrankaPandaRobot(robot_config, logger)
    ai      = AIController(ai_config, logger)
    planner = TaskPlanner(planner_config, logger)
    joints  = JointController(robot, logger)

    clip = CLIPDetector(logger)
    clip.load()

    # Add objects
    print("\nAdding objects...")
    robot.add_object("red",   x=0.4,  y=0.0,  color=[1, 0, 0, 1])
    robot.add_object("blue",  x=0.3,  y=0.3,  color=[0, 0, 1, 1])
    robot.add_object("green", x=0.3,  y=-0.3, color=[0, 1, 0, 1])

    # Add obstacles
    print("Adding obstacles...")
    robot.add_obstacle(
        "wall1", x=0.35, y=0.15,
        size=[0.02, 0.12, 0.15],
        color=[0.6, 0.6, 0.6, 0.9]
    )
    robot.add_obstacle(
        "wall2", x=0.15, y=-0.1,
        size=[0.02, 0.12, 0.15],
        color=[0.6, 0.6, 0.6, 0.9]
    )

    for _ in range(100):
        robot.step()

    print("Scanning scene...")
    detected = robot.scan_scene()
    ai.update_state(detected_objects=detected)
    planner.update_objects(detected)

    print("\nSystem ready!")
    print_help()

    while True:
        try:
            robot.step()
            user_input = input("Command: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Shutting down...")
                logger.get_session_summary()
                logger.save_session()
                robot.disconnect()
                break

            if handle_special_command(
                user_input, robot, ai, planner, joints, logger
            ):
                continue

            # AI command
            detected = robot.scan_scene()
            ai.update_state(
                robot_position=robot.get_end_effector_position(),
                detected_objects=detected
            )
            planner.update_objects(detected)

            print("AI thinking...")
            command    = ai.process_command(user_input, detected)
            confidence = command.get("confidence", 1.0)

            print(f"\n  AI: {command.get('response', 'executing...')}")
            if confidence < 0.6:
                print(f"  (confidence: {confidence:.0%} - may have misunderstood)")

            execute_plan(command, robot, ai, logger)
            planner.add_to_memory(
                user_input, command.get("steps", []), "completed"
            )

            pos = robot.get_end_effector_position()
            print(f"\n  Robot at: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        except KeyboardInterrupt:
            print("\nInterrupted - shutting down...")
            logger.get_session_summary()
            logger.save_session()
            robot.disconnect()
            break

        except Exception as e:
            logger.log_error(e, "Main loop")
            print(f"  Error: {e}")
            print("  Type 'home' to reset")


if __name__ == "__main__":
    main()