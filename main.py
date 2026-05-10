# =============================================================
#   MAIN - AI Powered Robot Control System
#   Full integration: Robot + AI + Vision + Motion + Obstacles
#   + Direct Joint Control
#   + Autonomous Agent Loop (--auto flag)
# =============================================================

import sys
import time
import argparse
import pybullet as p

import config as cfg

from utils.logger import RobotLogger
from core.robot import FrankaPandaRobot
from core.ai_controller import AIController
from core.task_planner import TaskPlanner
from core.joint_controller import JointController
from core.robot_tools import RobotTools
from perception.vision import ClaudeVision
from perception.clip_detector import CLIPDetector

# Agent loop imports
from agent_loop import WarehouseAgent, EMERGENCY_STOP
from reachability_viz import ReachabilityVisualiser


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
    intent    = plan.get("intent", "")
    logger.info(f"Executing: '{task_name}' ({len(steps)} steps)")

    if intent == "pick":
        target_obj = plan.get("target_object")
        if not target_obj:
            for step in steps:
                if step.get("action") == "pick" and step.get("object"):
                    target_obj = step["object"]
                    break
        if target_obj:
            logger.info(f"Direct pick: '{target_obj}'")
            success = robot.pick_object(target_obj)
            # Handle both bool and dict returns
            ok = success if isinstance(success, bool) else success.get("status") == "success"
            if ok:
                ai.update_state(
                    robot_position=robot.get_end_effector_position(),
                    holding=target_obj
                )
            return

    for i, step in enumerate(steps):
        action = step.get("action", "")
        desc   = step.get("description", action)
        logger.info(f"Step {i+1}/{len(steps)}: {desc}")

        try:
            if action == "pick":
                obj     = step.get("object", "")
                success = robot.pick_object(obj)
                ok = success if isinstance(success, bool) else success.get("status") == "success"
                if ok:
                    ai.update_state(
                        robot_position=robot.get_end_effector_position(),
                        holding=obj
                    )

            elif action == "place":
                x        = step.get("x", 0.0)
                y        = step.get("y", 0.4)
                stack_on = step.get("stack_on", None)
                robot.place_object(x, y, stack_on=stack_on)
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


def handle_special_command(user_input, robot, ai, planner, joints, logger,
                            vision=None, agent=None):
    cmd = user_input.lower().strip()

    # ---- Emergency stop ----
    if cmd in ["stop", "estop", "e-stop", "emergency stop", "halt"]:
        EMERGENCY_STOP.trigger("user command")
        print("  🛑 Emergency stop triggered. Type 'resume' to clear.")
        return True

    if cmd in ["resume", "clear stop"]:
        EMERGENCY_STOP.reset()
        return True

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

    # ---- Reachability check ----
    if cmd.startswith("reach ") or cmd.startswith("check reach"):
        parts = cmd.split()
        try:
            x = float(parts[-2])
            y = float(parts[-1])
            result = robot.check_reachability(x, y)
            icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}.get(
                result["severity"], "?"
            )
            print(f"\n  {icon} {result['message']}")
            if result["hint"]:
                print(f"  💡 {result['hint']}")
        except (IndexError, ValueError):
            print("  Usage: check reach <x> <y>")
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
        print(f"  Obstacles: {list(robot.obstacles.keys())}")
        print(f"  E-Stop:    {'ACTIVE' if EMERGENCY_STOP.is_active() else 'clear'}\n")
        return True

    # ---- Help ----
    if cmd in ["help", "h"]:
        print_help()
        return True

    # ---- Camera view commands ----
    if cmd.startswith("view") or cmd == "camera":
        parts = cmd.split()
        preset = parts[1] if len(parts) > 1 else "default"
        robot.set_camera(preset)
        return True

    # ---- Claude Vision commands ----
    if vision and cmd in ["see", "vision", "identify"]:
        if cmd in ["see", "vision"]:
            print("  Claude Vision analysing scene...")
            result = vision.analyse_scene(robot.robot)
            print(f"\n  Claude sees:\n  {result}\n")
        elif cmd == "identify":
            result = vision.identify_objects(robot.robot)
            print(f"\n  Objects identified:\n  {result}\n")
        return True

    return False


def print_help():
    print("""
========================================
  AI ROBOT COMMAND REFERENCE
========================================
  AUTONOMOUS AGENT MODE:
    auto <order>              run autonomous agent
    Examples:
      auto pick red and place at dispatch
      auto stack blue on green, then bring red to me
      auto fulfil order: sort all blocks by colour
    stop / estop              emergency stop agent
    resume                    clear emergency stop

  NATURAL LANGUAGE (AI powered):
    pick up red block
    move to position 0.3 0.2 0.5
    stack all blocks
    sort blocks by color

  DEBUG:
    check reach <x> <y>      check if position is reachable
    status                    robot state + e-stop status
    scan                      scan scene

  JOINT CONTROL:
    show joints               show all joint angles
    reset joints              return to home
    move joint 3 to 45        set joint 3 to 45 degrees
    wave                      robot waves hand
    joint help                full joint command list

  OBSTACLE COMMANDS:
    obstacles                 list all obstacles
    add obstacle <n> <x> <y>  add new wall
    remove obstacle <n>       remove obstacle

  BUILT-IN:
    home    go home           forget  clear memory
    ? <q>   ask AI question   suggest AI suggestion
    see     Claude Vision     identify  object ID
    help    show this menu    quit    exit
========================================
""")


def run_interactive(robot, ai, planner, joints, logger, vision, agent,
                    ai_config, viz=None):
    """Standard interactive command loop"""
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

            # ---- Autonomous agent mode ----
            if user_input.lower().startswith("auto "):
                order = user_input[5:].strip()
                if not order:
                    print("  Usage: auto <order description>")
                    continue

                print(f"\n  🤖 Starting autonomous agent...")
                result = agent.run(order)

                print(f"\n  📊 Agent result:")
                print(f"     Success:    {result['success']}")
                print(f"     Completed:  {result['completed']}")
                print(f"     Failed:     {result['failed']}")
                print(f"     Steps:      {result['steps_taken']}")
                print(f"     Summary:    {result['summary']}\n")
                continue

            if handle_special_command(
                user_input, robot, ai, planner, joints, logger, vision, agent
            ):
                if viz:
                    viz.update_objects()
                continue

            # AI command (existing behaviour)
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

            if viz:
                viz.update_objects()
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


def run_auto_demo(robot, ai_config, logger, order):
    """Run a single autonomous order and exit — for demo/submission"""
    agent = WarehouseAgent(robot, ai_config, logger)
    result = agent.run(order)

    print("\n" + "="*50)
    print("DEMO COMPLETE")
    print("="*50)
    print(f"Order:     {order}")
    print(f"Success:   {result['success']}")
    print(f"Completed: {result['completed']}")
    print(f"Failed:    {result['failed']}")
    print(f"Steps:     {result['steps_taken']}")
    print(f"Summary:   {result['summary']}")
    print("="*50)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="AI Robot Control System — Franka Panda + Claude AI"
    )
    parser.add_argument(
        "--auto",
        type=str,
        default=None,
        help='Run autonomous agent with given order. '
             'Example: --auto "pick red, stack blue on green"'
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Place objects at random positions before starting"
    )
    args = parser.parse_args()

    print("""
========================================
  AI ROBOT CONTROL SYSTEM
  Franka Panda + Claude AI + RRT
  + Autonomous Warehouse Agent
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
    tools   = RobotTools(robot, logger)
    ai.set_tools(tools)
    vision  = ClaudeVision(ai_config, logger)
    agent   = WarehouseAgent(robot, ai_config, logger)

    clip = CLIPDetector(logger)
    clip.load()

    # Add table — visual surface only (not an RRT obstacle)
    # Table top at z=0.1m, centred at (0.35, 0.0)
    # Size: 0.7m wide (x), 0.8m deep (y), 0.1m thick
    print("\nAdding table...")
    TABLE_Z = 0.1          # table top surface height
    TABLE_CX = 0.42        # table centre x
    TABLE_CY = 0.0         # table centre y
    TABLE_W  = 0.42        # half-width: 0.0 to 0.84m in x  (bigger)
    TABLE_D  = 0.50        # half-depth: -0.50 to 0.50m in y (bigger)
    TABLE_H  = 0.05        # half-thickness

    table_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_W, TABLE_D, TABLE_H]
    )
    table_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[TABLE_W, TABLE_D, TABLE_H],
        rgbaColor=[0.55, 0.35, 0.15, 1.0]   # warm wood brown
    )
    table_id = p.createMultiBody(
        baseMass=0,                          # static
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[TABLE_CX, TABLE_CY, TABLE_H]  # z = half-thickness
    )
    print(f"  Table surface at z={TABLE_Z:.2f}m  "
          f"x:[{TABLE_CX-TABLE_W:.2f} to {TABLE_CX+TABLE_W:.2f}]  "
          f"y:[{TABLE_CY-TABLE_D:.2f} to {TABLE_CY+TABLE_D:.2f}]")
    print(f"  Reachable zone: 0.25m - 0.75m from robot base")

    # Add objects ON the table surface
    # z = TABLE_Z + half-object-height (objects are 0.04m cubes -> z=0.14)
    OBJ_Z = TABLE_Z + 0.04   # sits on table top
    print("\nAdding objects on table...")
    import random
    if args.random:
        robot.add_object("red",
            x=round(random.uniform(0.20, 0.55), 2),
            y=round(random.uniform(-0.30, 0.30), 2),
            color=[1, 0, 0, 1], z=OBJ_Z)
        robot.add_object("blue",
            x=round(random.uniform(0.20, 0.55), 2),
            y=round(random.uniform(-0.30, 0.30), 2),
            color=[0, 0, 1, 1], z=OBJ_Z)
        robot.add_object("green",
            x=round(random.uniform(0.20, 0.55), 2),
            y=round(random.uniform(-0.30, 0.30), 2),
            color=[0, 1, 0, 1], z=OBJ_Z)
        print("  Objects placed at random positions on table")
    else:
        # Objects in safe green zone (0.30-0.65m from base)
        robot.add_object("red",   x=0.55, y=0.00,  color=[1, 0, 0, 1], z=OBJ_Z)
        robot.add_object("blue",  x=0.50, y=0.30,  color=[0, 0, 1, 1], z=OBJ_Z)
        robot.add_object("green", x=0.50, y=-0.30, color=[0, 1, 0, 1], z=OBJ_Z)

    # No walls — clean table scene for warehouse demo

    # ── CONVEYOR SLOTS ──
    # Three numbered slots at far edge of table.
    # No colours — avoids confusion with block colours.
    # Slot 1 = leftmost (y=+0.30), Slot 2 = centre (y=0.00), Slot 3 = rightmost (y=-0.30)
    SLOT_X  = 0.68    # x position of all slots (within table, near far edge)
    SLOT_Z  = 0.10    # same height as table surface
    SLOT_W  = 0.07    # half-width of slot marker
    SLOT_H  = 0.004   # very thin marker

    SLOTS = {
        1: (SLOT_X,  0.30),
        2: (SLOT_X,  0.00),
        3: (SLOT_X, -0.30),
    }

    # Conveyor base platform — dark grey
    conv_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.08, 0.44, 0.005])
    conv_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.08, 0.44, 0.005],
        rgbaColor=[0.15, 0.15, 0.15, 1.0])
    p.createMultiBody(baseMass=0,
                      baseCollisionShapeIndex=conv_col,
                      baseVisualShapeIndex=conv_vis,
                      basePosition=[SLOT_X, 0.0, SLOT_Z + 0.005])

    # Slot markers — all white/light grey (no colour coding)
    SLOT_COLOUR = [0.85, 0.85, 0.85, 1.0]   # light grey
    for slot_num, (sx, sy) in SLOTS.items():
        sc = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[SLOT_W, SLOT_W, SLOT_H])
        sv = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[SLOT_W, SLOT_W, SLOT_H],
            rgbaColor=SLOT_COLOUR)
        p.createMultiBody(baseMass=0,
                          baseCollisionShapeIndex=sc,
                          baseVisualShapeIndex=sv,
                          basePosition=[sx, sy, SLOT_Z + 0.012])

        # Slot number label (debug line from slot to above)
        p.addUserDebugText(
            text=str(slot_num),
            textPosition=[sx, sy, SLOT_Z + 0.08],
            textColorRGB=[1, 1, 1],
            textSize=1.5
        )
        p.addUserDebugLine(
            [sx - SLOT_W, sy - SLOT_W, SLOT_Z + 0.013],
            [sx + SLOT_W, sy - SLOT_W, SLOT_Z + 0.013],
            [0.9, 0.9, 0.9], 2
        )

    # Store slot positions so agent_loop can access them
    robot.conveyor_slots = SLOTS

    print(f"  Conveyor slots (numbered, no colour coding):")
    for n, (sx, sy) in SLOTS.items():
        print(f"    Slot {n}: ({sx}, {sy:+.2f})")

    for _ in range(100):
        robot.step()

    print("Scanning scene...")
    detected = robot.scan_scene()
    ai.update_state(detected_objects=detected)
    planner.update_objects(detected)

    # Draw reachability zones in PyBullet viewer
    viz = ReachabilityVisualiser(robot)
    viz.draw_zones()
    viz.draw_table_outline(cx=0.40, cy=0.0, w=0.38, d=0.45)
    viz.update_objects()
    print("  [REACH] Reachability zones drawn in viewer")

    # ---- Run mode ----
    if args.auto:
        # Autonomous demo mode
        print(f"\n  🤖 AUTO MODE: {args.auto}")
        result = run_auto_demo(robot, ai_config, logger, args.auto)
        logger.save_session()
        robot.disconnect()
    else:
        # Interactive mode
        run_interactive(robot, ai, planner, joints, logger, vision,
                        agent, ai_config, viz=viz)


if __name__ == "__main__":
    main()