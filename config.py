# =============================================================
#   CENTRAL CONFIGURATION FILE
#   All settings for the entire project in one place
#   Change values here — no need to touch other files
# =============================================================

# --------------------------------------------------------------
# API KEYS
# --------------------------------------------------------------
from dotenv import load_dotenv
import os
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# --------------------------------------------------------------
# MODE: Switch between simulation and real robot here
# --------------------------------------------------------------
MODE = "simulation"   # Options: "simulation" | "real_robot"

# --------------------------------------------------------------
# ROBOT SELECTION
# --------------------------------------------------------------
# Options:
#   "franka_panda"   - Franka Panda (simulation default)
#   "ur5e"           - Universal Robots UR5e
#   "ur10e"          - Universal Robots UR10e
#   "ur3e"           - Universal Robots UR3e
#   "kuka_iiwa"      - Kuka iiwa
#   "custom"         - Your own URDF file
ROBOT_TYPE = "franka_panda"

# If ROBOT_TYPE is "custom", provide path to your URDF file
CUSTOM_URDF_PATH = ""  # e.g. "C:/robots/my_robot.urdf"

# --------------------------------------------------------------
# SIMULATION SETTINGS
# --------------------------------------------------------------
SIM_GRAVITY         = -9.81
SIM_TIMESTEP        = 1 / 240
SIM_STEPS_PER_MOVE  = 300
SIM_GUI             = True       # False = headless (no window)

# --------------------------------------------------------------
# ROBOT PHYSICAL LIMITS
# --------------------------------------------------------------
ROBOT_LIMITS = {
    "franka_panda": {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
        "z": (0.05, 1.2),
        "max_velocity": 1.7,      # rad/s
        "max_force": 500,
        "end_effector_index": 11,
        "arm_joints": [0, 1, 2, 3, 4, 5, 6],
        "finger_joints": [9, 10],
        "urdf": "franka_panda/panda.urdf",
        "home_position": [0, -0.5, 0, -2.0, 0, 1.5, 0.7]
    },
    "ur5e": {
        "x": (-0.85, 0.85),
        "y": (-0.85, 0.85),
        "z": (0.05, 1.0),
        "max_velocity": 1.0,
        "max_force": 150,
        "end_effector_index": 6,
        "arm_joints": [0, 1, 2, 3, 4, 5],
        "finger_joints": [],
        "urdf": "ur5e/ur5e.urdf",  # Requires download
        "home_position": [0, -1.57, 1.57, -1.57, -1.57, 0]
    },
    "ur10e": {
        "x": (-1.2, 1.2),
        "y": (-1.2, 1.2),
        "z": (0.05, 1.3),
        "max_velocity": 1.0,
        "max_force": 330,
        "end_effector_index": 6,
        "arm_joints": [0, 1, 2, 3, 4, 5],
        "finger_joints": [],
        "urdf": "ur10e/ur10e.urdf",  # Requires download
        "home_position": [0, -1.57, 1.57, -1.57, -1.57, 0]
    },
    "kuka_iiwa": {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
        "z": (0.05, 1.0),
        "max_velocity": 1.7,
        "max_force": 300,
        "end_effector_index": 6,
        "arm_joints": [0, 1, 2, 3, 4, 5, 6],
        "finger_joints": [],
        "urdf": "kuka_iiwa/model.urdf",  # Built into PyBullet
        "home_position": [0, 0, 0, -1.57, 0, 1.57, 0]
    }
}

# --------------------------------------------------------------
# REAL ROBOT COMMUNICATION (Phase 5 - used later)
# --------------------------------------------------------------
REAL_ROBOT_SETTINGS = {
    "ur5e": {
        "ip": "192.168.1.100",   # Robot IP address on your network
        "port": 30004,            # RTDE port
        "frequency": 500,         # Hz
        "protocol": "rtde"        # rtde | ros2
    },
    "ur10e": {
        "ip": "192.168.1.101",
        "port": 30004,
        "frequency": 500,
        "protocol": "rtde"
    },
    "franka_panda": {
        "ip": "192.168.1.102",
        "port": 1337,
        "frequency": 1000,
        "protocol": "fci"         # Franka Control Interface
    }
}

# --------------------------------------------------------------
# SAFETY SETTINGS (critical for real robot)
# --------------------------------------------------------------
SAFETY = {
    "max_joint_velocity": 1.0,    # rad/s - hard limit
    "max_joint_acceleration": 0.5, # rad/s²
    "workspace_padding": 0.05,     # metres - buffer from limits
    "collision_threshold": 10.0,   # Nm - torque spike = collision
    "emergency_stop_key": "q",     # Press to instantly stop robot
    "enable_safety_monitor": True
}

# --------------------------------------------------------------
# CAMERA SETTINGS
# --------------------------------------------------------------
CAMERA = {
    "width": 640,
    "height": 480,
    "fov": 60,
    "position": [0, 0, 2.0],      # Overhead position
    "target": [0, 0, 0],
    "near": 0.1,
    "far": 10.0
}

# --------------------------------------------------------------
# AI SETTINGS
# --------------------------------------------------------------
AI = {
    "model": "claude-opus-4-5",
    "max_tokens": 500,
    "memory_length": 10,           # How many past commands to remember
    "temperature": 0.1             # Low = more deterministic robot commands
}

# --------------------------------------------------------------
# MOTION PLANNING SETTINGS
# --------------------------------------------------------------
MOTION = {
    "planner": "rrt",              # rrt | straight_line
    "rrt_max_iterations": 1000,
    "rrt_step_size": 0.05,
    "rrt_goal_threshold": 0.05,
    "smooth_trajectory": True,
    "trajectory_points": 50
}

# --------------------------------------------------------------
# LOGGING
# --------------------------------------------------------------
LOGGING = {
    "enabled": True,
    "log_file": "logs/robot_session.log",
    "log_level": "INFO",           # DEBUG | INFO | WARNING | ERROR
    "log_commands": True,
    "log_positions": True,
    "replay_enabled": True         # Save session for replay
}