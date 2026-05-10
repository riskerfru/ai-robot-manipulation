# AI Robot Manipulation — Franka Panda + Claude AI

A simulation of an AI-powered robot manipulation system built with PyBullet, featuring a Claude-driven autonomous warehouse agent, visual servoing, RRT* motion planning, and TOPP-RA trajectory generation.

---

## Demo

**Autonomous warehouse order fulfilment — all 6/6 subtasks completed:**

```
Command: auto dispatch all blocks to the conveyor

  🤖 WAREHOUSE AGENT STARTING
  📦 Order: dispatch all blocks to the conveyor

  ✅ pick red  →  place slot 1  ✅
  ✅ pick blue →  place slot 2  ✅
  ✅ pick green → place slot 3  ✅

  ORDER COMPLETE — 6/6 subtasks in 8 action steps
```

---

## Architecture

This project mirrors the KinetIQ layered agent architecture:

```
┌─────────────────────────────────────────────┐
│  System 3 — Natural Language Order          │
│  "dispatch all blocks to the conveyor"       │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│  System 2 — Claude AI Reasoning             │
│  agent_loop.py · WarehouseAgent             │
│  · Decomposes order into subtasks           │
│  · Observe → Think → Act → Verify loop     │
│  · Handles failures and replanning          │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│  System 1 — Robot Capabilities              │
│  core/robot.py                              │
│  · pick(object)    · place(x, y)            │
│  · place_slot(n)   · stack(obj, on)         │
│  · scan()          · home()                 │
└───────────────────┬─────────────────────────┘
                    │
┌───────────────────▼─────────────────────────┐
│  System 0 — PyBullet Physics                │
│  Franka Panda URDF · Rigid body dynamics    │
└─────────────────────────────────────────────┘
```

---

## Features

### Autonomous Agent Loop
- Natural language order decomposition via Claude API
- Structured observe → reason → act → verify cycle
- Automatic retry with replanning on failure (3 attempts per subtask)
- Known object list passed to decomposer — prevents hallucination
- Slot occupancy tracking across the full order

### Conveyor Slot System
- 3 numbered dispatch slots at fixed positions
- Auto-assignment fills slots in order (1 → 2 → 3)
- User can specify exact slot: `auto drop red at slot 2`
- Occupancy cleared when object is picked back up
- Slot state reset at start of each new order

### Motion Planning
- RRT* with 6 strategies (direct, arc-left, arc-right, high, elbow-up, reconfigure)
- TOPP-RA trajectory generation for smooth, time-optimal motion
- Collision-aware transport path planning with clearance buffer

### Visual Servoing
- Overhead camera with colour blob detection
- Oscillation prevention: skips servo if initial offset > 25px
- Damped corrections (0.3×) with 8mm per-step limit
- Drift safety: falls back to known position if servo moves > 3cm

### Reachability System
- Live reachability check before every pick (blocks unreachable attempts)
- PyBullet debug visualisation: ±110° working arc, green/orange/grey zones
- Robot footprint marker at base
- Per-object status pillars (green = safe, orange = edge, red = out of reach)

### Feedback System
- Grasp verification: checks object lifted after pick
- Place verification: checks drift from target after release
- Stack verification: checks height difference after stacking

### Emergency Stop
- Type `stop` at any time to halt agent mid-task
- Type `resume` to continue
- Also accessible via web interface

### Web Interface
- Live 2D top-down scene map with reachability zones
- Object position and status display
- Command input with quick command buttons
- E-stop button
- Live agent log

---

## Observation Format

Each agent reasoning step receives:

```
CONVEYOR SLOTS (place destinations):
  Slot 1: (0.63, +0.30) — empty
  Slot 2: (0.63,  0.00) — occupied by blue
  Slot 3: (0.63, -0.30) — empty

OBJECTS:
  red:   (0.55, 0.00, 0.14) | dist=0.55m | ✅ reachable
  blue:  (0.50, 0.30, 0.14) | dist=0.58m | ✅ reachable
  green: (0.50,-0.30, 0.14) | dist=0.58m | ✅ reachable

ROBOT:
  Position: (0.39, 0.00, 0.55)
  Holding:  nothing

LAST ACTION RESULT: success — picked red from (0.55, 0.00, 0.14)
COMPLETED: [scan, pick red]
CURRENT SUBTASK: place_slot slot 1
```

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `pick` | `object` | Pick named object from table |
| `place_slot` | `slot` (1/2/3) | Place in numbered conveyor slot |
| `place` | `x, y` | Place at arbitrary coordinates |
| `stack` | `object, on` | Stack object on top of another |
| `scan` | — | Scan scene, refresh object positions |
| `home` | — | Return robot to home position |

---

## Workspace

```
Robot base at origin (0, 0)
Table:  x [0.00 – 0.84m],  y [–0.50 – 0.50m]

Reachability zones (Franka Panda, ±110° arc):
  Grey zone:   dist < 0.25m  — too close, arm folds
  Green zone:  0.25 – 0.65m  — safe picking zone
  Orange zone: 0.65 – 0.75m  — edge of reach, may fail
  Red beyond:  > 0.75m       — out of reach

Conveyor slots (on table, far edge):
  Slot 1: (0.63,  0.30)  — dist 0.70m
  Slot 2: (0.63,  0.00)  — dist 0.63m
  Slot 3: (0.63, -0.30)  — dist 0.70m

Starting object positions:
  Red:   (0.55,  0.00)   — dist 0.55m
  Blue:  (0.50,  0.30)   — dist 0.58m
  Green: (0.50, -0.30)   — dist 0.58m
```

---

## Installation

```bash
git clone https://github.com/riskerfru/ai-robot-manipulation.git
cd ai-robot-manipulation
pip install pybullet anthropic numpy flask toppra
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

---

## Usage

```bash
python main.py
```

### Autonomous agent mode

```
auto dispatch all blocks to the conveyor
auto pick red and place in slot 2
auto stack blue on green, then move stack to slot 1
auto move red to 0.4 0.2
```

### Direct commands

```
pick up red block
move to position 0.4 0.0 0.5
stack all blocks
check reach 0.6 0.3
status
scan
home
```

### Web interface

```bash
python web_server.py
# Open http://localhost:5000
```

---

## Real-World Deployment Note

Object positions in simulation are provided by PyBullet physics (ground truth). In a real deployment this would be replaced by:

- **Intel RealSense D435** depth camera with point cloud segmentation
- **Calibrated overhead RGB camera** with homography projection
- **Force/torque sensing** for contact-based surface detection

The agent interface (observation format and action space) remains identical — only the perception layer changes. Visual servoing already handles the correction step for sensor noise (±3–5mm typical for depth cameras).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Physics simulation | PyBullet |
| Robot URDF | Franka Panda (pybullet_data) |
| AI reasoning | Claude claude-sonnet-4-20250514 |
| Motion planning | RRT* (6 strategies) |
| Trajectory generation | TOPP-RA |
| Visual perception | OpenCV colour segmentation |
| Web interface | Flask + vanilla JS |
| Language | Python 3.10+ |

---

## Project Structure

```
ai-robot-manipulation/
├── main.py                  # Entry point, scene setup
├── agent_loop.py            # Autonomous warehouse agent (System 2)
├── robot.py                 # Root copy (sync to core/)
├── config.py                # Robot and simulation config
├── core/
│   ├── robot.py             # FrankaPandaRobot — pick, place, move
│   ├── ai_controller.py     # Claude with tool use
│   ├── joint_controller.py  # Direct joint control
│   ├── task_planner.py      # High-level task decomposition
│   └── robot_tools.py       # Tool definitions for Claude API
├── motion/
│   ├── rrt_planner.py       # RRT* motion planner
│   ├── trajectory.py        # TOPP-RA trajectory generator
│   └── visual_servoing.py   # Camera-guided fine alignment
├── perception/
│   ├── camera.py            # PyBullet camera system
│   ├── vision.py            # Claude Vision integration
│   └── clip_detector.py     # CLIP object detection (optional)
├── reachability_viz.py      # PyBullet debug visualisation
├── web_server.py            # Flask web interface
├── index.html               # Web UI
└── utils/
    └── logger.py            # Logging system
```

---

## Author

Prajjwalit — MSc Advanced Manufacturing Systems, Brunel University London  
Production Support Engineer | Robotics Researcher

GitHub: [riskerfru/ai-robot-manipulation](https://github.com/riskerfru/ai-robot-manipulation)
