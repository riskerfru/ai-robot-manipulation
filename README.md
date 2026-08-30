# AI Robot Manipulation — Franka Panda + Claude AI

A simulation of an AI-powered robot manipulation system built with PyBullet, featuring a Claude-driven autonomous warehouse agent, physics-driven grasping, visual servoing, RRT* motion planning, TOPP-RA trajectory generation, live placement verification, and a wrist camera system.

---

## Demo

[![AI Robot Demo](https://img.youtube.com/vi/woKWzid6joI/0.jpg)](https://youtu.be/cJXKncODGrU)

*Autonomous warehouse agent completing a full dispatch order — 6/6 subtasks in under 10 action steps*

```
Command: auto dispatch all blocks to the conveyor

  🤖 WAREHOUSE AGENT STARTING
  📦 Order: dispatch all blocks to the conveyor

  [GRIPPER] Open (4cm)
  [DESCENT] From z=0.526m seeking contact...
  [DESCENT] Contact at z=0.162m torque=9.3
  [GRIPPER] Contact at gap=0.018m — gripping
  [GRASP] ✅ Object lifted to z=0.220m

  ✅ pick red  →  place slot 1  ✅
  ✅ pick blue →  place slot 2  ✅
  ✅ pick green → place slot 3  ✅

  ORDER COMPLETE — 6/6 subtasks in 9 action steps
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
│  · Live placement verification              │
│  · Self-calibrating slot offsets            │
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
│  Force/torque sensing · Contact detection   │
└─────────────────────────────────────────────┘
```

---

## Features

### Physics-Driven Grasping (New)
- **Force contact detection** — arm descends until joint torque spikes, no hardcoded heights
- **Force-controlled finger closing** — fingers stop when resistance felt from object
- **Grasp verification** — lifts slightly after grip, confirms object rose with arm
- **Gradual constraint release** — force reduces 40→20→5→0N before release, prevents drift
- Realistic open/close animation — fingers visibly spread (4cm) and close around object

### Wrist Camera System (New)
- Camera attached to gripper end effector, follows arm everywhere
- Captures gripper-eye view during pick and place operations
- Saves frames to `wrist_frames/` during descent
- `wrist_viewer.py` — live popup window showing gripper camera feed

### GUI Camera Controls (New)
- 7 camera toggle buttons in PyBullet window
- Overhead, Front, Top, Side, Close-up, Wrist, Wide presets
- Click any button to switch view instantly during operation

### Live Placement Verification (New)
- After every place, checks where object actually landed
- If drift > 8cm and reachable: picks up and retries with calibration correction
- If drift > 8cm and unreachable: attempts joint reconfiguration rescue
- Reports honest failure if truly unreachable

### Self-Calibrating Slot Offsets (New)
- Records drift after every placement per slot
- Calculates running average offset (last 5 placements)
- Saves to `slot_calibration.json` — persists across sessions
- Applies correction automatically next run
- Gets more accurate with each order

### Autonomous Agent Loop
- Natural language order decomposition via Claude API
- Clear observation format: DONE / DO NOW / NOT YET
- Slot protection — cannot pick from occupied conveyor slots
- Automatic retry with replanning on failure (3 attempts per subtask)
- Known object list passed to decomposer — prevents hallucination

### Motion Planning
- RRT* with 6 strategies (direct, arc-left, arc-right, high, elbow-up, reconfigure)
- TOPP-RA trajectory generation for smooth, time-optimal motion
- Collision-aware transport path planning with clearance buffer

### Visual Servoing
- Overhead camera with colour blob detection
- Oscillation prevention: skips servo if initial offset > 25px
- Damped corrections (0.3×) with 8mm per-step limit

### Reachability System
- Live reachability check before every pick
- PyBullet debug visualisation: ±110° working arc, green/orange/grey zones
- Per-object status icons: ✅ ok / ⚠️ edge / ❌ unreachable

---

## Observation Format

Each agent reasoning step receives a structured observation:

```
=== ROBOT STATE ===
Holding: nothing
Robot at: (0.39, 0.00, 0.55)

=== CONVEYOR SLOTS (NEVER PICK FROM THESE) ===
  Slot 1: (0.60, +0.25) — EMPTY ← can place here
  Slot 2: (0.60,  0.00) — occupied by blue
  Slot 3: (0.60, -0.25) — EMPTY ← can place here

=== OBJECTS ON TABLE (SAFE TO PICK) ===
  ✅ red:   pos=(0.40, 0.00) dist=0.4m reach=ok
  ✅ green: pos=(0.40,-0.30) dist=0.5m reach=ok

=== OBJECTS IN SLOTS (DO NOT PICK) ===
  🚫 blue: IN SLOT 2 — DO NOT PICK THIS

╔══════════════════════════════════════╗
║ TASKS ALREADY DONE (do not repeat): ║
║ [pick red, place slot 1, pick blue] ║
╠══════════════════════════════════════╣
║ DO NOW (execute only this):          ║
║ {'action': 'place_slot', 'slot': 2} ║
╠══════════════════════════════════════╣
║ NOT YET (do not do these yet):       ║
║ [pick green, place slot 3]           ║
╚══════════════════════════════════════╝
```

---

## Installation

```bash
git clone https://github.com/riskerfru/ai-robot-manipulation.git
cd ai-robot-manipulation
pip install pybullet anthropic numpy flask toppra pillow
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
auto place all the block in the slots of conveyer and then stop
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

### Wrist camera viewer (separate terminal)

```bash
python wrist_viewer.py
```

### Web interface

```bash
python web_server.py
# Open http://localhost:5000
```

---

## Camera Controls

Seven camera presets available as buttons in the PyBullet GUI panel:

| Button | View |
|--------|------|
| Camera: Overhead | Top-down view of full table |
| Camera: Front | Face-on view of robot |
| Camera: Top | Directly above scene |
| Camera: Side | Side profile |
| Camera: Close-up | Zoomed in on gripper area |
| Camera: Wrist View | Camera moves to gripper position |
| Camera: Wide | Wide angle full scene |

---

## Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `pick` | `object` | Pick named object from table only |
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
  Grey zone:   dist < 0.25m  — too close
  Green zone:  0.25 – 0.65m  — safe picking zone
  Orange zone: 0.65 – 0.75m  — edge of reach
  Red beyond:  > 0.75m       — out of reach

Conveyor slots:
  Slot 1: (0.60,  0.25)
  Slot 2: (0.60,  0.00)
  Slot 3: (0.60, -0.25)

Starting object positions:
  Red:   (0.40,  0.00)
  Blue:  (0.40,  0.30)
  Green: (0.40, -0.30)
```

---

## Real-World Deployment Note

Object positions in simulation are provided by PyBullet physics (ground truth). In a real deployment this would be replaced by:

- **Intel RealSense D435** depth camera with point cloud segmentation
- **Calibrated overhead RGB camera** with homography projection (already implemented on UR5 hardware with SensoPart VISOR V20, ±2mm accuracy, 98% detection rate)
- **Force/torque sensing** for contact-based surface detection — already simulated via joint torque monitoring in this project

The agent interface (observation format and action space) remains identical — only the perception layer changes.

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
| Placement verification | place_verifier.py |
| Calibration | slot_calibration.json |
| Wrist camera | perception/camera.py |
| Web interface | Flask + vanilla JS |
| Language | Python 3.10+ |

---

## Project Structure

```
ai-robot-manipulation/
├── main.py                  # Entry point, scene setup, camera controls
├── agent_loop.py            # Autonomous warehouse agent (System 2)
├── place_verifier.py        # Live placement verification + calibration
├── wrist_viewer.py          # Live wrist camera popup viewer
├── slot_calibration.json    # Learned slot offset corrections
├── config.py                # Robot and simulation config
├── core/
│   ├── robot.py             # FrankaPandaRobot — physics-driven pick/place
│   ├── ai_controller.py     # Claude with tool use
│   ├── joint_controller.py  # Direct joint control
│   └── robot_tools.py       # Tool definitions for Claude API
├── motion/
│   ├── rrt_planner.py       # RRT* motion planner (6 strategies)
│   ├── trajectory.py        # TOPP-RA trajectory generator
│   └── visual_servoing.py   # Camera-guided fine alignment
├── perception/
│   ├── camera.py            # Camera system + wrist camera
│   ├── vision.py            # Claude Vision integration
│   └── clip_detector.py     # CLIP object detection (optional)
├── reachability_viz.py      # PyBullet debug visualisation
├── web_server.py            # Flask web interface
└── utils/
    └── logger.py            # Logging system
```

---

## Author

Prajjwalit — MSc Advanced Manufacturing Systems, Brunel University London
Production Support Engineer | Robotics Researcher

GitHub: [riskerfru/ai-robot-manipulation](https://github.com/riskerfru/ai-robot-manipulation)