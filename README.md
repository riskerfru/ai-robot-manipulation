# AI-Powered Robot Manipulation System

An autonomous robot manipulation system that integrates Claude AI with a Franka Panda 7-axis robot simulation. The robot queries the physics simulation in real time before planning every movement, using grounded reasoning rather than blind text generation.

---

## What Makes This Different

Most LLM and robotics projects send a text description to an AI and hope the output is physically valid. This system gives Claude direct access to the simulation state through tool use. Before planning any movement, the robot calls real simulation functions to check object positions, obstacle locations, path clearance, and safe heights. Every plan is validated against physics before execution.

---

## Features

**Claude AI Tool Use**
The robot calls 11 simulation tools before planning, including get_object_position, check_path_clear, get_safe_height, and plan_pick_sequence. Claude reasons with real data, not descriptions.

**6-Strategy Motion Planner**
When a path is blocked the system escalates automatically through six strategies: direct path, rise over obstacle, expand sideways, arc around, full RRT* optimal search, and joint reconfiguration. Each strategy is more computationally expensive than the last.

**Redundancy Resolution**
Strategy 6 exploits the 7 degrees of freedom of the Franka Panda arm. When all geometric strategies fail, the robot tries 14 different arm configurations to find a new approach angle that clears the obstacle. This is the same principle used in production industrial robots.

**TOPP-RA Trajectory Generation**
All trajectories are parameterised using Time-Optimal Path Parameterisation with Reachability Analysis. The robot accelerates smoothly, reaches full speed, then decelerates to stop, respecting joint velocity and acceleration limits throughout.

**RRT* Optimal Planning**
The random tree planner uses the RRT* variant which rewires the tree toward lower cost nodes, finding the shortest collision-free path rather than just any path.

**Transport-Aware Planning**
When carrying an object the collision buffer is increased and the planner searches for the path with maximum clearance from all obstacles.

**Claude Vision**
PyBullet camera frames are sent directly to Claude for visual scene analysis. Type `look` to have Claude describe what it sees, or `suggest` to get an action recommendation based on the visual scene.

**Direct Joint Control**
All 7 joints can be controlled individually by name or number with natural language commands.

---

## Technical Stack

- Python 3.12
- Anthropic Claude API with tool use
- PyBullet physics simulation
- Custom RRT* motion planner
- TOPP-RA trajectory optimisation
- Franka Panda URDF

---

## Installation

```bash
git clone https://github.com/riskerfru/ai-robot-manipulation
cd ai-robot-manipulation
pip install -r requirements.txt
```

Create a `.env` file in the root directory:
```
ANTHROPIC_API_KEY=your_key_here
```

Get a free API key at console.anthropic.com

Run:
```bash
python main.py
```

---

## Commands

```
pick red                    pick up the red block
put blue on green           stack blue on green
drop red 0.2 0.3            place red at coordinates
move joint 4 to -45         direct joint control
show joints                 display all joint angles
reset joints                return to home position
look                        Claude Vision scene analysis
suggest                     Claude recommends next action
scan                        scan all objects and obstacles
add obstacle wall3 0.2 0.1  add new obstacle
remove obstacle wall1       remove obstacle
```

---

## Architecture

```
Natural Language Input
        |
Claude AI with Tool Use
  queries simulation state
  validates paths before planning
  reasons with real physics data
        |
6-Strategy Motion Planner
  1. Direct path
  2. Rise over obstacle
  3. Expand sideways
  4. Arc around
  5. Full RRT* search
  6. Joint reconfiguration
        |
TOPP-RA Trajectory
  time-optimal velocity profile
  respects joint limits
        |
PyBullet Execution
```

---

## Roadmap

- Intel RealSense camera integration for real object detection
- uFactory xArm hardware driver
- Hand-eye calibration pipeline
- MoveIt2 integration via ROS2 bridge
- Reinforcement learning in simulation

---

## Author

Prajjwalit — MSc Advanced Manufacturing Systems, Brunel University London

Open to robotics and manufacturing engineering roles in the UK.