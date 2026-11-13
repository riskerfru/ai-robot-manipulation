# Language-model robot cell with a deterministic safety layer

A pick-and-place cell in PyBullet (Franka Panda) that takes an order in plain
language, plans collision-free motion with RRT* and TOPP-RA, places parts on
a three-slot conveyor, and measures every placement before reporting it.

Around the language model sits the structure a production cell needs: a
deterministic safety gate, a PLC handshake, pre-approved recipes that run
without the model, and a tamper-evident audit trail.

**Status: validated in simulation. Not run on a real robot, not certified.**
See [docs/INDUSTRIAL_INTEGRATION.md](docs/INDUSTRIAL_INTEGRATION.md) for
exactly what is and is not done.

Demo video: https://youtu.be/cJXKncODGrU

## The idea

A language model is good at turning "dispatch the blocks" into steps. It is
not something you let command a robot directly. So:

- The model **reads** the cell through 11 tools and **requests** one of six
  actions. It never produces a trajectory, a joint target or a speed.
- Deterministic code decides whether the request is carried out: slot
  occupancy, reachability, workspace limits, cell interlocks.
- Classical planning (RRT*, TOPP-RA, visual servoing) produces the motion.
- A verifier measures where the part landed. In tolerance, or reported as a
  failure. Never assumed.
- The cell PLC stays in charge. If the model is unavailable, the cell runs
  recipes and says so.

```
 Operator (free text)        Cell PLC (master)
        |                          |
   Language model            Cell controller        READY / JOB_START / DONE / FAULT
   11 read-only tools              |
   6 actions                       | recipe steps
        +-----------+--------------+
                    v
             Agent executor         one task at a time, slot and reach checks
                    v
            Safety supervisor       interlocks, workspace, keep-out, speed; fail-closed
                    v
             Robot interface        PyBullet Panda | UR5e over RTDE (untested on hardware)
                    v
           Placement verifier       measure, retry, calibrate, or report failure

             Audit trail            every decision and refusal, hash-chained
```

## Results

Simulation only: 720 placements in physics, 2 cm tolerance, no language
model in the loop (orders run as recipes).

| Condition | No measurement | Measure, retry, calibrate |
|---|---|---|
| No disturbance | 93 % good, 7 % defects reported as placed | 100 % good, none escaped |
| 30 % of parts knocked 3 to 15 cm | 70 % good, 30 % escaped | 90 % good, none escaped, the rest reported as failures |
| 6 cm calibration error | 0 % good, 100 % escaped | 95 % good, none escaped |

Across 540 measured placements, no out-of-tolerance part was reported as
placed and no good part was rejected. 60 parts per cell, so "none" means
below about 6 % at 95 % confidence.

Full tables and method: [EVAL.md](EVAL.md).

## What went wrong while building it

Six ways the language-model agent failed, and the fixes:
[FAILURE_MODES.md](FAILURE_MODES.md). None was a planning or control
failure. Each was the model losing track of the cell: which task it was on,
which parts were where, what it could reach.

## Quick start

```bash
git clone https://github.com/riskerfru/ai-robot-manipulation.git
cd ai-robot-manipulation
pip install -r requirements.txt

python -m pytest -q                     # 86 tests, about 1 min
python run_cell.py                      # two PLC jobs, headless, no API key
python run_cell.py --scenario guard     # guard door opens mid-job
python run_cell.py --gui                # watch it
python eval_physics.py --orders 20      # the numbers above (about 35 min)
```

## Force-guided assembly

A second cell layout (`scene="assembly"`): a housing with three pockets is
fixed on the conveyor, and three supply stations each hold one component
(50, 40 and 35 mm). Recipe 3 seats each component in its pocket, which
leaves 1.5 mm per side.

Position alone does not do this. The component's position in the gripper
and the housing's position are each known only to a few millimetres, so
the arm lowers the component until it feels contact, tells from its own
height whether the part is in the pocket or on the rim, and if it is on the
rim searches in a spiral under light contact until the part drops in
(`motion/insertion.py`). A component that does not reach the pocket floor
is reported as a failure and returned to its station.

The force signal is the contact force between the held part and the
housing, read from the physics engine. On hardware that is a wrist
force/torque sensor (a UR5e has one built in). There is no screwing,
press-fit or snap-fit here.

```bash
python record_demo.py assembly            # three components seated
python record_demo.py assembly-compare    # position only, then force-guided
python record_demo.py order --scene assembly --text "assemble the unit: seat the motor, then the bearing, then the sensor"   # needs the API key
```

Demo videos with a live status panel (needs `ffmpeg`):

```bash
python record_demo.py normal            # one PLC job
python record_demo.py guard             # guard door opens mid-job
python record_demo.py verify            # a knocked part, without and with measurement
python record_demo.py order --text "dispatch all blocks to the conveyor"   # needs the API key
```

Videos are written to `demo_out/`. The panel is drawn from the audit trail
and the PLC signals.

Free-text orders need an Anthropic API key in `.env`
(`ANTHROPIC_API_KEY=...`, never commit it):

```bash
python main.py
```

```
auto dispatch all blocks to the conveyor
auto pick red and place in slot 2
auto stack blue on green, then move stack to slot 1
stop            emergency stop (latches)
resume          operator reset
```

More: [REPRODUCTION.md](REPRODUCTION.md).

## What `run_cell.py --scenario guard` shows

```
  cell started            state=READY safety=RUN             parts_ok=0 slots=000 fault=0
  PLC: JOB_START job=1001 recipe=1
  job 1001 finished       state=FAULT safety=PROTECTIVE_STOP parts_ok=1 slots=001 fault=10 SAFETY_STOP
  PLC: FAULT_RESET while the cause is still present
  reset attempt 1         state=FAULT safety=PROTECTIVE_STOP
  Operator clears the cause, then FAULT_RESET
  reset attempt 2         state=READY safety=RUN
  PLC: JOB_START job=1002 recipe=1
  job 1002 finished       state=DONE  safety=RUN             parts_ok=3 slots=111 fault=0
  Audit trail: 63 records, chain intact
  Safety gate: 10 commands authorised, 2 denied
```

The guard opens after the first part. The next command is refused, the job
faults, a reset with the guard still open is refused, and the cell only
returns to READY after the guard is closed and the fault acknowledged.

## Components

| Part | Where | Notes |
|---|---|---|
| Agent loop | `agent_loop.py` | Order decomposition, DONE / DO NOW / NOT YET observation, recipe and degraded modes |
| Model tools | `core/robot_tools.py` | 11 read-only tools: positions, obstacles, reachability, path checks, pick/place plans |
| Motion planning | `motion/rrt_planner.py`, `motion/trajectory.py` | RRT* with six strategies; TOPP-RA time-optimal trajectories |
| Visual servoing | `motion/visual_servoing.py` | Overhead camera alignment before the descent |
| Grasping | `core/robot.py` | Torque-based contact detection, closed-loop descent, grasp-capture check, gradual release |
| Placement verifier | `place_verifier.py` | Tolerance check (2 cm, `config.PLACEMENT`), retry, per-slot calibration, honest failure |
| Safety supervisor | `safety/` | Deterministic gate; reason codes; latching stops |
| PLC interface | `plc/` | Signal map, handshake, recipes, Modbus TCP, soft PLC for simulation |
| Audit trail | `audit/` | Hash-chained JSON Lines; `python -m audit.trail <file>` verifies |
| Robot backends | `hardware/` | PyBullet Panda; UR5e over RTDE (fake-tested only) |
| Cell definition | `cell/` | Layout, scene builder, factory, visual dressing for recordings |
| Assembly cell | `cell/assembly.py`, `motion/insertion.py` | Housing with pockets, supply stations, force-guided insertion |
| Demo recorder | `record_demo.py` | Headless video with a status panel driven by the audit trail |

Design reasoning: [DESIGN.md](DESIGN.md). Change history:
[CHANGELOG.md](CHANGELOG.md).

## Action space

| Action | Parameters | Description |
|---|---|---|
| `pick` | `object` | Pick a named part from the table |
| `place_slot` | `slot` | Place in a numbered conveyor slot |
| `place` | `x, y` | Place at coordinates (checked against the workspace) |
| `stack` | `object, on` | Stack one part on another |
| `scan` | | Refresh part positions |
| `home` | | Return to the home pose |

## Cell layout

Defined once in `cell/layout.py`.

```
Robot base at the origin. Table x 0.00 to 0.84 m, y -0.50 to 0.50 m.
Reach: under 0.25 m too close; 0.25 to 0.65 m normal; 0.65 to 0.75 m edge; beyond 0.75 m refused.
Conveyor slots: 1 (0.63, +0.30)   2 (0.63, 0.00)   3 (0.63, -0.30)
Infeed nests:   red (0.40, 0.00)  blue (0.40, +0.30)  green (0.40, -0.30)
```

## Limits

- Simulation only. Part positions come from the physics engine; a real cell
  needs a camera or sensors and their accuracy has to be validated.
- The UR5e backend has been tested against a fake RTDE interface, not URSim
  or a robot, and uses straight-line moves without the RRT* planner.
- The safety supervisor is a software interlock layer, not a safety-rated
  function. Emergency stop, guarding and speed and force limits belong in
  safety-rated hardware, following a risk assessment.
- The language-model path is not covered by the automated tests (it needs an
  API key). Recipe mode, degraded mode and a failing model client are.

## Other tools

`python wrist_viewer.py` shows the wrist camera. `python web_server.py`
starts a browser interface on port 5000.

## Author

Prajjwalit Singh. MSc Advanced Manufacturing Systems, Brunel University London.

GitHub: [riskerfru/ai-robot-manipulation](https://github.com/riskerfru/ai-robot-manipulation)
