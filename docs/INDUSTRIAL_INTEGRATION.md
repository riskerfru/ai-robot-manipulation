# Industrial integration

How this system is meant to sit in a PLC-controlled production cell, what is
implemented, what has been tested and how, and what still has to happen
before it runs next to people.

## Status in one table

| Item | State |
|---|---|
| Agent, planner, verifier in PyBullet | Working; evaluated in physics (`EVAL.md`) |
| Safety supervisor (software gate) | Implemented; unit-tested; exercised in physics |
| PLC handshake and signal map | Implemented; tested in memory and over Modbus TCP on localhost |
| Audit trail | Implemented; tamper detection tested |
| Recipe / degraded mode without the language model | Implemented; tested |
| UR5e backend over RTDE | Written; tested against a fake of the RTDE interface only |
| Run on a real robot or in URSim | **Not done** |
| Run against a real PLC | **Not done** |
| Risk assessment, safety-rated functions, CE/UKCA | **Not done; cannot be done in software** |

So: the architecture is the one a real cell needs, and it is validated in
simulation. It is not a certified or commissioned system.

## Architecture

```
  Operator HMI (free text)          Cell PLC (master)
          |                          | JOB_START, RECIPE, interlocks
          v                          v
  +---------------------+    +--------------------+
  | Language model      |    | Cell controller    |   plc/cell_controller.py
  | 11 read-only tools  |    | READY/BUSY/DONE/   |
  | 6 enumerated actions|    | FAULT handshake    |
  +----------+----------+    +---------+----------+
             | action request          | recipe steps
             v                         v
  +--------------------------------------------------+
  | Agent executor  (agent_loop.py)                  |
  | slot occupancy, reachability, one task at a time |
  +-------------------------+------------------------+
                            v
  +--------------------------------------------------+
  | Safety supervisor  (safety/)   deterministic     |
  | interlocks, workspace, reach, keep-out, speed    |
  | fail-closed, latching stop, operator reset only  |
  +-------------------------+------------------------+
                            v
  +--------------------------------------------------+
  | Robot interface  (hardware/interface.py)         |
  |   PyBullet Franka Panda   |   UR5e over RTDE     |
  +-------------------------+------------------------+
                            v
  +--------------------------------------------------+
  | Placement verifier (place_verifier.py)           |
  | measure, retry, calibrate, or report failure     |
  +--------------------------------------------------+

  Everything above writes to the audit trail (audit/trail.py).
```

### What the language model can and cannot do

It can read the scene through 11 tools (positions, obstacles, reachability,
path checks, pick/place plans) and ask for one of six actions: `pick`,
`place_slot`, `place`, `stack`, `scan`, `home`.

It cannot generate a trajectory, a joint target or a speed. It has no tool
that moves the robot, and none that reads, changes or resets the safety
layer (`tests/test_safety.py::test_language_model_tools_are_read_only`).
The one place it supplies numbers is `place(x, y)`, and those coordinates go
through the same supervisor as everything else.

## Safety concept

`safety/supervisor.py` is a software interlock layer. **It is not a
safety-rated function.** In a real cell the emergency stop, the guard
interlocks and the speed and force limits are implemented in safety-rated
hardware: the robot's own safety controller and a safety PLC, to the
performance level the risk assessment requires (ISO 10218-1/-2, ISO 13849-1,
and ISO/TS 15066 for collaborative operation). Those devices stop the robot
whatever this software does.

What the supervisor adds:

- It mirrors the safety signals, so the software does not command motion the
  safety system would reject, and does not resume by itself afterwards.
- It bounds what the agent may ask for: workspace box, reach, keep-out
  zones, speed, payload.
- It gives every refusal a reason code in the audit trail.

Rules it follows:

- **Fail closed.** A missing, stale or unreadable interlock denies motion.
  So does an error inside the supervisor.
- **The PLC must be seen alive.** Its heartbeat has to be observed changing
  before any permit is trusted, and has to keep changing.
- **Stops latch.** A guard opening causes a protective stop; closing the
  guard does not restart motion. An open emergency-stop circuit latches an
  emergency stop. Both need an operator reset, which is refused while the
  cause is present.
- **The agent cannot reset.** Reset comes from the PLC `FAULT_RESET` signal
  or an operator, never from a model output.
- **Limits are immutable at run time.**

One limit of this design to be clear about: the gate runs before each
command and checks the tool position after it. It does not interrupt a
motion that is already under way. Stopping mid-motion is the job of the
safety-rated stop circuit acting on the robot controller.

### Reason codes

| Code | Meaning |
|---|---|
| `S000` | OK |
| `S001` | ESTOP_LATCHED |
| `S002` | ESTOP_CIRCUIT_OPEN |
| `S003` | GUARD_OPEN |
| `S004` | NOT_IN_AUTO |
| `S005` | NO_PLC_PERMIT |
| `S006` | ROBOT_NOT_READY |
| `S007` | INTERLOCK_STALE |
| `S008` | PROTECTIVE_STOP |
| `S010` | OUTSIDE_WORKSPACE |
| `S011` | OUTSIDE_REACH |
| `S012` | KEEP_OUT_ZONE |
| `S013` | SPEED_LIMIT |
| `S014` | PAYLOAD_LIMIT |
| `S020` | INVALID_COMMAND |
| `S099` | INTERNAL_ERROR |

## PLC interface

The PLC owns the cell. The robot side never starts a job by itself.

```
PLC                               robot cell controller
                                  READY = 1
JOB_ID, RECIPE, JOB_START = 1 ->
                                  READY = 0, BUSY = 1
                                  ... job runs ...
                                  BUSY = 0, DONE = 1     (or FAULT = 1, FAULT_CODE)
JOB_START = 0                 ->
                                  DONE = 0, READY = 1
```

`DONE` means every part was measured within tolerance. A part that could not
be placed, a safety stop, an unknown recipe or a lost PLC heartbeat gives
`FAULT` with a code, and the cell stays faulted until the PLC drops
`JOB_START` and pulses `FAULT_RESET` with the cause cleared.

The PLC does not send text. It selects a **recipe** by number
(`plc/recipes.py`): a fixed, reviewed list of steps that needs no language
model. Free-text orders come from an operator and go through the model, then
through exactly the same executor, supervisor and verifier.

### Signal map (Modbus TCP, zero-based)

| Signal | Direction | Type | Address | Meaning |
|---|---|---|---|---|
| `ESTOP_OK` | PLC → robot | coil | 0 | Emergency stop circuit healthy (mirror of the safety relay) |
| `GUARD_CLOSED` | PLC → robot | coil | 1 | All guard doors closed and locked |
| `MODE_AUTO` | PLC → robot | coil | 2 | Cell mode selector in AUTO |
| `MOTION_PERMIT` | PLC → robot | coil | 3 | PLC permits robot motion (conveyor stopped, infeed ready) |
| `JOB_START` | PLC → robot | coil | 4 | Level: start the job in JOB_ID / RECIPE; hold until DONE or FAULT |
| `FAULT_RESET` | PLC → robot | coil | 5 | Operator acknowledged the fault (rising edge) |
| `JOB_ID` | PLC → robot | register | 0 | Job number for traceability |
| `RECIPE` | PLC → robot | register | 1 | Recipe number (see plc/recipes.py) |
| `PLC_HEARTBEAT` | PLC → robot | register | 2 | Counter incremented by the PLC every scan group |
| `READY` | robot → PLC | coil | 16 | Idle, healthy, able to accept a job |
| `BUSY` | robot → PLC | coil | 17 | Job in progress |
| `DONE` | robot → PLC | coil | 18 | Job finished, all parts verified in tolerance |
| `FAULT` | robot → PLC | coil | 19 | Job stopped; see FAULT_CODE. Needs FAULT_RESET |
| `DEGRADED` | robot → PLC | coil | 20 | Language model unavailable; running pre-approved recipes only |
| `ROBOT_READY` | robot → PLC | coil | 21 | Robot controller powered, no protective stop |
| `FAULT_CODE` | robot → PLC | register | 16 | 0 = none; see FaultCode |
| `PARTS_OK` | robot → PLC | register | 17 | Parts placed and verified in the current/last job |
| `PARTS_FAILED` | robot → PLC | register | 18 | Parts that could not be placed in tolerance |
| `SLOT_MASK` | robot → PLC | register | 19 | Bit n-1 set = conveyor slot n occupied |
| `ROBOT_HEARTBEAT` | robot → PLC | register | 20 | Counter incremented by the robot controller |
| `JOB_ID_ECHO` | robot → PLC | register | 21 | JOB_ID the robot is working on / last finished |

Fault codes: 0 none, 10 safety stop, 20 unknown recipe, 30 part not placed,
40 PLC heartbeat lost, 90 internal error.

The names are the interface; Modbus TCP is one transport. A plant on
PROFINET or OPC UA maps the same names onto its own I/O.

## When the language model is unavailable

No API key, no network, a timeout or an error does not stop the cell.

- PLC recipes never use the model.
- A free-text order that matches a recipe runs as that recipe.
- A model failure in the middle of an order switches the rest of that order
  to its already-decomposed steps.
- A free-text order that matches no recipe is refused; nothing moves.

The `DEGRADED` signal and a `degraded_mode` audit record say when this has
happened. The next order tries the model again.

## Audit trail

`audit/trail.py` writes one JSON line per event: order, decomposition, model
decision, tool call, safety decision, motion result, verification, handshake
state, fault, reset. Each line carries the SHA-256 of the previous one.

```
python -m audit.trail audit_logs/cell.jsonl
INTACT: 74 record(s) verified
```

An edited, deleted or reordered line breaks the chain, and the writer
refuses to append to a broken file.

## Running it

```
python run_cell.py                     # two PLC jobs, headless
python run_cell.py --scenario guard    # guard door opens mid-job
python run_cell.py --scenario estop    # emergency stop mid-job
python run_cell.py --scenario plc      # PLC heartbeat lost mid-job
python run_cell.py --gui               # watch it in the viewer
python eval_physics.py                 # the numbers in EVAL.md
python -m pytest -q                    # all tests
```

## The UR5e backend

`hardware/ur_rtde.py` implements the same robot interface for a Universal
Robots e-Series arm using the `ur_rtde` library. It checks the controller's
safety mode, robot mode and stop flags before every move, asks the
controller whether each pose is inside its configured safety limits, moves
at a configured commissioning speed, and fails a pick if a fitted
part-present sensor does not confirm the grip.

It has been run only against a fake RTDE object that records calls
(`tests/test_ur_backend.py`). That proves the sequence and the refusals. It
does not prove behaviour on a robot.

Two gaps on the hardware side:

1. **Part positions must be measured.** In simulation the physics engine
   says where each part is. A real cell needs a camera or slot sensors
   behind a `CallbackPoseSource`. With fixtures only
   (`FixturePoseSource`), the verifier refuses to report a verified
   placement, so a job cannot reach `DONE`. That is deliberate.
2. **Motion is point-to-point `moveL`.** The RRT* planner used in simulation
   is not connected to the UR backend. The cell layout has to keep the
   approach corridors clear until it is.

3. **Force-guided insertion is simulation only.** `motion/insertion.py`
   reads the contact force on the held part from the physics engine and
   steps the arm in position control. On a UR5e the same logic would read
   the built-in tool force/torque sensor over RTDE and should use the
   controller's force mode, which limits contact force far better than
   small position steps do. The UR backend does not implement it.

## Before this runs on a real cell

1. Run the UR backend against URSim and fix what that finds.
2. Survey the cell; set `URCellConfig` (frame, heights, I/O) and the
   supervisor limits from the survey.
3. Add a measuring pose source and validate its accuracy against the
   placement tolerance.
4. Risk assessment for the application (ISO 12100, ISO 10218-2), and from it
   the required safety functions and performance levels.
5. Implement those in the robot safety configuration and a safety PLC, wire
   the emergency stop and guards, and validate them.
6. Connect the cell PLC to the signal map; test every handshake path and
   fault on the real fieldbus.
7. Commission at reduced speed in manual mode, then run acceptance trials
   with the real parts and count escaped defects.
8. Decide how model and prompt versions are controlled: a changed model is a
   changed process and needs re-validation of free-text orders.
