# Trajectory 2 — runtime warehouse agent (Claude)

Reconstructed from `agent_loop.py` + the README recorded run.
Not a raw API dump (no keys in the repo).

## Instructions

`ORCHESTRATOR_PROMPT` (see `agent_instructions.md`).
User command: `auto dispatch all blocks to the conveyor`.

## Memory / context the agent sees every step

DONE / DO NOW / NOT YET box from `build_observation`. This is the change
that stopped Claude repeating picks (bugs 1 and 4).

## Step-by-step (9 action steps, 6 subtasks)

### Step 1 — inspect

- Observation: holding nothing; red/blue/green on table; slots 1–3 EMPTY.
- DO NOW: `pick red`
- Tools: `get_all_objects`, `get_object_position(red)`, `check_path_clear`
- Agent JSON: `{"action":"pick","params":{"object":"red"}}`
- Robot: torque-contact grasp. `[GRASP] Object lifted to z=0.220m`

### Step 2 — place

- DO NOW: `place_slot 1`
- Agent: `{"action":"place_slot","params":{"slot":1}}`
- Verifier: drift under 5 cm → success.
- Result: `pick red → place slot 1  Success`

### Step 3–4 — blue → slot 2

- Observation now: `🚫 red: IN SLOT 1 — DO NOT PICK THIS`
- DO NOW: `pick blue` then `place_slot 2`
- Same tool pattern. Success.

### Step 5–6 — green → slot 3

- DO NOW: `pick green` then `place_slot 3`
- Success.

### Steps 7–9 — scan / home / done

- Agent: `{"action":"done","message":"task done"}`
- Order complete.

## Feedback that shaped the next step

| Failure we had seen | Next observation |
|---|---|
| Agent picked from an occupied slot | Slot objects labelled `DO NOT PICK THIS` |
| Agent jumped ahead in the task list | Only one line in DO NOW |
| Place looked good, 10 cm off | `PlacementVerifier` runs after every place; >8 cm retry |

## Retries

Per subtask max 3 in the orchestrator. Verifier max 2 place retries.
This recorded run did not need a retry (6/6 in 9 steps).

## Human checkpoints

Operator types the one `auto …` line. Emergency stop lives in `EmergencyStop`
(`agent_loop.py`). No real Franka is connected (`hardware/` stubs).

## Result

```
ORDER COMPLETE — 6/6 subtasks in 9 action steps
```
