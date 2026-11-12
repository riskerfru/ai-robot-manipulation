# Development log

What was tried, what the evidence was, and what was kept.

## Industrial integration

| Change | Why | Evidence |
|---|---|---|
| Verifier reports failure when retries are exhausted | It used to return `success` so the agent loop would not stall; a part 9 cm off was reported as placed | `tests/test_verifier_honesty.py` |
| A rescue move is re-measured before it is called a recovery | It was reported as recovered without checking where the part ended up | same |
| Placement tolerance is a parameter | The 5 cm and 8 cm thresholds were constants; the tolerance belongs to the part drawing | same |
| Safety supervisor between every command and the robot | Model output, typed commands and verifier retries all reached the robot directly | `tests/test_safety.py`, `run_cell.py --scenario guard` |
| PLC handshake, signal map, recipes | A production cell is run by its PLC, not by a chat prompt | `tests/test_plc_handshake.py`, `tests/test_modbus_link.py` |
| Recipe and degraded modes | The cell must keep running when the model or network is down | `tests/test_agent_modes.py` |
| Hash-chained audit trail | Traceability of every decision and refusal | `tests/test_audit.py` |
| Robot interface with a UR5e RTDE backend | `hardware/` was seven empty files | `tests/test_ur_backend.py` (fake RTDE only) |
| One cell layout for every entry point | Slot coordinates were defined three times with two different values; the model was told (0.60, ±0.25) while the robot used (0.63, ±0.30) | `cell/layout.py` |
| A `place_slot` step is complete when the place succeeds | The loop needed an extra model call to say "done" after every placement | `tests/test_agent_modes.py` (6 steps for 6 subtasks) |
| Grasp attaches the part where it actually sits in the gripper | The grasp constraint used a world-frame offset with the wrong sign and no orientation, so a part was dragged sideways and twisted as it was lifted | `eval_physics.py`, nominal condition |
| Pick descends on the part, with closed-loop x/y hold | When visual servoing gave up, the descent used wherever the arm happened to be (up to 12 cm off), and the open-loop descent drifted a further 2 to 4 cm near the base | same |
| Gripper held vertical and square to the part on every approach, where the arm can reach that pose | The tool arrived tilted by up to 40 degrees at an arbitrary yaw; parts were knocked over, picked crooked and occasionally wedged in the gripper | same |
| Pick descends to the known grasp height; contact sensing is the guard | The descent ran until the fingertips hit the table | same |
| A grasp counts only if the part is between the fingers | The simulation attached a part to a gripper that had closed on air beside it and reported a successful pick | `core/robot.py`, `GRASP_CAPTURE_RADIUS` |
| Release is confirmed after every place | A wedged part rode back up with the gripper, was reported as placed, and was dropped somewhere else later | `core/robot.py`, `place_object` |
| A pick is refused while the gripper is holding something | Follows from the above | same |
| Undisturbed placements within 2 cm rose from 42 % to 93 % (median error 2.5 cm to 0.7 cm) | The seven changes above, measured on the same 20 orders | `EVAL.md` |
| Placement tolerance tightened from 8 cm to 2 cm, set in `config.PLACEMENT` | 8 cm was hiding the placement error above | `config.py` |
| Slot calibration measures against the commanded position, is applied on the first place, and ignores scattered errors | A corrected placement that landed on target was diluting the learned offset; random knocks were being learned as offsets | `tests/test_verifier_honesty.py` |
| Stored `slot_calibration.json` removed from the repository | It held a 5 cm offset learned from stub runs and was loaded on every start | `.gitignore` |
| Physics evaluation with repeated randomised trials | The 12-case check uses a stub robot that lands every retry perfectly | `eval_physics.py`, `EVAL.md` |

## Earlier iterations

| Stage | What was tried and why | Evidence | Decision |
|---|---|---|---|
| Baseline | Typed `pick` / `place`, one planning attempt. If the part is still on the table, call it done. | A part 15 cm off, or out of reach, still counts as placed. | Starting point. Looks fine in a screenshot; ships the wrong pose. |
| 1 | `PlacementVerifier` after every place: accept, warn, retry (max 2), hard fail if the part left the table. | Out-of-tolerance placements are retried or reported. | Kept. The largest single gain. |
| 2 | Slot calibration: average of the last five drifts per slot, applied on retry. | Corrects a repeatable offset (`EVAL.md`, biased condition). | Kept. Needs no language model. |
| 3 | Observe, think, act, verify with a DONE / DO NOW / NOT YET box. | The model stopped repeating picks and picking from the conveyor (`FAILURE_MODES.md` 1, 2, 4). | Kept. This is the agent design. |
| 4 | Trust the first planned path and skip verification, for a faster demo. | Parts looked placed and sat well off the slot. | Removed. The planner was not the bottleneck. |
| 5 | Joint-reconfiguration rescue when a drifted part is out of reach. | Attempted, then honest failure if it does not work. | Kept, with the re-measurement above. |

## Known limits

1. Simulation only. The UR5e backend has not been run on a robot or in URSim.
2. The language-model path cannot be tested without an API key; the tests
   cover recipe mode, degraded mode and a failing client.
3. In simulation the part positions come from the physics engine. A real
   cell needs a camera or sensors (`hardware/pose_source.py`).
4. The safety supervisor is software. It is not a safety-rated function.
