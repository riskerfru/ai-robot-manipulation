# Improvement Changelog

Format from the micro1 Agentic Workflows brief (stage / tried / evidence / decision).
Same eval method throughout: `python eval.py` (12 fixed placement cases) plus the
recorded GUI dispatch in the README (`6/6 subtasks in 9 steps`).

**Before this weekend:** PyBullet Franka Panda, RRT*, Claude tools, typed `pick`/`place`.
**Added this weekend:** verifier-first eval, tests, this changelog, reproduction guide, trajectories.

| Stage | What you tried and why | Evidence | Decision / learning |
|---|---|---|---|
| **Baseline** | Typed `pick` / `place` / one RRT shot. If the block is still on the table, call it done. Matches “the manual process people use today.” | `eval.py` **0.83 (10/12)**. Only `missing` and `off_table` fail. 15 cm miss and **unreachable** still count as success. | Established the starting point. Looks fine in a screenshot. Ships the wrong pose. |
| **Iteration 1** | Added `PlacementVerifier` after seeing blocks sit 10+ cm off the slot. Gates: <5 cm accept, 5–8 cm warn, >8 cm retry (max 2), z<0.05 hard fail. | Cases `drift_9cm` and `drift_15cm` recover to 0.0 cm. `off_table` is failed. `eval` advanced **1.00** on honest outcomes. | **Kept.** Biggest single gain. |
| **Iteration 2** | Slot calibration: last-5 average of place drift, saved in `slot_calibration.json`, applied on retry. Why: the same slot was drifting the same way every order. | `tests/test_verifier.py::test_calibration_average_and_apply`. Checked-in offset for slot 2. | **Kept.** Cheap memory. Does not need the LLM. |
| **Iteration 3** | Observe → Think → Act → Verify with a DONE / DO NOW / NOT YET box. Why: Claude was repeating picks and grabbing blocks already on the conveyor (bugs 1, 2, 4 in `agent_loop.py`). | README recorded run: **6/6 subtasks in 9 action steps**. Slot lines say `DO NOT PICK THIS`. | **Kept.** This is the agent design, not more RRT. |
| **Iteration 4 (removed)** | Trust the first RRT path and skip verify. Why: faster demo. | Blocks looked placed, then sat 10+ cm off the slot. Baseline still scores those as 1. | **Removed.** Taught us the planner was not the bottleneck. |
| **Iteration 5 (revised)** | Joint-reconfigure rescue when a drifted block is out of reach. | Hard case `unreachable`: rescue is attempted, then `status=failed` + “human intervention required”. | **Kept the honest fail.** Do not fake a place. |
| **Final** | Combine verifier + calibration + DO NOW box + typed baseline for the no-key path. | Offline: baseline **0.83** → advanced **1.00** on 12 cases. GUI: 6/6 dispatch (needs API key). | Main contribution: **verification after place**, not a fancier planner. |

## Main failure mode

1. No `ANTHROPIC_API_KEY` → natural-language dispatch cannot run. Typed baseline and `eval.py` still run.
2. After max retries the verifier currently returns `status=success` so the agent loop does not stall. That is the honesty gap to patch next.
3. `hardware/` is empty stubs. This entry is PyBullet only.

## Hot take

The planner was never the bottleneck. A nearest-slot script looks fine and still ships the wrong block. The 8 cm retry gate plus the DONE / DO NOW / NOT YET observation is what turned “it moved” into 6/6. Next agents should spend the first iteration on **verify**, not on another skill file.
