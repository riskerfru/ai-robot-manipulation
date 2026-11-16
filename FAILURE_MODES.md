# Failure modes of the language-model agent

Six ways the agent failed while this system was being built, what each one
looked like, why it happened, and the fix that is in the code now.

None of the six is a motion-planning or control failure. In every case the
arm did what it was told. What failed was the model's picture of the cell:
which task it was on, which parts were where, and what a destination is.
The fixes are therefore not in the planner. They are in what the model is
shown, and in deterministic checks that sit between the model and the robot.

These were observed during development and fixed one at a time. They were
not counted, so there are no occurrence rates here. `eval_physics.py`
measures the execution and verification layers; measuring how often each of
these recurs with the fixes in place needs runs with the language model
enabled.

| # | Failure | Kind | Fixed by |
|---|---|---|---|
| 1 | Repeats a finished step | Task state | DONE / DO NOW / NOT YET observation |
| 2 | Picks a part back off the conveyor | World state | Slot occupancy shown to the model, and enforced in code |
| 3 | Keeps trying an unreachable part | Physical grounding | Reachability gate before every pick |
| 4 | Invents a part that is not there | Perception grounding | Structured observation and known-parts list |
| 5 | Sends two parts to one slot | World state | Explicit occupancy, deterministic next-free-slot |
| 6 | Stacks instead of using a free slot | Instruction ambiguity | Separate `place_slot` and `place` actions |

## 1. Repeats a finished step

**Seen:** the agent picked a part, was asked for its next action, and
picked the same part again.

**Cause:** every step showed the model the whole remaining task list. It had
no reliable way to tell which entry was current, so it re-read the list from
the top.

**Fix:** the controller, not the model, owns the task pointer. Each
observation has three boxes: tasks already done, the one task to do now, and
tasks not to start yet (`build_observation` in `agent_loop.py`). The model is
asked for one action for the DO NOW task only.

## 2. Picks a part back off the conveyor

**Seen:** after placing a part in a slot, the agent picked it up again as if
it were a fresh part on the table.

**Cause:** a part in a slot and a part on the table looked the same in the
observation: a name and a coordinate.

**Fix:** two layers. The observation lists slot contents separately, marked
do-not-pick. Independently, `_execute_action` refuses a pick of any part
recorded in `slot_occupants`, whatever the model asked for.

## 3. Keeps trying an unreachable part

**Seen:** a part that had slid outside the arm's reach was attempted again
and again.

**Cause:** the model has no sense of the arm's workspace; a coordinate 0.9 m
away reads the same as one 0.4 m away.

**Fix:** reachability is computed before the model sees the part and shown
as a status. A pick of an unreachable part is refused in code with the
distance in the message, and the model is told to report failure. The safety
supervisor applies the same reach limit to every commanded target.

## 4. Invents a part that is not there

**Seen:** the agent planned an action on a "yellow block". There was no
yellow block.

**Cause:** raw coordinates with little structure, and an order phrased in
general terms ("all the blocks"), left room to fill in a plausible part.

**Fix:** the order decomposer is given the list of parts that exist and told
to use only those. A pick of an unknown part fails in code because there is
no position for it.

## 5. Sends two parts to one slot

**Seen:** a second part was placed onto a slot that already held one.

**Cause:** the model could not tell "this slot has a part in it" from "this
slot is where parts go".

**Fix:** occupancy is tracked by the controller and shown per slot. If the
model still names an occupied slot, the controller places in the next free
slot and reports which one it used; if none is free the action fails.

## 6. Stacks instead of using a free slot

**Seen:** told to place a part at the conveyor, the agent stacked it on a
part already there.

**Cause:** one `place` action covered "put in a slot", "put at these
coordinates" and "put on that part", and the model chose the wrong reading.

**Fix:** three separate actions with separate parameters: `place_slot(slot)`,
`place(x, y)` and `stack(object, on)`. A `place` with coordinates is never
silently turned into a slot placement, or the reverse.

## The seventh: a controller that reported success it had not checked

Not a model failure, but found in the same review and worth recording.
After two failed retries the placement verifier used to return `success`, so
that the agent loop would move on. A part 9 cm off its slot was therefore
reported as placed. The verifier now returns `failed` with "human
intervention required", the slot is not marked occupied, and the PLC sees
`FAULT`, not `DONE`. A rescue move is also now re-measured before it is
called a recovery.

## What carries over to a factory

- Give the model the state it cannot infer: what is done, what is where,
  what it can reach. Do not ask it to track these across turns.
- Treat every model output as a request. Occupancy, reachability, workspace
  and interlocks are checked in deterministic code after the model answers.
- Measure the result. A placement is reported as good only after it has
  been measured within tolerance.
