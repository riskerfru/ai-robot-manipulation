# Real ladder logic in place of the soft PLC — two-day plan

The aim: the cell PLC stops being a Python stand-in and becomes a
ladder program running on OpenPLC. The robot cell then answers to a
controller that can refuse what the language model asks for, and the
refusal lives in rungs rather than in Python.

Nothing about the robot side changes. That is the point of the
interface that is already there.

---

## What already exists — do not rebuild it

| File | What it is |
|---|---|
| `plc/signals.py` | The interface control document. Every signal, its direction, type and Modbus address. **This is the contract. The ladder program must match it exactly.** |
| `plc/bus.py` | `ModbusBus` — a Modbus TCP client that reads and writes those signals by name. Already written. |
| `plc/interlocks.py` | Reads the PLC's safety signals, with a heartbeat timeout. |
| `plc/cell_controller.py` | READY / BUSY / DONE / FAULT state machine on the robot side. |
| `plc/recipes.py` | Pre-approved jobs the PLC selects by number. |
| `plc/soft_plc.py` | The Python stand-in being replaced. **Keep it** — the tests use it, and it stays the offline fallback. |

The deliberate design decision in `signals.py`, worth preserving:

> Natural-language orders are never sent over the fieldbus; the PLC
> selects a pre-approved recipe by number.

The model does not get a channel to the controller. It picks a recipe
from a fixed list; the PLC decides whether that recipe may run. Keep
that boundary.

---

## What to build

An OpenPLC ladder program that plays the PLC side of the existing
signal map: drives the PLC-owned coils and registers, reads the
robot-owned ones, and enforces the interlocks.

### Behaviour the ladder must implement

**Safety chain.** `MOTION_PERMIT` may only be true when `ESTOP_OK`
AND `GUARD_CLOSED` AND `MODE_AUTO` are all true. Dropping any input
drops the permit in the same scan.

**Job sequencing.**
- Raise `JOB_START` only when the robot reports `READY`, the safety
  chain is healthy, and a valid recipe number is loaded.
- Hold `JOB_START` until the robot reports `DONE` or `FAULT` — it is
  a level, not a pulse.
- On `DONE`: drop `JOB_START`, increment `JOB_ID`, latch the counts
  from `PARTS_OK` / `PARTS_FAILED`.
- On `FAULT`: drop `JOB_START`, latch `FAULT_CODE`, and stay faulted
  until an operator pulses `FAULT_RESET`.

**Rejecting bad work.** A recipe number not in the approved list is
refused before `JOB_START` is ever raised — the robot is never asked.

**Heartbeat.** Increment `PLC_HEARTBEAT` every scan group. Watch
`ROBOT_HEARTBEAT`; if it stops, drop `MOTION_PERMIT`.

### Addressing

`signals.py` gives zero-based Modbus addresses: PLC→robot on coils
0–15 and holding registers 0–15, robot→PLC on coils 16–31 and
registers 16–31.

**Confirm how OpenPLC maps `%IX` / `%QX` / `%IW` / `%QW` onto Modbus
coils, discrete inputs and holding registers before writing rungs** —
read the OpenPLC docs, do not assume. Getting this wrong is the most
likely way to lose half a day. Write one coil, read it back with a
Python Modbus client, prove the mapping, then build on it.

---

## Day 1 — the controller

1. Install OpenPLC Editor and Runtime. Confirm the runtime's Modbus
   server is reachable from Python.
2. Prove the addressing with a throwaway script: write one coil, read
   it back, both directions.
3. Write the ladder: safety chain, heartbeat, job sequencing, recipe
   validation, fault latch.
4. Drive it from `plc/bus.py`'s `ModbusBus` with a small test client
   that fakes the robot side — report READY, watch JOB_START come up,
   report DONE.

**Day 1 is finished when:** a Python script pretending to be the robot
can run a full job handshake against real ladder logic, and opening
the guard (forcing `GUARD_CLOSED` false) drops `MOTION_PERMIT` and
stops the job.

That alone is the CV line. No AI involved yet.

## Day 2 — the boundary

5. Point the real cell at it: `ModbusBus` to the OpenPLC runtime
   instead of `InMemoryBus` + `SoftPLC`. A `--plc <host>` flag on the
   runner, defaulting to the soft PLC so the tests and the offline
   path still work.
6. Wire the planner: typed instruction → model → recipe number →
   written to `RECIPE` → PLC validates → `JOB_START` → cell runs.
7. Add the terminal log lines for the handshake, in the style the
   runner already uses: `[PLC] JOB_START raised, recipe 3`,
   `[PLC] MOTION_PERMIT dropped: guard open`.
8. Record the demo.

**Day 2 is finished when:** you can type an instruction, watch it
become a recipe number, watch the PLC accept it and the cell run it —
and then, with the guard forced open, watch the PLC refuse while the
model is told no.

---

## The thirty seconds that make the post

Guard open. Type a perfectly sensible instruction. The model produces
a perfectly sensible plan. The PLC refuses it in ladder logic, and
nothing the model says changes that.

Film the ladder rung going false next to the terminal, if you can.

---

## Honest limits to keep stating

- OpenPLC is a soft PLC. This is not a safety-rated controller and
  not a functional-safety architecture. The pattern is right; the
  certification is absent.
- The plant is still simulated. What is real here is the ladder logic
  and the Modbus integration.
- Do not claim Siemens or TIA Portal experience from this. If the CV
  needs that word, it needs its own build later.

## Scope discipline

Two days. Things that are **not** in scope: an HMI, OPC UA, a
Factory IO scene, PROFINET, redundancy, a second station. If Day 1
runs long, cut step 7 before cutting step 4.
