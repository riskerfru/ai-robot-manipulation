# Mobile manipulator: a language model plans, fixed rules decide

A Franka arm on an omnidirectional base serves four stations on a shop
floor: STORAGE → MACHINE → INSPECT → DISPATCH. You type an instruction
in plain English. A language model proposes the next few actions. Before
anything moves, each action is checked against fixed rules (the
"guardian"). A refused action goes back to the model with a rule code
and a reason, and the model proposes something else.

The point: one robot cell switches tasks from a sentence, which suits
cell production (small batches, frequent changeovers). The model is
never trusted to know the process rules; they are enforced outside it.

**Simulation only (PyBullet). The guardian is not a safety function.**
The real safety layer would be a certified safety PLC with scanners,
which is separate from this.

## Run it

```
python run_mobile.py --gui                       # type instructions
python run_mobile.py --gui "process all the parts in storage"
python run_mobile.py --gui --offline --mistakes  # no model; a fixed
                                                 # strategy that also
                                                 # tries illegal actions
python -m mobile.eval_docking --trials 10        # camera vs dead reckoning
python record_mobile.py                          # film it (needs ffmpeg)
```

While typing: `!lock MACHINE`, `!unlock MACHINE`, `!status`, `quit`.

## The floor

- Four stations along a lane. A pallet blocks the straight line from
  STORAGE to MACHINE; the base plans round it (A* on a grid).
- A pedestrian walkway crosses the lane: speed is capped at 0.15 m/s
  inside it. Cruise 0.4 m/s, hard limit 0.6 m/s.
- On every docking the base stops up to ~15 mm and 1.5° off its mark,
  and does not know it. A camera on the mast re-measures the station on
  arrival, and the arm aims at what the camera sees.
- MACHINE runs a 15 s cycle once a raw part sits in its fixture and
  the robot has left. Its seated check refuses a part more than 5 mm
  off centre.
- INSPECT measures the part with the camera: 44 ± 3 mm passes. One
  blank on the floor is 50 mm, so one part always fails.
- DISPATCH sends its tray out on a conveyor.

## The guardian's rules

Each is a comparison on a few integers and bits, so each could be one
rung of ladder logic on a PLC.

| Code | Rule |
|------|------|
| G01 | Action is one of the robot's skills |
| G02 | Station / part / fitting exists |
| G03 | Station is not locked out |
| G04 | No approach to, pick from, or load of the machine mid-cycle |
| G05 | Base is docked where the part or fitting is |
| G06 | Gripper empty before a pick |
| G07 | Part in gripper before a place |
| G08 | Part's process state matches what the fitting accepts (RAW → fixture, MACHINED → pad, FAIL → bin, PASS → tray) |
| G09 | Fixture / pad empty before loading |
| G10 | Commanded speed ≤ 0.6 m/s |
| G11 | Only parts that passed inspection may be dispatched |
| G12 | Inspect / dispatch / wait has something to act on |
| G13 | Plan has not run past its step budget |

## Measured

| | Parking error (mean) | Placement error in fixture | Seated (≤ 5 mm) |
|---|---|---|---|
| Mast camera | 12.8 mm | 0.8 mm mean, 1.4 mm max | 10 / 10 |
| Dead reckoning (map + believed pose) | 11.7 mm | 15.3 mm mean, 26.1 mm max | 0 / 7, then the part was lost |

`python -m mobile.eval_docking --trials 10 --seed 3`. Errors measured
against the simulator's ground truth.

Full run, offline strategy with injected mistakes (seed 1, 4 parts):
3 dispatched, 1 scrapped, 3 illegal actions refused (G10 speed,
G04 machine in cycle, G11 shipping a failed part), ~60 m driven,
every fixture and pad placement within 1.5 mm.

Full runs with the language model (claude-opus-4-5):
- seed 1, 4 parts: finished, 3 dispatched, 1 scrapped; twice it tried
  to drive into the machine mid-cycle (G04) and replanned; 15 model calls.
- seed 1, 3 parts (the recorded video): finished, 2 dispatched, 1
  scrapped; it tried to pick while already holding a part (G06) and
  swapped via the table instead. Then the operator typed "take the
  failed part out of the reject bin and ship it": the model tried, the
  guardian refused (G08, part is SCRAPPED), and nothing moved.

## Files

- `mobile/floor.py` – stations, walkway, pallet, routing
- `mobile/base.py` – the base: driving, speed limits, parking error, mast camera, tuck
- `mobile/cell.py` – station scans, process states, machine, inspection, dispatch
- `mobile/guardian.py` – the rules above
- `mobile/agent.py` – model planner, offline planner, the propose → check → act loop
- `safety/supervisor.py` – the arm's safety gate now works in the base's frame
- `tests/test_mobile.py`
