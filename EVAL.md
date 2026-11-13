# Evaluation

All numbers here come from the PyBullet simulation. Nothing has been
measured on a real robot.

## What was measured

`eval_physics.py` runs the complete cell headless and scores every placement
against the true part position from the physics engine. Orders run as
recipes, with no language model, so the numbers describe the execution,
verification and safety layers that every order passes through, whether it
was typed, sent by the PLC or produced by the model.

- 20 orders of 3 parts for each combination below: 720 placements in total.
- Part start positions are randomised per order from fixed seeds.
- Placement tolerance: 2 cm (`config.PLACEMENT`).

Four ways of handling a placement after the gripper opens:

| Arm | What it does |
|---|---|
| `open_loop` | Reports success without measuring |
| `verify_only` | Measures; out of tolerance is reported as a failure |
| `verify_retry` | Measures; picks the part up and places it again, up to twice |
| `retry_calib` | As `verify_retry`, plus a learned per-slot offset |

Three conditions:

| Condition | What happens |
|---|---|
| `nominal` | No disturbance |
| `disturbed` | 30 % of releases: the part is knocked 3 to 15 cm in a random direction |
| `biased` | Every release lands 6 cm off in +y with 1 cm noise, as a tool-centre-point calibration error would cause |

The disturbances are injected by the script after the gripper opens. They
are test inputs, not something the simulation produced on its own.

## Results

### nominal

| Arm | Good parts | Escaped defects | Caught | False alarms | Places / part | Mean drift |
|---|---|---|---|---|---|---|
| open_loop | 56/60 = 93% [84-97] | 4/60 = 7% [3-16] | 0 | 0 | 1.00 | 0.9 cm |
| verify_only | 56/60 = 93% [84-97] | 0/60 = 0% [0-6] | 4 | 0 | 1.00 | 0.9 cm |
| verify_retry | 60/60 = 100% [94-100] | 0/60 = 0% [0-6] | 0 | 0 | 1.07 | 0.7 cm |
| retry_calib | 60/60 = 100% [94-100] | 0/60 = 0% [0-6] | 0 | 0 | 1.03 | 0.7 cm |

### disturbed

| Arm | Good parts | Escaped defects | Caught | False alarms | Places / part | Mean drift |
|---|---|---|---|---|---|---|
| open_loop | 42/60 = 70% [57-80] | 18/60 = 30% [20-43] | 0 | 0 | 1.00 | 2.6 cm |
| verify_only | 42/60 = 70% [57-80] | 0/60 = 0% [0-6] | 18 | 0 | 1.00 | 2.6 cm |
| verify_retry | 55/60 = 92% [82-96] | 0/60 = 0% [0-6] | 5 | 0 | 1.38 | 1.5 cm |
| retry_calib | 54/60 = 90% [80-95] | 0/60 = 0% [0-6] | 6 | 0 | 1.40 | 1.6 cm |

### biased

| Arm | Good parts | Escaped defects | Caught | False alarms | Places / part | Mean drift |
|---|---|---|---|---|---|---|
| open_loop | 0/60 = 0% [0-6] | 60/60 = 100% [94-100] | 0 | 0 | 1.00 | 6.1 cm |
| verify_only | 0/60 = 0% [0-6] | 0/60 = 0% [0-6] | 60 | 0 | 1.00 | 6.1 cm |
| verify_retry | 0/60 = 0% [0-6] | 0/60 = 0% [0-6] | 60 | 0 | 2.98 | 6.4 cm |
| retry_calib | 57/60 = 95% [86-98] | 0/60 = 0% [0-6] | 3 | 0 | 1.60 | 2.1 cm |

- **Good parts:** reported placed and truly within tolerance.
- **Escaped defects:** reported placed but out of tolerance. This is the part
  that jams the line downstream.
- **Caught:** out of tolerance and reported as a failure.
- **False alarms:** within tolerance but reported as a failure.

Brackets are 95 % Wilson intervals. With 60 parts per row, zero observed
escapes means the true rate is below about 6 %, not that it is zero.

## What the numbers say

1. **Without measurement, defects are reported as good parts.** Open loop
   passed 7 % of parts out of tolerance with no disturbance, 30 % when
   disturbed and all of them under a calibration error.
2. **With measurement, no out-of-tolerance part was reported as placed** in
   any of the nine verified rows (540 placements), and no good part was
   rejected.
3. **Retrying recovers most disturbed parts**: 70 % good parts open loop,
   92 % with retry, at 1.38 placements per part.
4. **Retrying cannot fix a repeatable offset; calibration can.** Under the
   6 cm bias, retry alone used almost three placements per part and
   delivered nothing. With the learned slot offset, 95 % of parts were good
   at 1.6 placements per part.

## Placement accuracy before and after the pick-and-place fixes

Same 20 nominal orders, open loop, no disturbance:

| | Within 2 cm | Within 5 cm | Median error | Largest error |
|---|---|---|---|---|
| Before | 25/60 = 42 % | 41/60 = 68 % | 2.5 cm | 11.5 cm |
| After | 56/60 = 93 % | 60/60 = 100 % | 0.7 cm | 3.8 cm |

The fixes are listed in `CHANGELOG.md`: the grasp attachment, a vertical and
square tool on approach, closed-loop position hold on the descent, a known
grasp height, a grasp-capture check and release confirmation.

## What this does not show

- **Nothing about the language model.** How often the model picks the wrong
  action, and how often the six failure modes in `FAILURE_MODES.md` recur
  with the fixes in place, has not been measured. That needs runs with an
  API key.
- **Nothing about real hardware.** In simulation the verifier reads the true
  part position. A real cell measures with a camera or sensors, and the
  measurement error has to be smaller than the tolerance.
- **Cycle time.** Not measured.

## Verifier logic check

`python eval.py` runs 12 scripted cases against a stub robot in under two
seconds (`eval_results.txt`). It checks the verifier's decisions (accept,
retry, fail), not placement performance: the stub lands every retry exactly
on target. With no verification, 10 of the 12 cases are counted as placed,
including parts 9 cm and 15 cm off target and one 78 cm away and out of
reach. The verifier makes the right decision in all 12: it retries the first
two and reports the third as a failure.

## Reproduce

```bash
python eval_physics.py --orders 20 --workers 2    # about 35 minutes on two cores
python eval.py
```
