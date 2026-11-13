# Reproduction guide

From a clean machine to every number in this repository.

## What you need

| Item | Note |
|---|---|
| OS | Windows 10+, macOS or Linux |
| Python | 3.11 or newer (tested on 3.13) |
| Packages | `requirements.txt`; `requirements-industrial.txt` only for Modbus TCP or a real UR robot |
| API key | Not needed for the tests, `eval.py`, `eval_physics.py` or `run_cell.py`. Needed only for free-text orders in `main.py` |

## Setup

```bash
git clone https://github.com/riskerfru/ai-robot-manipulation.git
cd ai-robot-manipulation
python -m pip install -r requirements.txt
python -m pip install "pymodbus>=3.16,<4"     # optional: Modbus TCP test
```

## Tests

```bash
python -m pytest -q
```

Expected: **86 passed** in about a minute (83 if `pymodbus` is not
installed; those three are skipped). Two of the tests run the full cell in
physics.

## The cell under PLC control

```bash
python run_cell.py
python run_cell.py --scenario guard
python run_cell.py --scenario estop
python run_cell.py --scenario plc
```

Each run prints the handshake state after every step and finishes with the
audit-trail check. In the fault scenarios the first job stops with one part
placed, the first reset is refused, and the second job completes.

## Placement performance in physics

```bash
python eval_physics.py --orders 20 --workers 2
```

About 35 minutes on two cores. Writes `eval_out/physics_eval.md` and the raw
per-part records. The seeds are fixed, so the counts in `EVAL.md` should
reproduce on the same package versions; small differences across PyBullet
builds are possible.

## Verifier logic check

```bash
python eval.py
```

Under 2 seconds. Twelve scripted cases against a stub robot
(`eval_results.txt`).

## Free-text orders through the language model

Create `.env` with `ANTHROPIC_API_KEY=...` (never commit it), then:

```bash
python main.py
```

```text
auto dispatch all blocks to the conveyor
```

Headless: `SIM_GUI=0 python main.py`.
