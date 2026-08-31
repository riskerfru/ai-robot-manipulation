# Reproduction guide

For someone starting from a clean machine. Matches micro1 deliverable 02.

## What you need

| Item | Version / note |
|---|---|
| OS | Windows 10+, macOS, or Linux |
| Python | 3.11 or 3.12 |
| pip packages | see `requirements.txt` |
| Data | none. Scene is synthetic PyBullet blocks. `slot_calibration.json` is optional. |
| API key | **not** required for `pytest` or `eval.py`. Required only for `python main.py` → `auto dispatch…` |
| Cost | $0 for the scored offline eval. Live Claude dispatch is pay-as-you-go on your key. Do not put the key in git. |

## Setup

```bash
git clone https://github.com/riskerfru/ai-robot-manipulation.git
cd ai-robot-manipulation
python -m pip install -r requirements.txt
```

Or unzip this archive and `cd` into the folder, then the same `pip` line.

## Baseline

Typed pick/place with no verifier. In the GUI that is:

```text
pick up red block
```

Offline stand-in (same 12 cases as advanced):

```bash
python eval.py
```

Look at the `baseline` column.

## Advanced / agent solution

```bash
python eval.py
```

Look at the `advanced` column and the METRIC table.

Optional GUI (needs `.env` with `ANTHROPIC_API_KEY=…`, never commit it):

```bash
python main.py
```

Then type:

```text
auto dispatch all blocks to the conveyor
```

## Tests

```bash
python -m pytest -q
```

Expected: **10 passed**.

## Expected eval output (abridged)

```text
case           baseline  advanced  detail
accurate_s1    1         1         success, 0.0cm
…
off_table      0         1         failed, 999.0cm
unreachable    1         1         failed, 78.3cm

METRIC                 SIMPLE BASELINE     AGENT SOLUTION      CHANGE
Primary outcome        0.83 (10/12 ok)     1.00 (12/12 ok)     +0.17
```

If your numbers differ, you are not on this commit.

## Runtime

| Command | Approx. |
|---|---|
| `pytest -q` | < 3 s |
| `python eval.py` | < 2 s |
| GUI `auto dispatch…` | a few minutes + Claude latency |

## Agents in this repo

| Agent | File | When it runs |
|---|---|---|
| Warehouse orchestrator | `agent_loop.py` (`ORCHESTRATOR_PROMPT`) | GUI `auto …` |
| Tool-use controller | `core/ai_controller.py` (`SYSTEM_PROMPT`) | GUI natural language |
| Build agent (this packet) | `trajectories/01_build_agent.md` | already done |

Instructions that shape each agent: `agent_instructions.md`.
Traces: `trajectories/`.
