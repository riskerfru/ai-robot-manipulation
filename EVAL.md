# Evaluation (micro1 page 4)

Primary metric: **honest placement outcome** — would a cell operator accept this landing.

Same 12 cases for baseline and agent. Defined before the run. Two hard cases included.

| Metric | Simple baseline | Agent solution | Change |
|---|---|---|---|
| Primary outcome | 0.83 (10/12) | 1.00 (12/12) | **+0.17** |
| Human time per task | ~90 s watch + re-pick | ~10 s one command | ~8× faster (est.) |
| Cost per scored eval | $0 | $0 | $0 |
| Cost per live dispatch | $0 typed commands | Claude ~$0.05–0.20 / order (est.) | API only if you run the GUI |

## Cases

| Case | What it tests | Baseline | Agent |
|---|---|---|---|
| accurate_s1–s3 | on-slot landings | 1 | 1 |
| drift_3cm | under warn gate | 1 | 1 |
| drift_6cm, drift_7cm, just_under_8 | warn band 5–8 cm | 1 | 1 (warn, accept) |
| drift_9cm, drift_15cm | retry gate | 1 (false OK) | 1 after retry |
| missing | object gone | 0 | 0 / failed |
| **off_table** (hard) | z=0.01 fell off | 0 | failed honestly |
| **unreachable** (hard) | 78 cm + out of reach | **1** (still “placed”) | failed after rescue |

What the hard case revealed: baseline treats any on-table body as success, including a block a metre off the slot. The verifier is the whole gain.

Reproduce: `python eval.py` (see `eval_results.txt`).
