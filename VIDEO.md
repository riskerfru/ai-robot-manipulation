# Solution video (deliverable 03, ≤ 5 minutes)

VIDEO_URL=https://youtu.be/cJXKncODGrU

The linked clip is the advanced dispatch. If you re-record, use this cut so it
matches the PDF:

| Time | Show | Say |
|---|---|---|
| 0:00 | Problem | Cell operator. Scripted pick still misses the slot. |
| 0:40 | Baseline | Typed `pick red`. No verify. `eval.py` baseline column. |
| 1:20 | One full advanced run | `auto dispatch all blocks to the conveyor` start to 6/6. |
| 3:20 | Comparison | Same 12 cases: 0.83 → 1.00. |
| 4:00 | Changelog | Verifier was the keep. Trust-first-RRT was the remove. |
| 4:40 | Hard case | `unreachable` — baseline still “placed”, advanced fails honest. |

Highlight: **placement verifier**. Removed experiment: skip verify and trust RRT.
