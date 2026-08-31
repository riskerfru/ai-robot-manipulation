# HackerEarth form paste

**Repository:** https://github.com/riskerfru/ai-robot-manipulation

**Archive:** `ai-robot-manipulation-submit.zip` (no `.env`, no `__pycache__`)

**Video:** https://youtu.be/cJXKncODGrU

**Tools:** Grok coding agent (packet) + Claude Sonnet in `agent_loop.py` (runtime)

**How to run (from a clean machine):**

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python eval.py
```

Expected: 10 passed; primary outcome baseline **0.83** (10/12) vs advanced **1.00** (12/12).

**PDF checklist**

- [x] 01 Code + changelog table + user/bottleneck + failure mode + hot take
- [x] 02 Reproduction guide (commands, versions, runtime, cost)
- [x] 03 Video URL (re-record if you want the 5-min cut in VIDEO.md)
- [x] 04 Trajectories for both agents
- [x] 12 eval cases, two hard cases, metric table
- [x] What existed before vs added
- [x] No credentials

Do not upload `.env` or API keys.
