# Trajectory 1 — build agent (Grok)

## Instructions

Human: align the submission with the micro1 PDF (changelog table, 10+ eval
cases, repro guide with versions/cost, trajectories for every agent, video
slot). Do not commit secrets. Do not invent a 6/6 number.

## Tools

Read `place_verifier.py`, `agent_loop.py`, `eval.py`, the PDF.
Wrote/rewrote: `CHANGELOG.md`, `eval.py`, `REPRODUCTION.md`,
`agent_instructions.md`, this folder, `README.md`, `VIDEO.md`.
Ran: `python -m pytest -q`, `python eval.py`.

## Steps

1. PDF page 3 → changelog must be a table. Rewrote list into that table.
2. PDF page 4 → 10+ same cases, hard case, metric table. Expanded `eval.py` to 12 cases.
3. First eval run on 5 cases was too thin. Feedback: add warn-band, missing, two hard cases.
4. Unreachable rescue initially returned `recovered` because FakeRobot always moved. Human-visible bug. Fixed `move_to_position` to honour `reachable=False`.
5. PDF page 7.04 → trajectories were a stub. Split into this file + runtime file.

## Human checkpoints

- Chose this repo over a 2D tug demo.
- Forbade `.env` / API keys in git.
- Video URL: https://youtu.be/cJXKncODGrU

## Result

`pytest -q` → 10 passed.
`python eval.py` → baseline 0.83 vs advanced 1.00 on 12 cases.
Retry: FakeRobot rescue stub (once).
