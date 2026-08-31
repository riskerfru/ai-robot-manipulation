# Agent trajectories

Representative traces for every agent in this entry (micro1 deliverable 04).

| File | Agent | From → to |
|---|---|---|
| [01_build_agent.md](01_build_agent.md) | Grok coding agent | PDF gaps → this packet |
| [02_runtime_warehouse_agent.md](02_runtime_warehouse_agent.md) | Claude warehouse orchestrator | `auto dispatch…` → 6/6 in 9 steps |

Each file has: instructions, tools, step results, feedback that changed the next step, retries, human checkpoints.

No raw Anthropic HTTP dump (that would leak key headers). Runtime steps are reconstructed from `ORCHESTRATOR_PROMPT`, `build_observation`, and the README recorded log.
