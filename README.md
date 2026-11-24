# WAAM digital twin

A browser simulation of a robotic wire-arc additive manufacturing (WAAM) cell. An operator types a request in plain English; Claude turns it into a build plan, choosing only from a table of tested process settings; fixed rules check every plan before anything moves; a slicer generates the bead paths; and a simulated six-axis arm lays every bead while a heat model holds each layer to its interpass temperature.

Simulation only. Process values are illustrative, not calibrated.

## How it is put together

```
browser (GitHub Pages)          Cloudflare Worker                Anthropic API
index.html  -- POST /plan -->  worker/worker.js  -- key -->     Claude
            <-- plan JSON --   (holds the API key)
```

- `index.html` is the whole twin: 3D cell (three.js), slicer, rules R01-R09, heat model, rule-based fallback planner.
- `worker/worker.js` holds the API key. It answers only requests from the page's own address, only WAAM planning requests, with short replies, and limits each visitor to 20 planning calls per 10 minutes.
- If the Worker is unreachable or its limit is reached, the page uses its rule-based planner and says so in the log.

## Setting it up

### 1. An API key with a spending cap
In the Anthropic Console, create a workspace for this demo, set a low monthly spend limit on it, and create an API key in that workspace. The spend limit is what protects you if someone abuses the demo.

### 2. The Worker (no command line needed)
1. Sign in at dash.cloudflare.com (a free account is enough).
2. Workers & Pages > Create > Create Worker. Name it `waam-planner`, then Deploy.
3. Edit code: replace everything with the contents of `worker/worker.js`, then Deploy.
4. Settings > Variables and Secrets > Add:
   - `ANTHROPIC_API_KEY` as a **Secret**: the key from step 1
   - `ALLOWED_ORIGIN` as Text: `https://riskerfru.github.io`
5. Note the Worker's address, for example `https://waam-planner.<your-name>.workers.dev`. Opening it in a browser should show `{"ok":true,"service":"waam-planner"}`.

### 3. Point the page at the Worker
In `index.html`, set the line near the top of the script:

```js
const PLAN_API='https://waam-planner.<your-name>.workers.dev';
```

### 4. GitHub Pages
Repository Settings > Pages > Deploy from a branch > `main`, folder `/ (root)`. The page appears at `https://riskerfru.github.io/ai-robot-manipulation/`.

## Limitations
- Process values are illustrative; a real cell uses bead trials for each wire, gas and machine.
- One lumped temperature: no melt pool, distortion or residual stress.
- Identical beads: no layer-height variation or scanning.
- Generic arm, torch held vertical, no collision checking; three part types at any size within limits.

## Next steps
1. Layer-height scanning with closed-loop correction of the next layer.
2. A per-layer thermal model, so bead geometry changes with temperature.
3. Process data from bead trials or published data.
4. STL import and slicing of real CAD.
5. Several arms on one part, with spacing and heat rules between them.
