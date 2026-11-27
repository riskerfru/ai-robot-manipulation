"""
Browser view of the mobile cell.

For machines where the PyBullet window cannot open (for example WSL on
Windows): the simulation renders its own pictures off-screen, a few
times a second, and serves them with the event log and the cell state
on http://localhost:8090.

    python run_mobile.py --web --parts 3

Rendering happens in the simulation's own thread, on its physics steps;
the web server only hands out the latest picture.
"""

import collections
import io
import json
import logging
import threading
import time

import numpy as np
import pybullet as p
from PIL import Image

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mobile cell</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#111417;color:#e6e9ec;font:15px/1.45 system-ui,sans-serif}
header{padding:10px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:20px;margin:0 12px 0 0}
button{font:600 14px system-ui;padding:8px 12px;border-radius:7px;border:1px solid #2b3238;background:#1b1f23;color:#e6e9ec;cursor:pointer}
button.on{background:#ff5a1f;border-color:#ff5a1f;color:#fff}
main{display:grid;grid-template-columns:minmax(0,1.7fr) minmax(0,1fr);gap:12px;padding:0 16px 16px}
img{width:100%;border-radius:8px;background:#000;display:block}
.panel{background:#191d21;border-radius:8px;padding:10px 12px}
.log{font:12.5px/1.45 ui-monospace,Menlo,Consolas,monospace;height:300px;overflow:auto;margin-top:12px}
.log div{white-space:pre-wrap;overflow-wrap:anywhere}
.GUARD{color:#e0b341}.REJ{color:#ff6b6b;font-weight:700}.LLM,.PLAN{color:#7fd3ff}.FORCE,.FAULT{color:#ff6b6b}
.INSPECT{color:#d79cff}.NAV{color:#8fb5ff}.OPERATOR{color:#fff;font-weight:700}.LOCKOUT{color:#ff8a65}
table{width:100%;border-collapse:collapse;font-size:14px}td{padding:3px 4px;border-bottom:1px solid #2b3238}
td:first-child{color:#9aa4ad;white-space:nowrap}
@media(max-width:900px){main{grid-template-columns:minmax(0,1fr)}}
</style></head><body>
<header><h1>Mobile cell</h1>
<button data-v="follow" class="on">Follow robot</button>
<button data-v="close">Close-up</button>
<button data-v="overview">Whole floor</button>
<span id="stamp" style="color:#9aa4ad;font-size:13px"></span></header>
<main>
<section><img id="cam" alt="simulation view"><div class="panel log" id="log"></div></section>
<aside class="panel"><table id="state"></table></aside>
</main>
<script>
{
const cam=document.getElementById('cam'),logEl=document.getElementById('log'),stateEl=document.getElementById('state');
let after=0,busy=false;
function frame(){if(busy)return;busy=true;const i=new Image();i.onload=()=>{cam.src=i.src;busy=false};i.onerror=()=>{busy=false};i.src='/frame.jpg?t='+Date.now()}
setInterval(frame,150);
async function poll(){try{const r=await fetch('/log?after='+after);const d=await r.json();
 for(const [n,tag,text] of d){after=n;const div=document.createElement('div');div.className=(/REJECT/.test(text)?'REJ':tag);div.textContent=tag.padEnd(9)+text;logEl.append(div)}
 if(d.length)logEl.scrollTop=logEl.scrollHeight;
 const s=await (await fetch('/state')).json();stateEl.innerHTML='';
 for(const [k,v] of Object.entries(s)){const tr=stateEl.insertRow();tr.insertCell().textContent=k;tr.insertCell().textContent=typeof v==='string'?v:JSON.stringify(v)}
 document.getElementById('stamp').textContent='updated '+new Date().toLocaleTimeString()}catch(e){}}
setInterval(poll,600);poll();
for(const b of document.querySelectorAll('button[data-v]'))b.addEventListener('click',async()=>{
 await fetch('/view/'+b.dataset.v,{method:'POST'});for(const o of document.querySelectorAll('button[data-v]'))o.classList.toggle('on',o===b)});
}
</script></body></html>"""


class WebView:

    def __init__(self, cell, port=8090, fps=5, size=(640, 360)):
        self.cell = cell
        self.port = port
        self.period = 1.0 / fps
        self.w, self.h = size
        self.mode = "follow"
        self.lock = threading.Lock()
        self.jpeg = None
        self.last = 0.0               # when the last picture finished
        self.gap = self.period        # wait this long before the next one
        self.seq = 0
        self.lines = collections.deque(maxlen=400)
        self.state = {}
        self.idle = False             # set while the program waits for input
        self.proj = p.computeProjectionMatrixFOV(50, self.w / self.h,
                                                 0.05, 30)

    # ---- called from the simulation thread --------------------------------
    def log(self, tag, text):
        with self.lock:
            self.seq += 1
            self.lines.append((self.seq, str(tag), str(text)))

    def tick(self):
        # Rendering on the CPU is slow; leave the simulation at least
        # two thirds of the time, so the robot keeps moving at speed.
        if time.time() - self.last < self.gap:
            return
        t0 = time.time()
        self.render()
        self.last = time.time()
        self.gap = max(self.period, 2.0 * (self.last - t0))

    def _view(self):
        x, y, _ = self.cell.base.true_pose
        if self.mode == "close":
            from mobile.base import compose
            tx, ty = compose(self.cell.base.true_pose, 0.52, 0.0)
            return p.computeViewMatrixFromYawPitchRoll(
                [tx, ty, 0.12], 1.25, -55, -42, 0, 2)
        if self.mode == "overview":
            return p.computeViewMatrixFromYawPitchRoll(
                [0.6, -1.0, 0.0], 4.6, -90, -55, 0, 2)
        return p.computeViewMatrixFromYawPitchRoll(
            [x + 0.35, y, 0.25], 2.5, -60, -34, 0, 2)

    def render(self):
        try:
            raw = p.getCameraImage(self.w, self.h, self._view(), self.proj,
                                   renderer=p.ER_TINY_RENDERER, shadow=0,
                                   lightDirection=[0.6, -0.5, 1.6])
            img = Image.fromarray(np.reshape(raw[2], (self.h, self.w, 4))
                                  [:, :, :3].astype("uint8"))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            from mobile.agent import state_summary
            state = self._flat_state(state_summary(self.cell))
        except Exception:
            return
        with self.lock:
            self.jpeg = buf.getvalue()
            self.state = state

    @staticmethod
    def _flat_state(s):
        out = {"robot at": s.get("robot_at"), "holding": s.get("holding"),
               "machine": s.get("machine", {}).get("status"),
               "locked out": ", ".join(s.get("locked_out", [])) or "none",
               "blocked": ", ".join(s.get("blocked_fittings", [])) or "none",
               "dispatched": s.get("dispatched")}
        for part in s.get("parts", []):
            text = part.get("state", "")
            if part.get("measured_mm") is not None:
                text += f", {part['measured_mm']} mm"
            if part.get("at"):
                text += f", at {part['at']}"
            out[f"part {part['part']}"] = text
        return out

    # ---- web server -------------------------------------------------------
    def start(self):
        from flask import Flask, Response, request
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app = Flask(__name__)

        @app.get("/")
        def page():
            return Response(PAGE, mimetype="text/html")

        @app.get("/frame.jpg")
        def frame():
            with self.lock:
                data = self.jpeg
            if data is None:
                return Response(status=204)
            return Response(data, mimetype="image/jpeg",
                            headers={"Cache-Control": "no-store"})

        @app.get("/log")
        def log():
            after = int(request.args.get("after", 0))
            with self.lock:
                rows = [r for r in self.lines if r[0] > after]
            return Response(json.dumps(rows), mimetype="application/json")

        @app.get("/state")
        def state():
            with self.lock:
                data = dict(self.state)
            return Response(json.dumps(data, default=str),
                            mimetype="application/json")

        @app.post("/view/<mode>")
        def view(mode):
            if mode in ("follow", "close", "overview"):
                self.mode = mode
                self.last = 0.0       # next physics step renders
                if self.idle:         # nothing is stepping: render now
                    self.render()
            return Response(status=204)

        threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=self.port,
                                   threaded=True, use_reloader=False),
            daemon=True).start()
        self.cell.hub.callbacks.append(self.tick)
        self.render()
        return f"http://localhost:{self.port}"
