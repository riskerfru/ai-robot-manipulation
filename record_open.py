#!/usr/bin/env python3
# =============================================================
#   RECORD OPEN - films a session at the open table
#
#   Two scenes. In the first, a typed instruction moves a few
#   things into a container. In the second, the rest go in and
#   the conveyor takes the lot away.
#
#   The right-hand panel is the terminal: the camera's
#   measurements, the model's reading of the picture, the
#   checker's verdict, every waypoint, every force reading in
#   newtons, every safety decision.
#
#       python record_open.py                 (needs ffmpeg)
#       python record_open.py --seed 4 --out demo_out/open.mp4
#
#   With ANTHROPIC_API_KEY set, the plans come from the model.
#   Without one, --scripted replays plans of the same shape so
#   the film can still be made; the panel says which was used
#   and the robot, camera, forces and gate are real either way.
# =============================================================

import argparse
import contextlib
import io
import os
import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw

import commands as C
from record_demo import (BG, BLUE, CARD, FPS, GREEN, H, LINE, MUTED, RED,
                         SCENE_W, TEXT, W, Recorder, font)

F_LOG = font("mono", 13)
F_CMD  = font("bold", 26)
F_CMD_S = font("bold", 20)
F_HINT = font("regular", 14)
F_HEAD = font("bold", 15)
F_TAG = font("bold", 11)
PANEL_X = SCENE_W + 1
PANEL_W = W - PANEL_X
LINES = 31

TAG_COLOUR = {"CAMERA": BLUE, "LLM": (178, 140, 255), "CHECK": GREEN,
              "GATE": GREEN, "DENIED": RED, "FORCE": (242, 179, 61),
              "TOUCH": (242, 179, 61), "GRIP": TEXT, "MOVE": TEXT,
              "PATH": MUTED, "BELT": BLUE, "AUDIT": MUTED, "you": GREEN}


class Panel:
    """The terminal, drawn beside the scene."""

    def __init__(self, source):
        self.rows = []
        self.sim_time = 0.0
        self.speed = 1.0
        self.source = source
        self.command = ""
        self.heading = "waiting"

    def log(self, tag, text, colour=None):
        for piece in self._wrap(text):
            self.rows.append((tag, piece, colour))
            tag = ""
        self.rows = self.rows[-400:]

    @staticmethod
    def _wrap(text, width=41):
        out, line = [], ""
        for word in str(text).split():
            if len(line) + len(word) + 1 > width:
                out.append(line)
                line = "  " + word
            else:
                line = f"{line} {word}".strip()
        out.append(line)
        return out or [""]

    def draw(self, scene):
        img = Image.new("RGB", (W, H), BG)
        img.paste(scene, (0, 0))
        d = ImageDraw.Draw(img)

        # What was typed, large, across the top of the scene, so it
        # stays readable while the arm works.
        if self.command:
            font_cmd = F_CMD if len(self.command) < 46 else F_CMD_S
            d.rectangle([0, 0, SCENE_W, 88], fill=(10, 13, 17))
            d.line([(0, 88), (SCENE_W, 88)], fill=LINE)
            d.text((28, 14), "TYPED BY THE OPERATOR", font=F_TAG,
                   fill=MUTED)
            d.text((28, 40), f"> {self.command}", font=font_cmd,
                   fill=GREEN)
        d.rectangle([PANEL_X, 0, W, H], fill=(12, 15, 19))
        d.line([(PANEL_X, 0), (PANEL_X, H)], fill=LINE)

        d.rectangle([PANEL_X, 0, W, 64], fill=CARD)
        d.text((PANEL_X + 14, 12), "TERMINAL", font=F_TAG, fill=MUTED)
        d.text((PANEL_X + 14, 30), self.heading[:40], font=F_HEAD,
               fill=TEXT)
        d.text((W - 96, 12), f"x{self.speed:.0f} speed", font=F_TAG,
               fill=MUTED)
        d.text((W - 96, 32), f"{self.sim_time:5.1f} s", font=F_LOG,
               fill=MUTED)
        d.line([(PANEL_X, 64), (W, 64)], fill=LINE)

        y = 76
        for tag, text, colour in self.rows[-LINES:]:
            if tag:
                d.text((PANEL_X + 12, y), f"[{tag:<6}]", font=F_LOG,
                       fill=TAG_COLOUR.get(tag, MUTED))
            d.text((PANEL_X + 80, y), text, font=F_LOG,
                   fill=colour or TEXT)
            y += 19

        d.line([(PANEL_X, H - 30), (W, H - 30)], fill=LINE)
        d.text((PANEL_X + 14, H - 22), self.source, font=F_TAG, fill=MUTED)
        return img


def type_out(rec, panel, text):
    """Show the instruction appearing as if typed."""
    panel.command = ""
    rec.hold(0.8)
    for index in range(len(text) + 1):
        panel.command = text[:index] + ("|" if index % 2 else "")
        rec.frame()
        rec.hold(0.055)
    panel.command = text
    rec.hold(2.2)


def scripted_plan(command, cell):
    """
    Plans of the shape the model returns, for when no key is set.
    Chosen from what the camera measured, not from fixed numbers.
    """
    oc = cell.robot.open_cell
    tray = oc.conveyor_container()
    loose = [t for t in oc.listing()
             if not t.container and not t.held and t.inside is None
             and oc.why_not_grip(t) is None]

    def name(thing):
        r, g, b = (float(v) for v in thing.rgb)
        colour = ("red" if r > .6 and g < .4 and b < .4 else
                  "orange" if r > .7 and g > .4 and b < .3 else
                  "yellow" if r > .7 and g > .6 and b < .35 else
                  "green" if g > .5 and r < .4 else
                  "blue" if b > .6 and r < .4 else
                  "cyan" if b > .6 and g > .6 else
                  "purple" if r > .3 and b > .5 else "pale")
        shape = ("cylinder" if thing.roundness > 0.88 else
                 "bar" if thing.long - thing.short > 0.012 else "cube")
        return f"{colour} {shape}"

    names = {str(t.id): name(t) for t in oc.listing() if not t.container}
    for box in (t for t in oc.listing() if t.container):
        names[str(box.id)] = ("brown tray on the conveyor"
                              if box is tray else "grey bin")

    if "long" in command:
        chosen = [t for t in loose if t.long - t.short > 0.012][:2]
        understood = ("the long bar-shaped things go into the brown "
                      "container on the conveyor")
    else:
        chosen = loose[:3]
        understood = ("everything still on the table goes into the brown "
                      "container, then the conveyor takes it away")
    steps = []
    for thing in chosen:
        steps += [{"step": "pick", "thing": thing.id},
                  {"step": "place", "on": tray.id}]
    if "send" in command or "dispatch" in command:
        steps.append({"step": "dispatch"})
    return steps, understood, names


def run_instruction(rec, panel, cell, command, client, names):
    type_out(rec, panel, command)
    panel.log("you", command, GREEN)
    panel.heading = "asking the model"
    rec.hold(0.4)

    if client is None:
        steps, understood, new_names = scripted_plan(command, cell)
        panel.log("LLM", "picture + measurements sent; reply replayed "
                         "(no API key in this environment)")
    else:
        panel.log("LLM", f"sending the picture with "
                         f"{len(cell.robot.open_cell.listing())} numbered "
                         f"things, the measurements, and your words")
        rec.hold(0.3)
        steps, understood, new_names, attempts, usage = C.plan(
            command, cell, client)
        panel.log("LLM", f"reply: {usage['input']} tokens in, "
                         f"{usage['output']} out"
                         + (", corrected once" if attempts > 1 else ""))
    names.update(new_names)
    rec.hold(0.5)

    panel.log("LLM", "it identified: " + ", ".join(
        f"{k}={v}" for k, v in sorted(
            new_names.items(), key=lambda kv: int(kv[0]))))
    rec.hold(1.6)
    panel.log("LLM", f"understood: {understood}", BLUE)
    rec.hold(2.4)

    panel.heading = "checking the plan"
    for number, step in enumerate(steps, 1):
        panel.log("", f"{number}. " + describe(step, names), MUTED)
    rec.hold(1.8)
    problems = C.validate(steps, cell)
    panel.log("CHECK", "every step checked against the camera's "
                       "measurements: " + ("allowed" if not problems
                                           else "; ".join(problems)),
              GREEN if not problems else RED)
    rec.hold(2.2)

    panel.heading = "running"
    with contextlib.redirect_stdout(io.StringIO()):
        done, said, failure = C.execute(steps, cell, log=panel.log)
    for text in said:
        panel.log("robot", text, BLUE)
    panel.log("", f"done: {len(done)} steps"
                  if not failure else f"stopped: {failure}",
              GREEN if not failure else RED)

    from audit.trail import verify_records
    intact = verify_records(cell.audit.records)[0]
    panel.log("AUDIT", f"{len(cell.audit.records)} records, chain "
                       f"{'intact' if intact else 'BROKEN'}; gate "
                       f"{cell.safety.authorized_count} authorised, "
                       f"{cell.safety.denied_count} denied")
    panel.heading = "idle"
    rec.hold(3.0)


def describe(step, names):
    def called(ident):
        label = names.get(str(ident))
        return f"{ident} ({label})" if label else str(ident)

    kind = step.get("step")
    if kind == "pick":
        return f"pick up thing {called(step.get('thing'))}"
    if kind == "place":
        target = step.get("on")
        if isinstance(target, str) and not target.strip().isdigit():
            return "put it on a clear place on the table"
        return f"put it in container {called(target)}"
    if kind == "dispatch":
        return "conveyor: send the container out"
    if kind == "look":
        return "look again with the camera"
    return "answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out/open_table.mp4")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--scripted", action="store_true",
                    help="do not call the model even if a key is set")
    args = ap.parse_args()

    import config as cfg
    client = None
    if cfg.ANTHROPIC_API_KEY and not args.scripted:
        import anthropic
        client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY,
                                     timeout=90.0, max_retries=1)
    source = ("model: " + cfg.AI["model"] if client
              else "model replies replayed (no API key set)")

    from cell.factory import build_sim_cell
    from utils.logger import NullLogger

    rec = Recorder(args.out)
    panel = Panel(source)
    rec.hud = panel
    rec.steps_per_frame = 30      # half speed: easier to follow
    rec.camera(target=(0.42, 0.0, 0.18), distance=1.55, yaw=52, pitch=-40)

    rec.card("A table of things nobody described",
             ["Shapes, colours, sizes and positions are random.",
              "The robot is told none of them.",
              "A camera measures what is there; the language model",
              "reads the picture and your words."], 4.2)

    with contextlib.redirect_stdout(io.StringIO()):
        cell = build_sim_cell(NullLogger(), scene="open", seed=args.seed,
                              open_count=args.objects)
    oc = cell.robot.open_cell

    audit_log = cell.audit.log

    def logged(event, **payload):
        record = audit_log(event, **payload)
        if event == "safety_decision":
            allowed = payload.get("allowed")
            target = payload.get("target") or ()
            panel.log("GATE" if allowed else "DENIED",
                      f"{payload.get('kind')} "
                      f"{'authorised' if allowed else 'DENIED'} at ("
                      + ", ".join(f"{v:.3f}" for v in target) + ")"
                      + ("" if allowed else f": {payload.get('reason')}"),
                      GREEN if allowed else RED)
        return record
    cell.audit.log = logged

    panel.heading = "looking"
    panel.log("CAMERA", "arm parked clear of the view, taking a picture")
    with contextlib.redirect_stdout(io.StringIO()):
        found = oc.scan()
    for thing in found:
        kind = "container" if thing.container else "thing"
        panel.log("CAMERA", f"{kind} {thing.id}: ({thing.x:.3f}, "
                            f"{thing.y:.3f}) "
                            f"{1000 * thing.long:.0f}x"
                            f"{1000 * thing.short:.0f}mm "
                            f"h{1000 * (thing.top - thing.ground):.0f} "
                            f"rgb({', '.join(str(int(255 * float(v))) for v in thing.rgb)}) "
                            f"round {thing.roundness:.2f}")
        rec.hold(0.45)
    panel.log("", "measured, not recognised: no names anywhere", MUTED)
    panel.heading = "idle"
    rec.hold(2.2)

    names = {}
    run_instruction(rec, panel, cell,
                    "put the long ones in the brown tray", client, names)
    rec.hold(2.5)
    run_instruction(rec, panel, cell,
                    "put the rest in as well, then send it out", client,
                    names)

    rec.hold(1.5)
    rec.card("What was real", [
        "Positions, sizes and colours: measured by the camera.",
        "Contact: found by the force sensor, not assumed.",
        "Every motion: passed a safety gate first.",
        "Every step: checked before the arm moved.",
        "Simulation only. Not run on real hardware.",
    ], 5.5)
    rec.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
