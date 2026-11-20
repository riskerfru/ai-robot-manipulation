#!/usr/bin/env python3
# =============================================================
#   RECORD MOBILE - films the mobile manipulator
#
#   Left: a camera that follows the base. Right: the terminal -
#   what the model proposed, the guardian's ACCEPT / REJECT with
#   its rule code, the drive, the parking error, what the mast
#   camera measured, the machine, the inspection.
#
#       python record_mobile.py                       (needs ffmpeg)
#       python record_mobile.py --offline --mistakes  (no model)
# =============================================================

import argparse
import contextlib
import io
import math

import pybullet as p
from PIL import Image, ImageDraw

import record_open as RO
from record_demo import (BG, BLUE, GREEN, H, MUTED, RED, SCENE_W, TEXT, W,
                         Recorder)

AMBER, VIOLET, CYAN = (242, 179, 61), (178, 140, 255), (90, 200, 220)
RO.TAG_COLOUR.update({
    "OPERATOR": GREEN, "LLM": VIOLET, "PLAN": VIOLET, "GUARD": AMBER,
    "NAV": BLUE, "CAMERA": CYAN, "MACHINE": AMBER, "INSPECT": VIOLET,
    "ROBOT": GREEN, "DISPATCH": GREEN, "FAULT": RED, "ERROR": RED,
    "LOCKOUT": RED})


class Panel(RO.Panel):
    def log(self, tag, text, colour=None):
        for piece in self._wrap(text, 38):
            self.rows.append((tag, piece, colour))
            tag = ""
        self.rows = self.rows[-400:]

    def draw(self, scene):
        img = super().draw(scene)
        # redraw the log with wider tags
        d = ImageDraw.Draw(img)
        d.rectangle([RO.PANEL_X + 1, 66, W, H - 31], fill=(12, 15, 19))
        y = 76
        for tag, text, colour in self.rows[-RO.LINES:]:
            if tag:
                d.text((RO.PANEL_X + 10, y), f"{tag:<8}", font=RO.F_LOG,
                       fill=RO.TAG_COLOUR.get(tag, MUTED))
            d.text((RO.PANEL_X + 92, y), text, font=RO.F_LOG,
                   fill=colour or TEXT)
            y += 19
        return img


class FollowRecorder(Recorder):
    base = None

    def frame(self):
        if self.base is not None:
            x, y, _ = self.base.true_pose
            self.view = p.computeViewMatrixFromYawPitchRoll(
                [x + 0.30, y, 0.30], 2.35, 62, -30, 0, 2)
        super().frame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out/mobile.mp4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--mistakes", action="store_true")
    ap.add_argument("--steps-per-frame", type=int, default=60)
    ap.add_argument("instructions", nargs="*", default=[
        "process all the parts in storage",
        "take the failed part out of the reject bin and ship it"])
    args = ap.parse_args()

    import config as cfg
    from mobile.agent import LLMPlanner, ScriptedPlanner, run
    from mobile.cell import build_mobile_cell
    from mobile.guardian import Guardian

    if args.offline or not cfg.ANTHROPIC_API_KEY:
        planner = ScriptedPlanner(mistakes=args.mistakes)
        source = "fixed strategy, no language model" + (
            " - illegal actions injected on purpose" if args.mistakes else "")
    else:
        import anthropic
        planner = LLMPlanner(anthropic.Anthropic(
            api_key=cfg.ANTHROPIC_API_KEY, timeout=90.0, max_retries=1))
        source = "model: " + planner.model + " | simulation, not safety-rated"

    with contextlib.redirect_stdout(io.StringIO()):
        w = build_mobile_cell(seed=args.seed, parts=args.parts)
    cell = w.cell

    rec = FollowRecorder(args.out)
    rec.base = w.base
    panel = Panel(source)
    rec.hud = panel
    rec.steps_per_frame = args.steps_per_frame

    rec.card("A robot arm on wheels, run by a language model",
             ["Four stations: storage, machine, inspection, dispatch.",
              "The model decides each next action from plain English.",
              "A fixed rule layer - like a PLC - checks every action",
              "first, and refuses any that breaks the process."], 5.0)

    def log(tag, text):
        if tag == "GUARD" and "REJECT" in text:
            panel.log(tag, text, RED)
            panel.heading = "REFUSED by the guardian"
            rec.hold(2.6)
            return
        if tag == "GUARD":
            panel.log(tag, text, GREEN)
            return
        panel.log(tag, text)
        if tag in ("LLM", "PLAN"):
            panel.heading = "model proposed the next actions"
            rec.hold(1.6)
        elif tag == "NAV" and text.startswith("docked"):
            panel.heading = text.split(":")[0]
        elif tag in ("INSPECT", "DISPATCH", "MACHINE"):
            rec.hold(1.0)

    cell.log = log
    panel.heading = "looking at STORAGE"
    with contextlib.redirect_stdout(io.StringIO()):
        cell.scan()
    log("CAMERA", "mast camera at STORAGE: " + ", ".join(
        f"part #{t.id} {t.long * 1000:.0f} mm"
        for t in cell.at_station() if not t.container))
    rec.hold(1.5)

    refused = []
    for text in args.instructions:
        RO.type_out(rec, panel, text)
        with contextlib.redirect_stdout(io.StringIO()):
            out = run(text, cell, planner, log, guardian=Guardian(cell),
                      max_rounds=30)
        rej = out["rejected"]
        refused += [r[1] for r in rej]
        panel.heading = "done" if out["finished"] else "stopped"
        panel.log("", f"{out['executed']} actions, {len(rej)} refused "
                      f"({', '.join(r[1] for r in rej) or 'none'}), "
                      f"{cell.dispatched} dispatched, "
                      f"{w.base.odometer:.0f} m driven", MUTED)
        rec.hold(3.0)
        print(text, "->", {k: v for k, v in out.items() if k != "rejected"},
              [r[1:] for r in rej], flush=True)

    vis = [p_["error_mm"] for p_ in cell.placements]
    rec.card("What it showed",
             [f"Base parked up to ~15 mm off every time; the mast camera",
              f"re-measured each station, so parts landed "
              f"{sum(vis) / max(len(vis), 1):.1f} mm from centre on average.",
              f"The guardian refused {len(refused)} proposed actions "
              f"({', '.join(sorted(set(refused))) or 'none'}); "
              "the model replanned each time.",
              "Simulation (PyBullet). Not safety-rated. Not real hardware."],
             5.0)
    rec.close()
    w.robot.disconnect()
    print("wrote", args.out, rec.frames, "frames")


if __name__ == "__main__":
    main()
