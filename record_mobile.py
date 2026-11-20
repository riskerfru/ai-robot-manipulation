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


F_BANNER = RO.font("bold", 30)
F_BANNER_S = RO.font("bold", 19)
F_INSET = RO.font("bold", 13)
F_STATION = RO.font("bold", 22)
F_FIT = RO.font("bold", 14)
RENDER_SCALE = 0.55             # render smaller, scale up: much faster


class FollowRecorder(Recorder):
    base = None
    cell = None
    shot = "follow"             # or "close" for a station close-up
    banner = None               # (title, subtitle, colour)
    inset = None                # mast camera picture, PIL image

    def _view(self):
        x, y, yaw = self.base.true_pose
        if self.shot == "close":
            from mobile.base import compose
            tx, ty = compose(self.base.true_pose, 0.52, 0.0)
            return p.computeViewMatrixFromYawPitchRoll(
                [tx, ty, 0.12], 1.25, -55, -42, 0, 2)
        return p.computeViewMatrixFromYawPitchRoll(
            [x + 0.35, y, 0.25], 2.5, -60, -34, 0, 2)

    def frame(self):
        sw, sh = int(SCENE_W * RENDER_SCALE), int(H * RENDER_SCALE)
        view = self._view()
        raw = p.getCameraImage(sw, sh, view, self.proj,
                               renderer=p.ER_TINY_RENDERER, shadow=0,
                               lightDirection=[0.6, -0.5, 1.6])
        import numpy as np
        scene = Image.fromarray(np.reshape(raw[2], (sh, sw, 4))[:, :, :3]
                                .astype("uint8")).resize((SCENE_W, H))
        self._decorate(scene, view)
        self.hud.speed = self.steps_per_frame / 240 * 24
        self.last = self.hud.draw(scene)
        self._write(self.last)

    def _project(self, view, x, y, z):
        import numpy as np
        V = np.array(view).reshape(4, 4).T
        P = np.array(self.proj).reshape(4, 4).T
        v = P @ V @ np.array([x, y, z, 1.0])
        if v[3] <= 0:
            return None
        v = v[:3] / v[3]
        if not (-1 < v[0] < 1 and -1 < v[1] < 1):
            return None
        return (v[0] + 1) / 2 * SCENE_W, (1 - v[1]) / 2 * H

    def _labels(self, d, view):
        from mobile import floor as F
        from mobile.base import compose
        names = {"STORAGE": "STORAGE - raw parts", "MACHINE": "MACHINE",
                 "INSPECT": "INSPECTION", "DISPATCH": "DISPATCH"}
        fittings = {"fixture": "machine fixture", "pad": "measuring pad",
                    "bin": "reject bin", "tray": "outgoing tray"}
        for st, sy in F.STATIONS.items():
            pos = self._project(view, 0.62, sy, 0.62)
            if pos:
                text = names[st]
                w = d.textlength(text, font=F_STATION)
                x, y = pos[0] - w / 2, pos[1]
                d.rectangle([x - 10, y - 6, x + w + 10, y + 30],
                            fill=(10, 13, 17), outline=(250, 250, 250))
                d.text((x, y - 2), text, font=F_STATION, fill=(255, 255, 255))
            for role, (ox, oy), _, _ in F.EQUIPMENT.get(st, []):
                wx, wy = compose((F.DOCK_X, sy, 0.0), ox, oy)
                pos = self._project(view, wx, wy, F.TABLE_Z + 0.06)
                if pos:
                    text = fittings[role]
                    w = d.textlength(text, font=F_FIT)
                    x, y = pos[0] - w / 2, pos[1] - 30
                    d.rectangle([x - 6, y - 3, x + w + 6, y + 19],
                                fill=(250, 250, 250))
                    d.text((x, y - 1), text, font=F_FIT, fill=(20, 22, 26))
        pos = self._project(view, F.LANE_X, 0.0, 0.02)
        if pos:
            d.text((pos[0] - 70, pos[1] + 14), "pedestrian walkway",
                   font=F_FIT, fill=(255, 225, 90))

    def _decorate(self, scene, view=None):
        d = ImageDraw.Draw(scene)
        if view is not None:
            self._labels(d, view)
        st = self.base.station if self.base is not None else None
        names = {"STORAGE": "STORAGE", "MACHINE": "MACHINE",
                 "INSPECT": "INSPECTION", "DISPATCH": "DISPATCH"}
        text = (f"ROBOT IS AT: {names[st]}" if st else
                "ROBOT IS DRIVING")
        w = d.textlength(text, font=F_STATION)
        d.rectangle([14, 100, 34 + w, 136], fill=(10, 13, 17))
        d.text((24, 104), text, font=F_STATION, fill=(255, 225, 90))
        if self.inset is not None:
            iw, ih = self.inset.size
            x0, y0 = SCENE_W - iw - 16, 104
            d.rectangle([x0 - 3, y0 - 22, x0 + iw + 3, y0 + ih + 3],
                        fill=(10, 13, 17))
            d.text((x0 + 4, y0 - 19), "MAST CAMERA VIEW", font=F_INSET,
                   fill=CYAN)
            scene.paste(self.inset, (x0, y0))
        if self.banner:
            title, sub, colour = self.banner
            d.rectangle([0, H - 92, SCENE_W, H], fill=(10, 13, 17))
            d.rectangle([0, H - 92, 8, H], fill=colour)
            d.text((26, H - 84), title, font=F_BANNER, fill=colour)
            d.text((26, H - 42), sub, font=F_BANNER_S, fill=TEXT)

    def mast_picture(self, mark=None, size=300):
        """What the mast camera sees, with a box round one part."""
        import numpy as np
        cam = self.base.camera_config()
        view = p.computeViewMatrix(cam["position"], cam["target"], [0, 1, 0])
        proj = p.computeProjectionMatrixFOV(cam.get("fov", 42), 1.0,
                                            0.05, 3.0)
        raw = p.getCameraImage(size, size, view, proj,
                               renderer=p.ER_TINY_RENDERER)
        img = Image.fromarray(np.reshape(raw[2], (size, size, 4))[:, :, :3]
                              .astype("uint8"))
        if mark is not None:
            V = np.array(view).reshape(4, 4).T
            P = np.array(proj).reshape(4, 4).T
            pts = []
            hx = mark.long / 2 + 0.006
            for dx in (-hx, hx):
                for dy in (-hx, hx):
                    v = P @ V @ np.array([mark.x + dx, mark.y + dy,
                                          mark.top, 1.0])
                    v = v[:3] / v[3]
                    pts.append(((v[0] + 1) / 2 * size,
                                (1 - v[1]) / 2 * size))
            xs, ys = [q[0] for q in pts], [q[1] for q in pts]
            ImageDraw.Draw(img).rectangle(
                [min(xs), min(ys), max(xs), max(ys)], outline=(255, 220, 0),
                width=3)
        return img


def hook_stages(rec, cell, base):
    """Banner and camera for each stage, without touching the logic."""
    real = {n: getattr(cell, n) for n in ("pick", "place", "inspect",
                                          "dispatch")}
    go = base.go_to_station

    def goto(station, *a, **k):
        rec.shot, rec.inset = "follow", None
        rec.banner = (f"DRIVING TO {station}",
                      "parks up to ~15 mm off; the mast camera re-measures "
                      "on arrival", BLUE)
        try:
            return go(station, *a, **k)
        finally:
            rec.banner = None

    def pick(thing, *a, **k):
        st = base.station
        rec.shot = "close"
        rec.banner = (f"PICKING PART {thing.id} AT {st}",
                      "position from the mast camera, not from a map", GREEN)
        try:
            return real["pick"](thing, *a, **k)
        finally:
            rec.banner = None

    def place(role, *a, **k):
        held = cell.held()
        ident = held.id if held else "?"
        words = {"fixture": ("LOADING THE MACHINE",
                             f"part {ident} into the fixture - must sit "
                             f"within 5 mm or the machine will not clamp it"),
                 "pad": ("INSPECTION",
                         f"part {ident} onto the measuring pad"),
                 "bin": ("REJECTED PART",
                         f"part {ident} into the red reject bin"),
                 "tray": ("DISPATCH",
                          f"part {ident} into the outgoing tray "
                          f"(passed inspection)"),
                 "table": ("PUTTING DOWN",
                           f"part {ident} on the table for now")}
        colour = {"bin": RED, "tray": GREEN, "pad": VIOLET}.get(role, AMBER)
        title, sub = words.get(role, (role.upper(), ""))
        rec.shot, rec.banner = "close", (title, sub, colour)
        try:
            out = real["place"](role, *a, **k)
            rec.hold(1.2)
            return out
        finally:
            rec.banner = None

    def inspect(*a, **k):
        parts = cell.parts_in("pad", "INSPECT")
        rec.shot = "close"
        rec.banner = ("INSPECTION", "mast camera measuring the part on the "
                      "pad...", VIOLET)
        if parts:
            rec.inset = rec.mast_picture(parts[0])
        rec.hold(2.0)
        r = real["inspect"](*a, **k)
        if r:
            ok = r["result"] == "PASS"
            rec.banner = (f"INSPECTION: PART {r['part']} {r['result']}",
                          f"measured {r['size_mm']} mm, allowed 44 +/- 3 mm"
                          + ("  -> goes to DISPATCH" if ok
                             else "  -> goes to the REJECT BIN"),
                          GREEN if ok else RED)
            rec.hold(3.5)
        rec.banner, rec.inset = None, None
        return r

    def dispatch(*a, **k):
        riders = [t.id for t in cell.parts_in("tray", "DISPATCH")]
        rec.shot = "close"
        rec.banner = ("DISPATCH: TRAY LEAVING",
                      f"parts {', '.join(map(str, riders))} - all passed "
                      f"inspection - out on the conveyor", GREEN)
        try:
            n = real["dispatch"](*a, **k)
            rec.hold(1.5)
            return n
        finally:
            rec.banner = None

    base.go_to_station = goto
    cell.pick, cell.place = pick, place
    cell.inspect, cell.dispatch = inspect, dispatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out/mobile.mp4")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parts", type=int, default=3)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--mistakes", action="store_true")
    ap.add_argument("--steps-per-frame", type=int, default=50)
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
    rec.card("Why it drives back and forth",
             ["Every part must visit: storage -> machine -> inspection",
              "-> dispatch. It carries one part at a time, the machine",
              "takes one part per 15 s cycle, and the robot only sees",
              "the station it is parked at."], 5.0)

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
    hook_stages(rec, cell, w.base)
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
