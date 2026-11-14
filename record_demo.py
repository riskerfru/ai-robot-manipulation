#!/usr/bin/env python3
# =============================================================
#   RECORD DEMO - renders the cell to a video with a status panel
#
#   Runs the simulated cell headless and writes an MP4: the cell
#   on the left, and on the right what the controller knows at
#   that moment (order, task list, cell and safety state, the
#   last placement measurement, recent audit events).
#
#   python record_demo.py normal      one PLC job, three parts
#   python record_demo.py guard       guard door opens mid-job
#   python record_demo.py verify      the same knocked part, without
#                                     and with placement measurement
#   python record_demo.py packing     FMCG mixed-case packing with a format
#                                     changeover, planned from typed orders
#                                     by the language model (needs
#                                     ANTHROPIC_API_KEY; --offline uses the
#                                     built-in parser instead)
#   python record_demo.py assembly    three components seated in a housing
#                                     by force-guided insertion
#   python record_demo.py assembly-compare
#                                     the same assembly, position only
#                                     and then force-guided
#   python record_demo.py order --text "dispatch all blocks to the conveyor"
#                                     free-text order through the language
#                                     model (needs ANTHROPIC_API_KEY)
#
#   Needs ffmpeg on the PATH. The panel is drawn from the audit
#   trail and the PLC signals, so it shows what was recorded,
#   not a script written for the video.
# =============================================================

import argparse
import contextlib
import io
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

import config as cfg
from cell import layout as L
from cell.dressing import CellDressing
from cell.factory import build_sim_cell
from plc.cell_controller import CellState
from plc.recipes import RECIPES
from utils.logger import NullLogger

W, H = 1280, 720
SCENE_W = 848
PANEL_X = SCENE_W
FPS = 24
STEPS_PER_FRAME = 60            # 0.25 s of simulated time per frame
SPEED = STEPS_PER_FRAME / 240 * FPS

BG     = (18, 22, 28)
CARD   = (28, 34, 43)
LINE   = (48, 56, 68)
TEXT   = (232, 236, 241)
MUTED  = (139, 148, 162)
GREEN  = (61, 190, 123)
AMBER  = (242, 179, 61)
RED    = (229, 84, 88)
BLUE   = (104, 160, 255)

FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu",
             "C:/Windows/Fonts", "/System/Library/Fonts/Supplemental",
             "/Library/Fonts"]
FONT_FILES = {
    "regular": ["Inter-Regular.otf", "DejaVuSans.ttf", "segoeui.ttf", "Arial.ttf"],
    "bold":    ["Inter-SemiBold.otf", "DejaVuSans-Bold.ttf", "segoeuib.ttf",
                "Arial Bold.ttf"],
    "mono":    ["DejaVuSansMono.ttf", "consola.ttf", "Courier New.ttf"],
}


def font(kind, size):
    for name in FONT_FILES[kind]:
        for folder in FONT_DIRS:
            path = os.path.join(folder, name)
            if os.path.exists(path):
                return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font("bold", 22)
F_LABEL = font("bold", 12)
F_BODY  = font("regular", 16)
F_BOLD  = font("bold", 16)
F_SMALL = font("regular", 13)
F_MONO  = font("mono", 13)
F_CARD_T = font("bold", 44)
F_CARD_B = font("regular", 24)


def wrap(text, width=44):
    lines = [""]
    for word in text.split():
        if lines[-1] and len(lines[-1]) + 1 + len(word) > width:
            lines.append("")
        lines[-1] = (lines[-1] + " " + word).strip()
    return lines


def short(part):
    """'lemon_s1' -> 'lemon'; other names unchanged."""
    head, _, tail = part.partition("_")
    return head if tail[:1] in ("s", "t") and tail[1:].isdigit() else part


def describe(subtask):
    if subtask.get("label"):
        return subtask["label"][:1].upper() + subtask["label"][1:]
    action = subtask.get("action")
    if action == "pick":
        return f"Pick {subtask.get('object')}"
    if action == "place_slot":
        return f"Place in slot {subtask.get('slot')}"
    if action == "place":
        return f"Place at ({subtask.get('x')}, {subtask.get('y')})"
    if action == "insert":
        return f"Seat {subtask.get('object')} in its pocket"
    if action == "stack":
        return f"Stack {subtask.get('object')} on {subtask.get('on')}"
    return str(action).capitalize()


class Hud:
    """Everything the panel shows. Filled from audit events."""

    def __init__(self, title, subtitle, footer):
        self.title, self.subtitle, self.footer = title, subtitle, footer
        self.order_source = ""
        self.order_text = ""
        self.tasks = []                  # [text, status]
        self.cell_state = "READY"
        self.safety_state = "RUN"
        self.guard_closed = True
        self.measure = ("No placement yet", MUTED)
        self.events = []                 # (time_s, text, colour)
        self.truth = None                # [(part, drift_cm, ok)]
        self.sim_time = 0.0
        self.measured = True
        self.free_text = False           # order typed, not a PLC job
        self.speed = SPEED
        self.force = None                # live contact force (N), assembly
        self.force_trace = []
        self.result_label = "Last placement"
        self.plan_line = None            # packing: the validated plan
        self.show_guard = True

    def event(self, text, colour=TEXT):
        self.events.append((self.sim_time, text, colour))
        del self.events[:-6]

    # ---- audit trail -> panel ------------------------------------
    def on_audit(self, name, d):
        if name == "job_start":
            self.order_source = f"PLC job {d['job_id']}  ·  recipe {d['recipe']}"
            self.order_text = d.get("name", "")
            self.truth = None
            self.event(f"PLC: JOB_START job {d['job_id']}", BLUE)
        elif name == "order_text":
            self.order_source = "Operator, typed order"
            self.order_text = d["order"]
            self.free_text = True
            self.cell_state = "BUSY"
            self.tasks = []
            self.truth = None
            self.plan_line = ("planning...", MUTED)
        elif name == "order_plan":
            items = ", ".join(f"{q} {f}" for f, q in d["items"])
            self.plan_line = (f"Plan: {d['format']} tray, {items}", GREEN)
            self.event(f"Plan from {d['source']}", BLUE)
            self.event(f"Plan checked: fills {d['nests']} nests, stock OK",
                       GREEN)
        elif name == "order_refused":
            self.plan_line = ("Order refused", RED)
            self.cell_state = "READY"
            self.measure = (d["reason"][:52], RED)
            self.event(f"REFUSED: {d['reason'][:34]}", RED)
        elif name == "changeover":
            self.event(f"Changeover {d['from_format']} to {d['to_format']}: "
                       f"{d['carton_mm']} mm", AMBER)
            self.event(f"New pattern {d['pattern']}, from the SKU master",
                       AMBER)
        elif name == "tray_out":
            self.event(f"Tray out: {d['cartons']} cartons", MUTED)
        elif name == "tray_in":
            self.event(f"Empty {d['format']} tray in position", MUTED)
        elif name == "tray_check":
            self.event(f"Tray check: {d['seated']} of {d['nests']} seated",
                       GREEN if d["seated"] == d["nests"] else RED)
        elif name == "order_received":
            if not d.get("recipe"):
                self.order_source = "Operator, free text"
                self.free_text = True
                self.cell_state = "BUSY"
                self.truth = None
            if self.plan_line is None:
                self.order_text = d.get("order", self.order_text)
        elif name == "order_decomposed":
            self.tasks = [[describe(s), "pending"] for s in d["subtasks"]]
        elif name in ("recipe_decision", "llm_decision"):
            for task in self.tasks:
                if task[1] in ("pending", "current"):
                    task[1] = "current"
                    break
            if name == "llm_decision":
                action = d.get("decision", {}).get("action", "?")
                self.event(f"Model requests: {action}", BLUE)
        elif name == "tool_call":
            self.event(f"Model reads: {d.get('tool')}", MUTED)
        elif name == "action_result":
            ok = d["result"].get("status") == "success"
            for task in self.tasks:
                if task[1] == "current":
                    task[1] = "done" if ok else "failed"
                    break
            if not ok:
                self.event(f"Failed: {d['result'].get('detail', '')[:44]}", RED)
        elif name == "safety_decision" and not d.get("allowed", True):
            self.event(f"Safety gate DENIED {d['kind']}: {d['reason']}", RED)
        elif name == "safety_trip":
            self.safety_state = d.get("state", self.safety_state)
            self.event(f"SAFETY STOP: {d.get('reason', '')}", RED)
        elif name == "verification":
            r = d["result"]
            part, slot = d["part"], d["slot"]
            if not self.measured:
                self.measure = (f"{part} in slot {slot}: not measured, "
                                f"reported as placed", AMBER)
            elif r["status"] == "success":
                extra = " after re-placing" if r.get("retried") else ""
                self.measure = (f"{part} in slot {slot}: {r['drift_cm']:.1f} cm, "
                                f"in tolerance{extra}", GREEN)
                self.event(f"Measured {part}: {r['drift_cm']:.1f} cm, OK", GREEN)
            else:
                self.measure = (f"{part}: {r['detail'][:46]}", RED)
                self.event(f"Measured {part}: FAILED", RED)
        elif name == "insertion":
            r = d["result"]
            part = short(d["part"])
            if r.get("unverified"):
                self.measure = (f"{part}: released at nominal, not checked",
                                AMBER)
                self.event(f"{part}: released, not checked", AMBER)
            elif r["status"] == "success":
                how = (f"search {r['search_offset_mm']:.0f} mm, "
                       if r.get("searched") else
                       "straight in, " if r.get("start_offset_mm") else "")
                self.measure = (f"{part} seated: {how}"
                                f"peak {r['peak_force_n']:.0f} N", GREEN)
                self.event(f"Seated {part}: {how}peak "
                           f"{r['peak_force_n']:.0f} N", GREEN)
            else:
                self.measure = (f"{part}: not seated", RED)
                self.event(f"{part}: NOT seated, reported as failure", RED)
        elif name == "job_done":
            self.event(f"PLC: DONE, {d['parts_ok']} parts reported placed", GREEN)
        elif name == "job_fault":
            self.event(f"PLC: FAULT {d['code']} ({d['reason']})", RED)
        elif name == "fault_reset":
            if d.get("allowed"):
                self.safety_state = "RUN"
                self.event("Fault reset accepted", GREEN)
            else:
                self.event(f"Reset refused: {d['reason'].replace('cannot reset: ', '')}",
                           AMBER)
        elif name == "cell_state":
            self.cell_state = d["state"]
        elif name == "order_result" and self.free_text:
            self.cell_state = "DONE" if d.get("success") else "FAULT"
            self.event("Order complete" if d.get("success")
                       else "Order FAILED", GREEN if d.get("success") else RED)
        elif name == "degraded_mode":
            self.event("Language model unavailable: recipe mode", AMBER)

    # ---- drawing -------------------------------------------------
    def draw(self, scene):
        img = Image.new("RGB", (W, H), BG)
        img.paste(scene, (0, 0))
        d = ImageDraw.Draw(img)
        x0, x1 = PANEL_X + 24, W - 24
        y = 22

        d.text((x0, y), self.title, font=F_TITLE, fill=TEXT)
        y += 30
        d.text((x0, y), self.subtitle, font=F_SMALL, fill=MUTED)
        y += 30

        def label(text):
            nonlocal y
            d.line([(x0, y), (x1, y)], fill=LINE, width=1)
            y += 12
            d.text((x0, y), text.upper(), font=F_LABEL, fill=MUTED)
            y += 22

        label("Order")
        d.text((x0, y), self.order_source or "Waiting for a job",
               font=F_SMALL, fill=BLUE)
        y += 20
        for line in (wrap(self.order_text)[:2] or ["-"]):
            d.text((x0, y), line, font=F_BODY, fill=TEXT)
            y += 21
        if self.plan_line:
            d.text((x0, y + 1), self.plan_line[0][:52], font=F_SMALL,
                   fill=self.plan_line[1])
            y += 20
        y += 9

        compact = len(self.tasks) > 4      # long task lists: tighter panel
        label("Tasks")
        for text, status in self.tasks[:7] or [["-", "pending"]]:
            self._mark(d, x0, y, status)
            d.text((x0 + 24, y - 2), text,
                   font=F_BOLD if status == "current" else F_BODY,
                   fill=TEXT if status != "pending" else MUTED)
            y += 20 if compact else 23
        y += 8

        if compact:
            d.line([(x0, y), (x1, y)], fill=LINE, width=1)
            y += 12
        else:
            label("Cell")
        self._pill(d, x0, y, "CELL " + self.cell_state,
                   {"READY": GREEN, "BUSY": BLUE, "DONE": GREEN,
                    "FAULT": RED}.get(self.cell_state, MUTED))
        self._pill(d, x0 + 150, y, "SAFETY " + self.safety_state.replace("_", " "),
                   GREEN if self.safety_state == "RUN" else RED)
        y += 34
        if self.show_guard:
            d.text((x0, y), "Guard door: "
                   + ("closed" if self.guard_closed else "OPEN"),
                   font=F_SMALL, fill=MUTED if self.guard_closed else RED)
            y += 20
        y += 8

        if self.force is not None:
            label("Contact force on the part")
            bar_w = x1 - x0 - 70
            level = min(1.0, self.force / 60.0)
            colour = (GREEN if self.force < 2 else AMBER
                      if self.force < 40 else RED)
            d.rounded_rectangle([x0, y + 2, x0 + bar_w, y + 16], radius=7,
                                fill=CARD)
            if level > 0:
                d.rounded_rectangle([x0, y + 2,
                                     x0 + max(14, int(bar_w * level)), y + 16],
                                    radius=7, fill=colour)
            d.text((x0 + bar_w + 10, y - 1), f"{self.force:4.0f} N",
                   font=F_BOLD, fill=TEXT)
            y += 24
            trace = self.force_trace[-120:]
            h = 24 if compact else 34
            if len(trace) > 1:
                pts = [(x0 + i * (x1 - x0) / 119,
                        y + h - min(1.0, f / 60.0) * h)
                       for i, f in enumerate(trace)]
                d.line(pts, fill=BLUE, width=2)
            y += h + 8

        label(self.result_label)
        d.text((x0, y), self.measure[0][:52], font=F_BODY, fill=self.measure[1])
        y += 28 if compact else 32

        if self.truth is not None:
            label("True position (simulator)")
            two_up = len(self.truth) > 4
            for i, (part, drift, ok) in enumerate(self.truth):
                text = (drift if isinstance(drift, str)
                        else f"{drift:.1f} cm from slot centre")
                tx = x0 + ((x1 - x0) // 2 if two_up and i % 2 else 0)
                self._mark(d, tx, y, "done" if ok else "failed")
                d.text((tx + 24, y - 2), f"{part}: {text}",
                       font=F_SMALL if two_up else F_BODY,
                       fill=TEXT if ok else RED)
                if not two_up or i % 2 or i == len(self.truth) - 1:
                    y += 22
        else:
            label("Audit trail")
            row = 19 if compact else 21
            room = max(1, (H - 44 - y) // row)       # stay above the footer
            for t, text, colour in self.events[-min(room, 5):]:
                d.text((x0, y), f"{t:6.1f}s", font=F_MONO, fill=MUTED)
                d.text((x0 + 68, y - 1), text[:40], font=F_SMALL, fill=colour)
                y += row

        d.text((x0, H - 30), self.footer, font=F_SMALL, fill=MUTED)
        d.text((16, H - 30), f"Simulation  ·  {self.speed:.0f}x speed  ·  "
               f"t = {self.sim_time:5.1f} s", font=F_SMALL, fill=(70, 76, 86))
        return img

    @staticmethod
    def _mark(d, x, y, status):
        """Status marks drawn as shapes, so no font has to supply them."""
        cx, cy = x + 7, y + 8
        if status == "done":
            d.line([(cx - 5, cy), (cx - 1, cy + 4), (cx + 6, cy - 5)],
                   fill=GREEN, width=2)
        elif status == "current":
            d.polygon([(cx - 4, cy - 6), (cx + 6, cy), (cx - 4, cy + 6)],
                      fill=BLUE)
        elif status == "failed":
            d.line([(cx - 5, cy - 5), (cx + 5, cy + 5)], fill=RED, width=2)
            d.line([(cx - 5, cy + 5), (cx + 5, cy - 5)], fill=RED, width=2)
        else:
            d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=MUTED, width=1)

    @staticmethod
    def _pill(d, x, y, text, colour):
        w = int(d.textlength(text, font=F_LABEL)) + 22
        d.rounded_rectangle([x, y, x + w, y + 24], radius=12, fill=colour)
        d.text((x + 11, y + 5), text, font=F_LABEL, fill=BG)


class Recorder:
    """Captures a frame every STEPS_PER_FRAME physics steps."""

    def __init__(self, path):
        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found on the PATH")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.proj = p.computeProjectionMatrixFOV(42, SCENE_W / H, 0.05, 6)
        self.camera()
        self.hud = None
        self.steps_per_frame = STEPS_PER_FRAME
        self.steps = 0
        self.frames = 0
        self.last = None
        self._orig_step = p.stepSimulation
        p.stepSimulation = self._step

    def camera(self, target=(0.36, 0.0, 0.26), distance=2.05, yaw=38,
               pitch=-36):
        self.view = p.computeViewMatrixFromYawPitchRoll(
            list(target), distance, yaw, pitch, 0, 2)

    def _step(self, *a, **k):
        out = self._orig_step(*a, **k)
        self.steps += 1
        if self.hud is not None:
            self.hud.sim_time = self.steps / 240
            if self.steps % self.steps_per_frame == 0:
                self.frame()
        return out

    def _write(self, img):
        self.proc.stdin.write(img.tobytes())
        self.frames += 1

    def frame(self):
        raw = p.getCameraImage(SCENE_W, H, self.view, self.proj,
                               renderer=p.ER_TINY_RENDERER, shadow=1,
                               lightDirection=[0.6, -0.5, 1.6])
        scene = Image.fromarray(
            np.reshape(raw[2], (H, SCENE_W, 4))[:, :, :3].astype("uint8"))
        self.hud.speed = self.steps_per_frame / 240 * FPS
        self.last = self.hud.draw(scene)
        self._write(self.last)

    def hold(self, seconds):
        """Freeze on the current state so it can be read."""
        self.frame()
        for _ in range(int(seconds * FPS)):
            self._write(self.last)

    def card(self, heading, lines, seconds=3.0):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((96, 250), heading, font=F_CARD_T, fill=TEXT)
        y = 330
        for line in lines:
            d.text((96, y), line, font=F_CARD_B, fill=MUTED)
            y += 38
        for _ in range(int(seconds * FPS)):
            self._write(img)

    def close(self):
        p.stepSimulation = self._orig_step
        self.proc.stdin.close()
        self.proc.wait()
        print(f"wrote {self.path}  ({self.frames / FPS:.0f} s, "
              f"{self.frames} frames)", file=sys.__stdout__)


# ---- cell set-up ---------------------------------------------

def make_cell(rec, hud, llm=False, scene="dispatch",
              housing_offset=(0.0, 0.0)):
    """Build a cell and connect it to the recorder and the panel."""
    with contextlib.redirect_stdout(io.StringIO()):
        cell = build_sim_cell(NullLogger(), llm_enabled=llm, scene=scene,
                              housing_offset=housing_offset)
    cell.dressing = CellDressing(cell.robot)
    cell.housing_offset = housing_offset
    if scene in ("assembly", "packing"):
        connect_assembly_view(cell, rec, hud)
    original_log = cell.audit.log

    def log(event, **payload):
        record = original_log(event, **payload)
        hud.on_audit(event, payload)
        light = ("red" if hud.cell_state == "FAULT" or hud.safety_state != "RUN"
                 else "amber" if hud.cell_state == "BUSY" else "green")
        cell.dressing.set_light(light)
        return record

    cell.audit.log = log
    rec.hud = hud
    return cell


WIDE_ASSEMBLY = dict(target=(0.47, 0.0, 0.20), distance=1.6, yaw=38,
                     pitch=-38)


def connect_assembly_view(cell, rec, hud):
    """Force read-out on the panel and a close-up during each insertion."""
    hud.force = 0.0
    hud.result_label = "Last insertion"
    hud.measure = ("No insertion yet", MUTED)
    rec.camera(**WIDE_ASSEMBLY)
    backend = cell.robot

    def on_force(z, force):
        hud.force = max(0.0, force)
        hud.force_trace.append(hud.force)

    backend.on_force = on_force
    original = backend.insert_part

    def insert_part(name, pocket, *a, **k):
        rec.camera(target=(pocket["x"], pocket["y"], 0.15), distance=0.62,
                   yaw=62, pitch=-24)
        hud.force_trace.clear()
        rec.steps_per_frame = 2 * STEPS_PER_FRAME   # insertion is slow
        try:
            return original(name, pocket, *a, **k)
        finally:
            rec.steps_per_frame = STEPS_PER_FRAME
            hud.force = 0.0
            rec.hold(1.2)
            rec.camera(**WIDE_ASSEMBLY)

    backend.insert_part = insert_part


def show_assembly_truth(cell, hud):
    """Is each component really seated? From the simulator's ground truth."""
    from cell import assembly as A
    off = cell.housing_offset
    centre = (A.HOUSING_CENTRE[0] + off[0], A.HOUSING_CENTRE[1] + off[1])
    truth = []
    for name in A.COMPONENTS:
        pocket = A.pocket_for(name, centre)
        pos = cell.robot.get_object_position(name)
        gap = 1000 * (pos[2] - pocket["half"] - pocket["floor_z"])
        side = 1000 * math.hypot(pos[0] - pocket["x"], pos[1] - pocket["y"])
        seated = abs(gap) <= 2.0 and side <= 1000 * pocket["clearance"] + 1.0
        truth.append((name, "seated in its pocket" if seated else
                      f"NOT seated, {gap:.0f} mm above the floor", seated))
    hud.truth = truth


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def set_guard(cell, hud, closed):
    (cell.soft_plc.close_guard if closed else cell.soft_plc.open_guard)()
    cell.dressing.set_guard(closed)
    hud.guard_closed = closed
    hud.event("Guard door closed" if closed else "Guard door OPENED",
              GREEN if closed else RED)


def show_truth(cell, hud, recipe=1):
    """Compare each part with its slot using the simulator's ground truth."""
    truth = []
    steps = RECIPES[recipe]["subtasks"]
    for pick, place in zip(steps[0::2], steps[1::2]):
        part = pick["object"]
        sx, sy = cell.robot.conveyor_slots[place["slot"]]
        pos = cell.robot.get_object_position(part)
        drift = 100 * math.hypot(pos[0] - sx, pos[1] - sy)
        truth.append((part, drift,
                      drift <= 100 * cfg.PLACEMENT["tolerance_m"]))
    hud.truth = truth


def show_truth_from_slots(cell, hud):
    """As show_truth, for an order whose steps the model chose."""
    truth = []
    for slot, part in sorted(cell.robot.slot_occupants.items()):
        sx, sy = cell.robot.conveyor_slots[slot]
        pos = cell.robot.get_object_position(part)
        drift = 100 * math.hypot(pos[0] - sx, pos[1] - sy)
        truth.append((part, drift,
                      drift <= 100 * cfg.PLACEMENT["tolerance_m"]))
    hud.truth = truth


def after_first_place(cell, action):
    """Run `action` once, right after the first part is released."""
    backend = cell.robot
    original = backend.place_object
    fired = {"done": False}

    def place(x, y, stack_on=None):
        out = original(x, y, stack_on=stack_on)
        if not fired["done"]:
            fired["done"] = True
            action()
        return out

    backend.place_object = place


def knock_part(cell, hud, on_place_number, dx, dy):
    """Shift a part after release, as a slip or a bump would."""
    backend = cell.robot
    original = backend.place_object
    count = {"n": 0}

    def place(x, y, stack_on=None):
        held = backend.grasped_object
        out = original(x, y, stack_on=stack_on)
        count["n"] += 1
        if count["n"] == on_place_number and held:
            body = backend.objects[held]["id"]
            pos, orn = p.getBasePositionAndOrientation(body)
            p.resetBasePositionAndOrientation(
                body, [pos[0] + dx, pos[1] + dy, pos[2] + 0.005], orn)
            p.resetBaseVelocity(body, [0, 0, 0], [0, 0, 0])
            hud.event(f"Disturbance: {held} knocked "
                      f"{100 * math.hypot(dx, dy):.0f} cm", AMBER)
            for _ in range(60):
                backend.step()
        return out

    backend.place_object = place


def announce_retries(cell, hud):
    """Put each out-of-tolerance measurement on the panel as it happens."""
    verifier = cell.verifier
    original = verifier.verify_and_fix

    def verify(name, tx, ty, slot_num=None, attempt=0, commanded=None):
        pos = cell.supervised.get_object_position(name)
        if pos:
            drift = math.hypot(pos[0] - tx, pos[1] - ty)
            if drift >= verifier.drift_retry and attempt < verifier.max_retries:
                hud.measure = (f"{name}: {100 * drift:.1f} cm, out of tolerance, "
                               f"re-placing ({attempt + 1}/{verifier.max_retries})",
                               AMBER)
                hud.event(f"Measured {name}: {100 * drift:.1f} cm, re-placing",
                          AMBER)
        return original(name, tx, ty, slot_num=slot_num, attempt=attempt,
                        commanded=commanded)

    verifier.verify_and_fix = verify


def run_job(cell, job_id, recipe=1):
    cell.soft_plc.request_job(job_id, recipe)
    return quiet(cell.controller.poll)


# ---- scenarios -----------------------------------------------

def scenario_normal(rec):
    hud = Hud("PLC-run job", "The cell PLC selects a recipe; every placement is measured",
              "Recipe mode: no language model in this run")
    cell = make_cell(rec, hud)
    announce_retries(cell, hud)
    quiet(cell.controller.poll)
    rec.hold(1.5)
    run_job(cell, 1001)
    show_truth(cell, hud)
    rec.hold(4.0)
    cell.soft_plc.stop()


def scenario_guard(rec):
    hud = Hud("Guard door opens mid-job", "The safety gate refuses motion; the PLC sees a fault",
              "Recipe mode: no language model in this run")
    cell = make_cell(rec, hud)
    announce_retries(cell, hud)
    after_first_place(cell, lambda: set_guard(cell, hud, False))
    quiet(cell.controller.poll)
    rec.hold(1.5)

    state = run_job(cell, 1001)
    rec.hold(3.0)
    if state is CellState.FAULT:
        cell.soft_plc.reset_fault()            # cause still present
        quiet(cell.controller.poll)
        rec.hold(3.0)
        cell.soft_plc.release_reset()
        quiet(cell.controller.poll)

        set_guard(cell, hud, True)
        import time
        time.sleep(0.1)
        quiet(cell.controller.poll)
        cell.soft_plc.reset_fault()
        quiet(cell.controller.poll)
        cell.soft_plc.release_reset()
        quiet(cell.controller.poll)
        rec.hold(2.5)

    # The interrupted order is finished by a new job on the same parts.
    hud.tasks = []
    run_job(cell, 1002, recipe=1)
    rec.hold(3.5)
    cell.soft_plc.stop()


def scenario_verify(rec):
    from eval_physics import OpenLoopVerifier
    knock = dict(on_place_number=2, dx=0.0, dy=0.09)

    rec.card("Without measurement",
             ["The second part is knocked 9 cm after the gripper lets go.",
              "The controller does not measure. It reports the job as done."])
    hud = Hud("Without measurement", "The part is knocked after release; nothing checks it",
              "Recipe mode: no language model in this run")
    hud.measured = False
    cell = make_cell(rec, hud)
    cell.agent.verifier = OpenLoopVerifier()
    knock_part(cell, hud, **knock)
    quiet(cell.controller.poll)
    rec.hold(1.0)
    run_job(cell, 2001)
    show_truth(cell, hud)
    rec.hold(5.0)
    cell.soft_plc.stop()
    rec.hud = None
    p.disconnect()

    rec.card("With measurement",
             ["The same knock, on the same part.",
              "The controller measures every placement before reporting it."])
    hud = Hud("With measurement", "Measure, re-place if out of tolerance, or report failure",
              "Recipe mode: no language model in this run")
    cell = make_cell(rec, hud)
    announce_retries(cell, hud)
    knock_part(cell, hud, **knock)
    quiet(cell.controller.poll)
    rec.hold(1.0)
    run_job(cell, 2002)
    show_truth(cell, hud)
    rec.hold(5.0)
    cell.soft_plc.stop()


def scenario_assembly(rec, offset_mm=(2.0, -1.5)):
    off = (offset_mm[0] / 1000, offset_mm[1] / 1000)
    hud = Hud("Force-guided assembly",
              "Three components seated in pockets with 1.5 mm clearance",
              "Recipe mode: no language model in this run")
    cell = make_cell(rec, hud, scene="assembly", housing_offset=off)
    quiet(cell.controller.poll)
    rec.hold(1.5)
    run_job(cell, 3001, recipe=3)
    show_assembly_truth(cell, hud)
    rec.hold(5.0)
    cell.soft_plc.stop()


def scenario_assembly_compare(rec, offset_mm=(3.0, 2.5)):
    off = (offset_mm[0] / 1000, offset_mm[1] / 1000)
    shift = math.hypot(*offset_mm)

    rec.card("Position only",
             [f"The housing sits {shift:.0f} mm from where the robot expects it.",
              "The pockets leave 1.5 mm per side.",
              "The robot goes to the taught position and lets go."])
    hud = Hud("Position only", "Taught positions, no force sensing, no check",
              "Recipe mode: no language model in this run")
    cell = make_cell(rec, hud, scene="assembly", housing_offset=off)
    cell.robot.use_force_sensing = False
    quiet(cell.controller.poll)
    rec.hold(1.0)
    run_job(cell, 3101, recipe=3)
    show_assembly_truth(cell, hud)
    rec.hold(5.0)
    cell.soft_plc.stop()
    rec.hud = None
    p.disconnect()

    rec.card("Force-guided",
             ["The same housing, the same 4 mm error.",
              "The robot feels the rim, searches, and seats each part.",
              "A part that does not seat is reported as a failure."])
    hud = Hud("Force-guided", "Feel the contact, search, seat, then check",
              "Recipe mode: no language model in this run")
    cell = make_cell(rec, hud, scene="assembly", housing_offset=off)
    quiet(cell.controller.poll)
    rec.hold(1.0)
    run_job(cell, 3102, recipe=3)
    show_assembly_truth(cell, hud)
    rec.hold(5.0)
    cell.soft_plc.stop()


def scenario_packing(rec, orders, offline=False):
    """Typed orders, a planned case, force-guided packing, a changeover."""
    from packing_orders import make_planner, run_order
    planner, label = make_planner(offline)
    hud = Hud("Mixed-case packing",
              "Typed order, planned case, force-guided packing",
              f"Planner: {label}")
    hud.show_guard = False
    cell = make_cell(rec, hud, scene="packing",
                     housing_offset=(0.002, -0.0015))
    hud.measure = ("No carton packed yet", MUTED)
    rec.hold(1.5)

    seated = total = 0
    refused = []
    for number, order in enumerate(orders):
        lines = wrap(order, 56)
        lines[0] = '"' + lines[0]
        lines[-1] += '"'
        rec.card("The operator types an order" if number == 0
                 else "The operator types the next order", lines,
                 seconds=4.0)
        hud.on_audit("order_text", {"order": order})
        hud.cell_state = "READY" if number == 0 else hud.cell_state
        rec.hold(1.0)
        out = quiet(run_order, cell, order, planner, tray_number=number)
        if "refused" in out:
            refused.append(out["refused"])
            rec.hold(5.0)
            continue
        hud.truth = [(f"nest {c['nest']} {c['flavour']}",
                      "seated" if c["seated"]
                      else f"NOT seated ({c['gap_mm']:.0f} mm up)",
                      c["seated"]) for c in out["check"]]
        seated += sum(c["seated"] for c in out["check"])
        total += len(out["check"])
        rec.hold(5.0)

    lines = [f"{seated} of {total} cartons seated in nests with 2 mm clearance.",
             "Each tray sat a few millimetres off its mark; the robot was "
             "not told.",
             "The format change came from a typed order and the SKU master."]
    if refused:
        lines.append(f"{len(refused)} order(s) refused: {refused[0][:60]}")
    lines.append(f"Planner: {label}. Simulation.")
    rec.card("Result", lines, seconds=6.0)
    cell.soft_plc.stop()


def scenario_order(rec, text, scene="dispatch"):
    if not cfg.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY is not set; this scenario needs the "
                 "language model")
    hud = Hud("Language-model order" if scene != "assembly"
              else "Language-model assembly",
              "The model plans the steps; code gates, measures and checks",
              f"Language model: {cfg.AI['model']}")
    off = (0.002, -0.0015) if scene == "assembly" else (0.0, 0.0)
    words, lines = text.split(), [""]
    for word in words:
        if len(lines[-1]) + len(word) > 58:
            lines.append("")
        lines[-1] = (lines[-1] + " " + word).strip()
    rec.card("The operator types an order",
             ['"' + lines[0]] + lines[1:-1]
             + ([lines[-1] + '"'] if len(lines) > 1 else [])
             if len(lines) > 1 else ['"' + text + '"'], seconds=3.5)
    cell = make_cell(rec, hud, llm=True, scene=scene, housing_offset=off)
    if scene != "assembly":
        announce_retries(cell, hud)
    rec.hold(1.5)
    quiet(cell.agent.run, text)
    if scene == "assembly":
        show_assembly_truth(cell, hud)
    else:
        show_truth_from_slots(cell, hud)
    rec.hold(5.0)
    cell.soft_plc.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", choices=["normal", "guard", "verify", "order",
                                         "assembly", "assembly-compare",
                                         "packing"])
    ap.add_argument("--orders", nargs="+", default=None,
                    help="packing: the orders to type, in sequence")
    ap.add_argument("--offline", action="store_true",
                    help="packing: plan with the built-in parser instead "
                         "of the language model")
    ap.add_argument("--scene", default="dispatch",
                    choices=["dispatch", "assembly"],
                    help="cell used by the 'order' scenario")
    ap.add_argument("--text", default="dispatch all blocks to the conveyor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = args.out or os.path.join("demo_out", f"{args.scenario}.mp4")
    rec = Recorder(out)
    try:
        if args.scenario == "normal":
            scenario_normal(rec)
        elif args.scenario == "guard":
            scenario_guard(rec)
        elif args.scenario == "verify":
            scenario_verify(rec)
        elif args.scenario == "assembly":
            scenario_assembly(rec)
        elif args.scenario == "assembly-compare":
            scenario_assembly_compare(rec)
        elif args.scenario == "packing":
            from run_packing import ORDER_1, ORDER_2
            scenario_packing(rec, args.orders or [ORDER_1, ORDER_2],
                             args.offline)
        else:
            scenario_order(rec, args.text, args.scene)
    finally:
        rec.close()


if __name__ == "__main__":
    main()
