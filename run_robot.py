#!/usr/bin/env python3
# =============================================================
#   RUN ROBOT - type any instruction
#
#   Nothing about the scene is written down for the robot. Shapes,
#   colours and positions are random. The camera measures what is
#   there; the language model is shown the picture, decides what
#   each thing is and what you meant, and plans the steps. Code
#   checks every step before the arm moves.
#
#   python run_robot.py --gui              window, then type
#   python run_robot.py --gui --seed 7     a different table
#   python run_robot.py --gui --objects 8  more things on it
#   python run_robot.py --quiet            results only, no log
# =============================================================

import argparse
import contextlib
import io
import sys
import time

EXAMPLES = [
    "what can you see?",
    "put the yellow one in the brown box",
    "pick the blue and the red thing and dispatch them",
    "put all the round ones in the grey bin",
    "everything long goes in the tray, then send it out",
    "move the green one out of the way",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instructions", nargs="*",
                    help="run these and stop, instead of asking")
    ap.add_argument("--seed", type=int, default=0,
                    help="a different random table")
    ap.add_argument("--objects", type=int, default=6,
                    help="how many loose things (up to 9)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--quiet", action="store_true",
                    help="hide the robot's log")
    args = ap.parse_args()

    # Windows consoles choke on characters outside their code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    import config as cfg
    if not cfg.ANTHROPIC_API_KEY:
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in the .env file "
                 "in this folder.\nThe language model is what reads the "
                 "picture and your instructions.")
    import anthropic
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY,
                                 timeout=max(90.0, cfg.AI.get("timeout_s", 90.0)),
                                 max_retries=cfg.AI.get("api_retries", 1))

    import commands as C
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger

    started = time.time()

    def log(tag, text):
        if not args.quiet:
            print(f"  {time.time() - started:6.1f}s [{tag:<6}] {text}",
                  flush=True)

    def robot_log():
        """The robot's own lines go to the terminal unless --quiet."""
        return contextlib.redirect_stdout(io.StringIO() if args.quiet
                                          else sys.__stdout__)

    print("Building the cell...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        cell = build_sim_cell(NullLogger(), gui=args.gui, scene="open",
                              seed=args.seed,
                              open_count=max(1, min(9, args.objects)))
    oc = cell.robot.open_cell

    # Every safety decision is shown as it is made.
    audit_log = cell.audit.log

    def logged(event, **payload):
        record = audit_log(event, **payload)
        if event == "safety_decision":
            verdict = "authorised" if payload.get("allowed") else "DENIED"
            target = payload.get("target") or ()
            where = ", ".join(f"{v:.3f}" for v in target)
            log("GATE", f"{payload.get('kind')} {verdict} at ({where})"
                        + ("" if payload.get("allowed")
                           else f": {payload.get('reason')}"))
        return record
    cell.audit.log = logged

    print(f"Random table, seed {args.seed}: {args.objects} loose things "
          f"and two containers.")
    print(f"The robot has been told nothing about them. "
          f"Language model: {cfg.AI['model']}")

    todo = list(args.instructions)
    if not todo:
        print("\nThe cell is set up. Nothing has moved yet."
              + ("  Drag things about in the window if you like."
                 if args.gui else ""))
        print("Type  start  when you are ready.")
        while True:
            try:
                first = input("ready> ").strip().lower()
            except EOFError:
                first = "start"
            if first in ("start", "go", "begin", ""):
                break
            if first in ("quit", "exit", "q"):
                cell.soft_plc.stop()
                return 0
            print("Type  start  to begin, or  quit  to stop.")

    log("CAMERA", "arm parked clear, taking the first picture")
    with robot_log():
        found = oc.scan()
    cell.audit.log("scan", things=len(found))
    for t in found:
        kind = "container" if t.container else "thing"
        log("CAMERA", f"{kind} {t.id}: at ({t.x:.3f}, {t.y:.3f})  "
                      f"{1000 * t.long:.0f} x {1000 * t.short:.0f} mm, "
                      f"{1000 * (t.top - t.ground):.0f} mm tall, colour "
                      f"rgb({', '.join(str(int(255 * float(v))) for v in t.rgb)})"
                      f", roundness {t.roundness:.2f}")
    print(f"\nThe camera found {sum(not t.container for t in found)} loose "
          f"things and {sum(t.container for t in found)} containers. It "
          f"has measured them; it does not know what they are.")
    if args.gui:
        print("Each one has its number floating above it in the window.")

    print("\nTell the robot what to do, in your own words. For example:")
    for example in EXAMPLES:
        print(f"    {example}")
    print("Type  quit  to stop.")

    names = {}
    while True:
        if todo:
            instruction = todo.pop(0)
            print(f"\nyou> {instruction}")
        else:
            try:
                instruction = input("\nyou> ").strip()
            except EOFError:
                break
            if instruction.lower() in ("quit", "exit", "q"):
                break
            if not instruction:
                continue

        cell.audit.log("instruction", text=instruction)
        log("LLM", f"sending the picture ({len(oc.listing())} numbered "
                   f"things), the measurements and your words")
        try:
            steps, understood, new_names, attempts, usage = C.plan(
                instruction, cell, client, log=log)
        except C.CommandError as exc:
            print(f"  refused: {exc}")
            cell.audit.log("instruction_refused", text=instruction,
                           reason=str(exc))
            continue
        except Exception as exc:
            print(f"  the language model could not be reached: "
                  f"{type(exc).__name__}: {exc}")
            continue
        names.update(new_names)
        cell.audit.log("instruction_plan", steps=steps,
                       understood=understood, names=new_names,
                       attempts=attempts)
        if usage["input"]:
            log("LLM", f"reply received: {usage['input']} tokens in, "
                       f"{usage['output']} out"
                       + (", corrected once after the checker refused "
                          "its first plan" if attempts > 1 else ""))
        if new_names:
            log("LLM", "it identified: " + ", ".join(
                f"{k} = {v}" for k, v in sorted(
                    new_names.items(), key=lambda kv: int(kv[0])
                    if kv[0].isdigit() else 0)))
        print(f"  understood: {understood}")
        print("  plan:")
        for number, step in enumerate(steps, 1):
            print(f"    {number}. {describe(step, names)}")
        log("CHECK", "every step checked against the camera's "
                     "measurements: allowed")

        with robot_log():
            done, said, failure = C.execute(steps, cell, log=log)
        for text in said:
            print(f"  robot: {text}")
        if failure:
            print(f"  stopped: {failure}")
            cell.audit.log("instruction_failed", reason=failure)
        else:
            print(f"  done: {len(done)} step"
                  f"{'' if len(done) == 1 else 's'}")
        from audit.trail import verify_records
        log("AUDIT", f"{len(cell.audit.records)} records, chain "
                     f"{'intact' if verify_records(cell.audit.records)[0] else 'BROKEN'}"
                     f"; safety gate {cell.safety.authorized_count} "
                     f"authorised, {cell.safety.denied_count} denied")
        print("  on the table now: " + summary(oc, names))

    cell.soft_plc.stop()
    return 0


def describe(step, names):
    def called(ident):
        label = names.get(str(ident))
        return f"{ident} ({label})" if label else str(ident)

    name = step.get("step")
    if name == "pick":
        return f"pick up thing {called(step.get('thing'))}"
    if name == "place":
        target = step.get("on")
        if isinstance(target, str) and not target.strip().isdigit():
            return "put it on a clear place on the table"
        return f"put it in container {called(target)}"
    if name == "dispatch":
        return "conveyor: send the container out"
    if name == "look":
        return "look again with the camera"
    if name == "say":
        return "answer"
    return str(name)


def summary(oc, names):
    parts = []
    for t in oc.listing():
        label = names.get(str(t.id), "unnamed")
        if t.container:
            inside = [str(o.id) for o in oc.listing() if o.inside == t.id]
            parts.append(f"{t.id} {label} [holds "
                         f"{', '.join(inside) if inside else 'nothing'}]")
        elif t.held:
            parts.append(f"{t.id} {label} (in the gripper)")
        elif t.inside is None:
            parts.append(f"{t.id} {label}")
    return "; ".join(parts) if parts else "nothing"


if __name__ == "__main__":
    sys.exit(main())
