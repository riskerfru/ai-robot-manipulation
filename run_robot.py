#!/usr/bin/env python3
# =============================================================
#   RUN ROBOT - type any instruction, the robot works it out
#
#   The language model is briefed on the cell and turns what you
#   type into steps. Code checks every step before the robot moves.
#
#   python run_robot.py --gui                 window, then type
#   python run_robot.py                       no window, faster
#   python run_robot.py --scene-file scenes/cosmetics.json --gui
#   python run_robot.py --seed 7              a different lay-out
# =============================================================

import argparse
import contextlib
import io
import sys

import commands as C
from cell import packing as P

EXAMPLES = [
    "fill the tray with two lemon and two mint",
    "put the yellow one in the reject bin",
    "pack a full tray, mostly berry",
    "change over to the travel size and fill it",
    "what can you see?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instructions", nargs="*",
                    help="run these and stop, instead of asking")
    ap.add_argument("--scene-file", dest="scene", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true",
                    help="no scatter, sensor error or damaged items")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not cfg_key():
        sys.exit("ANTHROPIC_API_KEY is not set. Put it in the .env file "
                 "in this folder.\nThis one needs the language model: it "
                 "is what reads your instructions.")
    import anthropic
    import config as cfg
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY,
                                 timeout=cfg.AI.get("timeout_s", 20.0),
                                 max_retries=cfg.AI.get("api_retries", 1))

    from packing_orders import quiet_scan
    from run_packing import build_packing_cell
    quiet = not args.verbose

    def hush():
        return contextlib.redirect_stdout(io.StringIO() if quiet
                                          else sys.__stdout__)

    with hush():
        cell = build_packing_cell(args, gui=args.gui)
    line = cell.robot.packing_line
    print("\n".join(P.describe_scene()))
    print(f"Language model: {cfg.AI['model']}")
    print("Conditions: " + ("clean" if args.clean
                            else f"real-line (seed {args.seed})"))

    todo = list(args.instructions)
    if not todo:
        print("\nThe cell is set up. Nothing has moved yet."
              + ("  Drag things around in the window if you like."
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

    print("\nLooking at the cell with the overhead camera...")
    with hush():
        quiet_scan(cell)

    print("\nTell the robot what to do, in your own words. For example:")
    for example in EXAMPLES:
        print(f"    {example}")
    print("Type  quit  to stop.")

    while True:
        if todo:
            instruction = todo.pop(0)
            print(f"\nyou> {instruction}")
        else:
            seen = ", ".join(f"{k} {v}" for k, v in
                             line.stock_levels().items() if v)
            held = cell.robot.grasped_object
            print(f"\n[camera sees {seen or 'nothing'}"
                  f"{', holding ' + held if held else ''}]")
            try:
                instruction = input("you> ").strip()
            except EOFError:
                break
            if instruction.lower() in ("quit", "exit", "q", ""):
                break

        cell.audit.log("instruction", text=instruction)
        try:
            with hush():
                steps, note, attempts = C.plan(instruction, cell, client)
        except C.CommandError as exc:
            print(f"  refused: {exc}")
            cell.audit.log("instruction_refused", text=instruction,
                           reason=str(exc))
            continue
        except Exception as exc:
            print(f"  the language model could not be reached: "
                  f"{type(exc).__name__}: {exc}")
            continue

        cell.audit.log("instruction_plan", steps=steps, note=note,
                       attempts=attempts)
        plural = "" if len(steps) == 1 else "s"
        print(f"  plan ({len(steps)} step{plural}"
              f"{', corrected once' if attempts > 1 else ''}): "
              + ", ".join(describe(s) for s in steps))
        if note:
            print(f"  note: {note}")

        with hush():
            done, said, failure = C.execute(steps, cell, say=lambda *a: None)
        for text in said:
            print(f"  robot: {text}")
        if failure:
            print(f"  stopped: {failure}")
        else:
            print(f"  done ({len(done)} step"
                  f"{'' if len(done) == 1 else 's'})")
        for check in line.check_tray():
            print(f"    nest {check['nest']}: {check['flavour']:<8} "
                  f"{'seated' if check['seated'] else 'NOT SEATED'}")

    from audit.trail import verify_records
    print(f"\nAudit trail: {len(cell.audit.records)} records, chain "
          f"{'intact' if verify_records(cell.audit.records)[0] else 'BROKEN'}")
    print(f"Safety gate: {cell.safety.authorized_count} authorised, "
          f"{cell.safety.denied_count} denied")
    cell.soft_plc.stop()
    return 0


def describe(step):
    name = step.get("step")
    if name == "insert":
        return f"seat {step.get('product')} in nest {step.get('nest')}"
    if name == "pick":
        return f"pick {step.get('product')}"
    if name == "place":
        return f"place in {step.get('where')}"
    if name == "change_format":
        return f"change to {step.get('format')}"
    if name == "report":
        return "answer"
    return name


def cfg_key():
    import config as cfg
    return bool(cfg.ANTHROPIC_API_KEY)


if __name__ == "__main__":
    sys.exit(main())
