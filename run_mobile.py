#!/usr/bin/env python3
# =============================================================
#   RUN MOBILE - a robot arm on a moving base serves four stations
#
#   STORAGE -> MACHINE -> INSPECT -> DISPATCH, along a lane with a
#   pedestrian walkway and a pallet in the way. Type an instruction;
#   the language model decides the next actions; a fixed rule layer
#   (the guardian) accepts or refuses every one before anything
#   moves, and tells the model why when it refuses.
#
#   python run_mobile.py --gui                    window, then type
#   python run_mobile.py --gui "process all the parts in storage"
#   python run_mobile.py --gui --offline          no language model:
#                                                 a fixed strategy
#   python run_mobile.py --gui --offline --mistakes
#                                                 the fixed strategy
#                                                 also tries illegal
#                                                 actions, to show
#                                                 the guardian refusing
#
#   While typing:  !lock MACHINE   !unlock MACHINE   (operator lockout)
#                  !status         quit
#
#   Simulation only. The guardian is not a safety function.
# =============================================================

import argparse
import contextlib
import io
import sys
import time

EXAMPLES = [
    "process all the parts in storage",
    "machine one part and inspect it",
    "ship everything that has passed",
    "what is the robot carrying?",
]

COLOURS = {"OPERATOR": "\033[1;97m", "LLM": "\033[96m", "PLAN": "\033[96m",
           "GUARD": "\033[93m", "NAV": "\033[94m", "CAMERA": "\033[95m",
           "MACHINE": "\033[33m", "INSPECT": "\033[35m", "ROBOT": "\033[92m",
           "DISPATCH": "\033[92m", "FAULT": "\033[91m", "ERROR": "\033[91m",
           "LOCKOUT": "\033[91m"}
RESET = "\033[0m"


def make_log(start, colour=True):
    out = sys.stdout            # the arm's own prints are hidden; this is not

    def log(tag, text):
        t = time.time() - start
        c = COLOURS.get(tag, "") if colour else ""
        if tag == "GUARD" and "REJECT" in text and colour:
            c = "\033[1;91m"
        print(f"{t:6.1f}s {c}[{tag:8}]{RESET if c else ''} {text}",
              file=out, flush=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instructions", nargs="*")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--parts", type=int, default=4)
    ap.add_argument("--offline", action="store_true",
                    help="no language model; a fixed strategy")
    ap.add_argument("--mistakes", action="store_true",
                    help="with --offline, try illegal actions too")
    ap.add_argument("--dead-reckoning", action="store_true",
                    help="aim from the map, not the mast camera")
    ap.add_argument("--no-colour", action="store_true")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    import config as cfg
    from mobile.agent import LLMPlanner, ScriptedPlanner, run
    from mobile.cell import build_mobile_cell
    from mobile.guardian import Guardian

    if args.offline:
        planner = ScriptedPlanner(mistakes=args.mistakes)
    else:
        if not cfg.ANTHROPIC_API_KEY:
            sys.exit("ANTHROPIC_API_KEY is not set (.env). Or run with "
                     "--offline for the fixed strategy.")
        import anthropic
        planner = LLMPlanner(anthropic.Anthropic(
            api_key=cfg.ANTHROPIC_API_KEY,
            timeout=max(90.0, cfg.AI.get("timeout_s", 90.0))))

    start = time.time()
    log = make_log(start, colour=not args.no_colour)
    print("Building the shop floor...", flush=True)
    with contextlib.redirect_stdout(io.StringIO()):
        w = build_mobile_cell(gui=args.gui, seed=args.seed, parts=args.parts,
                              use_vision=not args.dead_reckoning, log=log)
        w.cell.scan()
    cell = w.cell
    log("CAMERA", "mast camera at STORAGE: " + ", ".join(
        f"part #{t.id} ({t.long * 1000:.0f} mm)"
        for t in cell.at_station() if not t.container))
    print(f"\nPlanner: {planner.name}. "
          f"Aiming: {'dead reckoning' if args.dead_reckoning else 'mast camera'}."
          f"\nTry: " + " | ".join(EXAMPLES[:3]) + "\n", flush=True)

    def one(text):
        text = text.strip()
        if not text:
            return
        if text.startswith("!"):
            parts = text[1:].split()
            cmd = parts[0].lower() if parts else ""
            st = parts[1].upper() if len(parts) > 1 else ""
            if cmd in ("lock", "unlock") and st:
                cell.lockout(st, cmd == "lock")
                log("LOCKOUT", f"operator {'locked out' if cmd == 'lock' else 'released'} {st}")
            elif cmd == "status":
                from mobile.agent import state_summary
                import json
                print(json.dumps(state_summary(cell), indent=1))
            return
        with contextlib.redirect_stdout(io.StringIO()):
            out = run(text, cell, planner, log, guardian=Guardian(cell))
        rej = out["rejected"]
        print(f"\n  -> {'finished' if out['finished'] else 'stopped'}: "
              f"{out['executed']} actions done, {out['checked']} checked, "
              f"{len(rej)} refused by the guardian"
              + (f" ({', '.join(r[1] for r in rej)})" if rej else "")
              + f"; {out['dispatched']} part(s) dispatched; "
              f"{out['driven_m']:.1f} m driven; {out['seconds']:.0f} s\n",
              flush=True)
        if isinstance(planner, LLMPlanner):
            u = planner.usage
            print(f"     language model: {u['calls']} calls, "
                  f"{u['input']} in / {u['output']} out tokens\n")

    for text in args.instructions:
        one(text)
    if args.instructions:
        return
    while True:
        try:
            text = input("instruction> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip().lower() in ("quit", "exit", "q"):
            break
        one(text)


if __name__ == "__main__":
    main()
