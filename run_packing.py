#!/usr/bin/env python3
# =============================================================
#   RUN PACKING - the packing cell from the terminal
#
#   python run_packing.py --gui --interactive  a window, then type your
#                                              own orders one by one
#   python run_packing.py                      default scene and orders
#   python run_packing.py --interactive        type orders one by one
#   python run_packing.py --scene-file scenes/cosmetics.json --interactive
#   python run_packing.py --seed 7             a different run: cartons,
#                                              trays and damage all move
#   python run_packing.py --offline            built-in parser, no model
#   python run_packing.py --clean              no real-line conditions
#   python run_packing.py --gui                with the simulator window
#   python run_packing.py "two lemon, one mint and one berry" "..."
# =============================================================

import argparse
import contextlib
import io
import sys

from cell import packing as P
from packing_orders import make_planner, run_order

ORDER_1 = ("Pack a mixed case of standard cartons: two lemon, one mint "
           "and one berry.")
ORDER_2 = ("Change over to the travel size and pack two lemon, two mint "
           "and two berry.")
DEFAULT_ORDERS = [ORDER_1, ORDER_2]


def add_cell_arguments(ap):
    ap.add_argument("--scene-file", dest="scene", default=None,
                    help="scene file (JSON); default is the tea scene")
    ap.add_argument("--seed", type=int, default=0,
                    help="changes carton positions, tray offsets and "
                         "random damage")
    ap.add_argument("--clean", action="store_true",
                    help="no scatter, sensor error or damaged cartons")
    ap.add_argument("--offline", action="store_true",
                    help="plan with the built-in parser, no language model")


def build_packing_cell(args, gui=False):
    """Build the cell for the parsed arguments. Exits on a bad scene."""
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger
    try:
        P.load_scene(args.scene)
    except P.SceneError as exc:
        sys.exit(f"Scene refused: {exc}")
    return build_sim_cell(NullLogger(), gui=gui, scene="packing",
                          packing_scene=args.scene, seed=args.seed,
                          line_noise=None if args.clean else P.LINE_NOISE)


def report(out):
    if "refused" in out:
        print(f"  REFUSED: {out['refused']}")
        return False
    plan = out["plan"]
    if out.get("seen"):
        print("  Camera sees: " + ", ".join(
            f"{n} {c}" for n, c in out["seen"].items() if c))
    print(f"  Plan ({plan.source}): {plan.format} tray, {plan.summary()}")
    for r in out.get("rejected", []):
        print(f"  rejected at the pick: {r['flavour']} "
              f"({r['seen_mm']} mm seen, {r['width_mm']} mm in the grip, "
              f"nominal {r['nominal_mm']} mm)"
              + (", replaced from stock" if r["substitute"]
                 else ", no replacement in stock"))
    for c in out["check"]:
        print(f"  nest {c['nest']}: {c['flavour']:<8} "
              f"{'seated' if c['seated'] else 'NOT SEATED'}")
    missing = out["planned"] - len(out["check"])
    if missing:
        print(f"  {missing} nest(s) left empty")
    return (out["result"]["success"] and not missing
            and all(c["seated"] for c in out["check"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orders", nargs="*")
    add_cell_arguments(ap)
    ap.add_argument("--interactive", action="store_true",
                    help="type orders one after another")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    planner, label = make_planner(args.offline)
    quiet = not args.verbose

    def hush():
        return contextlib.redirect_stdout(io.StringIO() if quiet
                                          else sys.__stdout__)

    from packing_orders import quiet_scan
    with hush():
        cell = build_packing_cell(args, gui=args.gui)
    line = cell.robot.packing_line
    print("\n".join(P.describe_scene()))
    print(f"Planner: {label}")
    print("Conditions: " + ("clean" if args.clean else
                            f"real-line (seed {args.seed})"))

    interactive = args.interactive or (not args.orders and args.scene)
    if interactive:
        # Nothing moves and nothing is looked at until you say so.
        # With the simulator window open this is the moment to drag
        # the cartons and the tray wherever you want them.
        print("\nThe cell is set up. Move things around if you like,"
              + (" then" if args.gui else ""))
        print("type  start  and press Enter when you are ready.")
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

    orders = list(args.orders)
    if not orders and not interactive:
        orders = DEFAULT_ORDERS
    ok = True
    number = 0
    while True:
        if orders:
            order = orders.pop(0)
            print(f"\nORDER: {order}")
        elif interactive:
            stock = "; ".join(
                f"{fmt}: " + ", ".join(f"{n} {k}" for k, n in
                                       line.stock_levels(fmt).items()
                                       if n)
                for fmt in P.FORMATS)
            print(f"\nThe camera sees: {stock}")
            print("Type an order, or 'quit' to stop.")
            try:
                order = input("order> ").strip()
            except EOFError:
                break
            if order.lower() in ("", "quit", "exit", "q"):
                break
        else:
            break
        with hush():
            out = run_order(cell, order, planner, tray_number=number)
        number += 1
        ok = report(out) and ok

    from audit.trail import verify_records
    print(f"\nAudit trail: {len(cell.audit.records)} records, chain "
          f"{'intact' if verify_records(cell.audit.records)[0] else 'BROKEN'}")
    print(f"Safety gate: {cell.safety.authorized_count} commands authorised, "
          f"{cell.safety.denied_count} denied")
    cell.soft_plc.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
