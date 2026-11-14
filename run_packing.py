#!/usr/bin/env python3
# =============================================================
#   RUN PACKING - the packing cell from the terminal
#
#   python run_packing.py                 two default orders, language model
#   python run_packing.py --offline       same, with the built-in parser
#   python run_packing.py --gui           with the simulator window
#   python run_packing.py "two lemon, one mint and one berry" \
#                         "change over to travel size: two of each flavour"
# =============================================================

import argparse
import contextlib
import io
import sys

from packing_orders import make_planner, run_order

ORDER_1 = ("Pack a mixed case of standard cartons: two lemon, one mint "
           "and one berry.")
ORDER_2 = ("Change over to the travel size and pack two lemon, two mint "
           "and two berry.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("orders", nargs="*", default=[ORDER_1, ORDER_2])
    ap.add_argument("--offline", action="store_true",
                    help="plan with the built-in parser, no language model")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    planner, label = make_planner(args.offline)
    from cell.factory import build_sim_cell
    from utils.logger import NullLogger
    quiet = not args.verbose

    def hush():
        return contextlib.redirect_stdout(io.StringIO() if quiet
                                          else sys.__stdout__)

    with hush():
        cell = build_sim_cell(NullLogger(), gui=args.gui, scene="packing",
                              housing_offset=(0.002, -0.0015))
    print(f"Planner: {label}")
    ok = True
    for number, order in enumerate(args.orders):
        print(f"\nORDER: {order}")
        with hush():
            out = run_order(cell, order, planner, tray_number=number)
        if "refused" in out:
            print(f"  REFUSED: {out['refused']}")
            ok = False
            continue
        plan = out["plan"]
        print(f"  Plan ({plan.source}): {plan.format} tray, {plan.summary()}")
        for c in out["check"]:
            state = "seated" if c["seated"] else "NOT SEATED"
            print(f"  nest {c['nest']}: {c['flavour']:<6} {state}")
        ok = ok and out["result"]["success"] and all(
            c["seated"] for c in out["check"])
    from audit.trail import verify_records
    print(f"\nAudit trail: {len(cell.audit.records)} records, chain "
          f"{'intact' if verify_records(cell.audit.records)[0] else 'BROKEN'}")
    print(f"Safety gate: {cell.safety.authorized_count} commands authorised, "
          f"{cell.safety.denied_count} denied")
    cell.soft_plc.stop()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
