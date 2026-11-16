#!/usr/bin/env python3
# =============================================================
#   RUN CELL - the cell as a PLC would run it
#
#   Starts the simulated cell with the safety supervisor, the
#   audit trail and a soft PLC, then runs jobs through the
#   READY / JOB_START / DONE handshake.
#
#   python run_cell.py                    two normal jobs
#   python run_cell.py --scenario guard   guard door opens mid-job
#   python run_cell.py --scenario estop   emergency stop mid-job
#   python run_cell.py --scenario plc     PLC heartbeat lost mid-job
#   python run_cell.py --gui              same, in the 3D viewer
#
#   No API key is needed: jobs are PLC-selected recipes. Free-text
#   orders through the language model use main.py.
# =============================================================

import argparse
import contextlib
import io
import os
import sys
import time

import config as cfg
from audit.trail import verify_file
from cell.factory import build_sim_cell
from cell.scene import refill_infeed
from plc.cell_controller import CellState
from plc.signals import FaultCode

FAULT_NAMES = {v: k for k, v in vars(FaultCode).items() if k.isupper()}
STATUS = ["READY", "BUSY", "DONE", "FAULT", "FAULT_CODE", "PARTS_OK",
          "PARTS_FAILED", "SLOT_MASK", "DEGRADED"]


def say(text=""):
    print(text, file=sys.__stdout__, flush=True)


def show(cell, label):
    v = {k: cell.bus.read(k) for k in STATUS}
    code = v["FAULT_CODE"]
    say(f"  {label:<34} state={cell.controller.state.value:<5} "
        f"safety={cell.safety.state.value:<15} "
        f"parts_ok={v['PARTS_OK']} failed={v['PARTS_FAILED']} "
        f"slots={v['SLOT_MASK']:03b} "
        f"fault={code} {FAULT_NAMES.get(code, '')}")


QUIET = True


def poll(cell):
    with contextlib.redirect_stdout(io.StringIO() if QUIET
                                    else sys.__stdout__):
        return cell.controller.poll()


def run_job(cell, job_id, recipe):
    cell.soft_plc.request_job(job_id, recipe)
    return poll(cell)


def install_hazard(cell, scenario):
    """After the first part is placed, trigger the scenario's hazard."""
    if scenario == "none":
        return
    backend = cell.robot
    original = backend.place_object
    fired = {"done": False}

    def place(x, y, stack_on=None):
        out = original(x, y, stack_on=stack_on)
        if not fired["done"]:
            fired["done"] = True
            if scenario == "guard":
                cell.soft_plc.open_guard()
            elif scenario == "estop":
                cell.soft_plc.press_estop()
            elif scenario == "plc":
                cell.soft_plc.stall_heartbeat()
                time.sleep(cell.interlocks.heartbeat_timeout_s + 0.2)
        return out

    backend.place_object = place


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="none",
                    choices=["none", "guard", "estop", "plc"])
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--scene", default="dispatch",
                    choices=["dispatch", "assembly"],
                    help="assembly: housing with pockets, recipe 3, "
                         "force-guided insertion (one job)")
    ap.add_argument("--verbose", action="store_true",
                    help="show the robot's own motion log")
    ap.add_argument("--audit", default="audit_logs/cell.jsonl")
    args = ap.parse_args()
    global QUIET
    QUIET = quiet = not args.verbose

    from utils.logger import NullLogger, RobotLogger
    logger = RobotLogger(cfg.LOGGING) if args.verbose else NullLogger()

    if os.path.exists(args.audit):
        os.remove(args.audit)
    with contextlib.redirect_stdout(io.StringIO() if quiet
                                    else sys.__stdout__):
        if args.scene == "assembly":
            cell = build_sim_cell(logger, gui=args.gui, audit_path=args.audit,
                                  scene="assembly",
                                  housing_offset=(0.002, -0.0015))
        else:
            cell = build_sim_cell(logger, gui=args.gui, audit_path=args.audit)
    recipe = 3 if args.scene == "assembly" else 1
    jobs = 1 if args.scene == "assembly" else args.jobs
    install_hazard(cell, args.scenario)

    say("=" * 100)
    say(f"  CELL RUN   scenario={args.scenario}   audit={args.audit}")
    say("=" * 100)
    poll(cell)
    show(cell, "cell started")

    job_id = 1000
    for n in range(jobs):
        job_id += 1
        say(f"\n  PLC: JOB_START job={job_id} recipe={recipe}")
        state = run_job(cell, job_id, recipe)
        show(cell, f"job {job_id} finished")

        if state is CellState.FAULT:
            say("  PLC: FAULT_RESET while the cause is still present")
            cell.soft_plc.reset_fault()
            poll(cell)
            show(cell, "reset attempt 1")
            cell.soft_plc.release_reset()
            poll(cell)

            say("  Operator clears the cause, then FAULT_RESET")
            cell.soft_plc.close_guard()
            cell.soft_plc.release_estop()
            cell.soft_plc.resume_heartbeat()
            time.sleep(0.1)
            poll(cell)
            cell.soft_plc.reset_fault()
            poll(cell)
            show(cell, "reset attempt 2")
            cell.soft_plc.release_reset()
        else:
            cell.soft_plc.acknowledge_done()
            poll(cell)
            show(cell, "PLC acknowledged")

        with contextlib.redirect_stdout(io.StringIO()):
            if args.scene != "assembly":
                refill_infeed(cell.robot)
        poll(cell)

    cell.soft_plc.stop()
    ok, count, _ = verify_file(args.audit)
    say(f"\n  Audit trail: {count} records, "
        f"{'chain intact' if ok else 'CHAIN BROKEN'}")
    say(f"  Safety gate: {cell.safety.authorized_count} commands authorised, "
        f"{cell.safety.denied_count} denied")
    cell.robot.disconnect()


if __name__ == "__main__":
    main()
