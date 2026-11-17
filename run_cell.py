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
#   python run_cell.py --plc 127.0.0.1    against the real ladder
#                                         program (OpenPLC, Modbus TCP)
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
from plc.plc_link import LivePlcLink, SoftPlcLink, refusal_reason
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


START_WAIT_S = 2.0       # how long the PLC gets to raise JOB_START


def run_job(cell, link, job_id, recipe):
    """
    Request a job and poll until the robot has run it. Returns the
    controller state (DONE / FAULT), or None if the PLC never raised
    JOB_START. A refused request is withdrawn, so it cannot start
    later on its own.
    """
    link.request_job(job_id, recipe)
    deadline = time.monotonic() + START_WAIT_S
    while True:
        state = poll(cell)               # runs the job once JOB_START is high
        if state is not CellState.READY:
            return state
        if time.monotonic() > deadline:
            link.withdraw()
            return None
        time.sleep(0.01)


def wait_ready(cell, timeout_s=2.0):
    """Poll until the controller is back to READY (PLC dropped JOB_START)."""
    deadline = time.monotonic() + timeout_s
    state = poll(cell)
    while state is not CellState.READY and time.monotonic() < deadline:
        time.sleep(0.01)
        state = poll(cell)
    return state


def recover(cell, link):
    """The operator's fault recovery, as the PLC would see it."""
    say("  PLC: FAULT_RESET while the cause is still present")
    link.reset_fault()
    poll(cell)
    show(cell, "reset attempt 1")
    link.release_reset()
    poll(cell)

    say("  Operator clears the cause, then FAULT_RESET")
    link.close_guard()
    link.release_estop()
    link.resume_heartbeat()
    time.sleep(0.1)
    poll(cell)
    link.reset_fault()
    poll(cell)
    show(cell, "reset attempt 2")
    link.release_reset()
    poll(cell)


def install_hazard(cell, link, scenario):
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
                say("  [PLANT] guard door opened")
                link.open_guard()
            elif scenario == "estop":
                say("  [PLANT] emergency stop pressed")
                link.press_estop()
            elif scenario == "plc":
                link.stall_heartbeat()
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
    ap.add_argument("--plc", metavar="HOST", default=None,
                    help="run against a real PLC over Modbus TCP (the "
                         "OpenPLC ladder program) instead of the soft PLC")
    ap.add_argument("--plc-port", type=int, default=502)
    args = ap.parse_args()
    if args.plc and args.scenario == "plc":
        ap.error("--scenario plc stalls the soft PLC's heartbeat; a real "
                 "PLC's heartbeat cannot be stalled from here")
    global QUIET
    QUIET = quiet = not args.verbose

    from utils.logger import NullLogger, RobotLogger
    logger = RobotLogger(cfg.LOGGING) if args.verbose else NullLogger()

    if os.path.exists(args.audit):
        os.remove(args.audit)
    with contextlib.redirect_stdout(io.StringIO() if quiet
                                    else sys.__stdout__):
        plc = dict(plc_host=args.plc, plc_port=args.plc_port)
        if args.scene == "assembly":
            cell = build_sim_cell(logger, gui=args.gui, audit_path=args.audit,
                                  scene="assembly",
                                  housing_offset=(0.002, -0.0015), **plc)
        else:
            cell = build_sim_cell(logger, gui=args.gui, audit_path=args.audit,
                                  **plc)
    if args.plc:
        link = LivePlcLink(cell.bus)
        # Between polls the controller is idle; keep the robot heartbeat
        # going so the ladder's watchdog does not drop MOTION_PERMIT.
        cell.controller.start_heartbeat()
    else:
        link = SoftPlcLink(cell.soft_plc, cell.bus)
    link.start()
    recipe = 3 if args.scene == "assembly" else 1
    jobs = 1 if args.scene == "assembly" else args.jobs
    install_hazard(cell, link, args.scenario)

    plc_name = (f"OpenPLC at {args.plc}:{args.plc_port}" if args.plc
                else "soft PLC")
    say("=" * 100)
    say(f"  CELL RUN   scenario={args.scenario}   plc={plc_name}   "
        f"audit={args.audit}")
    say("=" * 100)
    poll(cell)
    show(cell, "cell started")

    job_id = 1000
    for n in range(jobs):
        job_id += 1
        say(f"\n  PLC: job request recipe={recipe}")
        state = run_job(cell, link, job_id, recipe)
        if state is None:
            say(f"  PLC: JOB_START withheld for recipe {recipe}: "
                f"{refusal_reason(cell.bus, recipe)}")
            continue
        show(cell, f"job {cell.bus.read('JOB_ID_ECHO')} finished")

        if state is CellState.FAULT:
            recover(cell, link)
        else:
            link.acknowledge_done()
            wait_ready(cell)
            show(cell, "PLC acknowledged")

        with contextlib.redirect_stdout(io.StringIO()):
            if args.scene != "assembly":
                refill_infeed(cell.robot)
        poll(cell)

    if args.plc:
        cell.controller.stop_heartbeat()
    link.stop()
    ok, count, _ = verify_file(args.audit)
    say(f"\n  Audit trail: {count} records, "
        f"{'chain intact' if ok else 'CHAIN BROKEN'}")
    say(f"  Safety gate: {cell.safety.authorized_count} commands authorised, "
        f"{cell.safety.denied_count} denied")
    cell.robot.disconnect()


if __name__ == "__main__":
    main()
