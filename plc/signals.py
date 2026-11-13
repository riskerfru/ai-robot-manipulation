# =============================================================
#   PLC SIGNAL MAP  (interface control document, in code)
#
#   The cell PLC owns the cell. The robot cell controller (this
#   software) never starts a job on its own: it reports READY,
#   waits for the PLC to raise JOB_START, and reports DONE or
#   FAULT. Natural-language orders are never sent over the
#   fieldbus; the PLC selects a pre-approved recipe by number.
#
#   Modbus TCP mapping (zero-based addresses):
#     PLC -> robot : coils 0..15,  holding registers 0..15
#     robot -> PLC : coils 16..31, holding registers 16..31
#   The same names map onto OPC UA nodes or PROFINET I/O in a
#   plant that uses those instead; only the transport changes.
# =============================================================

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    name: str
    kind: str        # "coil" (bool) or "register" (uint16)
    address: int
    writer: str      # "plc" or "robot"
    description: str


SIGNALS = [
    # ---- PLC -> robot ----------------------------------------
    Signal("ESTOP_OK",        "coil", 0, "plc",
           "Emergency stop circuit healthy (mirror of the safety relay)"),
    Signal("GUARD_CLOSED",    "coil", 1, "plc",
           "All guard doors closed and locked"),
    Signal("MODE_AUTO",       "coil", 2, "plc",
           "Cell mode selector in AUTO"),
    Signal("MOTION_PERMIT",   "coil", 3, "plc",
           "PLC permits robot motion (conveyor stopped, infeed ready)"),
    Signal("JOB_START",       "coil", 4, "plc",
           "Level: start the job in JOB_ID / RECIPE; hold until DONE or FAULT"),
    Signal("FAULT_RESET",     "coil", 5, "plc",
           "Operator acknowledged the fault (rising edge)"),
    Signal("JOB_ID",          "register", 0, "plc",
           "Job number for traceability"),
    Signal("RECIPE",          "register", 1, "plc",
           "Recipe number (see plc/recipes.py)"),
    Signal("PLC_HEARTBEAT",   "register", 2, "plc",
           "Counter incremented by the PLC every scan group"),

    # ---- robot -> PLC ----------------------------------------
    Signal("READY",           "coil", 16, "robot",
           "Idle, healthy, able to accept a job"),
    Signal("BUSY",            "coil", 17, "robot",
           "Job in progress"),
    Signal("DONE",            "coil", 18, "robot",
           "Job finished, all parts verified in tolerance"),
    Signal("FAULT",           "coil", 19, "robot",
           "Job stopped; see FAULT_CODE. Needs FAULT_RESET"),
    Signal("DEGRADED",        "coil", 20, "robot",
           "Language model unavailable; running pre-approved recipes only"),
    Signal("ROBOT_READY",     "coil", 21, "robot",
           "Robot controller powered, no protective stop"),
    Signal("FAULT_CODE",      "register", 16, "robot",
           "0 = none; see FaultCode"),
    Signal("PARTS_OK",        "register", 17, "robot",
           "Parts placed and verified in the current/last job"),
    Signal("PARTS_FAILED",    "register", 18, "robot",
           "Parts that could not be placed in tolerance"),
    Signal("SLOT_MASK",       "register", 19, "robot",
           "Bit n-1 set = conveyor slot n occupied"),
    Signal("ROBOT_HEARTBEAT", "register", 20, "robot",
           "Counter incremented by the robot controller"),
    Signal("JOB_ID_ECHO",     "register", 21, "robot",
           "JOB_ID the robot is working on / last finished"),
]

BY_NAME = {s.name: s for s in SIGNALS}


class FaultCode:
    NONE              = 0
    SAFETY_STOP       = 10   # protective or emergency stop during a job
    UNKNOWN_RECIPE    = 20
    PART_NOT_PLACED   = 30   # at least one part out of tolerance / lost
    PLC_HEARTBEAT     = 40   # PLC stopped updating its heartbeat
    INTERNAL          = 90


def signal_table_markdown():
    rows = ["| Signal | Direction | Type | Address | Meaning |",
            "|---|---|---|---|---|"]
    for s in SIGNALS:
        direction = "PLC → robot" if s.writer == "plc" else "robot → PLC"
        rows.append(f"| `{s.name}` | {direction} | {s.kind} | "
                    f"{s.address} | {s.description} |")
    return "\n".join(rows)
