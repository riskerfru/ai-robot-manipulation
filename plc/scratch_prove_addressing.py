# =============================================================
#   THROWAWAY SCRIPT - Day 1 step 2
#
#   Proves the %QX / %QW <-> Modbus coil / holding-register
#   mapping before any ladder logic is written. Delete this file
#   once the mapping is confirmed; it is not part of the product.
#
#   Prerequisite: the OpenPLC Runtime is running, with a program
#   deployed whose Buffer Mapping assigns coils 0-31 to %QX0.0-
#   %QX3.7 and holding registers 0-31 to %QW0-%QW31 (all as %QX/
#   %QW, never %IX/%IW - those are Modbus-read-only).
#
#   Usage: python plc/scratch_prove_addressing.py [host] [port]
# =============================================================

import sys

from pymodbus.client import ModbusTcpClient

from plc.bus import ModbusBus
from plc.signals import SIGNALS


def raw_probe(host, port):
    print(f"\n--- raw pymodbus probe at {host}:{port} ---")
    client = ModbusTcpClient(host, port=port, timeout=1.0)
    if not client.connect():
        print("FAILED to connect - is the runtime started and listening?")
        sys.exit(1)

    checks = [
        ("coil",     0,  True),   # %QX0.0  - low end,  PLC-owned range
        ("coil",     16, True),   # %QX2.0  - low end,  robot-owned range
        ("coil",     31, True),   # %QX3.7  - top of the 32-coil map
        ("register", 0,  1234),   # %QW0    - low end,  PLC-owned range
        ("register", 16, 5678),   # %QW16   - low end,  robot-owned range
        ("register", 31, 65535),  # %QW31   - top of the 32-register map
    ]

    ok = True
    for kind, addr, value in checks:
        if kind == "coil":
            wr = client.write_coil(addr, value, device_id=1)
            rr = client.read_coils(addr, count=1, device_id=1)
            got = bool(rr.bits[0]) if not rr.isError() else None
        else:
            wr = client.write_register(addr, value, device_id=1)
            rr = client.read_holding_registers(addr, count=1, device_id=1)
            got = int(rr.registers[0]) if not rr.isError() else None

        passed = (not wr.isError()) and (not rr.isError()) and got == value
        ok &= passed
        print(f"  {kind:8s} addr {addr:2d}  wrote {value!r:>6}  "
              f"read back {got!r:>6}  {'OK' if passed else 'MISMATCH'}")

    client.close()
    if not ok:
        print("\nAddressing is NOT proven - stop and fix the Buffer "
              "Mapping before writing any rungs.")
        sys.exit(1)
    print("\nRaw addressing confirmed both directions.")


def named_probe(host, port):
    print(f"\n--- plc/bus.py + plc/signals.py probe at {host}:{port} ---")
    bus = ModbusBus(host, port=port)

    # One PLC-owned signal, one robot-owned signal - exercises the
    # low half and the high half of both the coil and register maps
    # through the exact code path cell_controller.py will use.
    samples = [
        ("ESTOP_OK",        True),
        ("MODE_AUTO",       True),
        ("JOB_ID",          7),
        ("READY",           True),
        ("ROBOT_HEARTBEAT", 42),
        ("PARTS_OK",        3),
    ]

    ok = True
    for name, value in samples:
        bus.write(name, value)
        got = bus.read(name)
        passed = got == value
        ok &= passed
        sig = next(s for s in SIGNALS if s.name == name)
        print(f"  {name:16s} ({sig.kind:8s} addr {sig.address:2d})  "
              f"wrote {value!r:>6}  read back {got!r:>6}  "
              f"{'OK' if passed else 'MISMATCH'}")

    bus.close()
    if not ok:
        print("\nName-level addressing does NOT match signals.py - "
              "fix the Buffer Mapping before writing rungs.")
        sys.exit(1)
    print("\nNamed-signal addressing confirmed - plc/bus.py matches "
          "the Buffer Mapping.")


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 502
    raw_probe(host, port)
    named_probe(host, port)
