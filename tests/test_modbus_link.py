"""Modbus TCP transport against a local soft-PLC server (loopback)."""

import asyncio
import socket
import threading
import time

import pytest

pymodbus = pytest.importorskip("pymodbus")

from plc.bus import ModbusBus                          # noqa: E402
from plc.interlocks import PLCInterlocks               # noqa: E402
from safety.supervisor import MotionCommand, SafetySupervisor  # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    """A soft PLC register image served over Modbus TCP on localhost."""
    from pymodbus.server import ModbusTcpServer
    from pymodbus.simulator import DataType, SimData, SimDevice

    port = _free_port()
    device = SimDevice(id=1, simdata=(
        [SimData(0, count=64, values=False, datatype=DataType.BITS)],
        [SimData(0, count=64, values=False, datatype=DataType.BITS)],
        [SimData(0, count=64, values=0, datatype=DataType.REGISTERS)],
        [SimData(0, count=64, values=0, datatype=DataType.REGISTERS)],
    ))
    loop = asyncio.new_event_loop()
    holder = {}

    async def serve():
        holder["srv"] = ModbusTcpServer(device, address=("127.0.0.1", port))
        await holder["srv"].serve_forever()

    def run():
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(serve())
        except Exception as exc:               # surfaced by the wait below
            holder["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline and "error" not in holder:
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    if "error" in holder:
        pytest.skip(f"local Modbus server unavailable: {holder['error']}")
    yield port
    try:
        asyncio.run_coroutine_threadsafe(
            holder["srv"].shutdown(), loop).result(5)
    except Exception:
        pass
    t.join(2)


def test_signals_round_trip_over_modbus_tcp(server):
    plc_side = ModbusBus("127.0.0.1", server)
    robot_side = ModbusBus("127.0.0.1", server)
    try:
        plc_side.write("JOB_ID", 4711)
        plc_side.write("RECIPE", 1)
        plc_side.write("JOB_START", True)
        assert robot_side.read("JOB_ID") == 4711
        assert robot_side.read("JOB_START") is True
        robot_side.write("BUSY", True)
        robot_side.write("FAULT_CODE", 30)
        assert plc_side.read("BUSY") is True
        assert plc_side.read("FAULT_CODE") == 30
        assert plc_side.read("DONE") is False
    finally:
        plc_side.close()
        robot_side.close()


def test_interlocks_over_modbus_gate_motion(server):
    plc_side = ModbusBus("127.0.0.1", server)
    robot_side = ModbusBus("127.0.0.1", server)
    try:
        for name in ("ESTOP_OK", "GUARD_CLOSED", "MODE_AUTO",
                     "MOTION_PERMIT"):
            plc_side.write(name, True)
        interlocks = PLCInterlocks(robot_side)
        safety = SafetySupervisor(interlock_source=interlocks)
        plc_side.write("PLC_HEARTBEAT", 1); interlocks.read()
        plc_side.write("PLC_HEARTBEAT", 2)
        assert safety.authorize(MotionCommand("home")).allowed
        plc_side.write("GUARD_CLOSED", False)
        plc_side.write("PLC_HEARTBEAT", 3)
        assert not safety.authorize(MotionCommand("home")).allowed
    finally:
        plc_side.close()
        robot_side.close()


def test_unreachable_plc_is_an_error_not_a_silent_pass():
    with pytest.raises(Exception):
        ModbusBus("127.0.0.1", _free_port(), timeout=0.2)
