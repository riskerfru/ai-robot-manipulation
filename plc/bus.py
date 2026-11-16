# =============================================================
#   SIGNAL BUS - transport-independent access to PLC signals
#   InMemoryBus : simulation and tests
#   ModbusBus   : Modbus TCP to a real or soft PLC
# =============================================================

import threading

from plc.signals import BY_NAME


class SignalBus:
    def read(self, name):
        raise NotImplementedError

    def write(self, name, value):
        raise NotImplementedError

    def read_many(self, names):
        return {n: self.read(n) for n in names}


class InMemoryBus(SignalBus):
    """Shared signal image; both the soft PLC and the robot use one."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values = {name: (False if s.kind == "coil" else 0)
                        for name, s in BY_NAME.items()}

    def read(self, name):
        with self._lock:
            return self._values[name]

    def write(self, name, value):
        sig = BY_NAME[name]
        with self._lock:
            self._values[name] = (bool(value) if sig.kind == "coil"
                                  else int(value) & 0xFFFF)


class ModbusBus(SignalBus):
    """
    Modbus TCP client side. Requires `pymodbus` (see
    requirements-industrial.txt). A failed read raises, and the
    interlock reader turns that into "deny motion".
    """

    def __init__(self, host, port=502, timeout=0.5, device_id=1):
        from pymodbus.client import ModbusTcpClient
        self._client = ModbusTcpClient(host, port=port, timeout=timeout)
        self._device_id = device_id
        self._lock = threading.Lock()
        if not self._client.connect():
            raise ConnectionError(f"cannot reach PLC at {host}:{port}")

    def read(self, name):
        sig = BY_NAME[name]
        with self._lock:
            if sig.kind == "coil":
                rr = self._client.read_coils(
                    sig.address, count=1, device_id=self._device_id)
                if rr.isError():
                    raise IOError(f"Modbus read {name} failed: {rr}")
                return bool(rr.bits[0])
            rr = self._client.read_holding_registers(
                sig.address, count=1, device_id=self._device_id)
            if rr.isError():
                raise IOError(f"Modbus read {name} failed: {rr}")
            return int(rr.registers[0])

    def write(self, name, value):
        sig = BY_NAME[name]
        with self._lock:
            if sig.kind == "coil":
                rr = self._client.write_coil(
                    sig.address, bool(value), device_id=self._device_id)
            else:
                rr = self._client.write_register(
                    sig.address, int(value) & 0xFFFF,
                    device_id=self._device_id)
            if rr.isError():
                raise IOError(f"Modbus write {name} failed: {rr}")

    def close(self):
        self._client.close()
