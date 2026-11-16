# =============================================================
#   PLC INTERLOCK SOURCE
#   Feeds the SafetySupervisor from the PLC signal image. A
#   failed read, or a PLC heartbeat that has stopped changing,
#   yields "all interlocks false" and therefore denies motion.
# =============================================================

import time

from safety.supervisor import Interlocks


class PLCInterlocks:

    def __init__(self, bus, heartbeat_timeout_s=1.0, clock=None,
                 robot_ready=lambda: True):
        self.bus = bus
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self._clock = clock or time.monotonic
        self._robot_ready = robot_ready
        self._last_beat = None
        self._last_change = None       # None until a change is observed
        self.plc_alive = False

    def read(self):
        now = self._clock()
        try:
            v = self.bus.read_many(
                ["ESTOP_OK", "GUARD_CLOSED", "MODE_AUTO",
                 "MOTION_PERMIT", "PLC_HEARTBEAT"])
        except Exception:
            self.plc_alive = False
            return Interlocks()                    # fail closed

        # The PLC counts as alive only once its heartbeat has been
        # seen to change, and only while it keeps changing.
        beat = v["PLC_HEARTBEAT"]
        if self._last_beat is not None and beat != self._last_beat:
            self._last_change = now
        self._last_beat = beat
        self.plc_alive = (
            self._last_change is not None
            and (now - self._last_change) <= self.heartbeat_timeout_s)
        if not self.plc_alive:
            return Interlocks()                    # PLC silent: deny

        return Interlocks(
            estop_ok=bool(v["ESTOP_OK"]),
            guard_closed=bool(v["GUARD_CLOSED"]),
            mode_auto=bool(v["MODE_AUTO"]),
            plc_permit=bool(v["MOTION_PERMIT"]),
            robot_ready=bool(self._robot_ready()),
            timestamp=now,
        )

    def wait_until_alive(self, timeout_s=2.0, poll_s=0.01):
        """Block until the PLC heartbeat is seen moving. Start-up only."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.read()
            if self.plc_alive:
                return True
            time.sleep(poll_s)
        return False
