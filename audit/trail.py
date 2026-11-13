# =============================================================
#   AUDIT TRAIL
#
#   Append-only JSON Lines record of everything the cell did and
#   why: the order, every language-model decision and tool call,
#   every safety decision, every motion result, every
#   verification and every PLC handshake transition.
#
#   Each record carries the SHA-256 of the previous record, so an
#   edited, removed or reordered line is detectable with
#   verify_file(). This gives traceability for quality and
#   incident review; it is not a substitute for a validated
#   records system where regulation requires one.
# =============================================================

import hashlib
import json
import os
import threading
from datetime import datetime, timezone

GENESIS = "0" * 64


def _canonical(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      default=str)


def _digest(record):
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


class AuditTrail:

    def __init__(self, path=None, job_id=None):
        """path=None keeps records in memory only (tests, dry runs)."""
        self.path = path
        self.job_id = job_id
        self.records = []
        self._seq = 0
        self._prev = GENESIS
        self._lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)),
                        exist_ok=True)
            if os.path.exists(path) and os.path.getsize(path) > 0:
                ok, count, last = verify_file(path)
                if not ok:
                    raise ValueError(
                        f"audit file {path} failed integrity check at "
                        f"record {count}; refusing to append")
                self._seq, self._prev = count, last

    def log(self, event, **payload):
        with self._lock:
            record = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"),
                "job_id": self.job_id,
                "event": event,
                "data": payload,
                "prev": self._prev,
            }
            record["hash"] = _digest(record)
            self._seq += 1
            self._prev = record["hash"]
            self.records.append(record)
            if self.path:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(_canonical(record) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            return record

    def events(self, name=None):
        return [r for r in self.records if name is None or r["event"] == name]


class NullAudit:
    """Drop-in when no trail is configured."""
    job_id = None

    def log(self, event, **payload):
        return None


def verify_records(records):
    """Returns (ok, records_checked, last_hash)."""
    prev = GENESIS
    for i, rec in enumerate(records):
        body = {k: v for k, v in rec.items() if k != "hash"}
        if rec.get("seq") != i or rec.get("prev") != prev:
            return False, i, prev
        if _digest(body) != rec.get("hash"):
            return False, i, prev
        prev = rec["hash"]
    return True, len(records), prev


def verify_file(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                return False, len(records), None
    return verify_records(records)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m audit.trail <audit_file.jsonl>")
        sys.exit(2)
    ok, n, _ = verify_file(sys.argv[1])
    print(f"{'INTACT' if ok else 'TAMPERED'}: "
          f"{n} record(s) {'verified' if ok else 'valid before the break'}")
    sys.exit(0 if ok else 1)
