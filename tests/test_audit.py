"""Audit trail: append-only, hash-chained, tamper-evident."""

import json

from audit.trail import AuditTrail, verify_file, verify_records


def test_chain_verifies(tmp_path):
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(str(path), job_id=7)
    for i in range(5):
        trail.log("step", n=i)
    ok, n, _ = verify_file(str(path))
    assert ok and n == 5
    assert all(r["job_id"] == 7 for r in trail.records)


def test_edited_record_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    trail = AuditTrail(str(path))
    trail.log("verification", part="red", status="failed")
    trail.log("job_fault", code=30)
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["data"]["status"] = "success"            # rewrite history
    lines[0] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n")
    ok, n, _ = verify_file(str(path))
    assert not ok and n == 0


def test_deleted_record_is_detected():
    trail = AuditTrail()
    for i in range(4):
        trail.log("step", n=i)
    assert verify_records(trail.records)[0]
    assert not verify_records(trail.records[:1] + trail.records[2:])[0]


def test_appending_continues_an_existing_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditTrail(str(path)).log("a")
    AuditTrail(str(path)).log("b")
    ok, n, _ = verify_file(str(path))
    assert ok and n == 2


def test_refuses_to_append_to_a_tampered_file(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditTrail(str(path)).log("a")
    path.write_text(path.read_text().replace('"a"', '"z"'))
    try:
        AuditTrail(str(path))
        raised = False
    except ValueError:
        raised = True
    assert raised
