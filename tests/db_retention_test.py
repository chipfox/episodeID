"""History retention: the newest scans and plans survive, old finished jobs go.

A scan is ~0.5 MB for a 200-file library and nothing used to trim them, so the DB grew
with every scan. Retention keeps the 20 newest of each per series (only the latest plan
is ever shown), deletes finished jobs older than 30 days (their log lines cascade), and
refuses to delete a scan still referenced by a surviving plan or a job that is queued.
"""
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_retention"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import db  # noqa: E402

db.init()
sid = db.add_series(1, "Show", "2020", str(W / "media" / "Show"), None)
(W / "media" / "Show").mkdir(parents=True)

scan_ids = [db.add_scan(sid, {"files": [], "counts": {}}) for _ in range(25)]
plan_ids = [db.add_plan(sid, scan_ids[i], []) for i in range(25)]
latest_scan, latest_plan = scan_ids[-1], plan_ids[-1]

old = db.add_job("scan", sid, {})
db.update_job(old, status="done", finished=db.now())
db.append_job_log(old, ["old line"])
# age it past the cutoff (created is ISO text, so rewrite it directly)
cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
db.conn().execute("UPDATE jobs SET created=? WHERE id=?", (cutoff, old))
queued_old = db.add_job("scan", sid, {})
db.conn().execute("UPDATE jobs SET created=? WHERE id=?", (cutoff, queued_old))
recent = db.add_job("scan", sid, {})
db.update_job(recent, status="done", finished=db.now())

pruned = db.prune_history(sid)
assert pruned == {"scans": 5, "plans": 5}, pruned
assert db.conn().execute("SELECT count(*) FROM scans WHERE series_id=?", (sid,)).fetchone()[0] == 20
assert db.conn().execute("SELECT count(*) FROM plans WHERE series_id=?", (sid,)).fetchone()[0] == 20
assert db.get_scan(latest_scan) and db.get_plan(latest_plan), "the newest must survive"
assert db.get_scan(scan_ids[0]) is None, "the oldest scan should be gone"

expired = db.prune_jobs()
assert expired == 1, expired  # the finished old job, not the queued one
assert db.get_job(old) is None
assert db.conn().execute("SELECT count(*) FROM job_log WHERE job_id=?", (old,)).fetchone()[0] == 0, \
    "log lines must cascade"
assert db.get_job(queued_old) is not None, "a queued job must never be pruned"
assert db.get_job(recent) is not None

db.vacuum()
assert db.conn().execute("PRAGMA integrity_check").fetchone()[0] == "ok"
print("PASS (retention: 20 scans/plans kept, old jobs pruned, queued kept, vacuum ok)")
