"""Job bookkeeping must not cost more than the work it describes.

A scan logs and reports progress once per file. Written straight through, that was
~900 SQLite commits per library scan and measured 38 s of a 99 s scan, so log lines
are buffered and progress is throttled. This pins the behaviour *and* the thing the
buffer must never break: a job that dies still has its error in the log.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_job_log_batch"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
(W / "config").mkdir(parents=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db, jobs  # noqa: E402

db.init()


def logged(job_id):
    job = db.get_job(job_id)
    assert job, f"job {job_id} vanished"
    return job


# --- the real writer puts every line in, in one call ------------------------------
first = db.add_job("scan", None, {})
db.append_job_log(first, [f"line {i}" for i in range(30)])
lines = logged(first)["log"].splitlines()
assert len(lines) == 30 and lines[0].endswith("line 0") and lines[-1].endswith("line 29"), lines
db.append_job_log(first, [])
assert len(logged(first)["log"].splitlines()) == 30, "an empty flush must not touch the database"

# --- the context buffers instead of writing per line ------------------------------
writes = []
real_append = db.append_job_log
jobs.db.append_job_log = lambda job_id, ls: writes.append(list(ls))

ctx = jobs.JobContext(first, None)
for i in range(10):
    ctx.log(f"sampled file {i}")
assert writes == [], "log lines must be buffered, not written one at a time"
ctx.flush()
assert len(writes) == 1 and len(writes[0]) == 10, writes
ctx.flush()
assert len(writes) == 1, "flushing with nothing pending must be free"

for i in range(jobs.JobContext.FLUSH_LINES):
    ctx.log(f"more {i}")
assert len(writes) == 2 and len(writes[1]) == jobs.JobContext.FLUSH_LINES, writes
ctx._pending.clear()

# A progress report that names a file used to write every time, which is exactly the
# per-file storm; the last frame must still always get through.
updates = []
real_update = db.update_job
jobs.db.update_job = lambda job_id, **f: updates.append(f)
ctx._last_progress = 0.0
for i in range(20):
    ctx.progress(i / 40, f"Reading file {i}")
assert len(updates) == 1, f"a burst of progress updates must collapse: {updates}"
ctx.progress(1.0, "done")
assert len(updates) == 2 and updates[1]["progress"] == 1.0, updates

# A status change carries the log with it, so the UI never shows a new phase while
# the lines that explain it are still in the buffer.
writes.clear()
ctx.log("why this phase matters")
ctx.status("Matching")
assert len(writes) == 1 and "why this phase matters" in writes[0][0], writes

# --- a job that fails still has its diagnostics -----------------------------------
jobs.db.append_job_log = real_append
jobs.db.update_job = real_update


def boom(ctx, params):
    ctx.log("about to fail on purpose")
    raise RuntimeError("kaboom")


jobs._handlers["boom"] = boom
doomed = db.add_job("boom", None, {})
jobs._run(doomed)
job = logged(doomed)
assert "about to fail on purpose" in job["log"] and "kaboom" in job["log"], job["log"]
assert job["status"] == "failed", job["status"]

print("PASS")
