"""Every episode reports how well its file matched: the score the scan gave the
segment that points at it, which file that was, and whether it settled the row.

The latest plan is the verdict (promoted evidence, title cards, overrides), so its
segments win; a series with no plan falls back to its last scan.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_episode_scores"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"),
                  WHISPER_ENABLED="false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import db, main  # noqa: E402

db.init()
for name in ("Planned", "Scanned", "Empty"):
    (W / "media" / name / "Season 1").mkdir(parents=True)

eps = [{"season": 1, "episode": n, "title": f"Episode {n}", "aired": "2020-01-01"} for n in (1, 2, 3, 4)]
planned = db.add_series(1, "Planned", "2020", str(W / "media" / "Planned"), None, options={})
db.update_series(planned, episodes=eps)
scanned = db.add_series(2, "Scanned", "2020", str(W / "media" / "Scanned"), None, options={})
db.update_series(scanned, episodes=eps)
empty = db.add_series(3, "Empty", "2020", str(W / "media" / "Empty"), None, options={})
db.update_series(empty, episodes=eps)

two_part = "Season 1/Planned - S01E01E02 - Two Parter.mkv"
wrong = "Season 1/Planned - S01E03 - Not This One.mkv"
scan_id = db.add_scan(planned, {"files": [], "counts": {}})
db.add_plan(planned, scan_id, [
    {"kind": "ok", "status": "OK", "source": two_part, "scope": "sample",
     "segments": [
         {"code": "S01E01", "score": 0.4, "margin": 0.31, "confidence": "high", "scope": "sample",
          "start": 0.0, "end": 600.0},
         {"code": "S01E02", "score": 0.35, "margin": 0.26, "confidence": "high", "scope": "sample",
          "start": 600.0, "end": 2700.0},
     ]},
    {"kind": "review", "status": "LOW_CONFIDENCE", "source": wrong, "scope": "sample",
     "segments": [{"code": "S01E03", "score": 0.12, "margin": 0.01, "confidence": "low",
                   "scope": "sample", "start": 0.0, "end": 600.0}]},
])

db.add_scan(scanned, {"files": [{
    "rel": "Season 1/Scanned - S01E01 - Pilot.mkv", "status": "OK", "text_source": "sidecar x.srt",
    "scope": "full", "segments": [{"code": "S01E01", "score": 0.22, "margin": 0.1, "confidence": "high"}]}],
    "counts": {"OK": 1}})

c = TestClient(main.app)


def episode_map(sid):
    body = c.get(f"/api/series/{sid}").json()
    return {e["code"]: e for e in body["episodes"]}


# The plan's segments decide, and each episode gets its own score.
m = episode_map(planned)
assert m["S01E01"]["match"]["score"] == 0.4, m["S01E01"]["match"]
assert m["S01E01"]["match"]["confidence"] == "high"
assert m["S01E01"]["match"]["margin"] == 0.31
assert m["S01E01"]["match"]["source"] == two_part
assert m["S01E02"]["match"]["score"] == 0.35, m["S01E02"]["match"]
assert m["S01E02"]["match"]["end"] == 2700.0
assert m["S01E03"]["match"]["score"] == 0.12 and m["S01E03"]["match"]["kind"] == "review"
assert m["S01E03"]["match"]["status"] == "LOW_CONFIDENCE"
assert m["S01E04"]["match"] is None, "an episode with no file has no score"

# No plan: the scan's own segments are used.
s = episode_map(scanned)
assert s["S01E01"]["match"]["score"] == 0.22 and s["S01E01"]["match"]["status"] == "OK"
assert s["S01E01"]["match"]["source"] == "Season 1/Scanned - S01E01 - Pilot.mkv"
assert s["S01E02"]["match"] is None

# Neither plan nor scan: no scores at all.
assert all(e["match"] is None for e in episode_map(empty).values())

# The transcript dialog's picker carries the same scores.
ev = c.get(f"/api/series/{planned}/evidence", params={"code": "S01E01"}).json()
by = {e["code"]: e for e in ev["episodes"]}
assert by["S01E01"]["match"]["score"] == 0.4 and by["S01E02"]["match"]["score"] == 0.35
assert by["S01E04"]["match"] is None
print("PASS (per-episode match scores: plan, scan fallback, none)")
