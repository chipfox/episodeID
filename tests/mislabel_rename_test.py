"""A mislabelled file is offered as a rename to the episode the dialogue points at,
in that episode's own season folder — and the user can override the episode by hand.

Four shapes: a file in the wrong season, two files holding each other's episode (a
swap), a weak match that must stay a review, and a two-parter whose parts play in the
wrong order (a split, first target = the part that plays first).
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_mislabel"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import db, main, pipeline, planner  # noqa: E402
from app.jobs import JobContext  # noqa: E402

root = W / "media" / "tv" / "Show"
for d in ("Season 01", "Season 02"):
    (root / d).mkdir(parents=True, exist_ok=True)

eps = [{"season": s, "episode": e, "title": f"Title {s}x{e:02d}", "aired": "2020-01-01"}
       for s, n in ((1, 10), (2, 5)) for e in range(1, n + 1)]
series = {"path": str(root), "tvdb_id": 1, "options": {}, "episodes": eps}


def seg(code, start=0.0, end=600.0, confidence="high", score=0.5, margin=0.3, words=200):
    return {"start": start, "end": end, "code": code, "score": score, "margin": margin,
            "confidence": confidence, "words": words}


wrong_season = "Season 02/Show - S02E01 - Wrong Name.mkv"      # dialogue says S01E03
swapped = "Season 01/Show - S01E03 - Title 1x03.mkv"           # dialogue says S02E01
weak = "Season 01/Show - S01E07 - Title 1x07.mkv"              # nothing convincing
two_parts = "Season 01/Show - S01E05-E06 - Two Parts.mkv"      # plays E06 then E05
files = [
    {"rel": wrong_season, "expected": ["S02E01"], "status": "MISMATCH", "duration": 2700,
     "text_source": "remote parakeet", "segments": [seg("S01E03")],
     "note": "Name does not match content"},
    {"rel": swapped, "expected": ["S01E03"], "status": "MISMATCH", "duration": 2700,
     "text_source": "remote parakeet", "segments": [seg("S02E01")],
     "note": "Name does not match content"},
    {"rel": weak, "expected": ["S01E07"], "status": "LOW_CONFIDENCE", "duration": 2700,
     "text_source": "remote parakeet", "segments": [seg("S01E08", confidence="low", score=0.12, margin=0.01)],
     "note": "close to S01E08"},
    {"rel": two_parts, "expected": ["S01E05", "S01E06"], "status": "MISMATCH", "duration": 5400,
     "text_source": "remote parakeet",
     "segments": [seg("S01E06", 0.0, 2700.0), seg("S01E05", 2700.0, 5400.0)],
     "note": "The parts play out of order"},
]
for rel in (wrong_season, swapped, weak, two_parts):
    (root / rel).write_bytes(b"x")

plan = {it["source"]: it for it in planner.build_plan(series, {"files": files})}

# 1. Wrong season folder: renamed to the episode's own season and number.
a = plan[wrong_season]
assert a["kind"] == "rename", a["kind"]
assert a["codes"] == ["S01E03"], a["codes"]
assert a["targets"][0]["codes"] == ["S01E03"]
assert a["targets"][0]["path"].startswith("Season 01/"), a["targets"][0]["path"]
assert "S01E03" in a["targets"][0]["path"], a["targets"][0]["path"]
assert a["selected"] is True, "dialogue-settled renames are pre-ticked"
assert a["expected"] == ["S02E01"], "the old name stays on the row for the log"

# 2. Two files holding each other's episode both point at the other number/season.
b = plan[swapped]
assert b["kind"] == "rename", b["kind"]
assert b["targets"][0]["path"].startswith("Season 02/"), b["targets"][0]["path"]
assert "S02E01" in b["targets"][0]["path"], b["targets"][0]["path"]

# 3. Weak evidence: no rename offered, the user is asked instead.
c = plan[weak]
assert c["kind"] == "review", c["kind"]
assert not c.get("targets"), c.get("targets")

# 4. Two-parter out of order: split, targets in playing order, each with its number.
d = plan[two_parts]
assert d["kind"] == "split", d["kind"]
targets = sorted(d["targets"], key=lambda t: t.get("start") or 0)
assert [t["codes"] for t in targets] == [["S01E06"], ["S01E05"]], targets
assert all(t["path"].startswith("Season 01/") for t in targets)
assert "S01E06" in targets[0]["path"] and "S01E05" in targets[1]["path"], targets

# 5. Applying the two cross-season renames must actually move each file into the
# other's season folder (this is the swap the plan promised).
db.init()
sid = db.add_series(1, "Show", "2020", str(root), None, options={})
db.update_series(sid, episodes=eps)
db.add_scan(sid, {"files": files, "counts": {}})
scan = db.latest_scan(sid)
assert scan is not None, "the scan just added must exist"
scan_id = scan["id"]
series_now = db.get_series(sid)
assert series_now is not None
items = planner.build_plan(series_now, {"files": files})
for it in items:
    it["selected"] = it["source"] in (wrong_season, swapped)
pid = db.add_plan(sid, scan_id, items)
job = db.add_job("apply", sid, {"plan_id": pid})
result = pipeline.job_apply(JobContext(job, sid), {"plan_id": pid})
assert result["applied"] == 2, result
after = sorted(str(p.relative_to(root)) for p in root.rglob("*.mkv"))
assert any(p.startswith("Season 01/") and "S01E03" in p for p in after), after
assert any(p.startswith("Season 02/") and "S02E01" in p for p in after), after
assert not any("Wrong Name" in p for p in after), after
assert (root / "Season 01" / "Show - S01E03 - Title 1x03.mkv").exists(), after

# 6. The user can set the episode by hand, and the rename follows their choice.
root2 = W / "media" / "tv" / "Show2"
(root2 / "Season 01").mkdir(parents=True, exist_ok=True)
manual_rel = "Season 01/Show2 - S02E01 - Manual.mkv"
(root2 / manual_rel).write_bytes(b"x")
sid2 = db.add_series(2, "Show2", "2020", str(root2), None, options={})
db.update_series(sid2, episodes=eps)
db.add_scan(sid2, {"files": [{"rel": manual_rel, "expected": ["S02E01"], "status": "MISMATCH",
                              "duration": 2700, "text_source": "remote parakeet",
                              "segments": [seg("S01E03")]}], "counts": {}})
client = TestClient(main.app)
body = client.post(f"/api/series/{sid2}/overrides",
                   json={"source": manual_rel, "codes": ["S01E05"]}).json()
row = next(it for it in body["items"] if it["source"] == manual_rel)
assert row["codes"] == ["S01E05"], row["codes"]
assert row["targets"][0]["path"].startswith("Season 01/"), row["targets"][0]["path"]
assert "S01E05" in row["targets"][0]["path"], row["targets"][0]["path"]
assert "S01E03" not in row["targets"][0]["path"], row["targets"][0]["path"]
assert row["kind"] == "rename" and row.get("ai") is False, row

print("PASS (mislabel → season+episode rename, swap, weak=review, split order, manual override)")
