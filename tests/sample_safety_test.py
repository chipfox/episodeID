"""A sampled file may certify a clear match, but a mismatch is only a suggestion.

These cases pin the boundary: a high-confidence sample that agrees with the
filename is OK; a high-confidence sample on a misnamed file is an unticked
rename; weak or multi-episode sample evidence stops at review.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_sample_safety"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import planner  # noqa: E402

root = W / "media" / "Sample Show"
(root / "Season 01").mkdir(parents=True)
eps = [{"season": 1, "episode": n, "title": f"Episode {n}", "runtime": 44}
       for n in range(1, 5)]
series = {"path": str(root), "tvdb_id": 123456, "year": "2020",
          "options": {}, "episodes": eps}

FILES = {
    "wrong": "Season 01/Sample Show - S01E01 - Episode 1.mkv",
    "multi": "Season 01/Sample Show - S01E01E02 - Episode 1 + Episode 2.mkv",
    "full": "Season 01/Sample Show - S01E01 - Episode 1.mkv",
}
for rel in set(FILES.values()):
    (root / rel).touch()


def build(*files):
    return {item["source"]: item for item in
            planner.build_plan(series, {"files": list(files)})}


def sample_segment(code, confidence="provisional", **extra):
    return {"start": 0, "end": 600, "code": code, "score": 0.61,
            "confidence": confidence, "scope": "sample", "alternatives": [], **extra}


# 1. A wrong-looking sample on a misnamed file is review, never a pre-ticked rename.
by = build({"rel": FILES["wrong"], "expected": ["S01E01"], "status": "LOW_CONFIDENCE",
            "duration": 600, "sample_end": 600, "segments": [sample_segment("S01E02")]})
item = by[FILES["wrong"]]
print("sample wrong:", item["kind"], item["selected"], item["suggested"])
assert item["kind"] == "review", item
assert item["selected"] is False, item
assert item["suggested"] == ["S01E02"], item
assert item["sample"] is True, item

# 2. A high-confidence sample on a misnamed file is a suggested rename, not a
#    pre-ticked action. The sample can identify it, but the human still checks it.
by = build({"rel": FILES["wrong"], "expected": ["S01E01"], "status": "MISMATCH",
            "duration": 600, "sample_end": 600,
            "segments": [sample_segment("S01E02", confidence="high")]})
item = by[FILES["wrong"]]
print("sample high mismatch:", item["kind"], item["selected"], item.get("codes"))
assert item["kind"] == "rename" and item["selected"] is False, item
assert item["codes"] == ["S01E02"], item
assert item.get("sample") is True, item

# 3. A high-confidence sample that agrees with the filename is OK.
by = build({"rel": FILES["wrong"], "expected": ["S01E01"], "status": "OK",
            "duration": 600, "sample_end": 600,
            "segments": [sample_segment("S01E01", confidence="high")]})
item = by[FILES["wrong"]]
print("sample high match:   ", item["kind"], item["selected"])
assert item["kind"] == "ok" and item["selected"] is False, item

# 4. A multi-episode filename with only sampled evidence is review, not a split
#    or a rename that drops one of its episodes.
by = build({"rel": FILES["multi"], "expected": ["S01E01", "S01E02"],
            "status": "LOW_CONFIDENCE", "duration": 600, "sample_end": 600,
            "segments": [sample_segment("S01E01")]})
item = by[FILES["multi"]]
print("sample multi:        ", item["kind"], item["selected"], item["suggested"])
assert item["kind"] == "review" and item["selected"] is False, item
assert item["suggested"] == ["S01E01"], item

# 5. A sample must not be upgraded by the process-of-elimination pass.
elim = []
for n in range(1, 5):
    rel = f"Season 01/Sample Show - S01E{n:02d} - Episode {n}.mkv"
    (root / rel).touch()
    if n == 2:
        elim.append({"rel": rel, "expected": ["S01E02"], "status": "LOW_CONFIDENCE",
                     "duration": 600, "sample_end": 600,
                     "segments": [sample_segment("S01E02")]})
    else:
        elim.append({"rel": rel, "expected": [f"S01E{n:02d}"], "status": "OK",
                     "duration": 2640,
                     "segments": [{"start": 0, "end": 2640, "code": f"S01E{n:02d}",
                                   "score": 0.9, "confidence": "high", "alternatives": []}]})
by = build(*elim)
item = by[elim[1]["rel"]]
print("sample elimination:", item["kind"], item["selected"], item.get("sample"))
assert item["kind"] == "review" and item["selected"] is False, item
assert item.get("sample") is True, item

# 5. A human override still works: an explicit choice is not sample evidence.
by = build({"rel": FILES["wrong"], "expected": ["S01E01"], "status": "MANUAL",
            "duration": 600, "sample_end": 600,
            "segments": [{"start": 0, "end": 600, "code": "S01E02", "override": "S01E02",
                          "score": 1.0, "confidence": "manual", "scope": "sample"}]})
item = by[FILES["wrong"]]
print("manual override:", item["kind"], item["selected"], item.get("codes"))
assert item["kind"] == "rename" and item["selected"] is True, item
assert item["codes"] == ["S01E02"], item

# 5. Full-file high confidence still works as before: a misnamed file is
#    pre-ticked, because this is not sample evidence.
by = build({"rel": FILES["full"], "expected": ["S01E01"], "status": "MISMATCH",
            "duration": 2640, "segments": [{"start": 0, "end": 2640, "code": "S01E02",
                                             "score": 0.72, "confidence": "high",
                                             "alternatives": []}]})
item = by[FILES["full"]]
print("full-file high:", item["kind"], item["selected"], item.get("codes"))
assert item["kind"] == "rename" and item["selected"] is True, item
assert item["codes"] == ["S01E02"], item

# 6. The strength mapping makes the boundary explicit: sample high stays high,
#    while provisional/low sample evidence stays untrusted.
assert planner._segment_choice({"code": "S01E02", "confidence": "high",
                                "scope": "sample"}) == ("S01E02", "high", False)
assert planner._segment_choice({"code": "S01E02", "confidence": "provisional"}) == \
    ("S01E02", "provisional", False)
print("PASS")
