"""Duration is evidence and a qualification gate, not merely a displayed number.

The gate deliberately calibrates a series against files a prior scan already settled,
because TVDB runtimes sometimes include ads or combine multi-episode listings.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_duration"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import planner  # noqa: E402

root = W / "media" / "Duration Show"
root.mkdir(parents=True, exist_ok=True)
EPISODES = [{"season": 1, "episode": n, "title": f"Episode {n}", "runtime": 15} for n in range(1, 21)]
EPS = {f"S01E{n:02d}": e for n, e in enumerate(EPISODES, start=1)}


def file_for(code, status, duration, segments, **extra):
    rel = f"Duration Show - {code} - Episode.mkv"
    (root / rel).touch()
    return {"rel": rel, "expected": [code], "status": status, "duration": duration,
            "segments": segments, **extra}


# A settled cohort establishes that this series' files run at roughly 7/15 of TVDB.
calibration = [file_for(f"S01E{n:02d}", "OK", 420, []) for n in range(1, 6)]
correct = file_for("S01E06", "MISMATCH", 420, [{
    "start": 0, "end": 420, "code": "S01E06", "score": .60, "confidence": "high",
    "evidence": "dialogue"}])
impossible = file_for("S01E07", "MISMATCH", 1500, [{
    "start": 0, "end": 1500, "code": "S01E07", "score": .60, "confidence": "high",
    "evidence": "dialogue"}])
wrong_length = file_for("S01E08", "MISMATCH", 1500, [{
    "start": 0, "end": 1500, "code": "S01E09", "score": .60, "confidence": "high",
    "evidence": "dialogue"}])
files = calibration + [correct, impossible, wrong_length]
series = {"path": str(root), "tvdb_id": 999001, "options": {}, "episodes": EPISODES}
by = {it["source"]: it for it in planner.build_plan(series, {"files": files})}

factors = planner._duration_factors(files, EPS)
assert factors[(1, 1)] == 420 / (15 * 60), factors
assert by[correct["rel"]]["duration_check"]["state"] == "ok", by[correct["rel"]]
assert by[impossible["rel"]]["kind"] == "review", by[impossible["rel"]]
assert "Runtime does not qualify" in by[impossible["rel"]]["reason"], by[impossible["rel"]]
assert by[wrong_length["rel"]]["kind"] == "review", by[wrong_length["rel"]]
assert not by[wrong_length["rel"]].get("selected", False), by[wrong_length["rel"]]
for rel in (impossible["rel"], wrong_length["rel"]):
    method = next(m for m in by[rel]["methods"] if m["key"] == "duration")
    assert method["state"] == "fail", method

# Before there is a trustworthy cohort, the comparison is shown but cannot veto a plan.
lonely = file_for("S01E10", "MISMATCH", 2400, [{
    "start": 0, "end": 2400, "code": "S01E10", "score": .60, "confidence": "high",
    "evidence": "dialogue"}])
lonely_series = {"path": str(root), "tvdb_id": 999002, "options": {}, "episodes": EPISODES}
lonely_by = {it["source"]: it for it in planner.build_plan(lonely_series, {"files": [lonely]})}
assert lonely_by[lonely["rel"]]["kind"] == "ok", lonely_by[lonely["rel"]]
assert lonely_by[lonely["rel"]]["duration_check"]["calibrated"] is False
assert next(m for m in lonely_by[lonely["rel"]]["methods"] if m["key"] == "duration")["state"] == "warn"

# A multi-part listing can report the combined runtime on each part.
parts_series = {
    "path": str(root), "tvdb_id": 999003, "options": {},
    "episodes": [{"season": 2, "episode": n, "title": f"Part {n}", "runtime": 22} for n in (1, 2)],
}
parts_check = planner._duration_evidence({
    "duration": 1320, "codes": ["S02E01", "S02E02"], "decided_by": "parts"}, {},
    {"S02E01": {"season": 2, "episode": 1, "runtime": 22},
     "S02E02": {"season": 2, "episode": 2, "runtime": 22}})
assert parts_check["state"] == "info" and parts_check["expected_seconds"] == 1320, parts_check

print("duration match: calibrated cohort qualifies files; impossible runtimes become reviews")
