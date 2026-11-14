"""Repeated title-card reads are one episode, not a split."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_repeated_titlecard"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import pipeline, planner  # noqa: E402

root = W / "media" / "Show"
(root / "Season 6").mkdir(parents=True, exist_ok=True)
series = {"path": str(root), "tvdb_id": 999997, "options": {},
          "episodes": [{"season": 6, "episode": 6, "title": "Blood Pressure", "runtime": 15}]}
rel = "Season 6/Show - S06E06 - Blood Pressure - SDTV MPEG2 Opus 2.0.mkv"
(root / rel).touch()
cards = [
    {"code": "S06E06", "title": "Blood Pressure", "text": "BLOOD PRESSURE",
     "score": 1.0, "time": 1.0, "partial": False, "frames": 4},
    {"code": "S06E06", "title": "Blood Pressure", "text": "BLDOD PRESSURE",
     "score": 0.923, "time": 267.6, "partial": False, "frames": 3},
]

# No dialogue: applying both title-card reads must leave one segment, not two.
f = {"rel": rel, "expected": ["S06E06"], "status": "MISMATCH", "duration": 639.2,
     "text_source": "remote parakeet-test", "scope": "full", "segments": []}
pipeline._apply_title_cards(f, cards)
assert [sg["code"] for sg in f["segments"]] == ["S06E06"], f["segments"]
item = {it["source"]: it for it in planner.build_plan(series, {"files": [f]})}[rel]
assert item["kind"] == "ok", item
assert item["kind"] != "split" and not item.get("targets"), item

# An older scan may already have both reads as segments; the planner must still
# collapse them and keep the file as one episode.
f2 = {**f, "segments": [
    {"start": 0.0, "end": 177.6, "code": "S06E06", "score": 1.0, "margin": 0,
     "confidence": "high", "evidence": "title", "title_card": cards[0]},
    {"start": 262.6, "end": 639.2, "code": "S06E06", "score": 0.923, "margin": 0,
     "confidence": "high", "evidence": "title", "title_card": cards[1]},
]}
item2 = {it["source"]: it for it in planner.build_plan(series, {"files": [f2]})}[rel]
assert item2["kind"] == "ok", item2
assert not item2.get("targets"), item2

# A mid-file scene read ("BURGER WORLD" on a restaurant sign) must not turn a
# single-named 11-minute episode into two episodes just because another TVDB
# title matches the text. A real two-episode file is longer than 1.6x runtime.
series13 = {**series, "episodes": [
    {"season": 6, "episode": 13, "title": "No Service", "runtime": 15},
    {"season": 2, "episode": 3, "title": "Burger World", "runtime": 15},
]}
rel13 = "Season 6/Show - S06E13 - No Service - WEBDL-1080p AV1 Opus 2.0.mkv"
(root / rel13).touch()
f13 = {"rel": rel13, "expected": ["S06E13"], "status": "MISMATCH", "duration": 689.3,
       "text_source": "embedded #2", "scope": "full", "segments": []}
cards13 = [
    {"code": "S06E13", "title": "No Service", "text": "SERVICE", "score": 0.875,
     "time": 14.0, "partial": False, "frames": 3},
    {"code": "S02E03", "title": "Burger World", "text": "BURGER WORLD", "score": 1.0,
     "time": 324.6, "partial": False, "frames": 2},
]
pipeline._apply_title_cards(f13, cards13, {"S06E13": 15, "S02E03": 15})
assert [sg["code"] for sg in f13["segments"]] == ["S06E13"], f13["segments"]
item13 = {it["source"]: it for it in planner.build_plan(series13, {"files": [f13]})}[rel13]
assert item13["kind"] == "ok", item13
assert not item13.get("targets"), item13

print("repeated title-card reads: one episode, no split")
print("mid-file scene text: not a second episode")

