"""A sidecar may suggest an action, but never pre-tick one.

Sidecars are third-party files: one can be dropped next to a video that it does not
describe, so a match against its text is not proof of the file's contents. The plan
keeps the suggestion, clears the tick, and says why. Text read from the file itself
(embedded, local whisper, remote ASR) stays pre-ticked as before.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_sidecar_tick"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.planner import build_plan  # noqa: E402

root = W / "media" / "Show"
(root / "Season 1").mkdir(parents=True, exist_ok=True)
series = {"path": str(root), "tvdb_id": 999995, "options": {},
          "episodes": [{"season": 1, "episode": 1, "title": "One", "runtime": 45},
                       {"season": 1, "episode": 2, "title": "Two", "runtime": 45}]}


def plan(name, source):
    rel = f"Season 1/{name}"
    (root / rel).touch()
    f = {"rel": rel, "expected": ["S01E02"], "status": "MISMATCH", "duration": 2700,
         "text_source": source,
         "segments": [{"code": "S01E01", "start": 0, "end": 2400, "score": .8,
                       "words": 600, "confidence": "high"}]}
    return {it["source"]: it for it in build_plan(series, {"files": [f]})}[rel]


side = plan("Show - S01E02 - Wrong.mkv", "sidecar Show.S01E02.eng.srt")
assert side["kind"] == "rename", side
assert side["selected"] is False, side
assert "sidecar" in side["reason"], side

own = plan("Show - S01E02 - Wrong (own audio).mkv", "remote parakeet")
assert own["kind"] == "rename" and own["selected"] is True, own

print("sidecar_selection_test: OK — 2 cases")
