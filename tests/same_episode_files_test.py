"""One episode carried by two files: a split release, or a duplicate?

Two encodes of the same episode share their dialogue; the two halves of an episode
split across files do not. Overlap decides. When the text cannot be compared the
row is a review — never an automatic move into the duplicates folder, because
moving a real half is data loss. Sidecars are third-party text and cannot decide.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_same_episode"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.planner import build_plan  # noqa: E402

root = W / "media" / "Show"
(root / "Season 1").mkdir(parents=True, exist_ok=True)
series = {"path": str(root), "tvdb_id": 999996, "options": {},
          "episodes": [{"season": 1, "episode": 1, "title": "One", "runtime": 45}]}
HALF_A = "harbour boat captain storm anchor rope lighthouse " * 30
HALF_B = "courtroom witness judge evidence verdict gavel jury " * 30


def scan_file(name, text, source="remote parakeet-test", score=.8):
    rel = f"Season 1/{name}"
    (root / rel).touch()
    return {"rel": rel, "expected": ["S01E01"], "status": "OK", "duration": 2500,
            "text_source": source, "scope": "full", "sample_text": text,
            "segments": [{"code": "S01E01", "start": 0, "end": 2400, "score": score,
                          "words": 600, "confidence": "high"}]}


def rows(*files):
    return {it["source"]: it for it in build_plan(series, {"files": list(files)})}


# The two halves of one episode: different dialogue, both correct where they are.
r = rows(scan_file("Show - S01E01 - One (1).mkv", HALF_A, score=.80),
         scan_file("Show - S01E01 - One (2).mkv", HALF_B, score=.79))
assert {it["kind"] for it in r.values()} == {"ok"}, r
assert any("split across files" in it["reason"] for it in r.values()), r
assert not any(it["kind"] == "aside" for it in r.values()), r

# Two encodes of the same episode: the weaker one is a real duplicate.
r = rows(scan_file("Show - S01E01 - One.mkv", HALF_A, score=.80),
         scan_file("Show - S01E01 - One copy.mkv", HALF_A, score=.79))
assert {it["kind"] for it in r.values()} == {"ok", "aside"}, r
copy = next(it for it in r.values() if it["kind"] == "aside")
assert copy["selected"] is True and "Duplicate of S01E01" in copy["reason"], copy

# Text missing: cannot tell a split from a copy, so do not move anything.
r = rows(scan_file("Show - S01E01 - One.mkv", HALF_A, score=.80),
         scan_file("Show - S01E01 - Two.mkv", "", score=.79))
assert {it["kind"] for it in r.values()} == {"ok", "review"}, r
assert any("could not be compared" in it["reason"] for it in r.values()), r

# A sidecar is third-party text: it cannot prove the two files are copies.
r = rows(scan_file("Show - S01E01 - One.mkv", HALF_A, score=.80),
         scan_file("Show - S01E01 - Two.mkv", HALF_B, source="sidecar Show.S01E01.eng.srt",
                   score=.79))
assert {it["kind"] for it in r.values()} == {"ok", "review"}, r

print("same_episode_files_test: OK — 4 cases")
