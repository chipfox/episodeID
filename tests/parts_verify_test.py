"""One file holding two or three parts.

A two-part story stored as one file has references that overlap, so no score can
say which part a window came from. Its geometry can: each part owns a substantial,
ordered span and the whole file is about as long as the parts together. That proof
may only come from the file itself — embedded subtitles or ASR. A sidecar subtitle
added by a third party can describe a different release (or nothing at all), so it
must never turn a multi-part file into an automatic OK.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_parts"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import planner  # noqa: E402

root = W / "media" / "Show"
(root / "Season 1").mkdir(parents=True, exist_ok=True)
EPS = [(1, 1, "One"), (1, 2, "Two"), (1, 3, "Three")]
series = {"path": str(root), "tvdb_id": 999998, "options": {},
          "episodes": [{"season": s, "episode": e, "title": t, "runtime": 45}
                       for s, e, t in EPS]}


def seg(start, end, code, score=.55, words=900):
    return {"start": start, "end": end, "code": code, "score": score,
            "confidence": "low", "words": words}


def scan_file(name, expected, segments, duration=4500, source="remote parakeet-test", **extra):
    rel = f"Season 1/{name}"
    (root / rel).touch()
    return {"rel": rel, "expected": expected, "status": "LOW_CONFIDENCE",
            "duration": duration, "text_source": source, "scope": "full",
            "segments": segments, **extra}


def row(f):
    plan = {it["source"]: it for it in planner.build_plan(series, {"files": [f]})}
    assert f["rel"] in plan, plan
    return plan[f["rel"]]


def row_runtime(f, runtime):
    s = {**series, "episodes": [{"season": 1, "episode": n, "title": t, "runtime": runtime}
                                for n, t in ((1, "One"), (2, "Two"), (3, "Three"))]}
    return {it["source"]: it for it in planner.build_plan(s, {"files": [f]})}[f["rel"]]


TWO = [seg(0, 2400, "S01E01"), seg(2400, 4200, "S01E02", score=.72, words=700)]

# A file named for both parts, read from its own audio, is proved by geometry.
both = scan_file("Show - S01E01E02 - Story.mkv", ["S01E01", "S01E02"], TWO)
r = row(both)
assert r["kind"] == "ok" and r["decided_by"] == "parts", r
assert "in order" in r["reason"], r

# A sidecar is not the file: the same geometry proves nothing.
sidecar = scan_file("Show - S01E01E02 - Story (sidecar).mkv", ["S01E01", "S01E02"], TWO,
                    source="sidecar Show.S01E01E02.eng.srt")
r = row(sidecar)
assert r["kind"] == "review", r
assert "sidecar" in (r.get("note") or "") + r["reason"], r

# Thin evidence in a part is not proof of that part.
thin = scan_file("Show - S01E01E02 - Thin.mkv", ["S01E01", "S01E02"],
                 [seg(0, 2400, "S01E01", words=40), seg(2400, 4200, "S01E02", words=40)])
assert row(thin)["kind"] == "review"

# The file is much shorter than its parts together: not both parts.
short = scan_file("Show - S01E01E02 - Short.mkv", ["S01E01", "S01E02"], TWO, duration=1800)
assert row(short)["kind"] == "review"

# A file named for one part but covering two is renamed to both, not reviewed.
one_name = scan_file("Show - S01E01 - Story.mkv", ["S01E01"], TWO)
r = row(one_name)
assert r["kind"] == "rename", r
assert "S01E01-E02" in r["targets"][0]["path"], r

# Parts that are not consecutive are not a multi-part file.
jump = scan_file("Show - S01E01E03 - Jump.mkv", ["S01E01", "S01E03"],
                 [seg(0, 2400, "S01E01"), seg(2400, 4200, "S01E03")])
assert row(jump)["kind"] == "review", row(jump)

# TVDB sometimes lists the combined runtime on each part (90/90 for one 87-min
# story). The file fits the max runtime, not the sum, and is still proved.
combined = scan_file("Show - S01E01E02 - Combined.mkv", ["S01E01", "S01E02"],
                     [seg(0, 2400, "S01E01"), seg(2400, 4500, "S01E02", score=.72, words=700)])
r = row_runtime(combined, 90)
assert r["kind"] == "ok" and r["decided_by"] == "parts", r

# No runtime metadata at all: a part must still be a real block of the file.
noruntime = scan_file("Show - S01E01E02 - NoRuntime.mkv", ["S01E01", "S01E02"],
                      [seg(0, 2400, "S01E01"), seg(2400, 4500, "S01E02", score=.72, words=700)])
r = row_runtime(noruntime, 0)
assert r["kind"] == "ok" and r["decided_by"] == "parts", r

# An AI annotation on a low segment cannot displace the structural proof.
ai = scan_file("Show - S01E01E02 - AI.mkv", ["S01E01", "S01E02"],
               [dict(TWO[0], llm={"code": "S01E01", "confidence": .9}),
                dict(TWO[1], llm={"code": "S01E01", "confidence": .9})])
r = row(ai)
assert r["kind"] == "ok" and r["decided_by"] == "parts", r

# A part read by the local faster-whisper backend is the file's own audio too.
local = scan_file("Show - S01E01E02 - Local.mkv", ["S01E01", "S01E02"], TWO,
                  source="whisper small")
r = row(local)
assert r["kind"] == "ok" and r["decided_by"] == "parts", r

print("parts_verify_test: OK — 10 cases")
