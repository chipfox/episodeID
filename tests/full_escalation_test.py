"""Automatic full re-read when a sample cannot settle a file.

Driven through the pipeline pass that runs in a scan, not only the helper behind
it: the whole file is read once, matched again, and the deferral is kept when the
full transcript still does not settle it.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_full_escalation"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pipeline  # noqa: E402

EPS = {"S01E01": {"title": "Pilot", "overview": "The crew meets.", "season": 1,
                  "episode": 1, "runtime": 45},
       "S01E02": {"title": "Follow-Up", "overview": "The crew follows up.", "season": 1,
                  "episode": 2, "runtime": 45}}


class Ctx:
    def __init__(self):
        self.lines = []

    def log(self, line):
        self.lines.append(str(line))

    def progress(self, *a, **k):
        pass

    def check_cancel(self):
        pass


series = {"id": 1, "path": str(W / "media" / "Show"), "name": "Show"}


def entry(rel, **over):
    f = {"rel": rel, "size": 10, "duration": 600.0, "text_source": "audio",
         "expected": ["S01E01"], "scope": "sample", "sample_start": 0.0,
         "sample_end": 600.0, "sample_text": "weak lines", "segments": [], "whole": [],
         "status": "LOW_CONFIDENCE", "note": "weak sample evidence"}
    f.update(over)
    return f


class FullMatcher:
    def match_file(self, cues, duration):
        return {"segments": [{"start": 0.0, "end": duration, "code": "S01E01", "score": 0.9,
                              "margin": 0.6, "confidence": "high", "alternatives": []}],
                "whole": []}


class WeakMatcher:
    def match_file(self, cues, duration):
        return {"segments": [{"start": 0.0, "end": duration, "code": "S01E02", "score": 0.3,
                              "margin": 0.05, "confidence": "low", "alternatives": []}],
                "whole": []}


pipeline.get_dialogue = lambda v, ctx=None: {
    "cues": [[0.0, 120.0, "the crew meets at the crater"]], "duration": 2400.0,
    "source": "audio"}
pipeline.cached_dialogue = lambda v: None

# --- the pass re-reads the whole file, and skips what a full read cannot settle ----

weak = entry("Season 01/weak.mkv")
conflicted = entry("Season 01/conflicted.mkv", status="NO_MATCH",
                   reference_conflict=["S01E02"])
other_lang = entry("Season 01/other-lang.mkv", status="NO_MATCH", language="de")
videos = {f["rel"]: Path(series["path"]) / f["rel"]
          for f in (weak, conflicted, other_lang)}

ctx = Ctx()
re_read = pipeline._full_escalation_pass(series, [weak, conflicted, other_lang],
                                         FullMatcher(), EPS, {}, ctx, None, videos)
assert re_read == [weak], re_read
assert weak["scope"] == "full" and weak["status"] == "OK", weak
assert weak["full_reread"] and weak["note"].startswith("Full re-read after a weak sample")
assert conflicted["scope"] == "sample" and other_lang["language"] == "de"
assert any("Full re-read: 1 file(s)" in ln for ln in ctx.lines)
assert any("full re-read" in ln and "S01E01" in ln for ln in ctx.lines)

# --- a full read that still cannot settle the file keeps the deferral -------------

weak_again = entry("Season 01/weak.mkv")
ctx2 = Ctx()
re_read = pipeline._full_escalation_pass(series, [weak_again], WeakMatcher(), EPS, {}, ctx2,
                                         None, {"Season 01/weak.mkv": videos["Season 01/weak.mkv"]})
assert re_read == [weak_again]
assert weak_again["status"] == "LOW_CONFIDENCE", weak_again
assert weak_again["full_reread"] is True
# No model annotation exists any more: the scan's evidence and the AI fallback are
# the only things that speak about a file.
assert "ai_code" not in weak_again and "ai_confidence" not in weak_again, weak_again

print("ok full_escalation")
