"""Combined text needs independently labeled spans and whole-video coverage."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_combined_spans"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.matcher import Matcher, classify  # noqa: E402
from app.pipeline import _guard_combined_match  # noqa: E402
from app.planner import build_plan  # noqa: E402


def words(marker, n):
    return [f"{marker}{chr(97 + i % 26)}{chr(97 + i // 26)}" for i in range(n)]


def cues(tokens, start=0, end=600):
    chunks = [tokens[i:i + 6] for i in range(0, len(tokens), 6)]
    step = (end - start) / len(chunks)
    return [(start + i * step, start + (i + .9) * step, " ".join(w)) for i, w in enumerate(chunks)]


early, late, broad = "S09E03", "S09E04", "S09E05"
a, b = words("aa", 510), words("bb", 210)  # unequal dialogue density; no midpoint split
shared = " ".join(a + b)
other = " ".join(words("cc", 600))
config = get_settings()
no_anchors = Matcher({early: shared, late: shared, broad: other}, config)
for track, duration in ((cues(a), 600), (cues(a) + cues(b, 600, 1200), 1200)):
    result = no_anchors.match_file(track, duration)
    assert result["evidence"]["combined"]["state"] == "unresolved", result
    assert all(s["confidence"] == "low" for s in result["segments"]), result

anchors = {early: " ".join(a), late: " ".join(b)}
matcher = Matcher({broad: shared, "S09E06": other}, config, anchors=anchors)
whole = cues(a) + cues(b, 600, 1200)
result = matcher.match_file(whole, 1200)
assert result["evidence"]["combined"] == {"state": "verified", "reference": broad, "codes": [early, late]}, result
assert abs(result["segments"][0]["end"] - 600) < 45, result
assert classify([early, late], result["segments"], "remote test", result["evidence"])[0] == "OK"
reverse = matcher.match_file(cues(b) + cues(a, 600, 1200), 1200)
assert reverse["evidence"]["combined"]["codes"] == [late, early], reverse
single = matcher.match_file(cues(b), 600)
assert single["evidence"]["combined"]["codes"] == [late], single
incomplete = matcher.match_file(cues(a), 1200)
assert incomplete["evidence"]["combined"]["state"] == "partial", incomplete
assert not any(s["confidence"] == "high" for s in incomplete["segments"]), incomplete
assert incomplete["segments"][0]["code"] == early and incomplete["segments"][0]["end"] <= 600
nonmatching = matcher.match_file(cues(a) + cues(a, 600, 1200), 1200)
_guard_combined_match(nonmatching, [early, late], {"source": "embedded #1", "scope": "full"})
assert nonmatching["evidence"]["combined"]["state"] == "partial", nonmatching

for dialogue in ({"source": "sidecar demo.srt"}, {"source": "embedded #1", "scope": "sample"}):
    untrusted = matcher.match_file(whole, 1200)
    _guard_combined_match(untrusted, [early, late], dialogue)
    assert untrusted["evidence"]["combined"]["state"] == "partial"
    assert all(s["confidence"] == "low" for s in untrusted["segments"])

one_code = Matcher({broad: shared, "S09E06": other}, config).match_file(whole, 1200)
_guard_combined_match(one_code, [early, late], {"source": "embedded #1"})
assert one_code["evidence"]["combined"]["state"] == "unresolved", one_code

root = W / "media" / "Show"
root.mkdir(parents=True, exist_ok=True)
series = {"path": str(root), "tvdb_id": 999404, "options": {},
          "episodes": [{"season": 9, "episode": n, "title": str(n), "runtime": 10} for n in range(1, 10)]}

def plan_row(result, expected, duration, scope="full", baselines=()):
    file = {"rel": "Show.mkv", "expected": expected, "status": "OK", "duration": duration,
            "scope": scope, "text_source": "embedded #1", **result}
    return next(r for r in build_plan(series, {"files": [file, *baselines]}) if r["source"] == "Show.mkv")

assert plan_row(result, [early, late], 1200)["kind"] == "ok"
assert plan_row(single, [late], 1200)["kind"] == "review"
assert plan_row(result, [early, late], 1200, scope="sample")["kind"] == "review"
assert "unresolved" in plan_row(one_code, [early, late], 1200)["reason"]

confirmed = [{"rel": f"Show - S09E{n:02d}.mkv", "expected": [f"S09E{n:02d}"],
              "duration": 600, "status": "OK", "text_source": "embedded #1",
              "segments": [{"code": f"S09E{n:02d}", "confidence": "high", "score": 0.8}]}
             for n in (1, 2, 6, 7, 8)]
proposed = plan_row(result, [late, broad], 1200, baselines=confirmed)
assert proposed["kind"] == "review" and not proposed["selected"]
assert proposed["proposal"]["codes"] == [early, late], proposed
print("PASS (independent anchors, unequal spans, reversed order, partial/untrusted media, planner gates)")
