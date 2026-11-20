"""Half-position reasoning for double-episode references (app/matcher.py).

One OpenSubtitles text filed under both codes of an adjacent pair is broadcast
order: the earlier episode fills the first half of the text. The shared text
cannot say which episode was read — but WHERE a file's dialogue matched can.
Pins:
  * a file inside one half gets that half's episode as one high segment, and
    the conflict veto lifts because the position IS the proof;
  * a file holding both halves (a real double episode) is never pinned to one;
  * distant duplicate codes (one text under S05E24 AND S05E36) are a numbering
    duplicate, not a double — halves must not settle those;
  * adjacent references that merely resemble each other (not one text filed
    twice) must not settle by halves either.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_twin_halves"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    EVIDENCE_GATE="true",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.matcher import Matcher, classify, evidence_accepts  # noqa: E402

S = get_settings()


def words(marker: str, n: int) -> list[str]:
    return [f"{marker}{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}" for i in range(n)]


def cues(text_words: list[str]) -> list[tuple[float, float, str]]:
    out, t = [], 0.0
    for i in range(0, len(text_words), 6):
        out.append((t, t + 5.5, " ".join(text_words[i:i + 6])))
        t += 6.0
    return out


H1, H2 = words("aa", 600), words("bb", 600)
SHARED = " ".join(H1 + H2)
UNRELATED = words("cc", 600)
m = Matcher({"S09E03": SHARED, "S09E04": SHARED, "S09E06": " ".join(UNRELATED)}, S)

# First half only → the earlier twin's episode, high, veto lifted.
r = m.match_file(cues(H1), 600.0)
assert len(r["segments"]) == 1 and r["segments"][0]["code"] == "S09E03", r["segments"]
assert r["segments"][0]["confidence"] == "high", r["segments"]
assert "first half" in r["segments"][0]["why"], r["segments"]
assert r["evidence"]["conflicted"] is False and r["evidence"]["top1"] == "S09E03", r["evidence"]
assert r["evidence"]["half"]["halves"][0] > 0.9, r["evidence"]
assert classify(["S09E03"], r["segments"], "remote test", r["evidence"])[0] == "OK"

# Second half only → the later twin's episode.
r2 = m.match_file(cues(H2), 600.0)
assert r2["segments"][0]["code"] == "S09E04" and "second half" in r2["segments"][0]["why"], r2["segments"]
assert r2["evidence"]["half"]["winner"] == "S09E04", r2["evidence"]
assert evidence_accepts(["S09E04"], r2["segments"], r2["evidence"]) is True
assert classify(["S09E04"], r2["segments"], "remote test", r2["evidence"])[0] == "OK"
# ...and the gate still refuses when the filename names the OTHER twin.
assert classify(["S09E03"], r2["segments"], "remote test", r2["evidence"])[0] != "OK"

# A partial recording of only the opening of the shared text points one way but
# covers nothing episode-sized: the old identical-reference veto must stand.
part = m.match_file(cues(H1[:150]), 150.0)
assert "half" not in part["evidence"], part["evidence"]
assert part["evidence"]["conflicted"] is True, part["evidence"]
assert all(sg["confidence"] != "high" for sg in part["segments"]), part["segments"]

# Both halves → a real double episode: never pinned to one code, veto stays.
rd = m.match_file(cues(H1 + H2), 1200.0)
assert "half" not in rd["evidence"], rd["evidence"]
assert rd["evidence"]["conflicted"] is True, rd["evidence"]
assert {s["code"] for s in rd["segments"] if s["confidence"] == "high"} != {"S09E03"}, rd["segments"]

# Distant duplicate codes: one text under S05E24 AND S05E36 is a numbering
# duplicate, not a double episode — position must not settle it.
md = Matcher({"S05E24": SHARED, "S05E36": SHARED, "S05E10": " ".join(UNRELATED)}, S)
rdx = md.match_file(cues(H1), 600.0)
assert "half" not in rdx["evidence"], rdx["evidence"]
assert rdx["evidence"]["conflicted"] is True, rdx["evidence"]

# Adjacent references that merely resemble each other (shared story words,
# different texts) must not settle by halves either.
LOOK1 = " ".join(words("dd", 600) + words("ee", 300))
LOOK2 = " ".join(words("dd", 600) + words("ff", 300))
mk = Matcher({"S09E07": LOOK1, "S09E08": LOOK2, "S09E09": " ".join(UNRELATED)}, S)
rl = mk.match_file(cues(words("dd", 600) + words("ee", 300)), 900.0)
assert "half" not in rl["evidence"], rl["evidence"]

print("PASS (half-position settles adjacent twins; doubles, distant twins and lookalikes stay deferred)")
