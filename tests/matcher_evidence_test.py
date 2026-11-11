"""The evidence gate and the reference-conflict guard (app/matcher.py).

Pins the two behaviours that took a measured A/B to justify:
  * a deferral is accepted only when the whole transcript's best episode IS the
    filename's episode, most windows agree, and enough dialogue exists;
  * a segment whose winning reference has a near-identical twin is never high,
    because such a reference cannot prove which of the two episodes was read.

The gate test drives the real path (match_file -> classify), so a change that
neuters the gate inside classify fails here even if the predicate still works.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_matcher_evidence"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    EVIDENCE_GATE="true",
    GATE_MIN_WINDOW_SHARE="0.30",
    GATE_MIN_WORDS="120",
    REFERENCE_CONFLICT_SIMILARITY="0.70",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app.matcher as matcher_mod  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.matcher import Matcher, classify, evidence_accepts  # noqa: E402

S = get_settings()


def words(marker: str, n: int) -> list[str]:
    """n letter-only words that only one reference contains."""
    return [f"{marker}{chr(97 + i % 26)}{chr(97 + (i // 26) % 26)}"
            f"{chr(97 + (i // 676) % 26)}{chr(97 + (i // 17576) % 26)}"
            for i in range(n)]


def cues_over(text_words: list[str], seconds: float, per_cue: int = 2) -> list[list]:
    span = seconds / max(1, len(text_words) / per_cue)
    out = []
    for i in range(0, len(text_words), per_cue):
        st = round(i / per_cue * span, 1)
        out.append([st, st + span, " ".join(text_words[i:i + per_cue])])
    return out


REFS = {f"S01E0{i}": " ".join(words(m, 20000)) for i, m in
        enumerate(["aa", "bb", "cc", "dd", "ee", "ff"], start=1)}
M = Matcher(REFS, S)

# --- 1. thin-but-consistent sample: deferred without evidence, OK with it ----
thin_words = words("aa", 150)                    # 150 words out of a 4000-word reference
res = M.match_file(cues_over(thin_words, 600.0), 600.0)
ev = res["evidence"]
assert ev["top1"] == "S01E01", ev
assert ev["windows"] >= 4 and ev["words"] >= 60 and ev["rank_share"] >= 0.30, ev
assert ev["top1_score"] < S["min_score"], (
    f"fixture is not the case under test: absolute score {ev['top1_score']} "
    f"is not under min_score {S['min_score']}")
assert not any(sg["confidence"] == "high" for sg in res["segments"]), res["segments"]

status_without, _ = classify(["S01E01"], res["segments"], "embedded")
status_with, note_with = classify(["S01E01"], res["segments"], "embedded", ev)
assert status_without in ("NO_MATCH", "LOW_CONFIDENCE"), status_without
assert status_with == "OK", (status_with, note_with)

# The gate corroborates the filename; it never overrides it.
assert classify(["S01E02"], res["segments"], "embedded", ev)[0] != "OK"
# A multi-episode filename is never settled by one episode's evidence.
assert classify(["S01E01", "S01E03"], res["segments"], "embedded", ev)[0] != "OK"

# --- 2. thin evidence is refused by each floor in turn -----------------------
# 2a. Very little total dialogue, spread thin: the word floor refuses it.
sparse = M.match_file(cues_over(words("aa", 36), 600.0, per_cue=3), 600.0)
assert sparse["evidence"]["words"] < 60, sparse["evidence"]
assert classify(["S01E01"], sparse["segments"], "embedded", sparse["evidence"])[0] != "OK"

# 2b. Enough words, but all of it in one burst: the window quorum refuses it,
# because a single stretch of dialogue cannot vouch for a whole sample.
burst = M.match_file(cues_over(words("aa", 60), 60.0), 600.0)
assert burst["evidence"]["words"] >= 60, burst["evidence"]
assert burst["evidence"]["windows"] < 4, burst["evidence"]
assert classify(["S01E01"], burst["segments"], "embedded", burst["evidence"])[0] != "OK"

# --- 3. the kill switch and the contradiction check --------------------------
assert evidence_accepts(["S01E01"], [], ev)
real_settings = matcher_mod.get_settings
matcher_mod.get_settings = lambda: {**S, "evidence_gate": False}
try:
    assert not evidence_accepts(["S01E01"], [], ev)
finally:
    matcher_mod.get_settings = real_settings
assert not evidence_accepts(["S01E01"], [{"code": "S01E02", "confidence": "low"}], ev)
assert not evidence_accepts(["S01E01"], [], {**ev, "conflicted": True})
assert not evidence_accepts(["S01E01"], [], None)

# --- 4. duplicate reference: a full match cannot be high ---------------------
full = cues_over(words("aa", 4000), 1200.0)
good = M.match_file(full, 1200.0)
assert good["segments"], good
assert all(sg["confidence"] == "high" for sg in good["segments"]), good["segments"]
assert not good["evidence"]["conflicted"], good["evidence"]

twin_refs = dict(REFS)
twin_refs["S01E02"] = REFS["S01E01"]          # one wiki page filed under two numbers
twin = Matcher(twin_refs, S)
dup = twin.match_file(full, 1200.0)
assert dup["evidence"]["conflicted"] and dup["evidence"]["twin"] in ("S01E01", "S01E02"), \
    dup["evidence"]
assert dup["segments"], dup
assert all(sg["confidence"] == "low" for sg in dup["segments"]), dup["segments"]
assert any("near-identical" in (sg.get("why") or "") for sg in dup["segments"]), dup["segments"]
assert classify(["S01E01"], dup["segments"], "embedded", dup["evidence"])[0] != "OK"

# --- 5. no dialogue at all still reports a complete evidence block -----------
empty = M.match_file([], 0.0)
assert empty["segments"] == [] and empty["whole"] == []
assert empty["evidence"] == matcher_mod._no_evidence()

# The plan and the GUI need a segment, not just a status: promotion materialises
# an accepted-but-unsegmented result as one high segment over the file.
from app.matcher import promote  # noqa: E402

promoted = promote(["S01E01"], [], ev, 600.0)
assert promoted and promoted[0]["code"] == "S01E01" and promoted[0]["confidence"] == "high", promoted
assert "windows agree" in promoted[0]["why"], promoted
assert promote(["S01E01"], [{"code": "S01E01"}], ev, 600.0) == [{"code": "S01E01"}]  # never overwrites
assert promote(["S01E02"], [], ev, 600.0) == []  # a filename the evidence does not support

print("PASS")
