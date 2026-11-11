"""Reference-conflict detection, and the aggregate that must respect it.

The fixtures replay the two X-Files defects found in the live library: two
episodes whose Fandom transcripts are one and the same text, and a story pair
whose shared transcript is legitimate. Confusable references must never confirm
or deny a filename, and the sample aggregate must never let one accuse a rival.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_evidence_test"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import evidence  # noqa: E402
from app.pipeline import _aggregate_sample_evidence  # noqa: E402

FILLER = " ".join(["the agent filed a report with the office before the interview"] * 30)
BETTS = " ".join(["leonard betts cardiac regeneration tumor scalp salamander"] * 8) + " " + FILLER
# The real-world defect: S04E14's wiki page carried S04E12's transcript with a
# couple of lines of its own — a near-duplicate, not a distinct episode.
MEMENTO = BETTS + " zero chance survival"
NEVER = " ".join(["talking tattoo back Scranton scully manipulative landlord"] * 8) + " " + FILLER
DETOUR = " ".join(["yellow rain cryptid hiker mountain missing trail"] * 8) + " " + FILLER
JUMP = " ".join(["bank robbery jumpstart motor neuron coma"] * 10) + " " + FILLER

eps = {
    "S04E11": {"title": "Detour", "overview": "A hiker meets a creature after yellow rain.",
               "runtime": 45},
    "S04E12": {"title": "Leonard Betts",
               "overview": "An EMT regrows his severed head; Scully finds a healing tumour.",
               "runtime": 45},
    "S04E13": {"title": "Never Again", "overview": "Scully dates a man with a talking tattoo.",
               "runtime": 45},
    "S04E14": {"title": "Memento Mori",
               "overview": "Scully is diagnosed with terminal brain cancer.",
               "runtime": 45},
    "S09E01": {"title": "Jumpstart (1)", "overview": "A comatose man robs banks.",
               "runtime": 45},
    "S09E02": {"title": "Jumpstart (2)", "overview": "The jumper's story continues.",
               "runtime": 45},
}
refs = {"S04E11": DETOUR, "S04E12": BETTS, "S04E13": NEVER, "S04E14": MEMENTO,
        "S09E01": JUMP, "S09E02": JUMP}
story = lambda code: ["S09E01", "S09E02"] if code.startswith("S09") else [code]  # noqa: E731

conflicts = evidence.ReferenceConflicts(refs, story=story, conflict_similarity=0.85)

# One transcript under two numbers is a conflict; a story part is not.
assert conflicts.conflicting("S04E12") == ["S04E14"], conflicts.conflicting("S04E12")
assert conflicts.conflicting("S04E14") == ["S04E12"], conflicts.conflicting("S04E14")
assert conflicts.conflicting("S09E01") == [], conflicts.conflicting("S09E01")
# Story parts are still flagged for transcript repair — the identical copy is the
# whole story filed twice — but they never block anything.
assert conflicts.story_pair_for("S09E01") == ["S09E02", 1.0], conflicts.story_pair_for("S09E01")
assert [p["codes"] for p in conflicts.story_pair_conflicts()] == [["S09E01", "S09E02"]]
pairs = conflicts.conflict_pairs()
assert [p["codes"] for p in pairs] == [["S04E12", "S04E14"]], pairs
assert pairs[0]["similarity"] > 0.95, pairs

# A two-line stub is not evidence of a duplicate, however identical.
stub = evidence.ReferenceConflicts({"S01E01": "one two three", "S01E02": "one two three"},
                                   conflict_similarity=0.85)
assert stub.conflict_pairs() == [] and stub.story_pair_conflicts() == []

# Aggregate: a shared transcript neither confirms the claim nor accuses a rival.
SETTINGS = {"min_score": 0.12, "min_margin": 0.05}


def window(votes, start=0):
    return {"start": start, "end": start + 600, "source": "salm ASR", "words": 100,
            "has_text": True, "language": None, "votes": votes, "whole": votes,
            "text": "sample dialogue here"}


both = [{"code": "S04E12", "score": 0.30, "confidence": "high", "evidence": "whole"},
        {"code": "S04E14", "score": 0.295, "confidence": "high", "evidence": "whole"}]
agg = _aggregate_sample_evidence(["S04E12"], [window(both)], eps, SETTINGS, 2, {"S04E14"})
assert agg["status"] == "LOW_CONFIDENCE", agg
assert "shares" in agg["note"] and "cannot prove" in agg["note"], agg["note"]

rival = [{"code": "S04E14", "score": 0.3, "confidence": "high", "evidence": "whole"}]
agg = _aggregate_sample_evidence(["S04E12"], [window(rival, 0), window(rival, 1200)],
                                 eps, SETTINGS, 2, {"S04E14"})
assert agg["status"] == "LOW_CONFIDENCE", agg
assert "near-duplicate" in agg["note"], agg["note"]

# Without the conflict, the same votes behave as before: two windows → mismatch.
agg = _aggregate_sample_evidence(["S04E12"], [window(rival, 0), window(rival, 1200)],
                                 eps, SETTINGS, 2, None)
assert agg["status"] == "MISMATCH" and agg["code"] == "S04E14", agg

# A genuine (non-conflicting) rival still confirms the claim outright.
claim_only = [{"code": "S04E12", "score": 0.4, "confidence": "high", "evidence": "whole"}]
agg = _aggregate_sample_evidence(["S04E12"], [window(claim_only)], eps, SETTINGS, 2, {"S04E14"})
assert agg["status"] == "OK", agg

print("PASS")
