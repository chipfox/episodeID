"""Evidence selection and reference-conflict handling for the decision packet.

The fixtures replay the two X-Files defects found in the live library: two
episodes whose Fandom transcripts are one and the same text, and a story pair
whose shared transcript is legitimate. Confusable references must not confirm
or deny a filename; distinctive lines must reach Laya even when they sit in the
middle of a sample."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_evidence_test"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"), DECISION_ENABLED="true",
                  DECISION_BASE_URL="http://127.0.0.1:9/v1", DECISION_MODEL="laya")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import adjudication, evidence  # noqa: E402
from app.decision import _description  # noqa: E402

FILLER = " ".join(["the agent filed a report with the office before the interview"] * 30)
BETTS = " ".join(["leonard betts cardiac regeneration tumor scalp salamander"] * 8) + " " + FILLER
# The real-world defect: S04E14's wiki page carried S04E12's transcript with a
# couple of lines of its own — a near-duplicate, not a distinct episode.
MEMENTO = BETTS + " zero chance survival"
NEVER = " ".join(["talking tattoo back Scranton scully manipulative landlord"] * 8) + " " + FILLER
DETOUR = " ".join(["yellow rain cryptid hiker mountain missing trail"] * 8) + " " + FILLER
JUMP = " ".join(["bank robbery jumpstart motor neuron coma"] * 10) + " " + FILLER

eps = {
    "S04E11": {"title": "Detour", "overview": "A hiker encounters a mythical creature after yellow rain."},
    "S04E12": {"title": "Leonard Betts",
               "overview": "An EMT regrows his severed head. Scully finds a tumour with a healing power."},
    "S04E13": {"title": "Never Again", "overview": "Scully dates a manipulative man with a talking tattoo."},
    "S04E14": {"title": "Memento Mori",
               "overview": "Scully is diagnosed with terminal brain cancer: statistically zero chance."},
    "S09E01": {"title": "Jumpstart (1)", "overview": "A comatose man robs banks through another's body."},
    "S09E02": {"title": "Jumpstart (2)", "overview": "The jumper's story continues; Mulder hunts him."},
}
refs = {"S04E11": DETOUR, "S04E12": BETTS, "S04E13": NEVER, "S04E14": MEMENTO,
        "S09E01": JUMP, "S09E02": JUMP}
story = lambda code: ["S09E01", "S09E02"] if code.startswith("S09") else [code]  # noqa: E731

lex = evidence.Lexicon(eps, refs, story=story, conflict_similarity=0.85)

# One transcript under two numbers is a conflict; a story part is not.
assert lex.conflicting("S04E12") == ["S04E14"], lex.conflicting("S04E12")
assert lex.conflicting("S04E14") == ["S04E12"], lex.conflicting("S04E14")
assert lex.conflicting("S09E01") == [], lex.conflicting("S09E01")
assert lex.conflict_pairs() == [{"codes": ["S04E12", "S04E14"],
                                 "similarity": lex._conflicts["S04E12"][0][1]}], lex.conflict_pairs()

# A descriptor is one short line about what only this episode contains.
descriptor = lex.descriptor("S04E14")
assert "Memento Mori" in descriptor and "cancer" in descriptor, descriptor
assert len(descriptor) <= 110, descriptor
assert "matcher score" not in _description({"code": "S04E14", "title": "Memento Mori",
                                            "overview": "x" * 300, "score": 0.9,
                                            "descriptor": descriptor})

# Distinctive words outweigh the shared furniture of the show.
assert "tattoo" in lex.terms("S04E13") and "betts" in lex.terms("S04E12")
w12 = lex.weights("S04E12")
assert w12["betts"] > w12["office"] and w12["betts"] > w12["report"], w12

# Evidence selection puts the claim's lines first, tags where they were heard,
# and drops small talk that no episode's references make distinctive.
sample = ("[0–10 min] Hey buddy, we should grab dinner after the shift.\n"
          "[20–30 min] Scully, I have cancer. If it pushes into my brain, statistically zero chance.\n"
          "[40–50 min] The tattoo spoke again at the bar.")
packet = evidence.select_evidence(sample, lex, ["S04E14", "S04E13", "S04E11"], 700)
assert "cancer" in packet, packet
assert "grab dinner" not in packet, packet
assert "20–30 min" in packet, packet
assert len(packet) <= 700, packet
assert evidence.select_evidence("hello there friend words only", lex, ["S04E14"], 700) == ""
assert evidence.select_evidence(sample, None, ["S04E14"], 700) == ""

# With a lexicon the packet is the selected evidence, not the raw head/tail.
pool = adjudication.candidates({"rel": "Show - S04E14.mkv", "expected": ["S04E14"],
                                "whole": [{"code": "S04E14", "score": 0.2}], "segments": []},
                               eps, 4, lexicon=lex)
assert pool[0]["descriptor"].startswith("Memento Mori"), pool

adjudication.decision.available = lambda: True
asked = []
adjudication.decision.choose = lambda text, pool, instructions=None: asked.append(text) or {
    "code": "S04E13", "confidence": 0.42, "unknown": False, "probabilities": {}, "model": "laya"}
f = {"rel": "Show - S04E13.mkv", "scope": "sample", "expected": ["S04E13"], "cues": [],
     "whole": [{"code": "S04E13", "score": 0.2}], "segments": [], "sample_text": sample}
adjudication.adjudicate({"name": "Show"}, f, eps, lexicon=lex)
assert asked and "tattoo" in asked[-1] and "min]" in asked[-1], asked
assert "grab dinner" not in asked[-1], asked

# A claimed episode whose reference duplicates another's must not be asked about.
asked.clear()
dup = {"rel": "Show - S04E12.mkv", "scope": "sample", "expected": ["S04E12"], "cues": [],
       "whole": [{"code": "S04E12", "score": 0.5}], "segments": [], "sample_text": BETTS}
assert adjudication.adjudicate({"name": "Show"}, dup, eps, lexicon=lex) is None
assert not asked, asked
assert dup["reference_conflict"] == ["S04E14"], dup

# Aggregate: a shared transcript neither confirms the claim nor accuses a rival.
from app.pipeline import _aggregate_sample_evidence  # noqa: E402

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

# dialogue_text reads the cached [start, end, text] cue shape and clips to a
# head+tail, exactly as the packet budget requires (a dict-shaped cue also works).
from app.adjudication import dialogue_text  # noqa: E402

assert dialogue_text([[0, 2, "First line."], [3, 5, "Second line."]]) == "First line. Second line."
assert dialogue_text([{"text": "Dict line."}, [4, 6, "List line."]]) == "Dict line. List line."
clip = dialogue_text([[0, 1, "word " * 30]] * 60, limit=150)
assert len(clip) <= 150 + len(" … ") and "…" in clip, clip
print("PASS")
