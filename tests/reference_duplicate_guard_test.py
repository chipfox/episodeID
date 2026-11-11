"""A fetched transcript that repeats another episode's reference is refused.

The X-Files wiki's "Memento Mori/Transcript" page is the Leonard Betts transcript.
Accepting it makes two episodes unprovable, and on an upgrade (a wiki replacing an
OpenSubtitles reference) it would throw away the reference that actually worked — the
exact S04E14 regression. The fetch path asks duplicate_reference() before accepting.
Containment, not cosine: the page carries gallery/nav chrome the reference does not,
so the live pair measures 0.745 cosine but 0.911 bigram containment.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_dup_ref"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.references import duplicate_reference  # noqa: E402

BETTS = ("The ambulance crashes and the driver finds Leonard Betts decapitated on the "
         "road. His head is a few feet away. At the morgue Scully weighs the head and "
         "begins the autopsy; the eyes open and the mouth moves. Mulder thinks Betts "
         "regrew his head, that his tissue is riddled with cancer and that he visits "
         "cancer wards for access. His mother buried Albert Tanner six years ago and "
         "the same man now answers to Leonard Betts. ") * 8
MEMENTO = ("Scully sits with Penny Northern in the hospital and learns her own cancer "
           "has spread. She records a diary for her mother, refuses to stop working, "
           "and tells Mulder she is not going to die. The Lone Gunmen find a clinic "
           "trial; Penny dies; Scully keeps the cross and the promise she made. ") * 8
PAGE = BETTS + "Credits Gallery Transcript Contents SCENE one SCENE two navigation"

texts = {"S04E12": BETTS, "S04E14": MEMENTO, "S04E13": BETTS}
print("twin of S04E14:", duplicate_reference("S04E14", PAGE, texts, min_grams=20))

# The page fetched for S04E14 is S04E12's transcript (plus wiki chrome): refuse it.
assert duplicate_reference("S04E14", PAGE, texts, min_grams=20) in ("S04E12", "S04E13")
# A genuine transcript for this episode is not a duplicate of anything.
assert duplicate_reference("S04E14", MEMENTO, texts, min_grams=20) is None
# An episode is never compared with itself.
assert duplicate_reference("S04E12", BETTS, texts, min_grams=20) == "S04E13"
# No other references at all: nothing to compare, nothing refused.
assert duplicate_reference("S04E12", BETTS, {"S04E12": BETTS}, min_grams=20) is None
# Too short to judge: a 250-word transcript must not be condemned by chance overlap.
assert duplicate_reference("S04E14", "Mulder it is me", texts) is None
print("PASS")
