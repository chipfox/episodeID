"""Subtitle cleaning: the fansub watermark goes, the dialogue does not.

Measured in this library's own references, "Synch and correct by MemoryOnSmells
www.addic7ed.com" is pasted into a dozen season-9 transcripts, and season 9 is where
the confusions live: two transcripts carrying the same group's signature look more
alike than the episodes are. Removing it has to be precise, because "Owww." and
"murdered by a party favor" are dialogue, not junk.
"""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_text_clean"))
if W.exists():
    shutil.rmtree(W)
W.mkdir(parents=True)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.media_text import parse_srt_vtt  # noqa: E402


def one(text):
    """What one subtitle cue becomes after ingest (empty string: cue dropped)."""
    srt = f"1\n00:00:01,000 --> 00:00:04,000\n{text}\n\n"
    cues = parse_srt_vtt(srt)
    return " | ".join(c.text for c in cues)


# The watermark and its credit line, in the shapes the library actually contains.
assert one("Synch and correct by MemoryOnSmells www.addic7ed.com") == ""
assert one("Sync & corrected by honeybunny www.addic7ed.com") == ""
# A handle with a second word and dash delimiters can leave "--" behind. That is
# fine: the matcher tokenises letters only, so a residue contributes no signal,
# while the watermark words themselves must be gone.
residue = one("-- Sync by Rafael UPD -- www.addic7ed.com/")
assert "addic7ed" not in residue.lower() and "rafael" not in residue.lower(), residue
assert one("Subtitles by OpenSubtitles") == ""
mixed = one("translated by Maria, http://subs.example.org/x")
assert "subs.example.org" not in mixed and "translated" not in mixed.lower(), mixed

# A mark glued into the middle of real dialogue: the line stays, the mark goes.
assert one("The truth is out there, Sync by Rafael www.addic7ed.com") == "The truth is out there,"

# Dialogue that looks like the junk patterns must survive untouched.
keep = [
    "Owww. Burger World",
    "It's so tragic, to see these innocent creatures murdered by a party favor.",
    "Yeah. This is satanic country music.",
    "Ray Soames was the third victim.",
    "LAUREN KYTE takes the pictures off the wall.",
    "I got 999.55 dollars, and 2.5 seconds.",
]
for line in keep:
    assert one(line) == line, (line, one(line))

# What the parser already handled, pinned so a future edit cannot lose it.
assert one("<i>Don't</i> touch that") == "Don't touch that"
assert one("{\\an8}Hello there") == "Hello there"
assert one("[Music]") == ""
assert one("(gasps)") == ""
assert one("♪ we are the people ♪") == "we are the people"  # marks go, sung words stay
assert one("first line\\Nsecond line") == "first line second line"
assert one("&amp; also &quot;quoted&quot;") == "& also \"quoted\""

# An apostrophe word stays one word — this is why the hand-rolled strip was wrong.
cues = parse_srt_vtt("1\n00:00:01,000 --> 00:00:04,000\nDon't stop believing\n\n")
assert cues[0].text == "Don't stop believing", cues

print("PASS")
