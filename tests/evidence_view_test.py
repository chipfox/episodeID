"""The evidence view: one file's dialogue beside one episode's reference, so a
person can read both and judge a doubtful match themselves.

It must work from what a scan already has (no transcription), name the right
episode by default, and rank the references by how much their text looks like
the file's dialogue.
"""
import json
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_evidence_view"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"),
                  WHISPER_ENABLED="false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import db, main  # noqa: E402
from app.config import CACHE_DIR  # noqa: E402
from app.media_text import _cache_file  # noqa: E402
from app.textsim import shared_terms  # noqa: E402

db.init()
tv = W / "media" / "show"
(tv / "Season 1").mkdir(parents=True)
rel = "Season 1/Show - S01E01 - Pilot.mkv"
video = tv / rel
video.write_bytes(b"x" * 4096)

sid = db.add_series(1234, "Show", "2020", str(tv), None, options={})
db.update_series(sid, episodes=[
    {"season": 1, "episode": 1, "title": "Pilot", "aired": "2020-01-01"},
    {"season": 1, "episode": 2, "title": "The Lighthouse", "aired": "2020-01-08"},
])

refs = CACHE_DIR / "refs" / "1234"
refs.mkdir(parents=True)


def save_ref(code, text):
    (refs / f"{code}.json").write_text(json.dumps({"source": "opensubtitles", "text": text}))


import random

VOCAB = [f"{a}{b}" for a in "bcdfghjklmnprstvwz" for b in "aeiou"]


def transcript(n, markers, seed):
    """Varied dialogue-like text: unique word pairs, with the given markers recurring."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        out.extend(markers[:2] if i % 5 == 0 else [rng.choice(VOCAB)])
    return " ".join(out)


base = transcript(220, ("spaceship", "alaska", "pilot", "waitress"), 7)
save_ref("S01E01", base + " " + transcript(160, ("mulder", "scully", "diner", "report"), 8))
save_ref("S01E02", transcript(260, ("albatross", "lighthouse", "maine", "keeper"), 9))

# The sample is the first ten minutes of S01E01's dialogue: fully contained in that
# episode's reference, absent from the other's.
sample = base
files = [{
    "rel": rel, "status": "OK", "note": "Sample of 10 min — matches the filename",
    "text_source": "remote parakeet-tdt", "scope": "sample",
    "sample_start": 0.0, "sample_end": 600.0, "file_duration": 2700.0,
    "sample_text": sample, "expected": ["S01E01"],
    "segments": [{"code": "S01E01", "confidence": "high", "score": 0.4, "margin": 0.3,
                  "words": len(sample.split()), "scope": "sample"}],
}]
db.add_scan(sid, {"files": files, "counts": {"OK": 1}})

other = db.add_series(99, "Unscanned", "1999", str(W / "media" / "empty"), None, options={})
(W / "media" / "empty").mkdir(parents=True)

c = TestClient(main.app)

# Opening the view must never start a transcription: it only reads caches.
cache_dir = CACHE_DIR / "text"
before = set(cache_dir.glob("*.json")) if cache_dir.exists() else set()

# Default episode: the one the scan's own segment settled on.
d = c.get(f"/api/series/{sid}/evidence", params={"rel": rel}).json()
after = set(cache_dir.glob("*.json")) if cache_dir.exists() else set()
assert after == before, "the evidence view wrote a transcript cache"
assert d["code"] == "S01E01", d["code"]
f = d["file"]
# No cached sample here, so the view falls back to the clip the scan stored.
assert f["text"] == sample and f["scope"] == "excerpt" and f["scan_scope"] == "sample", f.keys()
assert f["words"] == len(sample.split()) and f["source"] == "remote parakeet-tdt"
assert f["sample_start"] == 0.0 and f["sample_end"] == 600.0
assert d["reference"]["code"] == "S01E01" and "spaceship" in d["reference"]["text"]
assert d["reference"]["words"] == len(d["reference"]["text"].split())
assert d["ranked"][0]["code"] == "S01E01", d["ranked"]
assert d["containment"] == d["ranked"][0]["containment"]
assert d["containment"] > 0.6, d["containment"]
assert d["file"]["bigrams"] >= d["bigram_floor"], d["file"]["bigrams"]
assert "spaceship" in d["shared"] and "albatross" not in d["shared"], d["shared"]

# Choosing the other episode swaps the reference text and containment collapses.
d2 = c.get(f"/api/series/{sid}/evidence", params={"rel": rel, "code": "S01E02"}).json()
assert d2["code"] == "S01E02" and "albatross" in d2["reference"]["text"]
assert d2["containment"] is not None and d2["containment"] < 0.35, d2["containment"]
assert d2["containment"] < d["containment"]
assert "albatross" not in d2["shared"]

# The picker only needs the episode list and which ones have a reference.
eps = {e["code"]: e for e in d2["episodes"]}
assert eps["S01E01"]["has_ref"] and eps["S01E02"]["has_ref"] and eps["S01E01"]["title"] == "Pilot"

# A reference on its own — what the Episodes & references table shows.
r = c.get(f"/api/series/{sid}/evidence", params={"code": "S01E02"}).json()
assert r["file"] is None and r["reference"]["code"] == "S01E02" and r["ranked"] == []

# Unknown file, and a series that was never scanned.
assert c.get(f"/api/series/{sid}/evidence", params={"rel": "nope.mkv"}).status_code == 404
bad = c.get(f"/api/series/{other}/evidence", params={"rel": rel})
assert bad.status_code == 404 and "scan" in bad.json()["detail"].lower(), bad.json()

# The cached sample window is what a scan actually matched, so the view must show
# all of it — not the ~1,400-character clip older scans stored.
window_text = transcript(400, ("diner", "spaceship"), 12)
window = [[i, i + 1, w] for i, w in enumerate(window_text.split())]
sample_cache = _cache_file(video, "sample", 0.0, 600.0)
sample_cache.parent.mkdir(parents=True, exist_ok=True)
sample_cache.write_text(json.dumps(
    {"source": "test-asr", "duration": 600.0, "cues": window,
     "scope": "sample", "sample_start": 0.0, "sample_end": 600.0, "file_duration": 2700.0}))
d3 = c.get(f"/api/series/{sid}/evidence", params={"rel": rel}).json()
assert d3["file"]["scope"] == "sample", d3["file"]["scope"]
assert "spaceship" in d3["file"]["text"] and d3["file"]["words"] > len(sample.split())
assert d3["file"]["sample_start"] == 0.0 and d3["file"]["sample_end"] == 600.0
assert d3["containment"] is not None

# A whole-file transcript wins over the sample when one is cached.
full_text = transcript(800, ("lighthouse", "keeper"), 13)
full_cues = [[i, i + 1, w] for i, w in enumerate(full_text.split())]
full_cache = _cache_file(video)
full_cache.parent.mkdir(parents=True, exist_ok=True)
full_cache.write_text(json.dumps(
    {"source": "test-asr", "duration": 2700.0, "cues": full_cues}))
d4 = c.get(f"/api/series/{sid}/evidence", params={"rel": rel}).json()
assert d4["file"]["scope"] == "full", d4["file"]["scope"]
assert d4["file"]["text"] == full_text
assert d4["file"]["sample_start"] is None and d4["file"]["sample_end"] is None

# A scan now stores the whole sample window, so the stored text and the cache are the
# same length. That is still "sample", not "excerpt" of one. (The full transcript from
# the previous case is removed first, or it would rightly win.)
full_cache.unlink(missing_ok=True)
db.add_scan(sid, {"files": [{
    "rel": rel, "status": "OK", "note": "sample", "text_source": "test-asr", "scope": "sample",
    "sample_start": 0.0, "sample_end": 600.0, "file_duration": 2700.0,
    "sample_text": window_text, "expected": ["S01E01"], "segments": []}], "counts": {}})
d5 = c.get(f"/api/series/{sid}/evidence", params={"rel": rel}).json()
assert d5["file"]["scope"] == "sample", d5["file"]["scope"]
assert d5["file"]["words"] == len(window_text.split())

# shared_terms is a reading aid: both texts, commonest shared words first (equal
# counts break alphabetically), no stop words, nothing shorter than four letters.
assert shared_terms("Leonard Betts, the EMT. Leonard is here.", "Betts! Leonard Betts, EMT.") == \
    ["betts", "leonard"]
assert shared_terms("a an the", "a an the") == []
# The reference-gram cache must notice an edited reference, or the view would keep
# scoring against yesterday's text.
save_ref("S01E01", transcript(200, ("albatross", "harbor"), 21))
after = c.get(f"/api/series/{sid}/evidence", params={"rel": rel}).json()
assert after["containment"] is not None and after["containment"] < 0.35, after["containment"]

print("PASS (evidence view + shared terms)")
