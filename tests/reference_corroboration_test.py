"""A replacement subtitle must match the episode's own file, not just its name.

The S09E02 failure: OpenSubtitles' best-looking candidate was named for part 2 but
carried part 1's dialogue. Repair compared replacements to each other, so two copies
of part 1 looked like a valid pair and both references stayed wrong. Corroboration
uses the file's own ASR/embedded text instead: download a couple of candidates and
keep the one that dialogue contains.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_corroborate"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.references as R  # noqa: E402
from app import pipeline  # noqa: E402

PART1 = "harbour boat captain storm anchor rope lighthouse " * 60
PART2 = "courtroom witness judge evidence verdict gavel jury " * 60
SERIES = {"tvdb_id": 900001, "name": "Show", "imdb_id": "tt123",
          "episodes": [{"season": 1, "episode": 1, "title": "One"},
                       {"season": 1, "episode": 2, "title": "Two"}]}


def srt(text):
    words = text.split()
    out = []
    for i in range(0, len(words), 10):
        n = i // 10 + 1
        out += [str(n), f"00:00:{i // 10:02d},000 --> 00:00:{i // 10 + 1:02d},000",
                " ".join(words[i:i + 10]), ""]
    return "\n".join(out)


def candidate(cid, ep, title, text, downloads):
    return {"id": cid, "attributes": {
        "feature_details": {"season_number": 1, "episode_number": ep, "title": title,
                            "parent_imdb_id": "123"},
        "files": [{"file_id": cid, "file_name": f"Show.S01E{ep:02d}.{title}.1080p.WEB-DL"}],
        "release": f"Show.S01E{ep:02d}.{title}.1080p.WEB-DL",
        "download_count": downloads, "machine_translated": False, "ai_translated": False}}


class FakeOS:
    configured = True

    def __init__(self):
        # E02: the popular candidate is part 1's text; the right one is ranked lower.
        self.by_ep = {
            1: [candidate(11, 1, "One", PART1, 900)],
            2: [candidate(21, 2, "Two", PART1, 1000),
                candidate(22, 2, "Two", PART2, 500)],
        }
        self.texts = {11: PART1, 21: PART1, 22: PART2}
        self.downloads = []

    def login(self):
        pass

    def close(self):
        pass

    def search(self, **q):
        return self.by_ep.get(q.get("episode_number"), [])

    def download(self, file_id):
        self.downloads.append(file_id)
        return srt(self.texts[file_id])


class Ctx:
    def log(self, _msg):
        pass

    def status(self, _msg):
        pass

    def check_cancel(self):
        pass


fake = FakeOS()
R.OpenSubtitles = lambda: fake

# Direct fetch: the corroborated candidate wins over the more downloaded one.
ref, note = R.fetch_opensubs(fake, SERIES, SERIES["episodes"][1], Ctx(),
                             corroborate=PART2, tries=2)
assert ref and "courtroom witness" in ref["text"] and "harbour boat" not in ref["text"], (ref, note)
assert ref["corroborated"] >= 0.9, ref

# A file whose dialogue no candidate contains: refuse to invent a reference.
ref, note = R.fetch_opensubs(fake, SERIES, SERIES["episodes"][1], Ctx(),
                             corroborate="volcano submarine glacier comet " * 60, tries=2)
assert ref is None and "matched the file's own transcript" in note, (ref, note)

# Repair: both references start as part 1. Corroboration should replace both with
# their own halves and record version-2 markers.
R._save(900001, "S01E01", {"source": "opensubtitles", "text": PART1})
R._save(900001, "S01E02", {"source": "opensubtitles", "text": PART1})
out = R.repair_reference_conflicts(SERIES, Ctx(), corroborate={"S01E01": PART1, "S01E02": PART2})
assert out["replaced"] == 2, out
refs = R.load_refs(900001)
assert "harbour boat" in refs["S01E01"]["text"] and "courtroom witness" in refs["S01E02"]["text"]
assert refs["S01E01"]["corroborated"] and refs["S01E02"]["corroborated"]
assert R._load_repairs(900001)["S01E02"].get("version") == 2

# Only trusted, full, long-enough dialogue can corroborate; a sidecar never can.
scan = {"results": {"files": [
    {"scope": "full", "text_source": "remote parakeet", "sample_text": PART1 * 2,
     "expected": ["S01E01"]},
    {"scope": "full", "text_source": "sidecar Show.S01E02.srt", "sample_text": PART2 * 2,
     "expected": ["S01E02"]},
    {"scope": "sample", "text_source": "remote parakeet", "sample_text": PART2 * 2,
     "expected": ["S01E03"]},
    {"scope": "full", "text_source": "whisper small", "sample_text": PART1 * 2,
     "expected": ["S01E04"]},
]}}
pipeline.db.latest_scan = lambda _sid: scan
checked = pipeline._corroborating_text(1)
assert list(checked) == ["S01E01", "S01E04"], checked

print("reference_corroboration_test: OK — 5 cases")
