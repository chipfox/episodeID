"""A candidate must be one named episode with plausible timings; preview saves nothing."""
import os
import sys
from pathlib import Path
from unittest.mock import patch

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_anchor_candidates"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from app import db, main  # noqa: E402
from app.media_text import chapter_hints  # noqa: E402
from app.references import _save, load_refs  # noqa: E402

root = W / "media" / "Show"
root.mkdir(parents=True, exist_ok=True)
db.init()
sid = db.add_series(987654, "Show", "", str(root), "tt1234")
db.update_series(sid, episodes=[{"season": 1, "episode": 1, "title": "Foreign Exchange", "runtime": 11},
                                {"season": 1, "episode": 2, "title": "Ball Breakers", "runtime": 11}])


def result(fid, name, rank):
    return {"attributes": {"release": name, "download_count": rank, "feature_details": {"parent_imdb_id": 1234},
                           "files": [{"file_id": fid, "file_name": name + ".srt"}]}}


def stamp(n):
    return f"00:{n // 60:02d}:{n % 60:02d},000"


def lines(prefix, n):
    return " ".join(prefix + chr(97 + i // 26) + chr(97 + i % 26) for i in range(n))


def subs(last, prefix):
    return "\n\n".join(f"{i}\n{stamp(t)} --> {stamp(t+2)}\n{lines(prefix + chr(97 + i % 26), 12)}"
                       for i, t in enumerate(range(8, last, 25), 1))


downloads = []


class Provider:
    configured = True

    def search(self, **kwargs):
        return [result(1, "Show.S01E01.S01E02.Foreign.Exchange", 100),
                result(4, "Show.S01E01.Foreign.Exchange.Duplicate", 95),
                result(2, "Show.S01E01.Foreign.Exchange", 90),
                result(3, "Show.S01E01.Foreign.Exchange.SDTV", 80)]

    def login(self):
        pass

    def download(self, fid):
        downloads.append(fid)
        return subs(1180 if fid == 2 else 640, "other" if fid == 4 else "own")

    def close(self):
        pass


client = TestClient(main.app)
_save(987654, "S01E02", {"source": "opensubtitles", "text": " ".join(lines("other" + chr(97 + i % 26), 12)
                                                                      for i in range(1, 27))})
with patch.object(main, "OpenSubtitles", Provider):
    base = f"/api/series/{sid}/references/S01E01"
    preview = client.post(base + "/candidate")
    assert preview.status_code == 200, preview.text
    candidate = preview.json()["candidate"]
    assert candidate and candidate["release"].endswith("SDTV") and downloads == [4, 2, 3]
    assert set(load_refs(987654)) == {"S01E02"}
    saved = client.post(base + "/text", json={"text": candidate["text"]})
    assert saved.status_code == 200 and load_refs(987654)["S01E01"]["source"] == "manual:S01E01.srt"

with patch("app.media_text.subprocess.run") as run:
    run.return_value.stdout = b'{"chapters":[{"start_time":"0"},{"start_time":"42"},{"start_time":"42"},{"start_time":"199"}]}'
    assert chapter_hints(root / "file.mkv", 200) == [42.0]
print("PASS (combined and oversized subtitles rejected, preview is read-only, boundary hints only)")
