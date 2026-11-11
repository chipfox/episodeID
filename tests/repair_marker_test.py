"""A repair marker only holds while the reference it was made for is still there.

"Already repaired" and "don't retry, the replacement was the other part" are only true
of the reference that was on disk when the marker was written. If that reference is
later replaced (wiki upgrade, manual upload, renumber) the marker must stop suppressing
repairs; legacy markers carry no hash and are treated as stale too.
"""
import json
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_repair_marker"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import references  # noqa: E402
from app.references import _ref_fingerprint, load_refs, repair_reference_conflicts  # noqa: E402


class Ctx:
    def log(self, msg):
        print("  log:", msg)

    def check_cancel(self):
        pass

    def status(self, _msg):
        pass


class NoOpenSubtitles:
    """Repair stops before any download; this test only exercises the markers."""
    configured = False
    remaining = None

    def login(self):
        raise AssertionError("must not try to log in when not configured")

    def close(self):
        pass


references.OpenSubtitles = NoOpenSubtitles

TEXT = " ".join(f"line {i} of a transcript that both episodes would share" for i in range(400))
for code in ("S01E01", "S01E02"):
    references._save(4242, code, {"source": "opensubtitles", "text": TEXT})

series = {"tvdb_id": 4242, "episodes": [{"season": 1, "episode": 1, "title": "One"},
                                        {"season": 1, "episode": 2, "title": "Two"}]}
repair_file = references._repair_file(4242)

# A stale marker (hash of some other text) is forgotten, and said so.
repair_file.write_text(json.dumps({"S01E01": {"keep": True, "ref": "0" * 16}}))
out = repair_reference_conflicts(series, Ctx())
assert out["conflicts"] == 1, out
left = json.loads(repair_file.read_text())
assert "S01E01" not in left, left

# A marker that matches its reference stays.
good = _ref_fingerprint(load_refs(4242)["S01E02"]["text"])
repair_file.write_text(json.dumps({"S01E02": {"keep": True, "why": "test", "ref": good}}))
repair_reference_conflicts(series, Ctx())
left = json.loads(repair_file.read_text())
assert left.get("S01E02", {}).get("ref") == good, left

# A legacy marker (no hash at all) is stale by definition.
repair_file.write_text(json.dumps({"S01E02": {"keep": True, "why": "old format"}}))
repair_reference_conflicts(series, Ctx())
assert "S01E02" not in json.loads(repair_file.read_text())
print("PASS (repair markers expire with their reference)")
