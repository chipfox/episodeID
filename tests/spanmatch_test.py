"""Local dialogue matches ignore extra reference text without inventing episode labels."""
import json
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_spanmatch"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from app import db, main, spanmatch  # noqa: E402
from app.planner import build_plan  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.matcher import Matcher  # noqa: E402
from app.media_text import _cache_file  # noqa: E402
from app.pipeline import _guard_combined_match, _label_local_from_card  # noqa: E402
from app.references import _save  # noqa: E402


def words(prefix, count):
    return [prefix + chr(97 + i // 26) + chr(97 + i % 26) for i in range(count)]


def cues(tokens, start=0, end=600):
    chunks = [tokens[i:i+6] for i in range(0, len(tokens), 6)]
    step = (end-start) / len(chunks)
    return [[start + i*step, start + (i+.92)*step, " ".join(chunk)] for i, chunk in enumerate(chunks)]


a, b, c, noise = words("apple", 350), words("beach", 210), words("circus", 280), words("noise", 300)
ref = " ".join(c + a)
local = spanmatch.find_span(cues(a), ref, 600)
assert local["whole_file"] and local["file_coverage"] > .98 and local["reference_start"] >= 270, local
assert spanmatch.reference_passage(ref.upper(), local).split()[20] == a[8].upper()
short = spanmatch.find_span(cues(noise, 0, 220) + cues(a[:110], 220, 360), ref, 600)
assert not short["whole_file"], short
assert spanmatch.find_span(cues(a[:50] * 5), ref, 600) is None

code_a, code_b, broad = "S01E01", "S01E02", "S01E05"
refs = {broad: ref, code_a: " ".join(a), code_b: " ".join(b), "S01E09": " ".join(noise)}

def detect(tokens, expected, anchors, end, source="embedded #1"):
    matcher = Matcher(refs, get_settings(), anchors={k: refs[k] for k in anchors})
    data = {"cues": tokens, "duration": end, "scope": "full", "source": source}
    res = {"whole": [{"code": broad}, {"code": code_a}, {"code": code_b}], "segments": [],
           "evidence": {"combined": {"state": "unresolved", "reference": broad}}}
    _guard_combined_match(res, expected, data, matcher)
    return res, data

unlabeled, data = detect(cues(a), [code_a], [], 600)
assert unlabeled["evidence"]["combined"]["state"] == "unresolved"
verified, _ = detect(cues(a), [code_a], [code_a], 600)
assert verified["evidence"]["combined"]["codes"] == [code_a]
double = cues(a) + cues(b, 600, 1200)
partial, full_data = detect(double, [code_a, code_b], [code_a], 1200)
ev = partial["evidence"]["combined"]
assert ev["local"]["code"] == code_a and ev["remaining"][0]["candidate"] == code_b, ev
assert ev["state"] != "verified" and "codes" not in ev
both, _ = detect(double, [code_a, code_b], [code_a, code_b], 1200)
assert both["evidence"]["combined"]["codes"] == [code_a, code_b], both
sidecar, _ = detect(cues(a), [code_a], [code_a], 600, "sidecar show.srt")
assert sidecar["evidence"]["combined"]["state"] != "verified"

file = {**unlabeled, "expected": [code_a], "text_source": data["source"], "scope": "full", "duration": 600,
        "title_cards": [{"code": code_a, "partial": False, "time": 25, "score": .95}]}
_label_local_from_card(file)
assert file["status"] == "OK" and file["evidence"]["combined"]["codes"] == [code_a]

root = W / "media" / "Show"
root.mkdir(parents=True, exist_ok=True)
video = root / "sample.mkv"
video.write_bytes(b"video")
cache = _cache_file(video)
cache.parent.mkdir(parents=True, exist_ok=True)
cache.write_text(json.dumps(full_data))
db.init()
sid = db.add_series(999202, "Show", "", str(root), None)
db.update_series(sid, episodes=[{"season": 1, "episode": n, "title": f"Title {n}", "runtime": 10} for n in (1, 2, 5)])
for code in (broad, code_a, code_b):
    _save(999202, code, {"source": "opensubtitles", "text": refs[code]})
db.add_scan(sid, {"files": [{**partial, "rel": video.name, "expected": [code_a, code_b], "scope": "full",
                             "status": "LOW_CONFIDENCE", "text_source": "embedded #1", "duration": 1200}]})
client = TestClient(main.app)
url = f"/api/series/{sid}/evidence"
matched = client.get(url, params={"rel": video.name, "focus": "matched"}).json()
gap = client.get(url, params={"rel": video.name, "focus": "remaining"}).json()
assert matched["reference"]["passage"] and matched["file"]["words"] < 400, matched.keys()
assert gap["code"] == code_b and gap["file"]["focused"]["kind"] == "remaining"
assert not gap["reference"]["passage"] and gap["file"]["words"] < 230

keep, _ = detect(cues(a), [code_a], [], 600)
keep["evidence"]["twin"] = code_a
scan = {"files": [{**keep, "rel": video.name, "status": "LOW_CONFIDENCE", "scope": "full",
                   "text_source": "embedded #1", "duration": 600, "expected": [code_a]}]}
scan_id = db.add_scan(sid, scan)
rows = build_plan(db.get_series(sid), scan)
assert rows[0]["proposal"]["action"] == "keep", rows
pid = db.add_plan(sid, scan_id, rows)
url = f"/api/plans/{pid}/items/{rows[0]['id']}/proposal"
preview = client.post(url, json={})
assert preview.status_code == 200, preview.text
assert db.get_series(sid)["options"] == {}  # preview is nonpersistent
done = client.post(url, json={"token": preview.json()["token"]})
assert done.status_code == 200, done.text
row = done.json()["items"][0]
assert row["kind"] == "ok" and not row["selected"] and not row.get("targets")
assert video.is_file() and not db.list_applies(sid)
print("PASS (local and residual matches, evidence-only untrusted, focused Compare)")
