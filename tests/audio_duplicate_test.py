"""Audio only corroborates a duplicate pair; approval stages without touching media."""
import os
import random
import sys
from pathlib import Path
from unittest.mock import patch

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_audio_duplicate"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from app import db, fingerprints, main, pipeline  # noqa: E402
from app.planner import build_plan, probe_audio_pairs  # noqa: E402

root = W / "media" / "Show"
root.mkdir(parents=True, exist_ok=True)
db.init()
sid = db.add_series(987456, "Show", "", str(root), None)
db.update_series(sid, episodes=[{"season": 1, "episode": 1, "title": "One", "runtime": 20}])


def file(name, score):
    p = root / name
    p.write_bytes(name.encode())
    st = p.stat()
    return {"rel": name, "expected": ["S01E01"], "status": "OK", "duration": 1200,
            "scope": "full", "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "segments": [{"code": "S01E01", "start": 0, "end": 1180, "score": score,
                          "words": 600, "confidence": "high"}]}


source = "Show - S01E01 - copy.mkv"
rival = "Show - S01E01 - original.mkv"
a, b = file(source, .8), file(rival, .95)

# Simulate raw subfingerprints with codec bit errors and a consistent offset.
rng = random.Random(67)
left = [[rng.getrandbits(32) for _ in range(620)] for _ in range(5)]
def shifted(windows):
    return [[0] * 9 + [v ^ rng.getrandbits(6) for v in seq[:-9]] for seq in windows]


right = shifted(left)
with patch.object(fingerprints, "_fingerprints", side_effect=[left, right]):
    assert fingerprints.compare(root, a, b) == "equivalent"
with patch.object(fingerprints, "_fingerprints", side_effect=[left, list(reversed(right))]):
    assert fingerprints.compare(root, a, b) != "equivalent"
with patch.object(fingerprints, "_fingerprints", side_effect=[left, [right[0], right[1], *left[1:4]]]), \
        patch.object(fingerprints, "_align", side_effect=[(.90, 0), (.88, 20), (.81, 0), (.84, 5), (.87, 2)]):
    assert fingerprints.compare(root, a, b) == "overlap"
with patch.object(fingerprints, "_fingerprints", side_effect=[left, right]), \
        patch.object(fingerprints, "_align", return_value=(.60, fingerprints.MAX_SHIFT)):
    assert fingerprints.compare(root, a, b) == "inconclusive"
with patch.object(fingerprints, "_fingerprints", side_effect=[left, [[0] * 620] * 5]):
    assert fingerprints.compare(root, a, b) == "inconclusive"
with patch.object(fingerprints, "_fingerprints") as fp:
    assert fingerprints.compare(root, a, {**b, "duration": 900}) == "inconclusive"
    fp.assert_not_called()

scan = {"files": [a, b]}
series = db.get_series(sid)
initial = build_plan(series, scan)
assert initial[0]["kind"] == "review" and initial[0]["audio_candidate"] == rival, initial


class Context:
    series_id = sid

    def check_cancel(self):
        pass


with patch("app.fingerprints.compare", return_value="equivalent") as probe:
    ctx = Context()
    scan["audio_pairs"] = probe_audio_pairs(root, initial, scan["files"], ctx)
    probe.assert_called_once_with(root, a, b, check_cancel=ctx.check_cancel)
    plan = build_plan(series, scan)
    assert probe.call_count == 1  # rebuilding the plan does not decode media
    assert not probe_audio_pairs(root, initial, [{**a, "mtime_ns": 1}, b], ctx)
    assert probe.call_count == 1
pair = scan["audio_pairs"][source]
different = build_plan(series, {**scan, "audio_pairs": {source: {**pair, "result": "different"}}})[0]
assert "No audio match" in different["reason"] and "proposal" not in different
assert len([i for i in plan if i["kind"] == "review"]) == 1
assert plan[0]["proposal"]["action"] == "aside" and not plan[0]["selected"]
sid_scan = db.add_scan(sid, scan)
pid = db.add_plan(sid, sid_scan, plan)
client = TestClient(main.app)
url = f"/api/plans/{pid}/items/{plan[0]['id']}/proposal"
preview = client.post(url, json={})
assert preview.status_code == 200, preview.text
assert db.get_series(sid)["options"] == {} and db.latest_plan(sid)["id"] == pid
done = client.post(url, json={"token": preview.json()["token"]})
assert done.status_code == 200, done.text
staged = next(i for i in done.json()["items"] if i["source"] == source)
assert staged["kind"] == "aside" and staged["selected"] and staged["audio_pair"]["rival"] == rival
assert "overrides" not in db.get_series(sid)["options"]
assert all((root / name).is_file() for name in (source, rival)) and not db.list_applies(sid)
params = {"plan_id": done.json()["id"]}
with patch.object(pipeline, "apply_plan", return_value={"applied": 0}) as executor, \
        patch.object(pipeline.autocheck, "record"):
    pipeline.job_apply(ctx, params)
    executor.assert_called_once()

# No approval on a replaced keeper; no proposal if a filename claims an unproved extra part.
(root / rival).write_bytes(b"replacement")
with patch.object(pipeline, "apply_plan") as executor:
    try:
        pipeline.job_apply(ctx, params)
    except RuntimeError as e:
        assert "reviewed audio duplicate changed" in str(e)
    else:
        raise AssertionError("stale audio match did not block Apply")
    executor.assert_not_called()
assert next(i for i in build_plan(db.get_series(sid), scan) if i["source"] == source)["kind"] == "review"
a["expected"].append("S01E02")
assert not any(i.get("audio_candidate") for i in build_plan(series, scan))
print("PASS (cached pair evidence, preview is read-only, stage does not move, stale keeper rejected)")
