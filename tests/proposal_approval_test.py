"""Preview is read-only; only the exact current proposal can be staged."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_proposal_approval"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402
from app import db, main, planner  # noqa: E402

root = W / "media" / "Example"
root.mkdir(parents=True, exist_ok=True)
src, weak, other = "Example S01E01.mkv", "Example S01E04.mkv", "Example S01E06.mkv"
for name in (src, weak, other):
    (root / name).write_bytes(b"video")

db.init()
sid = db.add_series(999001, "Example", "", str(root), None)
eps = [{"season": 1, "episode": n, "title": f"Title {n}", "runtime": 10} for n in range(1, 8)]
db.update_series(sid, episodes=eps)
scan = {"files": [
    {"rel": src, "expected": ["S01E01"], "status": "MISMATCH", "duration": 600, "hold": True,
     "text_source": "embedded #1", "segments": [{"code": "S01E02", "dialogue_code": "S01E02",
        "title_card": {"code": "S01E02"}, "confidence": "high", "start": 0, "end": 600, "score": .8}]},
    {"rel": weak, "expected": ["S01E04"], "status": "LOW_CONFIDENCE", "duration": 600,
     "text_source": "embedded #1", "segments": [{"code": "S01E05", "confidence": "low", "score": .2}]},
    {"rel": other, "expected": ["S01E06"], "status": "MISMATCH", "duration": 600,
     "text_source": "embedded #1", "segments": [{"code": "S01E07", "confidence": "high", "score": .8}]},
]}
scan_id = db.add_scan(sid, scan)
items = planner.build_plan(db.get_series(sid), scan)
by_source = {it["source"]: it for it in items}
assert by_source[src]["proposal"]["codes"] == ["S01E02"]
assert "proposal" not in by_source[weak]
assert by_source[other]["kind"] == "rename"
by_source[other]["selected"] = False
pid = db.add_plan(sid, scan_id, items)
client = TestClient(main.app)
url = f"/api/plans/{pid}/items/{by_source[src]['id']}/proposal"
initial = db.get_plan(pid)

if os.environ.get("EPID_BROWSER_PORT"):
    import uvicorn
    uvicorn.run(main.app, host="127.0.0.1", port=int(os.environ["EPID_BROWSER_PORT"]), lifespan="off")
    sys.exit(0)

def post(body):
    return client.post(url, json=body)

response = post({})
assert response.status_code == 200, response.text
preview = response.json()
assert [c["source"] for c in preview["changes"]] == [src]
assert db.latest_plan(sid) == initial and not db.get_series(sid)["options"].get("overrides")
assert client.post(f"/api/plans/{pid}/items/{by_source[weak]['id']}/proposal", json={}).status_code == 409

# Changing a draft selection, or replacing the physical source, invalidates a preview.
client.patch(f"/api/plans/{pid}/selection", json={"selected": {by_source[weak]["id"]: True}})
assert post({"token": preview["token"]}).status_code == 409
preview = post({}).json()
(root / src).write_bytes(b"replaced")
assert post({"token": preview["token"]}).status_code == 409

response = post({"token": post({}).json()["token"]})
assert response.status_code == 200, response.text
staged = {it["source"]: it for it in response.json()["items"]}
assert staged[src]["kind"] == "rename" and staged[src]["selected"]
assert staged[weak]["selected"] and not staged[other]["selected"]
assert (root / src).read_bytes() == b"replaced" and not db.list_applies(sid)
assert post({}).status_code == 409
print("PASS (proposal preview, stale rejection, staged selection and no media moves)")
