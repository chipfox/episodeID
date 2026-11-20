"""A filename whose episode code the TVDB list does not have used to be a dead-end
review ("reload the list"), even when the file's own dialogue decisively settled an
episode the list does have. The evidence must be applicable: such a row becomes an
unticked rename to the episode the evidence names — while weaker evidence (AI guess,
a second candidate, a reference conflict, a rival file that owns the episode) stays
a review.

Shapes:
1. Unknown filename code + high-confidence dialogue → unticked rename, applied
   together with the file's sidecar files.
2. Same file but the reference is in conflict → still the review asking to reload.
3. Evidence names two episodes → review (one code, or nothing).
4. The dialogue answer came from the AI → review.
5. Another file already owns S01E03 → no rename of this file.
6. The one-click UI path: POST /overrides with the evidence code pre-ticks a rename.
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_unknown_code"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient  # noqa: E402

from app import db, main, pipeline, planner  # noqa: E402
from app.jobs import JobContext  # noqa: E402

root = W / "media" / "tv" / "Show"
(root / "Season 01").mkdir(parents=True, exist_ok=True)
(root / "Season 02").mkdir(parents=True, exist_ok=True)

# Season 1 has exactly three episodes — S01E04 is not in the list, like the live
# Beavis TVDB list that stops at S01E03 "Balloon".
eps = [{"season": 1, "episode": e, "title": f"Title 1x{e:02d}", "aired": "2020-01-01"}
       for e in range(1, 4)]
series = {"path": str(root), "tvdb_id": 1, "options": {}, "episodes": eps}

UNKNOWN = "Season 01/Show - S01E04 - Balloon.mkv"


def file_row(rel=UNKNOWN, expected=("S01E04",), segments=None, **extra):
    return {"rel": rel, "expected": list(expected), "status": "MISMATCH", "duration": 750,
            "text_source": "remote parakeet",
            "segments": segments if segments is not None else
            [{"start": 0.0, "end": 750.0, "code": "S01E03", "score": 0.62, "margin": 0.55,
              "confidence": "high", "words": 684}],
            **extra}


(root / UNKNOWN).write_bytes(b"x")
plan = {it["source"]: it for it in planner.build_plan(series, {"files": [file_row()]})}

# 1. Decisive dialogue over an unknown filename code: an applicable rename, unticked.
a = plan[UNKNOWN]
assert a["kind"] == "rename", a
assert a["codes"] == ["S01E03"], a["codes"]
assert a["selected"] is False, "unknown-numbering renames wait for an explicit tick"
assert "isn't in the episode list" in a["reason"] and "S01E03" in a["reason"], a["reason"]
assert a["targets"][0]["path"].startswith("Season 01/") and "S01E03" in a["targets"][0]["path"], a
assert a["expected"] == ["S01E04"], "the old name stays on the row for the log"


def build_one(**kw):
    return planner.build_plan(series, {"files": [file_row(**kw)]})[0]


# 2. A conflicted reference cannot settle a rename: keep the reload-the-list review.
b = build_one(reference_conflict=["S01E02"])
assert b["kind"] == "review", b
assert "Reload the episode list" in b["reason"], b["reason"]

# 2b. But if that episode's reference is actively disputed (OpenSubtitles numbering
#     unverified), a rename would rest on the very source a renumber poisons → review.
refdir = W / "cache" / "refs" / "1"
refdir.mkdir(parents=True, exist_ok=True)
(refdir / "S01E03.json").write_text('{"source": "opensubtitles", "text": "x y z", "title_verified": false}')
b2 = build_one()
assert b2["kind"] == "review", b2
assert "Reload the episode list" in b2["reason"], b2["reason"]
(refdir / "S01E03.json").unlink()

# 3. Evidence naming two episodes is not "exactly one episode".
c = build_one(segments=[{"start": 0.0, "end": 375.0, "code": "S01E02", "score": 0.6,
                         "margin": 0.5, "confidence": "high", "words": 300},
                        {"start": 375.0, "end": 750.0, "code": "S01E03", "score": 0.6,
                         "margin": 0.5, "confidence": "high", "words": 300}])
assert c["kind"] == "review", c

# 4. An AI answer over an unknown code stays a review (the old behaviour).
d = build_one(segments=[{"start": 0.0, "end": 750.0, "code": None, "score": 0.0, "margin": 0.0,
                         "confidence": "low", "words": 0,
                         "llm": {"code": "S01E03", "confidence": 0.9}}])
assert d["kind"] == "review", d

# 5. A rival file already named for S01E03 owns the episode; this file gets no rename.
OWNED = "Season 01/Show - S01E03 - Title 1x03.mkv"
(root / OWNED).write_bytes(b"x")
rows = planner.build_plan(series, {"files": [file_row(),
                                             file_row(rel=OWNED, expected=("S01E03",))]})
e = next(it for it in rows if it["source"] == UNKNOWN)
assert e["kind"] != "rename", e
assert e["kind"] == "review", e  # the two texts cannot be compared here

# 6. Applying the promoted rename moves the file AND its sidecar files. The rival
#    file that was in the way goes to the backup folder WITH its own sidecars —
#    a stale .srt left on the path would re-pair with the renamed file.
(root / UNKNOWN).write_bytes(b"x")
(root / UNKNOWN).with_suffix(".en.srt").write_text("1\n00:00:01,00 --> 00:00:04,00\nmeega\n")
(root / OWNED).write_bytes(b"x")
(root / OWNED).with_suffix(".en.srt").write_text("1\n00:00:01,00 --> 00:00:04,00\ncornhue\n")
db.init()
sid = db.add_series(1, "Show", "2020", str(root), None, options={})
db.update_series(sid, episodes=eps)
db.add_scan(sid, {"files": [file_row()], "counts": {}})
scan = db.latest_scan(sid)
assert scan is not None
series_now = db.get_series(sid)
assert series_now is not None
items = planner.build_plan(series_now, {"files": [file_row()]})
assert len(items) == 1 and items[0]["kind"] == "rename"
assert items[0]["selected"] is False, "unticked by default, the user ticks it"
items[0]["selected"] = True  # what ticking the row does
pid = db.add_plan(sid, scan["id"], items)
job = db.add_job("apply", sid, {"plan_id": pid})
result = pipeline.job_apply(JobContext(job, sid), {"plan_id": pid})
assert result["applied"] == 1, result
moved = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
assert any("S01E03" in p and p.endswith(".mkv") for p in moved), moved
assert any(p.endswith(".en.srt") and "S01E03" in p for p in moved), \
    f"sidecar must move with the video: {moved}"
assert not any("S01E04" in p for p in moved), moved
conflicts = sorted(str(p) for p in (W / "media" / "tv" / "_episodeid_backup").rglob("*")
                   if p.is_file())
assert any("Title 1x03.mkv" in p and "conflicts" in p for p in conflicts), conflicts
assert any("Title 1x03.en.srt" in p and "conflicts" in p for p in conflicts), \
    f"the moved-aside file must take its sidecars: {conflicts}"

# 7. The one-click path the row's "Rename to <code>" button uses: the override
#    endpoint settles the row as a pre-ticked rename to the evidence code.
root2 = W / "media" / "tv" / "Show2"
(root2 / "Season 01").mkdir(parents=True, exist_ok=True)
rel2 = "Season 01/Show2 - S01E04 - Balloon.mkv"
(root2 / rel2).write_bytes(b"x")
row2 = dict(file_row(), rel=rel2)
sid2 = db.add_series(2, "Show2", "2020", str(root2), None, options={})
db.update_series(sid2, episodes=eps)
db.add_scan(sid2, {"files": [row2], "counts": {}})
client = TestClient(main.app)
body = client.post(f"/api/series/{sid2}/overrides",
                   json={"source": rel2, "codes": ["S01E03"]}).json()
row = next(it for it in body["items"] if it["source"] == rel2)
assert row["kind"] == "rename" and row["codes"] == ["S01E03"], row
assert row["selected"] is True, "a manual decision pre-ticks its rename"
assert "S01E03" in row["targets"][0]["path"], row["targets"][0]["path"]
assert row["decided_by"] == "you", row["decided_by"]

print("PASS (unknown filename code + decisive evidence → applicable rename; "
      "weak variants stay review; sidecar rides along; one-click override pre-ticks)")
