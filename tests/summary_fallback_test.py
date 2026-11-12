"""Planner decisions for a summary-only fallback."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_summary_fallback"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.planner import build_plan  # noqa: E402

root = W / "media" / "Show"
(root / "Season 1").mkdir(parents=True, exist_ok=True)
series = {"path": str(root), "tvdb_id": 999994, "year": "1993", "options": {},
          "episodes": [{"season": 1, "episode": 1, "title": "One", "runtime": 45},
                       {"season": 1, "episode": 2, "title": "Two", "runtime": 45}]}


def plan(name, expected, sm, **extra):
    rel = f"Season 1/{name}"
    (root / rel).touch()
    f = {"rel": rel, "expected": expected, "status": "NO_MATCH", "duration": 2700,
         "text_source": "remote parakeet", "scope": "full", "segments": [],
         "summary_match": sm, **extra}
    return {it["source"]: it for it in build_plan(series, {"files": [f]})}[rel]


def sm(code, auto, passes=True, mode="auto"):
    return {"code": code, "score": .27, "margin": .08, "terms": ["dolphin", "show"],
            "mode": mode, "passes": passes, "auto": auto, "calibrated": True}


# Auto gate: summary alone settles the file.
r = plan("Show - S01E01 - One.mkv", ["S01E01"], sm("S01E01", True))
assert r["kind"] == "ok" and r["decided_by"] == "summary", r

# Auto gate and a different episode: rename to what the summary says.
r = plan("Show - S01E02 - Wrong.mkv", ["S01E02"], sm("S01E01", True))
assert r["kind"] == "rename" and r["decided_by"] == "summary", r
assert "S01E01" in r["targets"][0]["path"], r

# Passes the suggestion gate but not the auto gate, no corroboration: review.
r = plan("Show - S01E02 - Wrong.mkv", ["S01E02"], sm("S01E01", False))
assert r["kind"] == "review" and r["decided_by"] != "summary", r
assert "summary suggests S01E01" in r["reason"], r

# A title card independently agrees: auto with the corroborating source named.
r = plan("Show - S01E02 - Wrong.mkv", ["S01E02"], sm("S01E01", False),
         title_cards=[{"code": "S01E01", "partial": False, "time": 12.0}])
assert r["kind"] == "rename" and r["decided_by"] == "summary+title", r

# A sidecar is a suggestion even if the numbers pass.
r = plan("Show - S01E02 - Wrong.mkv", ["S01E02"], sm("S01E01", False, mode="suggest"),
         text_source="sidecar Show.S01E02.srt")
assert r["kind"] == "review", r

print("summary_fallback_test: OK — 5 cases")
