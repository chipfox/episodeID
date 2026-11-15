"""Dialogue detector status must reflect evidence strength."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_dialogue_status"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings  # noqa: E402
from app import planner  # noqa: E402

settings = get_settings()
series = {"path": "/tmp", "tvdb_id": 999996, "options": {}, "episodes": []}


def dialogue_state(score, source="embedded #2"):
    f = {
        "rel": "Season 6/file.mkv",
        "expected": ["S06E13"],
        "status": "OK",
        "text_source": source,
        "evidence": {
            "top1": "S09E07",
            "top1_score": score,
            "windows": 20,
            "words": 100,
        },
        "segments": [],
    }
    methods = planner._detection_methods(f, settings, series, False, "")
    return next(method for method in methods if method["key"] == "dialogue")


weak = dialogue_state(0.06)
assert weak["state"] == "warn", weak
assert "threshold" in weak["detail"], weak

strong = dialogue_state(0.64)
assert strong["state"] == "ok", strong

conflicted = dialogue_state(0.64)
f_conflicted = {
    "rel": "Season 6/file.mkv",
    "expected": ["S06E13"],
    "status": "OK",
    "text_source": "embedded #2",
    "evidence": {
        "top1": "S09E07",
        "top1_score": 0.64,
        "conflicted": True,
        "twin": "S06E13",
        "windows": 20,
        "words": 100,
    },
    "segments": [],
}
assert next(method for method in planner._detection_methods(
    f_conflicted, settings, series, False, "")
    if method["key"] == "dialogue")["state"] == "warn"

sidecar = dialogue_state(0.64, source="sidecar srt")
assert sidecar["state"] == "warn", sidecar

print("dialogue status: weak matches and sidecar text stay amber")
