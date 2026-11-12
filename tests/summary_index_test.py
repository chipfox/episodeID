"""Summary index, cleaning, windows, and fallback mode."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_summary_index"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.summary import (  # noqa: E402
    Match, SummaryIndex, clean_overview, calibration_note, match_file,
    summary_mode)

EPS = {
    "S01E01": {"season": 1, "episode": 1, "title": "Dolphin Show",
               "overview": "The boys buy balloons to kill dolphins at a dolphin show.\r\n"
                           "videos:\r\nBand - Song"},
    "S01E02": {"season": 1, "episode": 2, "title": "Blood Drive",
               "overview": "The boys donate blood at a blood drive and get more than they bargained for."},
    "S01E03": {"season": 1, "episode": 3, "title": "Science Stuff",
               "overview": "Daria helps the boys with a science project about volcanoes."},
}
REFS = {
    "S01E01": "balloon dolphin show balloon dolphin the boys at the dolphin show " * 20,
    "S01E02": "blood drive donate blood the boys at the blood drive nurse needle " * 20,
    "S01E03": "science project volcano daria helps the boys with a volcano " * 20,
}

assert clean_overview(EPS["S01E01"]["overview"]) == (
    "The boys buy balloons to kill dolphins at a dolphin show.")
assert clean_overview("") == ""

idx = SummaryIndex(EPS, REFS)
best = idx.best("we went to the dolphin show with balloons")
assert best and best.code == "S01E01", best
assert "dolphin" in best.terms, best
assert idx.best("nothing about any episode here") is not None  # low score, still a ranking

long_ref = ("filler " * 800) + "blood drive donate blood nurse needle " * 30
win = idx.best_windows(long_ref)
assert win and win.code == "S01E02", win
assert win.window > 0, win

# Fallback mode: no reference vs no subtitle track vs sidecar.
base = {"text_source": "remote parakeet", "scope": "full", "status": "NO_MATCH",
        "expected": ["S01E01"]}
assert summary_mode(base, {}, EPS) == "auto"
assert summary_mode({**base, "text_source": "sidecar Show.S01E01.srt"}, {}, EPS) == "suggest"
assert summary_mode({**base, "scope": "sample"}, {}, EPS) == "suggest"
assert summary_mode({**base, "text_source": "none"}, {}, EPS) == "off"
assert summary_mode({**base, "status": "OK"}, REFS, EPS) == "off"
assert summary_mode({**base, "status": "NO_MATCH"}, REFS, EPS) == "suggest"
assert summary_mode({**base, "text_source": "embedded #2"}, {}, EPS) == "auto"

cal = {"available": True, "min_score": 0.05, "min_margin": 0.0,
       "precision_lb": 0.995, "target": 0.95, "coverage": 1.0, "n_refs": 25,
       "precision": 1.0, "oof_lb": 1.0}
settings = {"summary_auto_precision": 0.99}
m = match_file("dolphin show balloons", {**base, "expected": ["S01E01"]}, idx, cal, "auto", settings)
assert m and m["passes"] and m["auto"] and m["code"] == "S01E01", m
assert "dolphin" in m["terms"]
assert "LB precision" in calibration_note(cal)
# A weak gate can still auto an OK when the summary agrees with the filename,
# but it must not auto a rename to a different episode.
weak = {**cal, "precision_lb": 0.81, "oof_ok": False}
same = match_file("dolphin show balloons", {**base, "expected": ["S01E01"]},
                  idx, weak, "auto", settings)
assert same and same["auto"] and same["passes"], same
diff = match_file("dolphin show balloons", {**base, "expected": ["S01E02"]},
                  idx, weak, "auto", settings)
assert diff and not diff["auto"] and diff["passes"], diff
print("summary_index_test: OK — 7 groups")
