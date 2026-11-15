#!/usr/bin/env python3
"""A manual title-card check must report one definitive result, not just frames."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.pipeline import _titlecard_scope_all, _titlecard_skip_reason
from app.titlecard import _diagnostic_hit, _inspection_result, cards_from_frames

assert "disabled globally" in _titlecard_skip_reason(False, "on", True)
assert "turned off" in _titlecard_skip_reason(True, "off", False)
assert _titlecard_skip_reason(True, "off", True) == ""
assert _titlecard_scope_all({"title_cards_scope": "unconfirmed"}, False) is False
assert _titlecard_scope_all({"title_cards_scope": "unconfirmed"}, True) is True
assert _titlecard_scope_all({"title_cards_scope": "all"}, False) is True

matched_hit = {"code": "S01E01", "title": "Pilot", "text": "PILOT", "score": 0.91,
               "partial": False, "candidates": []}
diagnostic_hit = _diagnostic_hit(matched_hit)
assert diagnostic_hit is not None and diagnostic_hit["score"] == 0.91
grouped = cards_from_frames([{"t": 22.0, "hit": diagnostic_hit}])
assert grouped and grouped[0]["code"] == "S01E01", grouped

full = [{"code": "S01E01", "title": "Pilot", "text": "PILOT", "time": 22.0,
         "frames": 4, "score": 0.91}]
partial = [{"code": None, "title": None, "text": "PILO", "time": 30.0, "frames": 1,
            "partial": True, "candidates": ["S01E01", "S01E02"]}]
matched = _inspection_result(full, [])
assert matched["state"] == "ok" and matched["status"] == "Title card matched", matched
assert "S01E01 — Pilot" in matched["headline"]

incomplete = _inspection_result(partial, [])
assert incomplete["state"] == "warn" and incomplete["status"] == "Partial title card", incomplete
assert incomplete["codes"] == ["S01E01", "S01E02"]

large_text = [{"t": 12.0, "lines": [("RANDOM TEXT", 0.91, 0.36, 3, 5, 7, 0.44)]}]
unmatched = _inspection_result([], large_text)
assert unmatched["state"] == "warn" and unmatched["status"] == "No known episode matched", unmatched

empty = _inspection_result([], [{"t": 12.0, "lines": [("tiny", 0.5, 0.03, 1, 2, 3, 0.04)]}])
assert empty["state"] == "info" and empty["status"] == "No title card found", empty

print("title-card results: manual checks report a single definitive verdict")
