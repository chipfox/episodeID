#!/usr/bin/env python3
"""A manual title-card check must report one definitive result, not just frames."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app.titlecard import _inspection_result

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
