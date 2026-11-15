#!/usr/bin/env python3
"""A manual title-card check must report one definitive result, not just frames."""
import pathlib
import sys
import tempfile
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from app import db, pipeline
from app.pipeline import (_titlecard_effective_mode, _titlecard_scope_all,
                          _titlecard_skip_reason)
from app.titlecard import _diagnostic_hit, _inspection_result, cards_from_frames

assert "disabled globally" in _titlecard_skip_reason(False, "on", True)
assert "turned off" in _titlecard_skip_reason(True, "off", False)
assert _titlecard_skip_reason(True, "off", True) == ""
assert _titlecard_effective_mode("auto", False) == "auto"
assert _titlecard_effective_mode("auto", True) == "on"
assert _titlecard_effective_mode("off", True) == "on"
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


class _CheckCtx:
    job_id = 1

    def __init__(self, series_id):
        self.series_id = series_id
        self.statuses = []
        self.logs = []
        self.cancel_checks = 0

    def status(self, message):
        self.statuses.append(message)

    def log(self, message):
        self.logs.append(message)

    def check_cancel(self):
        self.cancel_checks += 1


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = pathlib.Path(tmp)
    db.DB_PATH = tmp_path / "test.db"
    db.CONFIG_DIR = tmp_path
    db._local = threading.local()
    db.init()
    root = tmp_path / "series"
    root.mkdir()
    (root / "Pilot.mkv").write_bytes(b"")
    sid = db.add_series(999, "Check Show", "2020", str(root), None,
                        {"include_specials": False})

    class _FakeIndex:
        def __init__(self, *args, **kwargs):
            pass

    original_available = pipeline.titlecard.available
    original_inspect = pipeline.titlecard.inspect
    original_safe = pipeline.safe_media_path
    original_index = pipeline.titlecard.TitleIndex
    try:
        pipeline.titlecard.available = lambda: True
        pipeline.safe_media_path = lambda value: pathlib.Path(value)
        pipeline.titlecard.TitleIndex = _FakeIndex

        def fake_inspect(video, start, length, index, decode, mode):
            return {
                "source": video.name, "start": start, "fps": 1, "count": 1,
                "mode": mode, "decode": decode, "ocr": "Test OCR",
                "cards": full, "frames": [{"t": 22.0, "size": [1280, 720],
                                           "lines": [], "hit": diagnostic_hit}],
                "result": matched,
            }

        pipeline.titlecard.inspect = fake_inspect
        ctx = _CheckCtx(sid)
        result = pipeline.job_titlecard_check(
            ctx, {"source": "Pilot.mkv", "start": 0, "end": 90,
                  "decode": "auto", "mode": "key"})
        assert result["result"]["status"] == "Title card matched", result
        assert any("Title card check: Pilot.mkv" in line for line in ctx.logs), ctx.logs
        assert any("Title card result: Title card matched — S01E01 — Pilot"
                   in line for line in ctx.logs), ctx.logs
        assert any("S01E01 Pilot" in line for line in ctx.logs), ctx.logs
        assert any("Title card check finished" in line for line in ctx.logs), ctx.logs
        assert ctx.statuses == ["Title cards: Pilot.mkv"], ctx.statuses
        assert ctx.cancel_checks >= 2, ctx.cancel_checks
    finally:
        pipeline.titlecard.available = original_available
        pipeline.titlecard.inspect = original_inspect
        pipeline.safe_media_path = original_safe
        pipeline.titlecard.TitleIndex = original_index
        conn = getattr(db._local, "conn", None)
        if conn is not None:
            conn.close()
        db._local = threading.local()

print("title-card results: manual checks report a single definitive verdict")
