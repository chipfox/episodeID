"""Regression tests for adaptive, windowed sample adjudication."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.media_text import detect_language
from app.pipeline import (_aggregate_sample_evidence, _sample_window_starts,
                          _story)

SETTINGS = {
    "min_score": 0.12,
    "min_margin": 0.04,
    "language": "en",
}


def _window(code, score, *, start=0, end=600, votes=None, whole=None):
    votes = [] if votes is None else votes
    if code:
        votes.append({"code": code, "score": score, "evidence": "segment"})
    return {
        "start": start,
        "end": end,
        "source": "remote",
        "words": 100,
        "has_text": True,
        "language": None,
        "votes": votes,
        "whole": whole or ([{"code": code, "score": score}] if code else []),
        "text": "dialogue",
    }


def test_window_plan_reaches_later_parts_without_overrunning_file():
    starts = _sample_window_starts(2700, 0, 600, 5)
    assert starts == [0, 1200, 600, 2100, 1800]  # 2400 clamps to 2100: the window must fit
    assert all(start + 600 <= 2700 for start in starts)
    assert len(starts) == len(set(starts))
    assert _sample_window_starts(450, 0, 600, 5) == [0]


def test_sparse_whole_window_can_confirm_the_filename():
    windows = [_window("S01E01", 0.27, votes=[{"code": "S01E01", "score": 0.27, "evidence": "whole"}])]
    agg = _aggregate_sample_evidence(["S01E01"], windows, {}, SETTINGS, 2)
    assert agg["status"] == "OK"
    assert agg["code"] == "S01E01"


def test_single_foreign_episode_vote_cannot_create_a_rename():
    windows = [_window("S01E02", 0.36)]
    agg = _aggregate_sample_evidence(["S01E01"], windows, {}, SETTINGS, 2)
    assert agg["status"] == "LOW_CONFIDENCE"


def test_two_independent_windows_are_required_for_a_sample_rename():
    windows = [_window("S01E02", 0.36, start=1200, end=1800),
               _window("S01E02", 0.43, start=600, end=1200)]
    agg = _aggregate_sample_evidence(["S01E01"], windows, {}, SETTINGS, 2)
    assert agg["status"] == "MISMATCH"
    assert agg["code"] == "S01E02"
    assert agg["votes"] == 2


def test_story_parts_do_not_count_as_opposing_evidence():
    eps = {
        "S09E01": {"season": 9, "episode": 1, "title": "Nothing Important Happened Today (1)"},
        "S09E02": {"season": 9, "episode": 2, "title": "Nothing Important Happened Today (2)"},
    }
    assert _story(eps, "S09E01") == ["S09E01", "S09E02"]
    windows = [_window("S09E02", 0.45)]
    agg = _aggregate_sample_evidence(["S09E01"], windows, eps, SETTINGS, 2)
    assert agg["status"] == "NO_MATCH"
    assert "story part" in agg["note"]
    assert not [c for c in agg["candidates"] if c["evidence"] == "sample"]


def test_non_target_audio_is_reported_instead_of_guessed():
    spanish = " ".join(["el que una para los con del pero esta son por como más"] * 10)
    assert detect_language([spanish], "en") == "es"
    assert detect_language(["the truth is out there and the files explain what happened here"] * 3, "en") is None
    windows = [
        {"start": 0, "end": 600, "source": "remote", "words": 100, "has_text": True,
         "language": "es", "votes": [], "whole": [], "text": spanish},
        {"start": 1200, "end": 1800, "source": "remote", "words": 100, "has_text": True,
         "language": "es", "votes": [], "whole": [], "text": spanish},
    ]
    agg = _aggregate_sample_evidence(["S05E15"], windows, {}, SETTINGS, 2)
    assert agg["status"] == "NO_MATCH"
    assert agg["language"] == "es"
    assert "not en" in agg["note"], agg["note"]  # names the chosen language, not "English"


for _name, _fn in sorted(globals().copy().items()):
    if _name.startswith("test_") and callable(_fn):
        _fn()
print("sample_confirmation_test: OK")
