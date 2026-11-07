"""Building and applying a bounded decision-model adjudication.

The matcher still owns the full reference index and produces the candidate
pool. This module only turns that pool into a compact packet for Laya and
records Laya's choice as provisional evidence.
"""
from __future__ import annotations

from pathlib import Path

from . import decision
from .config import get_settings


def candidates(file_result: dict, episodes: dict, limit: int) -> list[dict]:
    """Top candidates from the matcher, with the filename's claim included."""
    order: list[str] = []
    for whole in file_result.get("whole") or []:
        order.append(whole.get("code"))
    for segment in file_result.get("segments") or []:
        order.append(segment.get("code"))
        for alt in segment.get("alternatives") or []:
            order.append(alt.get("code"))
    order.extend(file_result.get("expected") or [])
    out: list[dict] = []
    for code in order:
        if not code or code not in episodes or any(c["code"] == code for c in out):
            continue
        score = next((w.get("score") for w in file_result.get("whole") or []
                      if w.get("code") == code), None)
        out.append({"code": code, "title": episodes[code].get("title") or code,
                    "overview": (episodes[code].get("overview") or "")[:180],
                    "score": score})
        if len(out) >= limit:
            break
    return out


def dialogue_text(cues: list, limit: int = 1400) -> str:
    text = " ".join(str(cue[2]) for cue in cues if len(cue) > 2 and cue[2])
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]} … {text[-(limit - head):]}"


def apply(file_result: dict, result: dict) -> None:
    """Attach a decision result without ever making it trusted evidence."""
    code = result.get("code")
    file_result["decision"] = {
        "engine": "laya",
        "model": result.get("model"),
        "code": code,
        "confidence": result.get("confidence"),
        "probabilities": result.get("probabilities") or {},
        "unknown": bool(result.get("unknown")),
    }
    if not code:
        return
    segments = file_result.get("segments") or []
    if not segments:
        segments = [{"start": 0, "end": float(file_result.get("duration") or 0),
                     "score": 0.0, "confidence": "provisional", "scope": "sample",
                     "alternatives": [], "why": "decision model candidate"}]
        file_result["segments"] = segments
    for segment in segments:
        segment["scope"] = file_result.get("scope") or "sample"
        segment["confidence"] = "provisional"
        segment["code"] = code
        segment["decision_code"] = code
    file_result["sample_suggestion"] = code


def adjudicate(series: dict, file_result: dict, episodes: dict, ctx=None) -> dict | None:
    """Ask Laya about one sample-scoped file. Returns None when unavailable."""
    if not decision.available():
        return None
    limit = int(get_settings()["decision_max_candidates"])
    pool = candidates(file_result, episodes, limit)
    if not pool:
        return None
    text = file_result.get("sample_text") or dialogue_text(file_result.get("cues") or [])
    if not text:
        return None
    try:
        result = decision.choose(text, pool)
    except decision.DecisionError as e:
        if ctx:
            ctx.log(f"  decision: {Path(file_result.get('rel', 'file')).name} — {e}")
        return None
    if result:
        apply(file_result, result)
        if ctx:
            what = result.get("code") or "unknown"
            ctx.log(f"  decision: {Path(file_result.get('rel', 'file')).name} → {what} "
                    f"(confidence {result.get('confidence', 0):.2f})")
    return result
