"""Typed decision model (Laya) as a fast candidate adjudicator.

Laya is not a chat model and never supplies prose evidence. It receives a
small, bounded packet and chooses one of at most four candidate episodes or
``unknown``. Its answer is always a suggestion: the planner must keep it
provisional and unticked.
"""
from __future__ import annotations

from typing import Iterable

import httpx

from .config import get_settings

UNKNOWN = "unknown"


class DecisionError(RuntimeError):
    pass


def available() -> bool:
    s = get_settings()
    return bool(s["decision_enabled"] and s["decision_base_url"] and s["decision_model"])


def _endpoint(settings: dict | None = None) -> str:
    s = settings or get_settings()
    base = str(s["decision_base_url"] or "").strip().rstrip("/")
    if not base:
        raise DecisionError("DECISION_BASE_URL is not set")
    return base if base.endswith("/decisions") else f"{base}/decisions"


def _description(candidate: dict) -> str:
    """One short line the model can actually read: a pre-computed distinctive
    descriptor when the scan has one, otherwise title plus a clipped summary.
    The matcher's numeric score is deliberately absent — TF-IDF points mean
    nothing to the model and eat the tiny input budget."""
    if candidate.get("descriptor"):
        return str(candidate["descriptor"])
    parts = [candidate.get("title") or candidate["code"]]
    if candidate.get("overview"):
        parts.append(str(candidate["overview"]).strip()[:90])
    return " — ".join(parts)


def _excerpt(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= limit:
        return text
    sep = " … "
    if limit <= len(sep):
        return text[:limit]
    body = limit - len(sep)
    head = body * 2 // 3
    tail = body - head
    return f"{text[:head]}{sep}{text[-tail:]}"


def choose(dialogue: str, candidates: Iterable[dict],
           instructions: str | None = None) -> dict | None:
    """Choose one candidate or unknown.

    candidates: [{"code", "title", "overview", "score"}]. Returns
    {"code": str|None, "unknown": bool, "confidence": float,
     "probabilities": {code: float}, "model": str}, or None if the packet is empty.
    """
    s = get_settings()
    caps = max(1, int(s["decision_max_candidates"]))
    seen: list[dict] = []
    for candidate in candidates:
        code = str(candidate.get("code") or "").strip()
        if not code or code == UNKNOWN or any(c["code"] == code for c in seen):
            continue
        seen.append({**candidate, "code": code})
        if len(seen) >= caps:
            break
    if not dialogue.strip() or not seen:
        return None

    limit = max(200, int(s["decision_max_input_chars"]))
    choices = [{"value": c["code"], "description": _description(c)} for c in seen]
    choices.append({"value": UNKNOWN,
                    "description": "No candidate is supported by the evidence."})
    body = {
        "model": s["decision_model"],
        "input": _excerpt(dialogue, limit),
        "questions": [{
            "type": "choice",
            "name": "episode",
            "instructions": instructions or (
                "Choose the candidate episode whose title and summary best match the "
                "dialogue. Choose unknown when the dialogue does not clearly support a "
                "candidate. Do not guess between close candidates."
            ),
            "criteria": (
                "Prefer specific plot events, character names, and locations in the "
                "dialogue. A generic opening, recap, or theme is not evidence."
            ),
            "choices": choices,
        }],
    }
    try:
        response = httpx.post(_endpoint(s), json=body, timeout=max(1, int(s["decision_timeout"])))
    except (httpx.HTTPError, ValueError) as e:
        raise DecisionError(f"decision request failed: {e}") from e
    if response.status_code < 200 or response.status_code >= 300:
        raise DecisionError(f"decision request failed ({response.status_code}): "
                            f"{response.text[:300]}")
    try:
        payload = response.json()
    except ValueError as e:
        raise DecisionError("decision endpoint returned non-JSON") from e
    answers = payload.get("answers")
    answer = answers[0] if isinstance(answers, list) and answers else None
    if not isinstance(answer, dict):
        raise DecisionError("decision endpoint returned no answer")
    choice = str(answer.get("choice") or answer.get("value") or "").strip()
    valid = {c["code"] for c in seen}
    if choice and choice != UNKNOWN and choice not in valid:
        raise DecisionError(f"decision endpoint chose an unknown candidate: {choice}")
    probs = {}
    for item in answer.get("probabilities") or []:
        if isinstance(item, dict) and item.get("value"):
            try:
                probs[str(item["value"])] = float(item.get("probability") or 0.0)
            except (TypeError, ValueError):
                continue
    try:
        confidence = float(answer.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    return {"code": None if not choice or choice == UNKNOWN else choice,
            "unknown": not choice or choice == UNKNOWN,
            "confidence": confidence,
            "probabilities": probs,
            "model": payload.get("model") or s["decision_model"]}


def test_connection() -> str:
    s = get_settings()
    if not (s["decision_base_url"] and s["decision_model"]):
        raise DecisionError("Set a decision base URL and model first")
    result = choose("A test file contains two agents discussing a case.",
                    [{"code": "S01E01", "title": "Pilot", "overview": "The first episode."},
                     {"code": "S01E02", "title": "Second", "overview": "The second episode."}])
    if result is None:
        raise DecisionError("decision endpoint returned no result")
    picked = result["code"] or UNKNOWN
    return f"{result['model']} replied {picked} (confidence {result['confidence']:.2f})"
