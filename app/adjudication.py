"""Adjudication: fast path proposes candidates; the decision model picks.

Candidate pool = the matcher's top whole-file hits (from a first ten-minute
sample), with the filename's claim guaranteed a slot, minus anything whose
reference is a near-duplicate of the claim. The evidence packet = the few sample
sentences that actually speak for the claim and for the strongest rivals, each
tagged with the window it was heard in. Laya never sees the file path, so it
cannot read the answer off the filename; the claim belongs in the instructions,
the way it belongs to a human reviewer, not to the model's eyes.

Laya's answer never applies itself: it only annotates the file
(``ai_code``/``ai_confidence``) for the planner. The planner decides what an
answer means (provisional match, rename suggestion, or review)."""
from __future__ import annotations

from pathlib import Path

from . import decision, evidence
from .config import get_settings
from .references import ep_code


def dialogue_text(cues: list, limit: int = 1400) -> str:
    """Clip dialogue so the model prompt stays small: head and tail, never a
    middle — the opening scene and the closing lines are where episodes
    differ most within a limited budget. Cues are [start, end, text] lists
    (the cache and Matcher shape) or dicts with a text key."""
    parts = []
    for cue in cues:
        text = cue.get("text") if isinstance(cue, dict) else (cue[2] if len(cue) > 2 else "")
        if text:
            parts.append(str(text).strip())
    text = " ".join(" ".join(parts).split())
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return f"{text[:head]} … {text[-(limit - head):]}"


def candidates(file_result: dict, episodes: dict[str, dict], limit: int,
               lexicon: evidence.Lexicon | None = None,
               blocked: set[str] | None = None) -> list[dict]:
    """The packet's candidate list.

    The claimed episode always keeps a slot: a wrongly-named file can only be
    fixed by an answer that *considered* it, and four stronger-looking wrong
    answers must not push the claim out of the packet. ``blocked`` holds
    episodes whose reference is a near-duplicate of another candidate's —
    asking the model to choose between two copies of one transcript is asking
    it to guess."""
    blocked = set(blocked or ())

    def entry(code: str) -> dict:
        ep = episodes[code]
        row = {"code": code, "title": ep.get("title") or code,
               "overview": (ep.get("overview") or "")[:180],
               "score": next((c.get("score") for c in (file_result.get("whole") or [])
                              if c.get("code") == code), 0.0)}
        if lexicon is not None:
            row["descriptor"] = lexicon.descriptor(code)
        return row

    order: list[str] = []
    for source in (file_result.get("whole") or [], file_result.get("segments") or []):
        for cand in sorted(source, key=lambda c: -float(c.get("score") or 0)):
            code = cand.get("code")
            if code and code not in order:
                order.append(code)
    order.extend(code for code in (file_result.get("expected") or []) if code)

    out: list[dict] = []
    for code in order:
        if not code or code not in episodes or code in blocked:
            continue
        if len(out) >= max(1, limit - 1):
            break
        out.append(entry(code))
    claim = next((code for code in (file_result.get("expected") or [])
                  if code in episodes and code not in blocked), None)
    if claim and all(candidate["code"] != claim for candidate in out):
        out.insert(0, entry(claim))
    return out


def apply(file_result: dict, result: dict | None, accept: float) -> None:
    """Store the model's suggestion on the file result (never a decision).

    ``code`` is the chosen episode code or None (unknown); confidence is the
    model's own probability for that choice. A weak answer is recorded as an
    explicit ``NO_MATCH`` so the UI can say why there is no AI suggestion."""
    if not result:
        return
    file_result["ai_model"] = result.get("model") or "decision"
    file_result["ai_probabilities"] = result.get("probabilities") or {}
    if result.get("unknown") or not result.get("code"):
        file_result["ai_code"] = "NO_MATCH"
        file_result["ai_confidence"] = float(result.get("confidence") or 0.0)
        return
    if float(result.get("confidence") or 0.0) < accept:
        file_result["ai_code"] = "LOW_CONFIDENCE"
        file_result["ai_confidence"] = float(result.get("confidence") or 0.0)
        return
    file_result["ai_code"] = str(result["code"])
    file_result["ai_confidence"] = float(result.get("confidence") or 0.0)


def adjudicate(series: dict, file_result: dict, episodes: dict[str, dict], ctx=None,
               lexicon: evidence.Lexicon | None = None) -> dict | None:
    """One decision call for one file, or None when there is nothing to ask.

    With a lexicon the packet becomes query-focused: the sentences that carry
    the claim episode's distinctive terms lead, the rivals' best lines follow,
    and each line shows the window it came from. Without one (or when none of
    the sample lines is distinctive) the caller's old head/tail excerpt is used,
    so turning the feature off restores exactly the previous behaviour."""
    if not decision.available():
        return None
    s = get_settings()
    limit = int(s["decision_max_candidates"])
    expected = next((code for code in (file_result.get("expected") or [])
                     if code in episodes), None)
    conflict = lexicon.conflicting(expected) if (lexicon is not None and expected) else []
    if conflict:
        file_result["reference_conflict"] = conflict[:2]
        if ctx:
            ctx.log(f"  decision: {Path(file_result['rel']).name} — not asked: the reference for "
                    f"{expected} is a near-duplicate of {conflict[0]}'s; one transcript cannot "
                    "settle either episode")
        return None
    pool = candidates(file_result, episodes, limit, lexicon=lexicon)
    if not pool:
        return None
    budget = max(200, int(s["decision_max_input_chars"]))
    text = ""
    if lexicon is not None and s["decision_evidence"]:
        text = evidence.select_evidence(file_result.get("sample_text") or "", lexicon,
                                        [candidate["code"] for candidate in pool],
                                        budget - 140)
    if not text:
        text = file_result.get("sample_text") or dialogue_text(file_result.get("cues") or [])
    if not text:
        return None
    instructions = None
    if expected:
        instructions = (
            f"The filename claims this file is {expected}. Pick the candidate episode whose "
            "title and summary the dialogue fits; pick unknown when none of them fits. Each "
            "evidence line is tagged with the point in the file where it was heard.")
    try:
        result = decision.choose(text, pool, instructions=instructions)
    except decision.DecisionError as exc:
        if ctx:
            ctx.log(f"  decision: {Path(file_result['rel']).name}: {exc}")
        return None
    apply(file_result, result, float(s["decision_min_confidence"]))
    return result
