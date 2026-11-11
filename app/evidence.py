"""Which episodes' references are too alike to tell their episodes apart.

Two episodes can share one transcript — a wiki page filed under the wrong number,
or both parts of a two-parter recorded once — and then neither reference can prove
its own episode. Pretending otherwise is how a confident wrong answer gets made,
so the matcher, the plan and the reference repair all read the verdicts from here.

Document sameness is measured by `app/textsim.py`, the one view the repair path
uses too, so the check that calls a pair a duplicate and the check that replaces
it can never disagree about what "the same transcript" means.
"""
from __future__ import annotations

import re

from . import textsim

try:  # the same English stopwords every other text view in this app uses
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOP = frozenset(ENGLISH_STOP_WORDS)
except Exception:  # noqa: BLE001 — filtering degrades to counting every word
    _STOP = frozenset()

_TOKEN = re.compile(r"(?u)\b[a-z][a-z']+\b")

MIN_CONFLICT_WORDS = 200     # a two-line stub is not evidence of a duplicate
STORY_PAIR_SIMILARITY = 0.6  # parts of one story legitimately share a cold open


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


class ReferenceConflicts:
    """Pairs of episodes whose reference texts cannot be told apart.

    conflicts: near-duplicates (a wrong or shared transcript). Neither can confirm
    its episode over the other, so a match on one is not evidence for either.
    story_pairs: parts of one story whose references overlap. Both genuinely
    resemble the file, so they never block anything — but their transcripts are
    worth re-fetching apart.
    """

    def __init__(self, refs: dict[str, str], story=None,
                 conflict_similarity: float = 0.85):
        self._refs = dict(refs)
        self._story = story or (lambda code: [code])
        self._conflicts: dict[str, list[tuple[str, float]]] = {}
        self._story_pairs: dict[str, list[tuple[str, float]]] = {}
        self._find_conflicts(float(conflict_similarity))

    def _find_conflicts(self, threshold: float) -> None:
        codes = sorted(code for code, text in self._refs.items()
                       if len(_tokens(text)) >= MIN_CONFLICT_WORDS)
        if len(codes) < 2:
            return
        sims = textsim.pairwise([self._refs[code] for code in codes])
        stories = [set(self._story(code)) for code in codes]
        for i, a in enumerate(codes):
            for j in range(i + 1, len(codes)):
                b = codes[j]
                sim = round(float(sims[i][j]), 3)
                if stories[i] & stories[j]:
                    # Two parts of one story may share a cold open, but a shared
                    # majority of the transcript means the reference is the whole
                    # story filed twice — worth replacing with per-part copies.
                    if sim >= STORY_PAIR_SIMILARITY:
                        self._story_pairs.setdefault(a, []).append((b, sim))
                        self._story_pairs.setdefault(b, []).append((a, sim))
                    continue
                if sim < threshold:
                    continue
                self._conflicts.setdefault(a, []).append((b, sim))
                self._conflicts.setdefault(b, []).append((a, sim))

    def conflicting(self, code: str) -> list[str]:
        return [other for other, _ in sorted(self._conflicts.get(code) or [],
                                             key=lambda item: -item[1])]

    def story_pair_for(self, code: str) -> list | None:
        """(other code, similarity) when this episode's reference overlaps its own
        story part's too much to tell them apart."""
        items = self._story_pairs.get(code) or []
        return list(max(items, key=lambda item: item[1])) if items else None

    def story_pair_conflicts(self) -> list[dict]:
        """Story parts whose references overlap each other's too much to tell the
        parts apart. Unlike plain conflicts these never block a question — both
        parts genuinely resemble the file — but their transcripts are worth
        re-fetching apart."""
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for a, items in self._story_pairs.items():
            for b, sim in items:
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"codes": list(key), "similarity": sim, "story": True})
        return sorted(out, key=lambda pair: -pair["similarity"])

    def conflict_pairs(self) -> list[dict]:
        seen: set[tuple[str, str]] = set()
        out: list[dict] = []
        for a, items in self._conflicts.items():
            for b, sim in items:
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"codes": list(key), "similarity": sim})
        return sorted(out, key=lambda pair: -pair["similarity"])
