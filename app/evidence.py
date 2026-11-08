"""Candidate-specific evidence selection for the decision packet.

The matcher says which episodes are plausible; this module decides which few
sentences of a sample actually prove something, and compresses each episode to
the one line that makes it distinctive. Laya reads a couple of hundred tokens
at a time, so the packet must carry the lines where the episodes *differ* —
the head and tail of whatever happened to be sampled mostly carries the show's
shared furniture, which is why the old packets kept picking the wrong episode
of an ambiguous pair.

The reference-conflict scan lives here too: when two episodes share one
transcript (a wiki page filed under the wrong number), neither reference can
prove its episode, and pretending otherwise manufactures confident answers.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from . import textsim

try:  # the same English stopwords every other vector view in this app uses
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    _STOP = frozenset(ENGLISH_STOP_WORDS)
except Exception:  # noqa: BLE001 — weights degrade to length-only filtering
    _STOP = frozenset()

_TOKEN = re.compile(r"(?u)\b[a-z][a-z']+\b")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WINDOW = re.compile(r"^\[(\d+)\u2013(\d+) min\]\s*(.*)$", re.S)

DESCRIPTOR_CHARS = 95      # title + the most distinctive sentence, trimmed
MIN_CONFLICT_WORDS = 200   # a two-line stub is not evidence of a duplicate
STORY_PAIR_SIMILARITY = 0.6  # parts of one story legitimately share a cold open;
                             # above this their references are worth re-fetching apart
TOP_TERMS = 80             # distinctive terms kept per episode


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def _sentences(text: str) -> list[str]:
    """Split transcript text into candidate lines; ASR without punctuation
    degrades to fixed word chunks so short samples still contribute evidence."""
    parts = [p.strip() for p in _SENTENCE.split(text or "") if p and p.strip()]
    if parts:
        return parts
    words = (text or "").split()
    return [" ".join(words[i:i + 30]) for i in range(0, len(words), 30)]


def _clip(sentence: str, limit: int) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit].rsplit(" ", 1)[0]
    return (cut or sentence[:limit]) + "…"


class Lexicon:
    """What each episode in a series is made of, from its own references.

    terms/weights: the words a reference makes distinctive for one episode —
    what we look for inside a sample, and what generic about them we ignore.
    descriptor: one short line telling a decision model what only this episode
    contains. conflicts: episodes whose references are near-duplicates of each
    other (a wrong or shared transcript); neither can confirm its episode over
    the other, so they must never face each other as rivals."""

    def __init__(self, eps: dict[str, dict], refs: dict[str, str], story=None,
                 conflict_similarity: float = 0.85):
        self._eps = eps
        self._refs = {code: text for code, text in refs.items() if code in eps}
        self._story = story or (lambda code: [code])
        self._weights: dict[str, dict[str, float]] = {}
        self._descriptors: dict[str, str] = {}
        self._conflicts: dict[str, list[tuple[str, float]]] = {}
        self._story_pairs: dict[str, list[tuple[str, float]]] = {}
        self._build_weights()
        self._build_descriptors()
        self._find_conflicts(float(conflict_similarity))

    # ------------------------------------------------------------------ terms
    def _build_weights(self) -> None:
        df: Counter = Counter()
        tf: dict[str, Counter] = {}
        for code in self._eps:
            ep = self._eps[code] or {}
            counts = Counter(_tokens(self._refs.get(code) or ""))
            for token in _tokens(ep.get("title") or ""):
                counts[token] += 3
            counts.update(_tokens(ep.get("overview") or ""))
            tf[code] = counts
            df.update(counts.keys())
        n = max(1, len(self._eps))
        for code, counts in tf.items():
            scored = [(token, (1.0 + math.log(count)) * math.log(1.0 + n / (1.0 + df[token])))
                      for token, count in counts.items() if len(token) > 3]
            scored.sort(key=lambda kv: -kv[1])
            self._weights[code] = dict(scored[:TOP_TERMS])

    def weights(self, code: str) -> dict[str, float]:
        return self._weights.get(code) or {}

    def terms(self, code: str) -> frozenset[str]:
        return frozenset(self.weights(code))

    # ------------------------------------------------------------ descriptors
    def _build_descriptors(self) -> None:
        for code, ep in self._eps.items():
            title = (ep.get("title") or code).strip()
            weights = self.weights(code)
            best, best_score = "", 0.0
            for sentence in _sentences(ep.get("overview") or ""):
                tokens = set(_tokens(sentence))
                if len(tokens) < 2:
                    continue
                score = sum(weights.get(token, 0.0) for token in tokens)
                if score > best_score:
                    best, best_score = sentence, score
            self._descriptors[code] = (f"{title} — {_clip(best, DESCRIPTOR_CHARS - len(title) - 3)}"
                                       if best and len(title) < DESCRIPTOR_CHARS - 12 else title)

    def descriptor(self, code: str) -> str:
        return self._descriptors.get(code) or code

    # -------------------------------------------------------------- conflicts
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


def select_evidence(sample_text: str, lexicon: Lexicon | None, codes: list[str],
                    char_budget: int) -> str:
    """Pick the sample sentences that speak for a candidate, tagged with when
    they were heard.

    Lines fitting the file's own claim lead, followed by the strongest line for
    each rival, so the decision model sees the best argument for the alternative
    instead of only the argument we already like. Sentences no candidate's
    distinctive terms touch are dropped — they are small talk, theme songs and
    the shared furniture of the show. Returns "" when the sample holds no
    distinctive lines at all, leaving the caller's old excerpt as fallback."""
    codes = [code for code in codes if code]
    if not sample_text or not codes or lexicon is None or char_budget < 80:
        return ""
    scored: list[dict] = []
    for block in re.split(r"\n(?=\[\d+\u2013\d+ min\])", sample_text):
        match = _WINDOW.match(block.strip())
        when, body = ((f"{match.group(1)}\u2013{match.group(2)} min", match.group(3))
                      if match else ("sample", block.strip()))
        for sentence in _sentences(body):
            tokens = set(_tokens(sentence))
            if len(tokens) < 2:
                continue
            per = {code: sum(lexicon.weights(code).get(token, 0.0) for token in tokens)
                   for code in codes}
            best_code, best = max(per.items(), key=lambda kv: kv[1], default=(None, 0.0))
            if not best_code or best <= 0:
                continue
            scored.append({"when": when, "sentence": sentence, "for": best_code, "score": best})
    if not scored:
        return ""
    expected, rivals = codes[0], codes[1:3]
    chosen: list[dict] = []

    def take(pred, want: int) -> None:
        got = 0
        for item in sorted(scored, key=lambda it: -it["score"]):
            if got >= want:
                break
            if any(item is existing for existing in chosen) or not pred(item):
                continue
            chosen.append(item)
            got += 1

    take(lambda it: it["for"] == expected, 2)
    for rival in rivals:
        take(lambda it, rival=rival: it["for"] == rival, 1)
    take(lambda it: True, max(0, 4 - len(chosen)))

    lines: list[str] = []
    used = 0
    for item in chosen:
        line = f"[{item['when']}] {item['sentence']}"
        if used + len(line) + 1 > char_budget:
            continue
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
