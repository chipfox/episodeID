"""One view of "do these two transcripts say the same thing".

Conflict detection, repair validation and reference verification all compare
reference transcripts. They must agree on what counts as a word, or a pair one
check calls a duplicate the other calls distinct. This is that one view.

The matcher keeps its own vectorizer on purpose: it scores phrase coverage
(1-2 grams) to find where an episode sits in a file, which is a different
question from "are these two documents the same".
"""
from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize


def vectorizer(n_docs: int) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                           stop_words="english",
                           max_df=0.6 if n_docs >= 10 else 1.0,
                           token_pattern=r"(?u)\b[a-z][a-z']+\b")


def matrix(docs: list[str]):
    """Normalised TF-IDF rows, so a dot product is a cosine."""
    return normalize(vectorizer(len(docs)).fit_transform([d or "" for d in docs]))


def similarity(a: str, b: str) -> float:
    m = matrix([a, b])
    return float((m @ m.T).toarray()[0][1])


def pairwise(docs: list[str]):
    """Cosine similarity of every document against every other."""
    m = matrix(docs)
    return (m @ m.T).toarray()


def scorer(docs: list[str]):
    """A function scoring one new text against every document, as cosines."""
    vec = vectorizer(len(docs))
    m = normalize(vec.fit_transform([d or "" for d in docs]))

    def score(text: str):
        q = normalize(vec.transform([text or ""]))
        return (q @ m.T).toarray()[0]

    return score


_WORD = re.compile(r"[a-z][a-z']{3,}")


def shared_terms(a: str, b: str, limit: int = 14) -> list[str]:
    """Words both texts use, the ones they use most first.

    A reading aid for a person comparing a file's dialogue with an episode's
    reference — names and places are what settle a match — not a score. Stop
    words and words shorter than four letters are ignored.
    """
    ca = Counter(w for w in _WORD.findall((a or "").lower()) if w not in ENGLISH_STOP_WORDS)
    cb = Counter(w for w in _WORD.findall((b or "").lower()) if w not in ENGLISH_STOP_WORDS)
    both = [(min(ca[w], cb[w]), ca[w] + cb[w], w) for w in ca.keys() & cb.keys()]
    both.sort(key=lambda t: (-t[0], -t[1], t[2]))
    return [w for _low, _total, w in both[:limit]]


# ------------------------------------------------------------------- containment
# Share of one text's word pairs that appear in another. This is the measure for
# "does this reference contain the dialogue we heard": it does not care that a
# 10-minute sample is a tenth the length of a full reference, which is exactly what
# a cosine punishes. Measured on the X-Files library (217 files): the episode's own
# reference 0.70-0.94 (median 0.84), the closest unrelated reference 0.18 (p90 0.23).

BIGRAM_FLOOR = 40  # fewer word pairs than this and a text is too short to judge


def bigrams(text: str) -> set[tuple[str, str]]:
    """The word pairs of a transcript, lowercased."""
    words = re.findall(r"[a-z][a-z']+", (text or "").lower())
    return set(zip(words, words[1:]))


def containment(query_grams: set[tuple[str, str]],
                document_grams: set[tuple[str, str]]) -> float:
    """Share of the query's word pairs that the document contains (0..1)."""
    if not query_grams:
        return 0.0
    return len(query_grams & document_grams) / len(query_grams)


def overlap(a_grams: set[tuple[str, str]], b_grams: set[tuple[str, str]]) -> float:
    """The larger share of either text found in the other, whichever is shorter.

    "Are these two the same text" (the duplicate-reference question), as opposed to
    containment's "is this text inside that one".
    """
    if not a_grams or not b_grams:
        return 0.0
    return len(a_grams & b_grams) / min(len(a_grams), len(b_grams))

