"""One view of "do these two transcripts say the same thing".

Conflict detection, repair validation and reference verification all compare
reference transcripts. They must agree on what counts as a word, or a pair one
check calls a duplicate the other calls distinct. This is that one view.

The matcher keeps its own vectorizer on purpose: it scores phrase coverage
(1-2 grams) to find where an episode sits in a file, which is a different
question from "are these two documents the same".
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
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
