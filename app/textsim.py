"""One view of "do these two transcripts say the same thing".

Conflict detection, repair validation and reference verification all compare
reference transcripts. They must agree on what counts as a word, or a pair one
check calls a duplicate the other calls distinct. This is that one view.

The matcher keeps its own vectorizer on purpose: it scores phrase coverage
(1-2 grams) to find where an episode sits in a file, which is a different
question from "are these two documents the same".
"""
from __future__ import annotations

import difflib
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.preprocessing import normalize


_QUOTE = str.maketrans({"’": "'", "‘": "'", "‛": "'", "ʼ": "'", "`": "'",
                        "´": "'", "′": "'", "“": '"', "”": '"',
                        "–": "-", "—": "-"})


def norm(text: str) -> str:
    """The one punctuation view every comparison in the app shares.

    Smart apostrophes are the killer: every tokenizer here matches words as
    [a-z][a-z']+, so a subtitle's DON’T tokenizes as "don" while an ASR's
    DON'T is "don't" — the same line scores as two different ones, in the
    matcher, the duplicate guard and the compare diff alike. A reference typo
    like I'‘m splits "I'm" the same way. Unify quotes, fold repeated
    apostrophes, lowercase, collapse whitespace.
    """
    t = (text or "").lower().translate(_QUOTE)
    t = re.sub(r"'{2,}", "'", t)
    return re.sub(r"\s+", " ", t)


def vectorizer(n_docs: int) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, strip_accents="unicode", sublinear_tf=True,
                           stop_words="english",
                           max_df=0.6 if n_docs >= 10 else 1.0,
                           token_pattern=r"(?u)\b[a-z][a-z']+\b")


def matrix(docs: list[str]):
    """Normalised TF-IDF rows, so a dot product is a cosine."""
    return normalize(vectorizer(len(docs)).fit_transform([norm(d) for d in docs]))


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
    ca = Counter(w for w in _WORD.findall(norm(a)) if w not in ENGLISH_STOP_WORDS)
    cb = Counter(w for w in _WORD.findall(norm(b)) if w not in ENGLISH_STOP_WORDS)
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
    words = re.findall(r"[a-z][a-z']+", norm(text))
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



# ---------------------------------------------------------------------- alignment
# Reading aid for the compare dialog: lay the two transcripts out sentence by
# sentence so the same line shares a row on both sides. "same" = identical once
# punctuation, case and speaker dashes are ignored; "near" = different words
# carrying the same line (a speech-to-text read rarely matches subtitle
# punctuation word for word); a blank side = the sentence appears only there.
# The thresholds keep near from pairing two exclamations that merely share a
# "yeah": a pair must agree in content (the Jaccard floor), in order.

NEAR_SIMILARITY = 0.4   # Jaccard of content words needed to call a pair the same line
NEAR_SHARED = 1         # ... and the minimum shared words it takes. One is enough
                        # only via containment — one side wholly inside a side at
                        # least twice as long ("Then it would rule." inside
                        # "Yeah, then it would rule!") — so the Jaccard floor,
                        # not the word count, is what rejects catchphrase pairs.
MAX_DIFF_ROWS = 4000    # a transcript this long in sentences is past readable anyway
_MAX_PAIR_CELLS = 120_000  # DP budget for pairing one changed block

_KW_WORD = re.compile(r"[a-z][a-z']+")
_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+|\s+-\s+")


def chunks(text: str) -> list[str]:
    """Sentence-sized display units of a transcript.

    Both sources arrive as one run-on line — Whisper with sentence punctuation,
    subtitles with " - " between speakers — so lines mean nothing here; split on
    sentence ends and speaker dashes, then fold fragments too short to stand
    alone into the sentence before them.
    """
    out: list[str] = []
    for line in re.split(r"[\r\n]+", text or ""):
        line = line.replace("*", " ")  # Whisper's music markers
        for part in _SENT_SPLIT.split(line):
            part = part.strip(" \t\"'“”‘’")
            if part:
                out.append(part)
    merged: list[str] = []
    for c in out:
        if merged and len(c.split()) < 3:
            merged[-1] = merged[-1] + " " + c
        else:
            merged.append(c)
    return merged


def _norm(chunk: str) -> str:
    return " ".join(_KW_WORD.findall(norm(chunk)))


def _keywords(chunk: str) -> frozenset:
    return frozenset(w for w in _KW_WORD.findall(norm(chunk))
                     if len(w) > 2 and w not in ENGLISH_STOP_WORDS)


def _line_similarity(ka: frozenset, kb: frozenset) -> float:
    shared = len(ka & kb)
    if shared < NEAR_SHARED:
        return 0.0
    jaccard = shared / len(ka | kb)
    if jaccard >= NEAR_SIMILARITY:
        return jaccard
    # A speech-to-text line often merges what the subtitles wrote as two lines:
    # the short subtitle line can sit inside the long transcript chunk without
    # lifting Jaccard. Let a short side that is nearly all contained count as a
    # pair — but only for a genuinely merged-size gap, or two lines merely
    # opening with the same catchphrase would pair.
    short, long_ = (len(ka), len(kb)) if len(ka) <= len(kb) else (len(kb), len(ka))
    if long_ >= 2 * short and shared / short >= 0.75:
        return 0.75
    return 0.0


def _pair_block(al: list[str], bl: list[str], near: float) -> list[tuple]:
    """Best order-preserving pairing of two runs of unmatched sentences.

    Needleman-Wunsch over sentence pairs: pairing two scores by their content-word
    Jaccard, skipping either side costs nothing. That keeps pairs monotonic (the
    diff never crosses itself) while still matching a garbled Whisper line to its
    subtitle counterpart.
    """
    n, m = len(al), len(bl)
    if not n or not m:
        return [(i, None) for i in range(n)] + [(None, j) for j in range(m)]
    ka = [_keywords(c) for c in al]
    kb = [_keywords(c) for c in bl]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]  # 0 pair, 1 file-only, 2 reference-only
    for i in range(1, n + 1):
        bt[i][0] = 1
    for j in range(1, m + 1):
        bt[0][j] = 2
    for i in range(1, n + 1):
        row_prev, row = dp[i - 1], dp[i]
        for j in range(1, m + 1):
            s = _line_similarity(ka[i - 1], kb[j - 1])
            pair = row_prev[j - 1] + (s if s >= near else float("-inf"))
            up, left = row_prev[j], row[j - 1]
            if pair >= up and pair >= left:
                row[j], bt[i][j] = pair, 0
            elif up >= left:
                row[j], bt[i][j] = up, 1
            else:
                row[j], bt[i][j] = left, 2
    pairs: list[tuple] = []
    i, j = n, m
    while i or j:
        t = bt[i][j]
        if t == 0:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif t == 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs


def align(a: str, b: str, near: float = NEAR_SIMILARITY) -> dict:
    """Align two transcripts into rows a two-column view can render directly.

    Rows are {"kind": same|near|file|ref, "left": str, "right": str} (empty when
    that side has no sentence), in reading order, and every sentence of both
    texts lands in exactly one row.
    """
    ac, bc = chunks(a), chunks(b)
    sm = difflib.SequenceMatcher(None, [_norm(c) for c in ac],
                                 [_norm(c) for c in bc], autojunk=False)
    rows: list[dict] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            rows.extend({"kind": "same", "left": ac[i1 + k], "right": bc[j1 + k]}
                        for k in range(i2 - i1))
        elif tag == "delete":
            rows.extend({"kind": "file", "left": ac[i], "right": ""} for i in range(i1, i2))
        elif tag == "insert":
            rows.extend({"kind": "ref", "left": "", "right": bc[j]} for j in range(j1, j2))
        elif (i2 - i1) * (j2 - j1) > _MAX_PAIR_CELLS:
            rows.extend({"kind": "file", "left": ac[i], "right": ""} for i in range(i1, i2))
            rows.extend({"kind": "ref", "left": "", "right": bc[j]} for j in range(j1, j2))
        else:
            for i, j in _pair_block(ac[i1:i2], bc[j1:j2], near):
                if i is not None and j is not None:
                    rows.append({"kind": "near", "left": ac[i1 + i], "right": bc[j1 + j]})
                elif i is not None:
                    rows.append({"kind": "file", "left": ac[i1 + i], "right": ""})
                else:
                    rows.append({"kind": "ref", "left": "", "right": bc[j1 + j]})
    counts = {"same": 0, "near": 0, "file": 0, "ref": 0}
    for r in rows:
        counts[r["kind"]] += 1
    return {"rows": rows[:MAX_DIFF_ROWS], "counts": counts,
            "truncated": len(rows) > MAX_DIFF_ROWS}
