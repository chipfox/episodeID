"""Dialogue matching.

Each file's dialogue is cut into overlapping windows (default 90 s every 45 s).
Every window is scored against every episode's reference text with TF-IDF
cosine similarity. A Viterbi pass then assigns one episode per window with a
penalty for switching, which yields clean segments even when individual windows
are noisy. Each segment is finally re-scored as a whole for its confidence.

This handles paraphrased subtitles, Whisper transcription errors, two-episode
files (including pairs in the wrong order), and files that are simply
mislabelled across seasons.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import get_settings
from . import textsim

EP_RE = re.compile(r"(?i)\bS(\d{1,3})((?:[ ._-]*E\d{1,4})+)(?:-(\d{1,4})\b)?")
EP_X_RE = re.compile(r"(?i)\b(\d{1,2})x(\d{2,3})(?:-(\d{2,3}))?\b")


def parse_filename_episodes(name: str) -> list[str]:
    m = EP_RE.search(name)
    if m:
        season = int(m[1])
        nums = [int(x) for x in re.findall(r"(?i)E(\d+)", m[2])]
        if m[3]:
            nums.append(int(m[3]))
        if len(nums) == 2 and nums[1] > nums[0] + 1 and "-" in m.group(0):
            nums = list(range(nums[0], nums[1] + 1))  # S01E01-E03 → 1,2,3
        return [f"S{season:02d}E{n:02d}" for n in dict.fromkeys(nums)]
    m = EP_X_RE.search(name)
    if m:
        season = int(m[1])
        nums = [int(m[2])] + ([int(m[3])] if m[3] else [])
        return [f"S{season:02d}E{n:02d}" for n in nums]
    return []


FEW_REFS = 5          # below this many references a match needs to be plainly that episode
PLAINLY_SAME = 0.35   # (= references.SAME_EPISODE_SCORE) similarity of the same episode


class Matcher:
    def __init__(self, refs: dict[str, str], settings: dict, background: list[str] | None = None):
        """background: the series' own files' dialogue. Word weights are learned from it as
        well as from the references, so what every episode says (the characters' names,
        catchphrases, the theme song) counts for little. With only a few references that
        is what tells one episode from another; the references alone can't."""
        if not refs:
            raise ValueError("No reference texts available — fetch references first.")
        self.codes = sorted(refs)
        self.index = {code: i for i, code in enumerate(self.codes)}
        self.refs = {code: refs[code] for code in self.codes}
        self._pair_cache: dict[tuple[str, str], float] = {}
        self.s = settings
        docs = [refs[c] for c in self.codes] + [t for t in background or [] if t.strip()]
        self.vec = TfidfVectorizer(
            lowercase=True, strip_accents="unicode", sublinear_tf=True,
            ngram_range=(1, 2), min_df=1, max_df=0.6 if len(docs) >= 10 else 1.0,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b", stop_words="english")
        self.vec.fit(docs)
        self.R = self.vec.transform([refs[c] for c in self.codes])  # l2-normalised

    def score_text(self, text: str) -> np.ndarray:
        return (self.vec.transform([text]) @ self.R.T).toarray()[0]

    def top(self, scores: np.ndarray, k: int = 3) -> list[dict]:
        idx = np.argsort(-scores)[:k]
        return [{"code": self.codes[i], "score": round(float(scores[i]), 4)} for i in idx]

    def _reference_similarity(self, a: str, b: str) -> float:
        """How much two references say the same thing (cached per pair).

        Document sameness goes through `app/textsim.py`, the one view the repair
        and conflict checks already use — the matcher's own 1-2 gram space exists
        to find where an episode sits in a file, not to compare documents, and it
        scores overlaps lower than this does.
        """
        key = (a, b) if a <= b else (b, a)
        if key not in self._pair_cache:
            self._pair_cache[key] = textsim.similarity(self.refs[a], self.refs[b])
        return self._pair_cache[key]

    def _conflict_twin(self, code: str, alternatives: list[dict]) -> str | None:
        """The runner-up whose reference is nearly the same text as this one's.

        Two references this alike (a wiki transcript filed under both numbers, or
        both parts of a two-parter) cannot prove which episode a transcript came
        from: a high score against one says exactly as much about the other.
        """
        if code not in self.refs:
            return None
        threshold = float(self.s["reference_twin_similarity"])
        for alt in alternatives:
            other = alt.get("code")
            if not other or other == code or other not in self.refs:
                continue
            if self._reference_similarity(code, other) >= threshold:
                return other
        return None

    # ------------------------------------------------------------------
    def match_file(self, cues: list, duration: float) -> dict:
        win = float(self.s["window_seconds"])
        step = float(self.s["window_step_seconds"])
        if not cues:
            return {"segments": [], "whole": [], "evidence": _no_evidence()}
        end_t = max(duration, cues[-1][1])
        starts = np.arange(0, max(end_t - win * 0.5, step), step)
        windows, texts = [], []
        for st in starts:
            en = st + win
            t = " ".join(c[2] for c in cues if c[1] > st and c[0] < en)
            windows.append((float(st), float(min(en, end_t))))
            texts.append(t)

        whole_text = " ".join(c[2] for c in cues)
        whole = self.top(self.score_text(whole_text), 5)

        S = (self.vec.transform(texts) @ self.R.T).toarray()  # windows × episodes
        win_words = np.array([len(t.split()) for t in texts])
        S[win_words < 8] = 0.0  # silence / music-only windows carry no evidence

        used = win_words >= 8
        evidence = {
            "windows": int(used.sum()),
            # How much dialogue the file actually offers — overlapping windows would
            # count the same speech two or three times, which would make a near-silent
            # episode look as substantial as a normal one.
            "words": len(whole_text.split()),
            "rank_share": 0.0,
            "top1": whole[0]["code"] if whole else None,
            "top1_score": whole[0]["score"] if whole else 0.0,
            "conflicted": False,
            "twin": None,
        }
        # Which episode each window votes for is evidence the segment gates throw
        # away: a sample can sit under the absolute score floor while most windows
        # still agree on one episode. classify() uses this to tell "weak but
        # consistent" apart from "not enough dialogue to say anything".
        top1 = evidence["top1"]
        if top1 and used.any():
            i = self.index.get(top1)
            if i is not None:
                evidence["rank_share"] = round(
                    float((S[used].argmax(axis=1) == i).mean()), 3)
            evidence["twin"] = self._conflict_twin(top1, whole[1:])
            evidence["conflicted"] = bool(evidence["twin"])

        labels = self._viterbi(S)
        runs = self._runs(labels, windows)
        min_seg = float(self.s["min_segment_seconds"])
        kept = [r for r in runs if r["label"] is not None and
                (r["end"] - r["start"] >= min_seg or r["end"] - r["start"] >= 0.35 * end_t)]
        # merge neighbouring runs that ended up with the same label
        merged: list[dict] = []
        for r in kept:
            if merged and merged[-1]["label"] == r["label"]:
                merged[-1]["end"] = r["end"]
            else:
                merged.append(dict(r))

        segments = []
        for r in merged:
            seg_text = " ".join(c[2] for c in cues if c[1] > r["start"] and c[0] < r["end"])
            sc = self.score_text(seg_text)
            alts = self.top(sc, 4)
            best = alts[0]
            second = alts[1] if len(alts) > 1 else {"code": None, "score": 0.0}
            margin = best["score"] - second["score"]
            why = []
            if best["score"] < self.s["min_score"]:
                why.append(f"weak match ({best['score']:.2f} < {self.s['min_score']:.2f})")
            if margin < self.s["min_margin"] or best["score"] < 1.4 * max(second["score"], 1e-6):
                why.append(f"close to {second['code']} ({best['score']:.2f} vs "
                           f"{second['score']:.2f})")
            if len(self.codes) < FEW_REFS and best["score"] < PLAINLY_SAME:
                why.append(f"only {len(self.codes)} reference{'s' if len(self.codes) > 1 else ''} to compare "
                           f"with — {best['score']:.2f} isn't enough to say it's that episode")
            twin = self._conflict_twin(best["code"], alts[1:])
            if twin:
                why.append(f"references for {best['code']} and {twin} are near-identical — "
                           "neither can prove which of them was read")
            high = not why
            segments.append({
                "start": round(r["start"], 1), "end": round(r["end"], 1),
                "code": best["code"], "score": best["score"],
                "margin": round(margin, 4), "confidence": "high" if high else "low",
                "alternatives": alts[1:],
                "words": len(seg_text.split()),
                "why": "; ".join(why),
            })
        # Collapse adjacent segments that re-scored to the same episode.
        out: list[dict] = []
        for sg in segments:
            if out and out[-1]["code"] == sg["code"]:
                out[-1]["end"] = sg["end"]
                out[-1]["score"] = max(out[-1]["score"], sg["score"])
            else:
                out.append(sg)
        return {"segments": out, "whole": whole, "evidence": evidence}

    def _viterbi(self, S: np.ndarray) -> list[int | None]:
        n_w, n_e = S.shape
        if n_w == 0:
            return []
        none_score = float(self.s["min_score"]) * 0.6
        # column n_e is "no episode"
        E = np.hstack([S, np.full((n_w, 1), none_score)])
        penalty = 0.22
        score = E[0].copy()
        back = np.zeros((n_w, n_e + 1), dtype=np.int64)
        for t in range(1, n_w):
            best_prev = int(np.argmax(score))
            switch = score[best_prev] - penalty
            stay = score
            use_switch = switch > stay
            back[t] = np.where(use_switch, best_prev, np.arange(n_e + 1))
            score = np.where(use_switch, switch, stay) + E[t]
        path = [int(np.argmax(score))]
        for t in range(n_w - 1, 0, -1):
            path.append(int(back[t][path[-1]]))
        path.reverse()
        return [None if p == n_e else p for p in path]

    def _runs(self, labels, windows) -> list[dict]:
        runs: list[dict] = []
        for lab, (st, en) in zip(labels, windows):
            code = None if lab is None else self.codes[lab]
            if runs and runs[-1]["label"] == code:
                runs[-1]["end"] = en
            else:
                if runs:
                    # windows overlap; split the overlap between neighbours
                    mid = (runs[-1]["end"] + st) / 2 if runs[-1]["end"] > st else st
                    runs[-1]["end"] = mid
                    st = mid
                runs.append({"label": code, "start": st, "end": en})
        return runs


def _no_evidence() -> dict:
    """The evidence block for a file with no dialogue at all."""
    return {"windows": 0, "words": 0, "rank_share": 0.0, "top1": None,
            "top1_score": 0.0, "conflicted": False, "twin": None}


def evidence_accepts(expected: list[str], segments: list[dict],
                     evidence: dict | None) -> bool:
    """A weak absolute score is not proof of absence.

    `min_score` is calibrated for a whole transcript; on a short sample the right
    episode can score under it while most 90-second windows still vote for it.
    Accept exactly that case — and only when every independent signal agrees:

    * the whole transcript's best episode *is* the file's one episode, so this is
      the filename being corroborated by weak evidence, never a filename override;
    * at least `gate_min_window_share` of the usable windows rank it first — a
      single lucky window out of thirteen is not agreement;
    * at least `gate_min_windows` windows carry real dialogue, and the file holds
      at least `gate_min_words` words of it — a near-silent episode produces one
      terse window and a 1.00 rank share, which proves nothing;
    * no surviving segment names another episode, which is a contradiction rather
      than a gap;
    * the winning reference has no near-duplicate twin, because a match on one of
      two near-identical references cannot say which of them was read.

    Anything that fails stays a deferral, which is the safe direction: the only
    outcome worse than "review this" is a confident wrong answer.
    """
    if not expected or len(expected) != 1 or not evidence:
        return False
    settings = get_settings()
    if not settings["evidence_gate"]:
        return False
    if evidence.get("top1") != expected[0] or evidence.get("conflicted"):
        return False
    if float(evidence.get("rank_share") or 0.0) < float(settings["gate_min_window_share"]):
        return False
    if int(evidence.get("windows") or 0) < int(settings["gate_min_windows"]):
        return False
    if int(evidence.get("words") or 0) < int(settings["gate_min_words"]):
        return False
    return all(seg.get("code") in expected for seg in segments)


def promote(expected: list[str], segments: list[dict], evidence: dict | None,
            duration: float) -> list[dict]:
    """Turn an accepted-but-unsegmented result into one high segment over the file.

    `classify()` already returns OK when the evidence gate accepts, but a file with
    no segments gives the plan nothing to show — it would be a review row whose
    status claims OK, which is worse than either answer. The promotion is
    materialised here as the segment the plan and the GUI display, carrying the
    evidence that justified it.
    """
    if segments or not evidence_accepts(expected, [], evidence):
        return segments
    ev = evidence or {}
    return [{
        "start": 0.0, "end": float(duration or 0), "code": expected[0],
        "score": float(ev.get("top1_score") or 0.0), "margin": 0.0,
        "confidence": "high", "alternatives": [], "words": int(ev.get("words") or 0),
        "evidence": "whole-file agreement",
        "why": (f"{int(ev.get('windows') or 0)} windows agree "
                f"({float(ev.get('rank_share') or 0):.0%}), "
                f"{int(ev.get('words') or 0)} words of dialogue"),
    }]


def classify(expected: list[str], segments: list[dict], text_source: str,
             evidence: dict | None = None) -> tuple[str, str]:
    status, note = _classify_segments(expected, segments, text_source)
    if status in ("NO_MATCH", "LOW_CONFIDENCE") and evidence_accepts(expected, segments, evidence):
        return "OK", ""
    return status, note


def _classify_segments(expected: list[str], segments: list[dict],
                       text_source: str) -> tuple[str, str]:
    if not segments:
        if text_source == "none":
            return "NO_TEXT", "No subtitles and no transcription available"
        return "NO_MATCH", "Dialogue did not match any reference"
    detected = [s["code"] for s in segments]
    if any(s["confidence"] != "high" for s in segments):
        return "LOW_CONFIDENCE", "At least one segment is a weak match"
    if detected == expected:
        return "OK", ""
    if len(set(detected)) == len(detected) and set(detected) == set(expected):
        return "OK_ORDER_DIFFERS", "Same episodes, different order inside the file"
    if not expected:
        return "MISMATCH", "Filename has no SxxEyy; identified from dialogue"
    return "MISMATCH", ""
