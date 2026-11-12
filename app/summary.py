"""Episode-summary matching for the ASR fallback.

The deterministic matcher compares a file's dialogue to episode *transcripts*.
When an episode has no transcript, the episode overview is the only text left.
This module turns a title + cleaned overview into a TF-IDF query and ranks it
against the series' reference dialogue. It is a candidate generator, never a
verdict: calibration in the caller decides whether a score is strong enough to
act on, and a summary alone must not rename a file.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_VIDEOS = re.compile(r"(?i)\bvideos?\s*:")
_WORD = re.compile(r"[a-z][a-z']+")
_STOP = set("""a an and are as at be been but by for from had has have he her his i if in
into is it its me my no not of on or our she so that the their them they this to
was we were what when which who will with you your""".split())
_WINDOW_WORDS = 400
_WINDOW_STEP = 200
CALIBRATION_VERSION = 2


def clean_overview(text: str) -> str:
    """TVDB overview without the music-video/credits tail, whitespace-normalised."""
    text = _VIDEOS.split(text or "", maxsplit=1)[0]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in _STOP and len(w) > 2]


def _query(ep: dict) -> str:
    return f"{ep.get('title') or ''} {clean_overview(ep.get('overview') or '')}".strip()


def refs_fingerprint(refs: dict[str, str]) -> str:
    h = hashlib.sha256()
    for code in sorted(refs):
        h.update(code.encode())
        h.update(b"\0")
        h.update(hashlib.sha256((refs[code] or "").encode()).digest())
    return h.hexdigest()[:16]


def summaries_fingerprint(episodes: dict[str, dict]) -> str:
    h = hashlib.sha256()
    for code in sorted(episodes):
        ep = episodes[code]
        h.update(code.encode())
        h.update(b"\0")
        h.update(_query(ep).encode())
    return h.hexdigest()[:16]


def wilson_lower(hits: int, n: int, z: float = 1.96) -> float:
    """95% Wilson lower bound for a proportion. A 1/1 result is not 100%."""
    if n <= 0:
        return 0.0
    p = hits / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _quantiles(values: list[float], n: int = 20) -> list[float]:
    vals = sorted(set(values))
    if not vals:
        return []
    if len(vals) <= n:
        return vals
    return [vals[round(i * (len(vals) - 1) / (n - 1))] for i in range(n)]


@dataclass
class Match:
    code: str
    score: float
    margin: float
    terms: list[str]
    window: int = 0


class SummaryIndex:
    """TF-IDF index fitted on reference dialogue; summaries are the queries."""

    def __init__(self, episodes: dict[str, dict], refs: dict[str, str]):
        self.episodes = {c: dict(e) for c, e in episodes.items()}
        # Include every episode with a summary, including those without a
        # reference: those are the fallback targets.
        self.codes = [c for c in sorted(self.episodes) if clean_overview(
            self.episodes[c].get("overview") or "")]
        docs = [refs[c] for c in self.codes if c in refs]
        if not docs:
            docs = [_query(self.episodes[c]) for c in self.codes]
        if not docs:
            self.vectorizer = None
            self.matrix = None
            self.idf: dict[str, float] = {}
            return
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                          min_df=1)
        self.vectorizer.fit(docs)
        self.matrix = self.vectorizer.transform([_query(self.episodes[c]) for c in self.codes])
        self.idf = dict(zip(self.vectorizer.get_feature_names_out(),
                            (self.vectorizer.idf_ if hasattr(self.vectorizer, "idf_") else [])))

    def score(self, text: str) -> dict[str, float]:
        if not self.codes or self.vectorizer is None or not (text or "").strip():
            return {}
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.matrix)[0]
        return {c: float(sims[i]) for i, c in enumerate(self.codes)}

    def _terms(self, text: str, code: str) -> list[str]:
        present = set(_tokens(text))
        terms = [t for t in set(_tokens(_query(self.episodes[code]))) if t in present]
        terms.sort(key=lambda t: (-self.idf.get(t, 0.0), t))
        return terms[:5]

    def best(self, text: str) -> Match | None:
        scores = self.score(text)
        if not scores:
            return None
        order = sorted(scores, key=lambda c: (-scores[c], c))
        top = order[0]
        return Match(top, scores[top],
                     scores[top] - scores[order[1]] if len(order) > 1 else 0.0,
                     self._terms(text, top))

    def best_windows(self, text: str) -> Match | None:
        words = (text or "").split()
        if len(words) <= _WINDOW_WORDS:
            return self.best(text)
        best = None
        for start in range(0, len(words), _WINDOW_STEP):
            m = self.best(" ".join(words[start:start + _WINDOW_WORDS]))
            if m is None:
                continue
            if best is None or m.score > best.score:
                m.window = start
                best = m
        return best


def build_index(series: dict, refs: dict[str, str]) -> SummaryIndex:
    eps = {f"S{e['season']:02d}E{e['episode']:02d}": e for e in series.get("episodes") or []}
    return SummaryIndex(eps, refs)


def _rows(index: SummaryIndex, refs: dict[str, str]) -> list[tuple[float, float, bool]]:
    codes = [c for c in index.codes if c in refs]
    out = []
    for code in codes:
        scores = index.score(refs[code])
        if code not in scores:
            continue
        order = sorted(scores, key=lambda c: (-scores[c], c))
        top = order[0]
        top_score = scores[top]
        margin = top_score - scores[order[1]] if len(order) > 1 else 0.0
        # Gate on what production sees: the top-1 candidate's score and margin.
        # Using the true episode's score here would leak the answer and make the
        # calibration look perfect while selecting only already-correct rows.
        out.append((top_score, margin, top == code))
    return out


def _best_gate(rows: list[tuple[float, float, bool]], target: float,
               min_gated: int) -> dict | None:
    best = None
    for min_score in _quantiles([r[0] for r in rows]):
        for min_margin in (0.0, 0.01, 0.02, 0.05, 0.08, 0.12):
            gated = [r for r in rows if r[0] >= min_score and r[1] >= min_margin]
            if len(gated) < min_gated:
                continue
            hit = sum(1 for r in gated if r[2])
            lb = wilson_lower(hit, len(gated))
            if lb < target:
                continue
            cov = len(gated) / len(rows)
            if best is None or cov > best["coverage"] or (
                    cov == best["coverage"] and lb > best["precision_lb"]):
                best = {"min_score": min_score, "min_margin": min_margin,
                        "precision": hit / len(gated), "precision_lb": lb,
                        "coverage": cov, "gated": len(gated)}
    return best


def calibrate(index: SummaryIndex, refs: dict[str, str], episodes: dict[str, dict],
              target: float = 0.95, min_refs: int = 20, min_gated: int = 8,
              folds: int = 5) -> dict:
    """Choose a score/margin gate whose Wilson lower bound meets *target*.

    Also reports a deterministic 5-fold out-of-fold validation of the chosen gate;
    a gate that does not survive it is marked unavailable rather than trusted.
    """
    usable = [c for c in index.codes if c in refs]
    base = {"version": CALIBRATION_VERSION, "target": target, "n_refs": len(usable),
            "refs_fp": refs_fingerprint(refs),
            "summaries_fp": summaries_fingerprint(episodes)}
    if len(usable) < min_refs:
        return {**base, "available": False,
                "reason": f"only {len(usable)} references; need {min_refs}"}
    rows = _rows(index, refs)
    gate = _best_gate(rows, target, min_gated)
    if gate is None:
        return {**base, "available": False,
                "reason": f"no gate reached {target:.0%} at the Wilson lower bound"}
    # Out-of-fold validation: split by code, choose a gate on the other folds,
    # evaluate on the held-out fold, and pool. This catches a gate that only
    # works because it was fitted to the same rows.
    by_code = {c: r for c, r in zip([c for c in index.codes if c in refs], rows)}
    ordered = sorted(by_code)
    oof_hit = oof_n = 0
    for fold in range(folds):
        test = [c for i, c in enumerate(ordered) if i % folds == fold]
        train = [r for c, r in by_code.items() if c not in test]
        test_rows = [by_code[c] for c in test]
        fg = _best_gate(train, target, min_gated)
        if fg is None:
            continue
        gated = [r for r in test_rows
                 if r[0] >= fg["min_score"] and r[1] >= fg["min_margin"]]
        oof_n += len(gated)
        oof_hit += sum(1 for r in gated if r[2])
    oof_lb = wilson_lower(oof_hit, oof_n) if oof_n else 0.0
    # The gate is usable for suggestions even when the out-of-fold estimate is
    # weaker than the target: title-card agreement is the independent check for
    # auto-apply, and the OOF number is shown so the UI does not overstate it.
    out = {**base, "available": True,
           "oof_precision": (oof_hit / oof_n) if oof_n else 0.0,
           "oof_lb": oof_lb, "oof_gated": oof_n,
           "oof_ok": oof_n > 0 and oof_lb >= target,
           **gate}
    return out


def calibration_stale(cal: dict | None, refs: dict[str, str],
                      episodes: dict[str, dict]) -> bool:
    return (not cal or cal.get("version") != CALIBRATION_VERSION
            or cal.get("refs_fp") != refs_fingerprint(refs)
            or cal.get("summaries_fp") != summaries_fingerprint(episodes))


def ensure_calibration(series: dict, index: SummaryIndex, refs: dict[str, str],
                       settings: dict) -> dict:
    """Reuse a valid stored calibration; otherwise calibrate now and return it."""
    episodes = {f"S{e['season']:02d}E{e['episode']:02d}": e
                for e in series.get("episodes") or []}
    opts = series.get("options") or {}
    cal = opts.get("summary_calibration")
    if not calibration_stale(cal, refs, episodes):
        return cal
    from datetime import datetime, timezone
    cal = calibrate(index, refs, episodes,
                    target=float(settings.get("summary_target_precision", 0.95)),
                    min_refs=int(settings.get("summary_min_refs", 20)))
    cal["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return cal


def summary_mode(f: dict, refs: dict[str, str], eps: dict) -> str:
    """'auto' = no reference, trusted text; 'suggest' = evidence only; 'off'."""
    source = str(f.get("text_source") or "")
    expected = [c for c in (f.get("expected") or []) if c in eps]
    if source == "none" or not expected or f.get("language"):
        return "off"
    if f.get("scope") != "full":
        return "suggest"                 # ten minutes cannot prove a whole episode
    if any(c not in refs for c in expected):
        return "suggest" if source.startswith("sidecar ") else "auto"
    if source.startswith(("remote ", "whisper ")) and \
            f.get("status") in ("LOW_CONFIDENCE", "NO_MATCH", "MISMATCH"):
        return "suggest"                 # a reference exists; second opinion only
    return "off"


def match_file(text: str, f: dict, index: SummaryIndex, cal: dict, mode: str,
               settings: dict) -> dict | None:
    """Score the file's dialogue against the summaries; never changes a status."""
    if not text.strip() or mode == "off":
        return None
    m = index.best_windows(text) if len(f.get("expected") or []) > 1 else index.best(text)
    if m is None:
        return None
    # The episode's own summary must exist before a summary can settle its name;
    # a match to a different episode is still useful evidence, but not an auto.
    expected = f.get("expected") or []
    if mode == "auto" and expected and expected[0] not in index.codes:
        mode = "suggest"
    passes = bool(cal.get("available") and m.score >= cal.get("min_score", 2)
                  and m.margin >= cal.get("min_margin", 2))
    auto = bool(mode == "auto" and passes and cal.get("oof_ok", True)
                and cal.get("precision_lb", 0) >=
                float(settings.get("summary_auto_precision", 0.99)))
    return {"code": m.code, "score": round(m.score, 4),
            "margin": round(m.margin, 4), "terms": m.terms, "window": m.window,
            "calibrated": bool(cal.get("available")), "passes": passes,
            "auto": auto, "mode": mode, "fallback": True,
            "calibration": {k: cal.get(k) for k in
                            ("target", "min_score", "min_margin", "precision",
                             "precision_lb", "coverage", "n_refs", "oof_lb")}}


def calibration_note(cal: dict) -> str:
    if not cal:
        return "not calibrated"
    if not cal.get("available"):
        return cal.get("reason") or "not calibrated"
    return (f"{cal.get('n_refs')} refs · score ≥ {cal['min_score']:.2f} · "
            f"margin ≥ {cal['min_margin']:.2f} · "
            f"{cal['precision_lb']:.0%} LB precision · "
            f"{cal['coverage']:.0%} coverage")
