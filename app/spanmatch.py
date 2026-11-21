"""Ordered dialogue overlap. A reference match is not an episode label."""
from bisect import bisect_left
from collections import Counter
import re

from . import textsim

TOKEN = re.compile(r"[a-z][a-z']+")
MIN_PAIRS = 65
MIN_COVERAGE = .48


def _words(text):
    return TOKEN.findall(textsim.norm(text))


def _chain(points):
    """Longest increasing chain of unique word-pair anchors in the reference."""
    tails, ends, previous = [], [], []
    for i, (_, pos) in enumerate(points):
        k = bisect_left(tails, pos)
        previous.append(ends[k - 1] if k else -1)
        if k == len(tails):
            tails.append(pos)
            ends.append(i)
        else:
            tails[k], ends[k] = pos, i
    out = []
    i = ends[-1] if ends else -1
    while i >= 0:
        out.append(points[i])
        i = previous[i]
    return out[::-1]


def find_span(cues, reference, duration):
    """Find one supported passage, scoring coverage in the file direction only.

    Ignore unused subtitle text before/after the passage. Repeated phrases aren't
    anchors; distant islands cannot be joined to manufacture a whole-file hit.
    Coordinates in the reference are normalized word offsets, not subtitle times.
    """
    words, times = [], []
    for start, end, text in cues:
        tokens = _words(text)
        words.extend(tokens)
        times.extend([(float(start), float(end))] * len(tokens))
    ref = _words(reference)
    q = list(zip(words, words[1:]))
    r = list(zip(ref, ref[1:]))
    if min(len(q), len(r)) < MIN_PAIRS or not duration:
        return None
    qcounts, rcounts = Counter(q), Counter(r)
    positions = {g: i for i, g in enumerate(r) if rcounts[g] == 1}
    chain = _chain([(i, positions[g]) for i, g in enumerate(q) if qcounts[g] == 1 and g in positions])
    blocks = []
    for i, j in chain:
        if not blocks:
            blocks.append([])
        elif blocks[-1]:
            pi, pj = blocks[-1][-1]
            if times[i][0] - times[pi + 1][1] > max(40, duration * .16) or j - pj > max(140, (i - pi) * 4):
                blocks.append([])
        blocks[-1].append((i, j))
    eligible = []
    for block in blocks:
        if len(block) < MIN_PAIRS:
            continue
        (a, x), (b, y) = block[0], block[-1]
        start, end = times[a][0], times[b + 1][1]
        coverage = len(block) / max(1, sum(qcounts[g] == 1 for g in q[a:b + 1]))
        file_coverage = len(block) / sum(n == 1 for n in qcounts.values())
        if coverage < MIN_COVERAGE or end - start < min(60, duration * .12):
            continue
        whole = (file_coverage >= MIN_COVERAGE and start <= duration * .18
                 and end >= duration * .82 and end - start >= duration * .7)
        eligible.append({"start": round(start, 1), "end": round(end, 1), "reference_start": x,
                         "reference_end": y + 2, "reference_words": len(ref), "pairs": len(block),
                         "coverage": round(coverage, 3), "file_coverage": round(file_coverage, 3),
                         "whole_file": whole})
    return max(eligible, key=lambda item: item["pairs"], default=None)


def reference_passage(text, span):
    """Crop the reference for Compare while retaining its original punctuation."""
    normalized = textsim.norm(text, lower=False)
    starts = [m.start() for m in re.finditer(TOKEN.pattern, normalized, re.I | re.ASCII)]
    start, end = max(0, span["reference_start"] - 12), span["reference_end"] + 12
    if start >= len(starts) or end <= start:
        return text
    return normalized[starts[start]:starts[end] if end < len(starts) else len(normalized)].strip()


def _labeled_match(matcher, cues, duration, candidates):
    """Use only separately labeled references, and reject near-tied labels."""
    hits = []
    for candidate in candidates:
        if candidate not in matcher.anchors:
            continue
        span = find_span(cues, matcher.refs[candidate], duration)
        if span and span["whole_file"] and span["pairs"] >= .40 * span["reference_words"]:
            hits.append((span["file_coverage"], candidate, span))
    hits.sort(reverse=True)
    if not hits or (len(hits) > 1 and hits[0][0] - hits[1][0] < .12):
        return None
    _, code, span = hits[0]
    return {**span, "code": code, "reference": code}


def inspect(matcher, result, dialogue, expected, combined):
    """Enrich a combined/partial result without treating a reference code as a label."""
    cues, duration = dialogue.get("cues") or [], float(dialogue.get("duration") or 0)
    if not cues or not duration or combined["state"] == "verified":
        return None
    ranked = [item["code"] for item in result.get("whole", [])][:5]
    candidates = list(dict.fromkeys([combined.get("reference"), *ranked]))
    passages = [{**span, "reference": code} for code in candidates if code in matcher.refs
                if (span := find_span(cues, matcher.refs[code], duration))]
    if not passages:
        return None
    local = max(passages, key=lambda p: (p["whole_file"], p["pairs"]))
    combined["local"] = local
    # Sidecar and sample text can be compared but cannot assign a label or a cut.
    if (dialogue.get("scope") == "sample" or dialogue.get("language")
            or not str(dialogue.get("source") or "").startswith(("embedded ", "remote ", "whisper "))):
        return None
    local["trusted"] = True
    labels = list(dict.fromkeys([*expected, *ranked]))
    if local["whole_file"] and len(expected) <= 1:
        hit = _labeled_match(matcher, cues, duration, labels)
        if hit:
            local["code"] = hit["code"]
            return [hit]
        return None
    start, end = local["start"], local["end"]
    if not local["whole_file"]:
        clip = [(max(a, start) - start, min(b, end) - start, t) for a, b, t in cues if start <= a < end]
        first = _labeled_match(matcher, clip, end - start, labels)
        if first:
            local["code"] = first["code"]
    else:
        first = None
    remaining = []
    verified = [{**first, "start": start, "end": end}] if first else []
    for lo, hi in ((0, start), (end, duration)):
        if hi - lo < max(90, duration * .17):
            continue
        clip = [(a - lo, min(b, hi) - lo, t) for a, b, t in cues if lo <= a < hi]
        item = {"start": round(lo, 1), "end": round(hi, 1)}
        if clip:
            ranked = [c["code"] for c in matcher.top(matcher.score_text(" ".join(c[2] for c in clip)), 5)]
            labels = [c for c in dict.fromkeys([*expected, *ranked]) if not first or c != first["code"]]
            hit = _labeled_match(matcher, clip, hi - lo, labels)
            if hit:
                item["code"] = hit["code"]
                verified.append({**hit, "start": hit["start"] + lo, "end": hit["end"] + lo})
            else:
                options = [{**s, "reference": c} for c in ranked
                           if matcher.refs[c] != matcher.refs[local["reference"]]
                           if (s := find_span(clip, matcher.refs[c], hi - lo))]
                if options:
                    item["candidate"] = max(options, key=lambda s: s["pairs"])["reference"]
        remaining.append(item)
    if remaining:
        combined["remaining"] = remaining
    if first and len(verified) >= max(2, len(expected)) and len(verified) == len(remaining) + 1 \
            and len({p["code"] for p in verified}) == len(verified):
        return sorted(verified, key=lambda p: p["start"])
    return None
