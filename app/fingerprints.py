"""Conservative sampled audio comparison. Results establish no episode label."""
from __future__ import annotations

import hashlib
import json
import statistics
import struct
import subprocess
import tempfile
from pathlib import Path

from .config import CACHE_DIR, safe_media_path


WINDOW = 80
MAX_SHIFT = 88  # ~11 seconds at Chromaprint's default ~8 subfingerprints/sec
MIN_FRAMES = 500


def _extract(video: Path, start: float) -> list[int]:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(video),
         "-t", str(WINDOW), "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "11025",
         "-f", "chromaprint", "-fp_format", "raw", "-"],
        capture_output=True, timeout=20, check=True,
    )
    return [x for (x,) in struct.iter_unpack("<I", result.stdout)]


def _fingerprints(video: Path, starts: list[float]) -> list[list[int]]:
    st = video.stat()
    key = repr((1, str(video), st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, starts))
    cache = CACHE_DIR / "fingerprints" / (hashlib.sha256(key.encode()).hexdigest() + ".json")
    try:
        return json.loads(cache.read_text())
    except (OSError, ValueError):
        pass
    windows = [_extract(video, start) for start in starts]
    tmp = None
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=cache.parent, delete=False) as f:
            tmp = Path(f.name)
            json.dump(windows, f)
        tmp.replace(cache)
    except OSError:
        pass  # Read-only cache still allows a read-only audio comparison.
    finally:
        if tmp:
            tmp.unlink(missing_ok=True)
    return windows


def _active(window: list[int]) -> bool:
    if len(window) < MIN_FRAMES or len(set(window)) < 80:
        return False
    return sum((a ^ b).bit_count() for a, b in zip(window, window[1:])) >= len(window)


def _align(a: list[int], b: list[int]) -> tuple[float, int]:
    best = (0.0, 0)
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        left, right = a[max(0, shift):], b[max(0, -shift):]
        n = min(len(left), len(right))
        if n < MIN_FRAMES:
            continue
        score = 1 - sum((x ^ y).bit_count() for x, y in zip(left, right)) / (32 * n)
        if score > best[0]:
            best = (score, shift)
    return best


def compare(root: Path, a: dict, b: dict) -> str:
    """Require comparable cuts, distributed active windows and one consistent offset.

    An inconclusive match never means different content; alternate cuts, tracks,
    commentary, silence and incomplete decodes can all fail fingerprint checks.
    """
    da, db = (float(f.get("duration") or 0) for f in (a, b))
    duration = min(da, db)
    if not 330 <= duration <= 4 * 3600 or abs(da - db) > min(10, .02 * duration):
        return "inconclusive"
    count = min(5, int((duration - 65) // WINDOW))
    starts = [round(30 + i * (duration - WINDOW - 65) / (count - 1), 2) for i in range(count)]
    try:
        paths = [safe_media_path(root / f["rel"]) for f in (a, b)]
        if any(not p.is_relative_to(root.resolve()) for p in paths):
            return "inconclusive"
        left, right = [_fingerprints(p, starts) for p in paths]
        if len(left) != count or len(right) != count or not all(map(_active, left + right)):
            return "inconclusive"
        scores, offsets = zip(*(_align(x, y) for x, y in zip(left, right)))
    except (OSError, ValueError, TypeError, struct.error, subprocess.SubprocessError):
        return "inconclusive"
    if min(scores) >= .77 and statistics.median(scores) >= .80 and max(offsets) - min(offsets) <= 8:
        return "equivalent"
    if max(scores) <= .62:
        return "different"
    return "inconclusive"
