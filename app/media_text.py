"""Getting timed dialogue out of a video file.

Order of preference when external sidecars are enabled:
  1. sidecar subtitle file next to the video (.srt/.ass/.ssa/.vtt)
  2. embedded text subtitle stream (subrip, ass, mov_text, webvtt)
  3. configured transcription backend (local faster-whisper by default, or a
     remote OpenAI-compatible ASR service)

When sidecar subtitles are disabled, embedded text tracks are preferred and
transcription remains the fallback. Results are cached under /cache/text keyed
by path+size+mtime and the active dialogue settings; cached sidecar results are
ignored when sidecars are disabled."""
from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import CACHE_DIR, get_settings

VIDEO_EXTS = {".mkv", ".mp4", ".m4v", ".avi", ".ts", ".m2ts", ".mov", ".wmv", ".webm", ".mpg"}
SUB_EXTS = (".srt", ".ass", ".ssa", ".vtt")
TEXT_SUB_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
IMAGE_SUB_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}


@dataclass
class Cue:
    start: float
    end: float
    text: str


# ---------------------------------------------------------------- language hints
#
# The ASR backend is configured for one subtitle language. That is normally a
# sufficient guard, but servers do not always report the language they heard.
# A small function-word detector prevents foreign-language samples from being
# scored against English references as if they were "no episode match".
_LANG_STOPWORDS = {
    "en": {"the", "and", "you", "your", "i", "to", "of", "a", "an", "in", "that",
           "it", "is", "are", "was", "were", "for", "on", "with", "have", "has",
           "this", "what", "we", "he", "she", "they", "me", "my", "do", "did"},
    "es": {"el", "la", "los", "las", "que", "de", "del", "al", "un", "una", "y",
           "en", "lo", "se", "por", "para", "con", "es", "esta", "está", "su",
           "me", "te", "si", "sí", "no", "como", "más", "mas", "pero", "fue",
           "ser", "son", "este", "ese", "esa", "esto", "hombre", "mujeres"},
    "fr": {"le", "la", "les", "de", "des", "et", "est", "que", "dans", "pour",
           "pas", "sur", "avec", "vous", "nous", "je", "tu", "il", "elle",
           "qui", "quoi", "une", "un", "au", "aux", "cette", "mon", "son"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "ich",
           "du", "sie", "wir", "mit", "auf", "für", "von", "zu", "den", "dem",
           "des", "wie", "was", "auch", "nur", "noch", "kann", "man"},
    "it": {"il", "lo", "la", "gli", "le", "di", "che", "e", "per", "con", "un",
           "una", "non", "sono", "questo", "questa", "come", "più", "piu",
           "anche", "quando", "dove", "lei", "lui", "noi", "voi"},
    "pt": {"o", "a", "os", "as", "de", "do", "da", "que", "e", "é", "em", "um",
           "uma", "não", "nao", "com", "para", "por", "como", "mais", "isso",
           "este", "esta", "meu", "minha", "você", "voce", "nos"},
    # Bazarr hands Indonesian subtitles out under .en.srt names often enough to
    # count as a property of the library, not an accident — detect_language must
    # recognize them or they poison an English reference comparison.
    "id": {"yang", "dan", "apa", "itu", "ini", "dengan", "untuk", "dari", "pada", "adalah",
           "bukan", "tidak", "saya", "anda", "kamu", "kami", "kita", "mereka", "dia", "nya",
           "aku", "atau", "juga", "sudah", "akan", "saja", "bagaimana", "kapan", "siapa",
           "mengapa", "bisa", "harus", "sangat", "seperti", "masih", "ada", "lagi", "hai",
           "kabarmu", "pakai", "pukul", "sedang", "kalau", "televisi", "sedikit", "memberi"},
}
_LANG_PREFIXES = {"spa": "es", "english": "en", "eng": "en", "fre": "fr", "fra": "fr",
                  "ger": "de", "deu": "de", "ita": "it", "por": "pt"}
_TOKEN = re.compile(r"(?u)\b[a-zà-ÿ']{2,}\b")


def _language_key(value: str | None) -> str:
    value = str(value or "").strip().lower()
    if not value:
        return "en"
    prefix = value[:3]
    return _LANG_PREFIXES.get(prefix, _LANG_PREFIXES.get(value, prefix))


def detect_language(cues: list | list[list], target_language: str = "en") -> str | None:
    """Return a non-target language only when the evidence is unambiguous.

    Samples are short and may contain names, sound effects or ASR noise, so this
    deliberately says `None` rather than guessing.
    """
    text = " ".join(
        str(item[2]) if isinstance(item, (list, tuple)) and len(item) > 2 else str(item)
        for item in cues
    )
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < 40:
        return None
    scores: dict[str, float] = {}
    for lang, stopwords in _LANG_STOPWORDS.items():
        hits = sum(1 for token in tokens if token in stopwords)
        if hits:
            scores[lang] = hits / len(tokens)
    target = _language_key(target_language)
    if not scores:
        return None
    best_lang, best = max(scores.items(), key=lambda item: item[1])
    target_score = scores.get(target, 0.0)
    if best_lang == target or best < 0.045 or best < target_score * 1.75:
        return None
    return best_lang


# ---------------------------------------------------------------- parsing

_TS = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")
_TAGS = re.compile(r"<[^>]+>|\{[^}]*\}")


def _ts(s: str) -> float:
    m = _TS.search(s)
    if not m:
        return 0.0
    h, mi, se, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms.ljust(3, "0")) / 1000


def _clean(text: str) -> str:
    text = html.unescape(_TAGS.sub(" ", text))
    text = text.replace("\\N", " ").replace("\\n", " ")
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|♪", " ", text)  # sound cues, lyrics marks
    return re.sub(r"\s+", " ", text).strip()


def parse_srt_vtt(raw: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", raw.replace("\r", "")):
        lines = [l for l in block.split("\n") if l.strip()]
        for i, line in enumerate(lines):
            if "-->" in line:
                a, b = line.split("-->", 1)
                text = _clean(" ".join(lines[i + 1:]))
                if text:
                    cues.append(Cue(_ts(a), _ts(b), text))
                break
    return cues


def parse_ass(raw: str) -> list[Cue]:
    cues = []
    fmt = None
    for line in raw.splitlines():
        if line.startswith("Format:") and fmt is None and "Text" in line:
            fmt = [f.strip().lower() for f in line[7:].split(",")]
        elif line.startswith("Dialogue:"):
            fields = fmt or ["layer", "start", "end", "style", "name", "marginl",
                             "marginr", "marginv", "effect", "text"]
            parts = line[9:].split(",", len(fields) - 1)
            if len(parts) < len(fields):
                continue
            d = dict(zip(fields, parts))
            m = re.match(r"\s*(\d+):(\d+):(\d+)\.(\d+)", d["start"])
            n = re.match(r"\s*(\d+):(\d+):(\d+)\.(\d+)", d["end"])
            if not (m and n):
                continue
            st = int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]) + int(m[4]) / 100
            en = int(n[1]) * 3600 + int(n[2]) * 60 + int(n[3]) + int(n[4]) / 100
            text = _clean(d["text"])
            if text:
                cues.append(Cue(st, en, text))
    cues.sort(key=lambda c: c.start)
    return cues


def parse_subtitle_text(raw: str, ext: str) -> list[Cue]:
    if ext in (".ass", ".ssa") or "[Script Info]" in raw[:200]:
        return parse_ass(raw)
    return parse_srt_vtt(raw)


def read_text_file(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", "replace")


# ------------------------------------------------------------------ probe

def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
         str(path)], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:300]}")
    return json.loads(r.stdout)


def duration(info: dict) -> float:
    try:
        return float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        return 0.0


def _lang_ok(stream: dict, lang: str, allow_untagged: bool = True) -> bool:
    tag = ((stream.get("tags") or {}).get("language") or "").lower()
    if not lang:
        return True
    aliases = {"en": {"en", "eng", "english"}, "es": {"es", "spa"}, "fr": {"fr", "fre", "fra"},
               "de": {"de", "ger", "deu"}, "it": {"it", "ita"}, "pt": {"pt", "por"},
               "nl": {"nl", "dut", "nld"}, "ja": {"ja", "jpn"}}
    if tag in ("", "und"):
        return allow_untagged
    return tag in aliases.get(lang, {lang})


# ---------------------------------------------------------------- extract

def _cache_key(path: Path, scope: str = "full", start: float = 0.0,
               duration: float | None = None) -> str:
    st = path.stat()
    # No path in the key: renames keep size+mtime, so Whisper transcripts survive an apply.
    key = f"{st.st_size}|{st.st_mtime_ns}"
    s = get_settings()
    # A sidecar subtitle added/changed later must invalidate the cache only
    # when sidecars are actually part of the evidence chain.
    stem = path.with_suffix("").name
    if s["sidecar_subtitles"]:
        for f in sorted(path.parent.glob(glob_escape(stem) + "*")):
            if f.suffix.lower() in SUB_EXTS:
                fs = f.stat()
                key += f"|{f.name[len(stem):]}|{fs.st_size}|{fs.st_mtime_ns}"
    # Switching between local Whisper and a remote model must not reuse a
    # transcript produced by the previous backend. Do not include API secrets.
    backend = str(s["whisper_backend"]).strip().lower()
    key += (f"|sidecars:{int(bool(s['sidecar_subtitles']))}|transcriber:{backend}"
            f"|local-model:{s['whisper_model']}|remote-model:{s['whisper_remote_model']}"
            f"|language:{s['whisper_language']}|remote-url:{s['whisper_base_url']}")
    if scope != "full":
        key += f"|scope:{scope}|start:{float(start):.3f}|duration:{float(duration or 0):.3f}"
    return hashlib.sha1(key.encode()).hexdigest()


def glob_escape(s: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", s)


def _cache_file(path: Path, scope: str = "full", start: float = 0.0,
                duration: float | None = None) -> Path:
    return CACHE_DIR / "text" / f"{_cache_key(path, scope, start, duration)}.json"


def sidecar_subtitle(video: Path, lang: str) -> Path | None:
    stem = video.with_suffix("").name
    candidates = []
    for f in video.parent.iterdir():
        if f.suffix.lower() in SUB_EXTS and f.name.startswith(stem):
            middle = f.name[len(stem):-len(f.suffix)].lower().strip(".")
            score = 0
            if not middle:
                score = 1
            elif lang and lang in middle.split("."):
                score = 2
            if "forced" in middle:
                score = -1
            candidates.append((score, f))
    candidates = [c for c in candidates if c[0] >= 0]
    return max(candidates, key=lambda c: c[0])[1] if candidates else None


def extract_embedded(video: Path, info: dict, lang: str,
                     allow_untagged: bool = True) -> tuple[list[Cue], str] | None:
    subs = [s for s in info.get("streams", []) if s.get("codec_type") == "subtitle"]
    text_subs = [s for s in subs if s.get("codec_name") in TEXT_SUB_CODECS]
    if not text_subs:
        return None

    def rank(s):
        disp = s.get("disposition") or {}
        title = ((s.get("tags") or {}).get("title") or "").lower()
        return (_lang_ok(s, lang, allow_untagged),
                not disp.get("forced") and "forced" not in title,
                "sdh" not in title, -s["index"])

    best = max(text_subs, key=rank)
    if not _lang_ok(best, lang, allow_untagged):
        return None
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", f"0:{best['index']}",
         "-f", "srt", "-"], capture_output=True, timeout=600)
    if r.returncode != 0:
        return None
    cues = parse_srt_vtt(r.stdout.decode("utf-8", "replace"))
    return (cues, f"embedded #{best['index']}") if cues else None


def _read_cached(path: Path, allow_sidecars: bool) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("source"), str):
        return None
    if not allow_sidecars and data["source"].startswith("sidecar "):
        return None
    return data


def cached_dialogue(video: Path) -> dict | None:
    """The dialogue already read from this file, or None — never reads the file."""
    return _read_cached(_cache_file(video), get_settings()["sidecar_subtitles"])


def dialogue_in_target_language(cues: list, lang: str) -> tuple[bool, str | None]:
    """Is this text in the language the references are written in?

    Embedded and sidecar subtitles are often untagged and sometimes belong to a
    different language — or to a different episode — so they must be checked
    before anyone treats them as this file's dialogue. Returns (False, lang)
    only when detect_language is certain; anything ambiguous stays usable.
    """
    other = detect_language(cues, lang)
    return other is None, other


def get_dialogue(video: Path, ctx=None, allow_whisper: bool = True) -> dict:
    """Returns {"source": str, "duration": float, "cues": [[start, end, text], ...]}.
    source is "none" when no text could be produced."""
    s = get_settings()
    lang = s["subtitle_language"]
    cache = _cache_file(video)
    data = _read_cached(cache, s["sidecar_subtitles"])
    if data is not None and "language" not in data:
        usable, other = dialogue_in_target_language(data["cues"], lang)
        if not usable and data["source"].startswith(("embedded ", "sidecar ")):
            data = None  # an untagged foreign subtitle track cached before this check
        else:
            data["language"] = None if usable else other
    if data is not None and (data["source"] != "none" or not allow_whisper or not s["whisper_enabled"]):
        return data

    info = probe(video)
    dur = duration(info)
    result = None

    if s["sidecar_subtitles"]:
        sc = sidecar_subtitle(video, lang)
        if sc:
            cues = parse_subtitle_text(read_text_file(sc), sc.suffix.lower())
            if cues:
                result = (cues, f"sidecar {sc.name}")
    if result is None:
        result = extract_embedded(video, info, lang)
    if result is not None:
        usable, other = dialogue_in_target_language(result[0], lang)
        if not usable:
            if ctx:
                ctx.log(f"  {result[1]} is {other}, not {lang} — reading the audio instead")
            result = None
    if result is None and allow_whisper and s["whisper_enabled"]:
        from . import transcribe as tr
        if ctx:
            ctx.log(f"  no text subtitles; transcribing with {tr.source_label(s)}…")
        cues = tr.transcribe(video, ctx)
        if cues:
            result = (cues, tr.source_label(s))

    language = None
    if result is not None:
        usable, other = dialogue_in_target_language(result[0], lang)
        if not usable:
            language = other
            if ctx:
                ctx.log(f"  the audio is {other}, not {lang} — dialogue kept, but flagged")

    if result is None:
        data = {"source": "none", "duration": dur, "cues": []}
    else:
        cues, source = result
        data = {"source": source, "duration": dur, "language": language,
                "cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in cues]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def _clip_cues(cues: list[Cue], start: float, end: float) -> list[Cue]:
    """Keep only the part of each cue inside [start, end], file-relative."""
    out = []
    for cue in cues:
        a, b = max(cue.start, start), min(cue.end, end)
        if b > a and cue.text.strip():
            out.append(Cue(round(a, 3), round(b, 3), cue.text.strip()))
    return out


def sample_dialogue(video: Path, ctx=None, allow_whisper: bool = True,
                    start: float | None = None,
                    duration: float | None = None) -> dict:
    """Read a bounded sample, preferring ASR, and mark it as provisional evidence.

    The returned dict has the same shape as get_dialogue(), plus scope,
    sample_start, sample_end and file_duration. A sample is not a full-file
    transcript and must never be treated as trusted evidence by the planner.
    """
    s = get_settings()
    start = max(0.0, float(s["decision_sample_start"] if start is None else start))
    window = float(s["decision_sample_seconds"] if duration is None else duration)
    cache = _cache_file(video, "sample", start, window)
    cached = _read_cached(cache, s["sidecar_subtitles"])
    if cached is not None and (cached["source"] != "none" or not allow_whisper
                               or not s["whisper_enabled"]):
        return cached

    info = probe(video)
    full = duration_of(info)
    end = min(full, start + window) if full else start + window
    sample_len = max(0.1, end - start)
    partial = not full or full > end + 1.0
    result = None
    fallback = ""

    if allow_whisper and s["whisper_enabled"]:
        try:
            from . import transcribe as tr
            if ctx:
                ctx.log(f"  sampling {start:.0f}–{end:.0f}s with {tr.source_label(s)}…")
            cues = tr.transcribe(video, ctx, start=start, duration=sample_len)
            if cues:
                result = (cues, tr.source_label(s))
        except Exception as e:  # noqa: BLE001
            fallback = f"ASR failed: {e}"
            if ctx:
                ctx.log(f"  sample ASR failed ({e}); trying subtitle evidence")
    if result is None:
        # With no ASR, only explicitly tagged English subtitles are acceptable
        # as a sample fallback. Untagged streams are often the wrong language.
        result = extract_embedded(video, info, s["subtitle_language"], allow_untagged=False)
        if result is None and s["sidecar_subtitles"]:
            sc = sidecar_subtitle(video, s["subtitle_language"])
            if sc:
                cues = parse_subtitle_text(read_text_file(sc), sc.suffix.lower())
                if cues:
                    result = (cues, f"sidecar {sc.name}")
        if result is not None:
            fallback = fallback or "ASR unavailable; used subtitle evidence"

    if result is None:
        data = {"source": "none", "duration": sample_len, "cues": [],
                "scope": "sample" if partial else "full",
                "sample_start": start, "sample_end": end, "file_duration": full}
        if fallback:
            data["fallback"] = fallback
    else:
        cues, source = result
        cues = _clip_cues(cues, start, end)
        data = {"source": source, "duration": sample_len,
                "cues": [[round(c.start, 2), round(c.end, 2), c.text] for c in cues],
                "scope": "sample" if partial else "full",
                "sample_start": start, "sample_end": end, "file_duration": full}
        if fallback:
            data["fallback"] = fallback
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def duration_of(info: dict) -> float:
    """Alias kept local to sampling so existing callers keep using duration()."""
    return duration(info)


def has_image_subs(info: dict) -> bool:
    return any(s.get("codec_name") in IMAGE_SUB_CODECS for s in info.get("streams", []))
