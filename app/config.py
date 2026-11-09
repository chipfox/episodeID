"""Settings: defaults < /config/settings.json < environment variables.

Settings edited in the web UI are written to /config/settings.json.
Environment variables (e.g. TVDB_API_KEY) always win, so secrets can live
in docker-compose / the Unraid template instead of the JSON file.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

MEDIA_ROOT = Path(os.environ.get("MEDIA_ROOT", "/media")).resolve()
CONFIG_DIR = Path(os.environ.get("CONFIG_DIR", "/config")).resolve()
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/cache")).resolve()

SETTINGS_FILE = CONFIG_DIR / "settings.json"

# key -> (default, env var name, is_secret)
SCHEMA: dict[str, tuple[object, str, bool]] = {
    # TVDB (required)
    "tvdb_api_key": ("", "TVDB_API_KEY", True),
    "tvdb_pin": ("", "TVDB_PIN", True),
    # OpenSubtitles
    "opensubtitles_api_key": ("", "OPENSUBTITLES_API_KEY", True),
    "opensubtitles_username": ("", "OPENSUBTITLES_USERNAME", False),
    "opensubtitles_password": ("", "OPENSUBTITLES_PASSWORD", True),
    # One language for everything that reads a file: the language audio is
    # transcribed in, the subtitle tracks worth reading, and the language the
    # reference text is written in. Two separate knobs here let ASR answer in a
    # language the references were not, so there is only this one.
    "language": ("en", "MEDIA_LANGUAGE", False),
    "sidecar_subtitles": (True, "SIDECAR_SUBTITLES", False),
    # Transcription: local faster-whisper by default, or an OpenAI-compatible
    # audio transcription endpoint (which can host Whisper, Parakeet, Canary, ...).
    "whisper_enabled": (True, "WHISPER_ENABLED", False),
    "whisper_backend": ("local", "WHISPER_BACKEND", False),
    "whisper_model": ("small", "WHISPER_MODEL", False),
    "whisper_device": ("auto", "WHISPER_DEVICE", False),
    "whisper_compute_type": ("default", "WHISPER_COMPUTE_TYPE", False),
    "whisper_base_url": ("", "WHISPER_BASE_URL", False),
    "whisper_api_key": ("", "WHISPER_API_KEY", True),
    "whisper_remote_model": ("whisper-1", "WHISPER_REMOTE_MODEL", False),
    "whisper_remote_timeout": (3600, "WHISPER_REMOTE_TIMEOUT", False),
    # How much audio the server will accept in one request. NeMo-Speech.cpp refuses
    # anything over its compiled kMaxUploadSeconds (3600), and asking before sending
    # beats encoding and uploading 87 minutes of a two-parter just to be told no.
    # Set 0 for a backend with no cap.
    "whisper_max_upload_seconds": (3600, "WHISPER_MAX_UPLOAD_SECONDS", False),
    # How many files' transcripts to fetch at once. Measured on the LAN ASR server:
    # 4 concurrent requests give ~1.3x its throughput, 8 give ~2.1x (it saturates
    # near 2x, so a bigger pool is just queueing). Only a file with no cached
    # transcript goes over the wire, so a warm scan never waits on this.
    "transcribe_workers": (4, "TRANSCRIBE_WORKERS", False),
    # LLM (OpenAI-compatible; Ollama: http://host:11434/v1)
    "llm_enabled": (False, "LLM_ENABLED", False),
    "llm_base_url": ("http://ollama:11434/v1", "LLM_BASE_URL", False),
    "llm_api_key": ("", "LLM_API_KEY", True),
    "llm_model": ("qwen3:8b", "LLM_MODEL", False),
    # Title cards (OCR of the episode title shown on screen)
    "titlecard_enabled": (True, "TITLECARD_ENABLED", False),
    "titlecard_scan_seconds": (180, "TITLECARD_SCAN_SECONDS", False),
    "titlecard_fps": (1.0, "TITLECARD_FPS", False),
    "titlecard_vision": (False, "TITLECARD_VISION", False),
    "titlecard_ocr_device": ("auto", "TITLECARD_OCR_DEVICE", False),   # auto | cpu
    "titlecard_hwaccel": ("auto", "TITLECARD_HWACCEL", False),         # auto | off | cuda | vaapi
    "llm_vision_model": ("", "LLM_VISION_MODEL", False),
    # Fast typed-decision model (Laya in LM Studio). It only rank-breaks the
    # matcher's candidates; it never becomes trusted evidence.
    "decision_enabled": (False, "DECISION_ENABLED", False),
    "decision_base_url": ("http://10.0.0.11:1234/v1", "DECISION_BASE_URL", False),
    "decision_model": ("laya", "DECISION_MODEL", False),
    "decision_timeout": (30, "DECISION_TIMEOUT", False),
    "decision_min_confidence": (0.15, "DECISION_MIN_CONFIDENCE", False),
    "decision_max_candidates": (4, "DECISION_MAX_CANDIDATES", False),
    "decision_max_input_chars": (800, "DECISION_MAX_INPUT_CHARS", False),
    "decision_evidence": (True, "DECISION_EVIDENCE", False),
    "decision_conflict_similarity": (0.85, "DECISION_CONFLICT_SIMILARITY", False),
    "reference_repair": (True, "REFERENCE_REPAIR", False),
    "decision_sample_seconds": (600, "DECISION_SAMPLE_SECONDS", False),
    "decision_sample_start": (0, "DECISION_SAMPLE_START", False),
    "decision_fast_path": (False, "DECISION_FAST_PATH", False),
    "decision_sample_expand": (False, "DECISION_SAMPLE_EXPAND", False),
    "decision_sample_max_windows": (5, "DECISION_SAMPLE_MAX_WINDOWS", False),
    "decision_mismatch_windows": (2, "DECISION_MISMATCH_WINDOWS", False),
    # Sonarr
    "sonarr_url": ("", "SONARR_URL", False),
    "sonarr_api_key": ("", "SONARR_API_KEY", True),
    "sonarr_rescan_after_apply": (True, "SONARR_RESCAN_AFTER_APPLY", False),
    # Scan a file as soon as Sonarr imports it (Sonarr → Connect → Webhook → /api/sonarr/webhook)
    "sonarr_import_scan": (True, "SONARR_IMPORT_SCAN", False),
    "check_imports": (True, "CHECK_IMPORTS", False),
    # Matching
    "window_seconds": (90, "WINDOW_SECONDS", False),
    "window_step_seconds": (45, "WINDOW_STEP_SECONDS", False),
    "min_segment_seconds": (240, "MIN_SEGMENT_SECONDS", False),
    "min_score": (0.12, "MIN_SCORE", False),
    "min_margin": (0.04, "MIN_MARGIN", False),
    # Naming (Sonarr "Standard" default)
    "naming_format": ("{series} - S{season:02d}E{episode:02d} - {title} {quality}", "NAMING_FORMAT", False),
    "multi_episode_style": ("prefixed_range", "MULTI_EPISODE_STYLE", False),
    # Filename cleaning, as Sonarr's Media Management → "Replace Illegal Characters" /
    # "Colon Replacement" (smart|delete|dash|space_dash|space_dash_space|custom)
    "replace_illegal_characters": (True, "REPLACE_ILLEGAL_CHARACTERS", False),
    "colon_replacement": ("smart", "COLON_REPLACEMENT", False),
    "colon_replacement_custom": ("", "COLON_REPLACEMENT_CUSTOM", False),
    "season_folder_format": ("Season {season:02d}", "SEASON_FOLDER_FORMAT", False),
    "specials_folder": ("Specials", "SPECIALS_FOLDER", False),
    "backup_folder": ("_episodeid_backup", "BACKUP_FOLDER", False),
    "allow_splits": (True, "ALLOW_SPLITS", False),
    # Look for added / replaced / removed files and rescan when there are any (0 = off)
    "auto_check_hours": (24, "AUTO_CHECK_HOURS", False),
    # Web UI login (HTTP Basic). Both set = login required; either empty = open.
    "auth_username": ("", "AUTH_USERNAME", False),
    "auth_password": ("", "AUTH_PASSWORD", True),
}

_lock = threading.Lock()

# Languages the app can honour. Media files tag audio and subtitle streams with
# ISO 639-2 (either the terminological or the bibliographic code) while settings
# use ISO 639-1, so both spellings live here together and the UI only offers what
# stream selection can actually match. Codes checked against the ISO 639-2 list.
LANGUAGES = {
    "en": ("English", ("eng", "english")),
    "es": ("Spanish", ("spa",)),
    "fr": ("French", ("fra", "fre")),
    "de": ("German", ("deu", "ger")),
    "it": ("Italian", ("ita",)),
    "pt": ("Portuguese", ("por",)),
    "nl": ("Dutch", ("nld", "dut")),
    "pl": ("Polish", ("pol",)),
    "sv": ("Swedish", ("swe",)),
    "da": ("Danish", ("dan",)),
    "fi": ("Finnish", ("fin",)),
    "no": ("Norwegian", ("nor",)),
    "cs": ("Czech", ("ces", "cze")),
    "hu": ("Hungarian", ("hun",)),
    "ro": ("Romanian", ("ron", "rum")),
    "el": ("Greek", ("ell", "gre")),
    "ru": ("Russian", ("rus",)),
    "uk": ("Ukrainian", ("ukr",)),
    "tr": ("Turkish", ("tur",)),
    "he": ("Hebrew", ("heb",)),
    "id": ("Indonesian", ("ind",)),
    "ja": ("Japanese", ("jpn",)),
    "ko": ("Korean", ("kor",)),
    "zh": ("Chinese", ("zho", "chi")),
}


def language_choices() -> list[tuple[str, str]]:
    """Dropdown options: (code, name) in listing order."""
    return [(code, name) for code, (name, _tags) in LANGUAGES.items()]


def language_tags(code: str) -> set[str]:
    """Every tag spelling that means this language, including the code itself."""
    code = (code or "").strip().lower()
    if not code:
        return set()
    _name, tags = LANGUAGES.get(code, ("", ()))
    return {code} | {t.lower() for t in tags}


def language_iso3(code: str) -> str:
    """The ISO 639-2/T spelling — what TVDB and other metadata services expect.

    Episode titles come back in this language and end up in suggested file names,
    so metadata should not drift from the language the dialogue was matched in.
    """
    _name, tags = LANGUAGES.get((code or "").strip().lower(), ("", ()))
    return tags[0] if tags else "eng"


def _coerce(value, default):
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int) and not isinstance(default, bool):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return "" if value is None else str(value)


# Older builds asked for the subtitle language, the transcription language and the
# metadata language separately. Those could disagree, and a transcript in a language
# the references were not written in matches the wrong episode with full confidence.
# An existing settings.json keeps whichever dialogue choice it recorded; the retired
# keys are dropped so they cannot quietly override the one selector.
_LEGACY_LANGUAGE = ("subtitle_language", "whisper_language")
_RETIRED_KEYS = _LEGACY_LANGUAGE + ("tvdb_language",)


def _fold_language(stored: dict) -> dict:
    if "language" not in stored:
        for key in _LEGACY_LANGUAGE:
            if stored.get(key):
                stored["language"] = stored[key]
                break
    return {k: v for k, v in stored.items() if k not in _RETIRED_KEYS}


def _load_file() -> dict:
    try:
        return _fold_language(json.loads(SETTINGS_FILE.read_text()))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def get_settings() -> dict:
    with _lock:
        stored = _load_file()
    out = {}
    for key, (default, env, _secret) in SCHEMA.items():
        val = stored.get(key, default)
        if env in os.environ:
            val = os.environ[env]
        out[key] = _coerce(val, default)
    # A blank language is not a choice, it is an accident: normalise it here, once,
    # so no caller can forward it to the ASR server, where it means "guess for
    # yourself" -- and a transcript in a language the references were not written
    # in matches the wrong episode with full confidence.
    out["language"] = (str(out["language"]).strip().lower() or SCHEMA["language"][0])
    return out


def env_locked_keys() -> list[str]:
    return [k for k, (_, env, _) in SCHEMA.items() if env in os.environ]


def public_settings() -> dict:
    """Settings for the UI: secrets are masked, never returned."""
    s = get_settings()
    out = {}
    for key, (_d, _e, secret) in SCHEMA.items():
        out[key] = ("********" if s[key] else "") if secret else s[key]
    return out


def save_settings(update: dict) -> None:
    with _lock:
        stored = _load_file()
        for key, val in update.items():
            if key not in SCHEMA:
                continue
            default, _env, secret = SCHEMA[key]
            if secret and val == "********":
                continue  # unchanged masked value
            stored[key] = _coerce(val, default)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        tmp = SETTINGS_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(stored, indent=2))
        tmp.replace(SETTINGS_FILE)


def safe_media_path(rel_or_abs: str | Path) -> Path:
    """Resolve a path and guarantee it stays inside MEDIA_ROOT."""
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = MEDIA_ROOT / p
    p = p.resolve()
    if p != MEDIA_ROOT and MEDIA_ROOT not in p.parents:
        raise ValueError(f"Path is outside the media root: {p}")
    return p
