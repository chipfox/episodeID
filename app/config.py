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
    # Two references this similar are one transcript filed twice (a wiki page under
    # the wrong number, or both parts of a two-parter): the matcher and the plan
    # both refuse to treat a match on one as proof of either.
    "reference_duplicate_similarity": (0.85, "REFERENCE_DUPLICATE_SIMILARITY", False),
    "reference_repair": (True, "REFERENCE_REPAIR", False),
    # Summary matching: the fallback when an episode has no reference transcript.
    # The index is fitted on reference dialogue; the gate is calibrated per series.
    "summary_match_enabled": (True, "SUMMARY_MATCH_ENABLED", False),
    "summary_min_refs": (20, "SUMMARY_MIN_REFS", False),
    "summary_target_precision": (0.95, "SUMMARY_TARGET_PRECISION", False),
    "summary_auto_precision": (0.99, "SUMMARY_AUTO_PRECISION", False),
    "summary_default_min_score": (0.25, "SUMMARY_DEFAULT_MIN_SCORE", False),
    "summary_default_min_margin": (0.05, "SUMMARY_DEFAULT_MIN_MARGIN", False),
    # Sampling: read the first N seconds of an eligible single-episode file instead of
    # the whole thing. These were once named decision_* (after the retired decision
    # model); the old keys and env names still work — see RENAMED / RENAMED_ENV below.
    "sample_seconds": (600, "SAMPLE_SECONDS", False),
    "sample_start": (0, "SAMPLE_START", False),
    "sampling_fast_path": (False, "SAMPLING_FAST_PATH", False),
    "sample_expand": (False, "SAMPLE_EXPAND", False),
    "sample_max_windows": (5, "SAMPLE_MAX_WINDOWS", False),
    "sample_mismatch_windows": (2, "SAMPLE_MISMATCH_WINDOWS", False),
    # When the samples still cannot settle a file, read the whole file once (cached
    # like any full read) and re-match it. Replaces hunting the Full-file match
    # button per review row; the button stays for deliberately-skipped cases.
    "sampling_full_escalate": (True, "SAMPLING_FULL_ESCALATE", False),
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
    # A weak absolute score is not proof of absence: on a short sample the right
    # episode can sit under min_score while most windows still agree on it. The
    # evidence gate accepts exactly that case, and only that case.
    "evidence_gate": (True, "EVIDENCE_GATE", False),
    "gate_min_window_share": (0.30, "GATE_MIN_WINDOW_SHARE", False),
    # A sample must carry dialogue across several windows, not one lucky one:
    # 4 × 90 s windows at a 45 s step is about three minutes of the sample.
    "gate_min_windows": (4, "GATE_MIN_WINDOWS", False),
    "gate_min_words": (60, "GATE_MIN_WORDS", False),
    # References this similar cannot tell their episodes apart (a duplicated wiki
    # transcript, or both parts of a two-parter): matching one proves nothing
    # about the other, so such a segment is never a confident match.
    "reference_twin_similarity": (0.70, "REFERENCE_TWIN_SIMILARITY", False),
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

# Settings renamed after the retired decision model, and to say what they actually
# measure. Old names keep working: an older settings.json is folded onto the new keys
# on read, and the old environment variables are still honoured, so an existing
# deployment (including its stack.env) needs no change. A new name wins if both are set.
RENAMED = {
    "decision_sample_seconds": "sample_seconds",
    "decision_sample_start": "sample_start",
    "decision_fast_path": "sampling_fast_path",
    "decision_sample_expand": "sample_expand",
    "decision_sample_max_windows": "sample_max_windows",
    "decision_mismatch_windows": "sample_mismatch_windows",
    "decision_full_escalate": "sampling_full_escalate",
    "decision_conflict_similarity": "reference_duplicate_similarity",
    "reference_conflict_similarity": "reference_twin_similarity",
}
_RENAMED_ENV = {
    "DECISION_SAMPLE_SECONDS": "sample_seconds",
    "DECISION_SAMPLE_START": "sample_start",
    "DECISION_FAST_PATH": "sampling_fast_path",
    "DECISION_SAMPLE_EXPAND": "sample_expand",
    "DECISION_SAMPLE_MAX_WINDOWS": "sample_max_windows",
    "DECISION_MISMATCH_WINDOWS": "sample_mismatch_windows",
    "DECISION_FULL_ESCALATE": "sampling_full_escalate",
    "DECISION_CONFLICT_SIMILARITY": "reference_duplicate_similarity",
    "REFERENCE_CONFLICT_SIMILARITY": "reference_twin_similarity",
}
# new env name -> old env name, for the lookup in get_settings/env_locked_keys.
_OLD_ENV = {SCHEMA[key][1]: old for old, key in _RENAMED_ENV.items()}


def _fold_renamed(stored: dict) -> dict:
    for old, new in RENAMED.items():
        if old in stored:
            stored.setdefault(new, stored[old])
            stored.pop(old, None)
    return stored


def _fold_language(stored: dict) -> dict:
    if "language" not in stored:
        for key in _LEGACY_LANGUAGE:
            if stored.get(key):
                stored["language"] = stored[key]
                break
    return {k: v for k, v in stored.items() if k not in _RETIRED_KEYS}


def _load_file() -> dict:
    try:
        return _fold_renamed(_fold_language(json.loads(SETTINGS_FILE.read_text())))
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
        else:
            old = _OLD_ENV.get(env)
            if old and old in os.environ:
                val = os.environ[old]
        out[key] = _coerce(val, default)
    # A blank language is not a choice, it is an accident: normalise it here, once,
    # so no caller can forward it to the ASR server, where it means "guess for
    # yourself" -- and a transcript in a language the references were not written
    # in matches the wrong episode with full confidence.
    out["language"] = (str(out["language"]).strip().lower() or SCHEMA["language"][0])
    return out


def env_locked_keys() -> list[str]:
    return [k for k, (_, env, _) in SCHEMA.items()
            if env in os.environ or (_OLD_ENV.get(env) or "") in os.environ]


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
            key = RENAMED.get(key, key)  # an older client may still send the old name
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
