"""Audio transcription.

The default backend is faster-whisper running in this container.  An
OpenAI-compatible remote backend is also supported: it uploads a compressed,
mono audio track to ``/audio/transcriptions`` and consumes timestamped
segments from the response.  That lets a LAN service host Whisper, Parakeet,
Canary, or another ASR model without adding its heavyweight runtime to the
EpisodeID image.

Audio is decoded with the ffmpeg binary rather than faster-whisper's built-in
PyAV decoder: faster-whisper 1.2.x calls av.open(metadata_errors=...), which
newer PyAV releases removed, and ffmpeg also handles every container/codec
the library might hold."""
from __future__ import annotations

import io
import re
import subprocess
import threading
import wave
from pathlib import Path

import httpx
import numpy as np

from .config import CACHE_DIR, get_settings
from .media_text import Cue, _lang_ok, duration_of, probe

SAMPLE_RATE = 16000  # what Whisper expects

_model = None
_model_key = None
_lock = threading.Lock()
_gpu_error: str | None = None  # set when GPU transcription failed; CPU is used afterwards


def _backend(settings: dict | None = None) -> str:
    """Return the normalized transcription backend name."""
    s = settings or get_settings()
    value = str(s.get("whisper_backend") or "local").strip().lower()
    return "openai" if value in {"openai", "remote", "openai-compatible", "api"} else "local"


def backend_name() -> str:
    """The backend shown in status and diagnostics."""
    return _backend()


def _transcription_model(settings: dict | None = None) -> str:
    s = settings or get_settings()
    if _backend(s) == "openai":
        return str(s.get("whisper_remote_model") or "whisper-1").strip() or "whisper-1"
    return str(s.get("whisper_model") or "small").strip() or "small"


def source_label(settings: dict | None = None) -> str:
    """Stable cache/plan label identifying the ASR implementation used."""
    s = settings or get_settings()
    if _backend(s) == "openai":
        return f"remote {_transcription_model(s)}"
    return f"whisper {_transcription_model(s)}"


def _load():
    global _model, _model_key
    s = get_settings()
    device = s["whisper_device"]
    compute = s["whisper_compute_type"]
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # noqa: BLE001
            device = "cpu"
    if _gpu_error:
        device = "cpu"
    if compute == "default":
        compute = "int8"
        if device == "cuda":
            # Pascal and older run float16 very slowly (or not at all): use the best
            # type this GPU supports rather than assuming float16.
            try:
                import ctranslate2
                supported = ctranslate2.get_supported_compute_types("cuda")
            except Exception:  # noqa: BLE001
                supported = {"float32"}
            compute = next(c for c in ("float16", "int8_float16", "int8_float32", "int8", "float32")
                           if c in supported or c == "float32")
    model_name = s["whisper_model"]
    if _gpu_error and re.match(r"(large|medium|distil-large)", model_name):
        model_name = "small"  # GPU-sized model would crawl on the CPU fallback
    key = (model_name, device, compute)
    if _model is None or _model_key != key:
        from faster_whisper import WhisperModel
        _model = WhisperModel(model_name, device=device, compute_type=compute,
                              download_root=str(CACHE_DIR / "whisper"))
        _model_key = key
    return _model


def _pick_audio_stream(video: Path, lang: str) -> str:
    """ffmpeg map spec for the best audio track: matching language, not commentary."""
    try:
        streams = [s for s in probe(video).get("streams", []) if s.get("codec_type") == "audio"]
    except Exception:  # noqa: BLE001
        return "0:a:0"
    if not streams:
        raise RuntimeError("file has no audio track")

    def rank(s):
        title = ((s.get("tags") or {}).get("title") or "").lower()
        disp = s.get("disposition") or {}
        return (_lang_ok(s, lang),
                "commentary" not in title and not disp.get("comment"),
                bool(disp.get("default")), -s["index"])

    return f"0:{max(streams, key=rank)['index']}"


def decode_audio(video: Path, lang: str = "", start: float = 0.0,
                 duration: float | None = None) -> np.ndarray:
    """Audio as 16 kHz mono float32, optionally clipped to a sample window."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video),
            "-map", _pick_audio_stream(video, lang), "-vn", "-sn", "-dn"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-acodec", "pcm_f32le", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=3600)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg could not decode audio: "
                           f"{r.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def _remote_endpoint(settings: dict) -> str:
    base = str(settings.get("whisper_base_url") or "").strip().rstrip("/")
    if not base:
        raise RuntimeError("WHISPER_BASE_URL is required when WHISPER_BACKEND=openai")
    return base if base.endswith("/audio/transcriptions") else f"{base}/audio/transcriptions"


def _encode_remote_audio(video: Path, lang: str, start: float = 0.0,
                         duration: float | None = None) -> bytes:
    """Encode a broadly accepted, compact upload for OpenAI-compatible APIs."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(video),
            "-map", _pick_audio_stream(video, lang), "-vn", "-sn", "-dn"]
    if duration is not None and duration > 0:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "libmp3lame",
            "-b:a", "64k", "-f", "mp3", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=3600)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"ffmpeg could not encode audio for remote transcription: "
                           f"{r.stderr.decode('utf-8', 'replace').strip()[-300:]}")
    return r.stdout


def _too_long(minutes: float) -> RuntimeError:
    """The one failure a re-run can never fix, said plainly.

    The server caps how long one request may be, so a two-part or feature-length
    file is refused whole. A plan reason should not hide that in a payload.
    """
    return RuntimeError(
        f"this audio is {minutes:.0f} min — longer than the transcription server "
        "accepts in one request, so its audio cannot be read")


def _refused(body: str, status: int, minutes: float | None = None) -> RuntimeError:
    """Say what to do about a refused upload instead of dumping the server's JSON."""
    if minutes is not None and status == 400 and "exceeds configured limit" in body:
        return _too_long(minutes)
    return RuntimeError(f"remote transcription failed ({status}): {body[:300]}")


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _remote_cues(payload, total: float, offset: float = 0.0) -> list[Cue]:
    """Accept verbose_json segments, word timestamps, or a plain text response.
    Times are shifted by offset so cues stay relative to the whole file."""
    if isinstance(payload, list):
        payload = {"segments": payload}
    if not isinstance(payload, dict):
        payload = {"text": str(payload or "")}

    segments = payload.get("segments")
    if isinstance(segments, list):
        cues = []
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            start = max(0.0, _float(segment.get("start"))) + offset
            end = _float(segment.get("end")) + offset
            if end <= start:
                next_start = (_float(segments[i + 1].get("start")) + offset
                              if i + 1 < len(segments) and isinstance(segments[i + 1], dict)
                              else offset)
                end = next_start if next_start > start else max(total + offset, start + 1.0)
            cues.append(Cue(start, end, text))
        if cues:
            return cues

    # A few compatible servers expose words but omit segments. Group them into
    # modest chunks so matching still has useful time locality.
    words = payload.get("words")
    if isinstance(words, list):
        cues, current = [], []
        current_start = current_end = 0.0
        for word in words:
            if not isinstance(word, dict) or not str(word.get("word") or "").strip():
                continue
            start = max(0.0, _float(word.get("start"), current_end - offset)) + offset
            end = _float(word.get("end"), start + 0.5) + (offset if word.get("end") else 0.0)
            if not current:
                current_start = start
            current.append(str(word.get("word")).strip())
            current_end = max(end, start + 0.1)
            if current_end - current_start >= 8.0 or len(current) >= 32:
                cues.append(Cue(current_start, current_end, " ".join(current)))
                current = []
        if current:
            cues.append(Cue(current_start, current_end, " ".join(current)))
        if cues:
            return cues

    text = str(payload.get("text") or "").strip()
    if text:
        # Preserve usable text even when a minimal server does not return
        # timestamps. A timestamped response remains strongly preferred.
        return [Cue(offset, offset + max(total, 1.0), text)]
    raise RuntimeError("remote transcription returned no text or timestamped segments")


def _remote_transcribe(video: Path, ctx=None, start: float = 0.0,
                       duration: float | None = None) -> list[Cue]:
    s = get_settings()
    lang = s["language"]
    model = _transcription_model(s)
    if ctx:
        ctx.status(f"Remote transcription: {video.name}")
    total = duration_of(probe(video))
    # How much audio we are about to put on the wire. Deciding this *before*
    # encoding costs one ffprobe; learning it from the server's refusal costs
    # 40 s of encoding and uploading an 87-minute file, on every single scan.
    send = total - max(0.0, start)
    if duration is not None:
        send = min(send, duration)
    limit = int(s.get("whisper_max_upload_seconds") or 0)
    if limit > 0 and send > limit:
        raise _too_long(send / 60)
    if start or duration is not None:
        audio = _encode_remote_audio(video, lang, start, duration)
    else:
        audio = _encode_remote_audio(video, lang)
    if ctx:
        ctx.check_cancel()
        where = f" {start:.0f}–{start + duration:.0f}s" if duration else ""
        ctx.log(f"  uploading {video.name}{where} to the remote transcription service ({model})…")
    headers = {}
    if s["whisper_api_key"]:
        headers["Authorization"] = f"Bearer {s['whisper_api_key']}"
    data = {"model": model, "response_format": "verbose_json", "language": lang}
    try:
        timeout = max(1, int(s["whisper_remote_timeout"]))
        response = httpx.post(
            _remote_endpoint(s), headers=headers, data=data,
            files={"file": (f"{video.stem}.mp3", audio, "audio/mpeg")}, timeout=timeout)
    except (httpx.HTTPError, ValueError) as e:
        raise RuntimeError(f"remote transcription request failed: {e}") from e
    if response.status_code < 200 or response.status_code >= 300:
        raise _refused(response.text, response.status_code, send / 60)
    try:
        payload = response.json()
    except ValueError:
        payload = {"text": response.text}
    # Some servers report the clip's own length and some report nothing; we already
    # know what we sent, so a second ffprobe here would be pure overhead.
    got = _float(payload.get("duration")) if isinstance(payload, dict) else 0.0
    cues = _remote_cues(payload, got if got > 0 else max(send, 1.0), offset=start)
    if ctx:
        ctx.check_cancel()
        ctx.status(f"Remote transcription: {video.name} done")
    return cues


def _test_audio() -> bytes:
    """Small valid WAV used by the Settings → Whisper connection test."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"\\0\\0" * (SAMPLE_RATE // 2))
    return out.getvalue()


def test_remote() -> str:
    """Make a tiny real request so the UI test checks reachability and auth."""
    s = get_settings()
    model = _transcription_model(s)
    data = {"model": model, "response_format": "json"}
    headers = {}
    if s["whisper_api_key"]:
        headers["Authorization"] = f"Bearer {s['whisper_api_key']}"
    try:
        response = httpx.post(
            _remote_endpoint(s), headers=headers, data=data,
            files={"file": ("episodeid-test.wav", _test_audio(), "audio/wav")},
            timeout=min(max(1, int(s["whisper_remote_timeout"])), 60))
    except (httpx.HTTPError, ValueError) as e:
        raise RuntimeError(f"remote transcription request failed: {e}") from e
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"remote transcription failed ({response.status_code}): "
                           f"{response.text[:300]}")
    return f"Remote transcription endpoint replied for '{model}'."


def transcribe(video: Path, ctx=None, start: float = 0.0,
               duration: float | None = None) -> list[Cue]:
    s = get_settings()
    global _gpu_error, _model
    if _backend(s) == "openai":
        with _lock:
            return _remote_transcribe(video, ctx, start, duration)
    lang = s["language"]
    audio = decode_audio(video, lang, start, duration)
    with _lock:
        try:
            return _run(_load(), audio, lang, video, ctx, offset=start)
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "Cancelled" or not _model_key or _model_key[1] != "cuda":
                raise
            # GPU transcription failed (unsupported GPU/driver, out of memory…): CPU from now on.
            _gpu_error = str(e).strip().splitlines()[-1][:300] if str(e).strip() else type(e).__name__
            _model = None
            if ctx:
                ctx.log(f"  Whisper on the GPU failed ({_gpu_error}); using the CPU instead "
                        "(with the 'small' model if a large one was selected).")
            return _run(_load(), audio, lang, video, ctx, offset=start)


def _run(model, audio, lang, video: Path, ctx, offset: float = 0.0) -> list[Cue]:
    segments, info = model.transcribe(
        audio, language=lang, vad_filter=True,
        beam_size=1, condition_on_previous_text=False)
    cues = []
    total = info.duration or (len(audio) / SAMPLE_RATE) or 1
    for seg in segments:
        text = seg.text.strip()
        if text:
            cues.append(Cue(seg.start + offset, seg.end + offset, text))
        if ctx:
            ctx.check_cancel()
            ctx.status(f"Whisper: {video.name} {seg.end / total:.0%}")
    return cues
