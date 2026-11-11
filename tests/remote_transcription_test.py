"""Remote OpenAI-compatible transcription returns EpisodeID cues."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_remote_transcription"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    WHISPER_BACKEND="openai",
    WHISPER_BASE_URL="http://asr.example.test:8000/v1",
    WHISPER_REMOTE_MODEL="nvidia/parakeet-tdt-0.6b-v3",
    WHISPER_LANGUAGE="en",
    WHISPER_REMOTE_TIMEOUT="17",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import transcribe  # noqa: E402

video = W / "media" / "Show" / "Season 01" / "Show - S01E01.mkv"
video.parent.mkdir(parents=True)
video.write_bytes(b"not a real video")
transcribe._encode_remote_audio = lambda _video, _lang, *_args: b"encoded-audio"
transcribe.probe = lambda _video: {"format": {"duration": "12.3"}, "streams": []}

calls = []


class Response:
    status_code = 200
    text = ""

    def json(self):
        return {
            "duration": 12.3,
            "segments": [
                {"start": 0.25, "end": 2.5, "text": " First line. "},
                {"start": 3, "end": 5.75, "text": "Second line."},
            ],
        }


def fake_post(url, *, headers, data, files, timeout):
    calls.append({"url": url, "headers": headers, "data": data, "files": files,
                  "timeout": timeout})
    return Response()


transcribe.httpx.post = fake_post
cues = transcribe.transcribe(video)
assert transcribe.backend_name() == "openai"
assert transcribe.source_label() == "remote nvidia/parakeet-tdt-0.6b-v3"
assert len(cues) == 2
assert (cues[0].start, cues[0].end, cues[0].text) == (0.25, 2.5, "First line.")
assert (cues[1].start, cues[1].end, cues[1].text) == (3.0, 5.75, "Second line.")
assert calls[0]["url"] == "http://asr.example.test:8000/v1/audio/transcriptions"
assert calls[0]["data"] == {
    "model": "nvidia/parakeet-tdt-0.6b-v3",
    "response_format": "verbose_json",
    "language": "en",
}
assert calls[0]["headers"] == {}
assert calls[0]["timeout"] == 17
name, content, content_type = calls[0]["files"]["file"]
assert name.endswith(".mp3") and content == b"encoded-audio" and content_type == "audio/mpeg"


class TextOnlyResponse(Response):
    def json(self):
        return {"text": "An untimed transcript."}


transcribe.httpx.post = lambda *args, **kwargs: TextOnlyResponse()
plain = transcribe.transcribe(video)
assert len(plain) == 1
assert plain[0].text == "An untimed transcript."
assert plain[0].start == 0.0 and plain[0].end == 12.3

# --- a span longer than one request is split into equal parts and joined --------
# The real case is the 87-minute joined two-parter: the server refuses it whole,
# so it was an ERROR row on every scan and its audio was never read.
transcribe.httpx.post = fake_post
transcribe.probe = lambda _video: {"format": {"duration": "7200"}, "streams": []}
os.environ["WHISPER_MAX_UPLOAD_SECONDS"] = "3600"
calls.clear()
joined = transcribe.transcribe(video)
assert len(calls) == 2, f"expected two uploads, got {len(calls)}"
assert len(joined) == 4, joined
assert (joined[0].start, joined[0].text) == (0.25, "First line.")
assert (joined[2].start, joined[2].text) == (3600.25, "First line."), joined[2]
# Order is preserved and the second part's cues sit after the first part's.
assert [c.start for c in joined] == sorted(c.start for c in joined), joined

# A span that exactly fits is still one request.
transcribe.probe = lambda _video: {"format": {"duration": "3600"}, "streams": []}
calls.clear()
transcribe.transcribe(video)
assert len(calls) == 1, calls
print("PASS")
