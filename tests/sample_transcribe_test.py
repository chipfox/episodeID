"""Remote ASR sample windows keep file-relative cue times."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_sample_transcribe"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    WHISPER_BACKEND="openai",
    WHISPER_BASE_URL="http://asr.example.test:8000/v1",
    WHISPER_REMOTE_MODEL="salm",
    MEDIA_LANGUAGE="en",
    WHISPER_REMOTE_TIMEOUT="17",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import transcribe  # noqa: E402

video = W / "media" / "Show" / "Season 01" / "Show - S01E01.mkv"
video.parent.mkdir(parents=True)
video.write_bytes(b"not a real video")

calls = []
transcribe.probe = lambda _video: {"format": {"duration": "2700"}, "streams": []}


def fake_encode(_video, _lang, start=0.0, duration=None):
    calls.append((_video, _lang, start, duration))
    return b"encoded-sample"


class Response:
    status_code = 200
    text = ""

    def json(self):
        return {"duration": 60, "segments": [
            {"start": 0.5, "end": 2.5, "text": "sample line one"},
            {"start": 3.0, "end": 5.0, "text": "sample line two"},
        ]}


def fake_post(url, *, headers, data, files, timeout):
    calls.append(("post", url, data, files, timeout))
    return Response()


transcribe._encode_remote_audio = fake_encode
transcribe.httpx.post = fake_post

cues = transcribe.transcribe(video, start=120, duration=60)
assert calls[0][2:] == (120.0, 60.0), calls[0]
assert (cues[0].start, cues[0].end) == (120.5, 122.5), cues
assert (cues[1].start, cues[1].end) == (123.0, 125.0), cues
assert calls[1][1] == "http://asr.example.test:8000/v1/audio/transcriptions"
assert calls[1][2]["model"] == "salm"

# The existing whole-file call shape is unchanged.
calls.clear()
whole = transcribe.transcribe(video)
assert calls[0][2:] == (0.0, None), calls[0]
assert whole[0].start == 0.5

# A minimal plain-text ASR response gets the sample offset too.
class TextOnly(Response):
    def json(self):
        return {"text": "plain sample transcript", "duration": 60}


transcribe.httpx.post = lambda *args, **kwargs: TextOnly()
plain = transcribe.transcribe(video, start=300, duration=60)
assert plain[0].start == 300 and plain[0].end == 360, plain[0]
print("PASS")

# A file the transcription server refuses because of its length must say so plainly:
# a re-run can never fix it, and the plan shows this text to a human.
class TooLong(Response):
    status_code = 400
    text = ('{"error":{"code":null,"message":"Show - S01E01.mkv: audio exceeds '
            'configured limit","param":null,"type":"invalid_request_error"}}')


transcribe.httpx.post = lambda *args, **kwargs: TooLong()
try:
    transcribe.transcribe(video)
    raise AssertionError("a refused upload must raise")
except RuntimeError as e:
    assert "45 min" in str(e) and "{" not in str(e), e

# Any other refusal still shows what the server actually said.
class Broken(Response):
    status_code = 500
    text = "model load failed"


transcribe.httpx.post = lambda *args, **kwargs: Broken()
try:
    transcribe.transcribe(video)
    raise AssertionError("a server error must raise")
except RuntimeError as e:
    assert "model load failed" in str(e), e
