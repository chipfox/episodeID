"""One language: the setting, the ASR request, the track picker and the dropdown agree.

This replaces the old pair of knobs (a subtitle language and an ASR language),
which could disagree — and a transcript in a language the references were not
written in matches the wrong episode with full confidence.
"""
import json
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_language"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    WHISPER_BACKEND="openai",
    WHISPER_BASE_URL="http://asr.example.test:8000/v1",
    WHISPER_REMOTE_MODEL="salm",
)
for gone in ("LANGUAGE", "MEDIA_LANGUAGE", "WHISPER_LANGUAGE", "SUBTITLE_LANGUAGE"):
    os.environ.pop(gone, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import config, transcribe  # noqa: E402
from app.media_text import _LANG_STOPWORDS, detect_language  # noqa: E402

CFG = Path(os.environ["CONFIG_DIR"])
CFG.mkdir(parents=True, exist_ok=True)

# Default is English with no settings file and no environment override.
assert config.get_settings()["language"] == "en"

# The dropdown and the stream picker read one table, so the UI cannot offer a
# language that track selection cannot match.
choices = config.language_choices()
assert choices and choices[0] == ("en", "English"), choices[:3]
for code, name in choices:
    assert name and code in config.language_tags(code), (code, name)
# Media files tag streams with ISO 639-2 (either spelling), settings use ISO 639-1.
assert {"deu", "ger"} <= config.language_tags("de")
assert {"spa", "es"} <= config.language_tags("es")
assert {"fra", "fre"} <= config.language_tags("fr")
assert config.language_tags("") == set()

# Episode titles are metadata too, so they follow the same choice. TVDB wants the
# ISO 639-2/T spelling, which is the first tag listed for each language.
assert config.language_iso3("de") == "deu" and config.language_iso3("zh") == "zho"
assert config.language_iso3("cy") == "eng"  # unlisted language: TVDB default, not a crash

# A settings file written before this change keeps its choice instead of being
# silently reset, and the retired keys do not survive the next save.
(CFG / "settings.json").write_text(json.dumps(
    {"subtitle_language": "fr", "whisper_language": "fr", "tvdb_language": "eng",
     "media_root": "/data/tv"}))
assert config.get_settings()["language"] == "fr"
assert "subtitle_language" not in config.get_settings()
config.save_settings({"whisper_enabled": False})
stored = json.loads((CFG / "settings.json").read_text())
assert stored.get("language") == "fr" and "subtitle_language" not in stored
assert "whisper_language" not in stored and stored.get("media_root") == "/data/tv"
assert "tvdb_language" not in stored  # metadata language is no longer a separate knob

# The transcription request always names a language.
(CFG / "settings.json").write_text(json.dumps({"language": "en"}))
video = W / "media" / "Show" / "Season 01" / "Show - S01E01.mkv"
video.parent.mkdir(parents=True, exist_ok=True)
video.write_bytes(b"not a real video")
transcribe.probe = lambda _v: {"format": {"duration": "60"}, "streams": []}
transcribe._encode_remote_audio = lambda *_a, **_k: b"encoded"


class Response:
    status_code = 200
    text = ""

    def json(self):
        return {"duration": 60, "segments": [{"start": 0.5, "end": 2.5, "text": "line one"}]}


sent = {}


def fake_post(url, *, headers, data, files, timeout):
    sent.update(data)
    return Response()


transcribe.httpx.post = fake_post
transcribe.transcribe(video)
assert sent.get("language") == "en", sent

(CFG / "settings.json").write_text(json.dumps({"language": "es"}))
transcribe.transcribe(video)
assert sent.get("language") == "es", sent

# A blank setting must not hand the language decision back to the ASR server.
(CFG / "settings.json").write_text(json.dumps({"language": ""}))
transcribe.transcribe(video)
assert sent.get("language") == "en", sent

# A secondary-language scene must not make the episode that language.
en = "the truth is out there and we have to talk about the case file tonight"
es = "la verdad esta ahi afuera y tenemos que hablar del expediente esta noche"
assert detect_language([en] * 9 + [es], "en") is None
assert detect_language([es] * 9 + [en], "en") == "es"
assert detect_language(["short line"], "en") is None

# A language the detector has no stopword list for must never report a foreign
# language: with nothing to compare against, the margin test would compare against
# zero and flag every file in the library as the wrong language.
assert detect_language([es] * 9 + [en], "nl") is None
assert detect_language([es] * 9 + [en], "ru") is None
# The two language tables must not drift: anything the detector scores has to be
# a language the app recognises, or its code would never reach detection at all.
assert set(_LANG_STOPWORDS) <= {code for code, _ in choices}, set(_LANG_STOPWORDS) - {c for c, _ in choices}

# The UI is served the same list rather than keeping its own copy, and sees the
# same effective language the backend uses.
(CFG / "settings.json").write_text(json.dumps({"language": "es"}))
from app import main  # noqa: E402

body = main.read_settings()
assert body["languages"] == choices, body.get("languages")
assert body["settings"]["language"] == "es"
for gone in ("whisper_language", "subtitle_language", "tvdb_language"):
    assert gone not in body["settings"], gone
(CFG / "settings.json").write_text(json.dumps({"language": ""}))
assert main.read_settings()["settings"]["language"] == "en"

# No user-facing string may name a language the setting does not. Two of these
# notes used to say "English references" unconditionally, which is precisely the
# wrong-language report this change exists to prevent.
offenders = []
for src in sorted(Path(__file__).resolve().parents[1].glob("app/*.py")):
    for n, line in enumerate(src.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if "English" in code and '"en"' not in code:
            offenders.append(f"{src.name}:{n}: {line.strip()}")
assert not offenders, offenders

print("language_test: OK")
