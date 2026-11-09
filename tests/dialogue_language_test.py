"""A subtitle track that is really another language must not become the dialogue.

The X-Files S05 repack ships an untagged Indonesian subrip track inside files
whose audio is English. get_dialogue() preferred the track, so the AI was asked
about the wrong episode's text and answered 99% for the episode in the filename.
"""
import json
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_dialogue_language"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    SIDECAR_SUBTITLES="false",   # Bazarr-managed sidecars are not evidence
    WHISPER_ENABLED="false",     # so a rejected track has to end up as "none"
)
for gone in ("LANGUAGE", "MEDIA_LANGUAGE"):
    os.environ.pop(gone, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import media_text  # noqa: E402

INDONESIAN = (
    "Ayo terus. Lanjutkan. Jangan berhenti. Dorong terus. Tiada hasil tanpa usaha. "
    "Lima langkah lagi, lalu kau selesai. Empat. Tiga. Bagus, bagus sekali. Ayo, "
    "perlahan saja. Kau sungguh membaik. Aku puas dengan perkembanganmu, Bobby. "
    "Mundurlah. Tetap siapkan semprotan merica, seperti yang kuberi tahu. Tuan "
    "Piero, apa ini semua benar-benar perlu? Tentu, aku yakin ini perlu. Jadi, "
    "siapa orang ini? Pembunuh polisi. Pembunuh warga sipil. Orang yang benar-benar "
    "tak berguna. Apa yang kau takutkan? Aku tidak takut. Kau tahu pelakunya Linda "
    "Bowman dan bukan Modell. Kau tahu jauh sebelum aku. Aku hampir membunuh rekanku."
)
ENGLISH = (
    "Okay, sweetheart. Just going in for a few things. We won't be long, okay, Polly? "
    "Mommy needs some groceries. I don't like this store, Mommy. We're only going to be "
    "a minute. I want to go home. Let's have fun. No, you cannot have that. Put it back. "
    "The doctors said she would be fine. The tissue quality or scarring is fairly recent. "
    "I would like to personally examine his injuries. There is nothing you can do about it now."
)

# The track picker is not involved here: the dialogue is already cached, which is
# how the pack's tracks reached the AI in the first place.
media_text.probe = lambda _v: {"format": {"duration": "2700"}, "streams": []}
media_text.extract_embedded = lambda *_a, **_k: None


def cached(video: Path, text: str, source: str) -> Path:
    """Write a cache entry the way a build from before this check would have left
    it: no `language` key, which is exactly what makes get_dialogue re-check it."""
    path = media_text._cache_file(video)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source, "duration": 2700.0,
                                "cues": [[0.0, 3.0, text]] * 8}))
    return path


def video_named(name: str) -> Path:
    v = W / "media" / "Season 05" / f"{name}.mkv"
    v.parent.mkdir(parents=True, exist_ok=True)
    v.write_bytes(b"not a real video")
    return v


def test_indonesian_text_is_not_the_english_dialogue():
    assert media_text.detect_language([[0.0, 3.0, INDONESIAN]], "en") == "id"


def test_english_text_is_the_english_dialogue():
    assert media_text.detect_language([[0.0, 3.0, ENGLISH]], "en") is None


def test_short_text_is_never_called_foreign():
    # Too little evidence to be sure: never discard the only text a file has.
    assert media_text.detect_language([[0.0, 3.0, "Ayo terus. Lanjutkan."]], "en") is None


def test_cached_foreign_track_is_dropped():
    video = video_named("poisoned")
    cached(video, INDONESIAN, "embedded #3")
    dlg = media_text.get_dialogue(video, allow_whisper=False)
    assert dlg["source"] == "none", f"foreign subtitle text was used as dialogue: {dlg['source']}"


def test_cached_target_language_track_is_kept():
    video = video_named("english")
    cached(video, ENGLISH, "embedded #3")
    dlg = media_text.get_dialogue(video, allow_whisper=False)
    assert dlg["source"] == "embedded #3", dlg
    assert dlg["language"] is None, dlg


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
