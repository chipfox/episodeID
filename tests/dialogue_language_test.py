"""A subtitle track that is really another language must not become the dialogue.

The X-Files S05 repack ships an untagged Indonesian subrip track inside files
whose audio is English. get_dialogue() preferred the track, so the AI was asked
about the wrong episode's text and answered 99% for the episode in the filename.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.media_text import dialogue_in_target_language  # noqa: E402

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


def test_indonesian_track_is_rejected():
    usable, other = dialogue_in_target_language([[0.0, 3.0, INDONESIAN]], "en")
    assert usable is False, "an Indonesian track must not pass as English dialogue"
    assert other == "id", other


def test_english_track_is_usable():
    usable, other = dialogue_in_target_language([[0.0, 3.0, ENGLISH]], "en")
    assert usable is True, f"English dialogue was rejected as {other}"
    assert other is None


def test_short_text_stays_usable():
    # Too little evidence to be sure: never discard the only text a file has.
    usable, other = dialogue_in_target_language([[0.0, 3.0, "Ayo terus. Lanjutkan."]], "en")
    assert usable is True
    assert other is None


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
