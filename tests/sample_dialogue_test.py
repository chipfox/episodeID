"""Sample dialogue is bounded, separately cached, and falls back safely."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_sample_dialogue"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    WHISPER_ENABLED="false",
    SIDECAR_SUBTITLES="true",
    SUBTITLE_LANGUAGE="en",
    DECISION_SAMPLE_SECONDS="600",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import media_text  # noqa: E402

root = W / "media" / "Show"
root.mkdir(parents=True)
video = root / "Show - S01E01.mkv"
video.write_bytes(b"not a real video")
sidecar = video.with_name(video.stem + ".en.srt")
sidecar.write_text(
    "1\n00:00:01,000 --> 00:00:02,000\nearly line\n\n"
    "2\n00:11:40,000 --> 00:11:42,000\nlate line\n")
media_text.probe = lambda _video: {"format": {"duration": "2700"}, "streams": []}
media_text.extract_embedded = lambda _video, _info, _lang, allow_untagged=True: (
    [media_text.Cue(100, 110, "embedded first"), media_text.Cue(700, 710, "outside sample")],
    "embedded #2")

key_full = media_text._cache_key(video)
key_sample = media_text._cache_key(video, "sample", 0, 600)
assert key_full != key_sample

first = media_text.sample_dialogue(video)
print("sample:", first["source"], first["duration"], first["cues"], first["scope"])
assert first["scope"] == "sample", first
assert first["sample_start"] == 0 and first["sample_end"] == 600, first
assert first["duration"] == 600, first
assert first["cues"] == [[100.0, 110.0, "embedded first"]], first
assert media_text._cache_file(video, "sample", 0, 600).exists()

# The full-file cache is a different file; sampling does not overwrite it.
assert media_text.cached_dialogue(video) is None
assert media_text._cache_file(video) != media_text._cache_file(video, "sample", 0, 600)

# If explicitly tagged embedded text is unavailable, the sidecar is a fallback
# and is clipped to the sample.
video2 = root / "Show - S01E02.mkv"
video2.write_bytes(b"not a real video")
sidecar2 = video2.with_name(video2.stem + ".en.srt")
sidecar2.write_text(
    "1\n00:00:01,000 --> 00:00:02,000\nsidecar early\n\n"
    "2\n00:11:40,000 --> 00:11:42,000\nsidecar late\n")
media_text.extract_embedded = lambda _video, _info, _lang, allow_untagged=True: None
second = media_text.sample_dialogue(video2)
print("fallback:", second["source"], second["cues"], second.get("fallback"))
assert second["source"].startswith("sidecar "), second
assert second["cues"] == [[1.0, 2.0, "sidecar early"]], second

# Untagged embedded text is not accepted as the sample's primary fallback.
video3 = root / "Show - S01E03.mkv"
video3.write_bytes(b"not a real video three")
os.utime(video3, ns=(1_700_000_000_000_000_000, 1_700_000_000_000_000_000))
media_text.extract_embedded = lambda _video, _info, _lang, allow_untagged=False: None
third = media_text.sample_dialogue(video3)
assert third["source"] == "none" and third["cues"] == [], third

# A file shorter than the sample window is a full read, not a sample.
media_text.probe = lambda _video: {"format": {"duration": "300"}, "streams": []}
video4 = root / "Show - S01E04.mkv"
video4.write_bytes(b"not a real video four")
os.utime(video4, ns=(1_700_000_001_000_000_000, 1_700_000_001_000_000_000))
media_text.extract_embedded = lambda _video, _info, _lang, allow_untagged=False: (
    [media_text.Cue(1, 2, "whole file")], "embedded #2")
fourth = media_text.sample_dialogue(video4)
assert fourth["scope"] == "full" and fourth["duration"] == 300, fourth
print("PASS")
