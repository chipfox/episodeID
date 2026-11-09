"""Reading dialogue for several files at once: same order, same errors, less waiting.

The dialogue read is the only part of a scan that waits on the ASR server, so it is
the one part worth parallelising. What must not change: file order, per-file error
capture, and a cancelled scan stopping promptly.
"""
import os
import shutil
import sys
import time
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_dialogue_pool"))
if W.exists():
    shutil.rmtree(W)
(W / "media").mkdir(parents=True)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import pipeline  # noqa: E402
from app.config import get_settings  # noqa: E402

ROOT = W / "media"
FILES = [ROOT / f"The.X.Files.S01E{n:02d}.mkv" for n in range(1, 9)]


class Ctx:
    def __init__(self, cancel_after=None):
        self.lines, self.progress_msgs = [], []
        self.cancel_after = cancel_after
        self.checks = 0

    def log(self, msg):
        self.lines.append(msg)

    def progress(self, frac, msg=None):
        self.progress_msgs.append(msg)

    def check_cancel(self):
        self.checks += 1
        if self.cancel_after is not None and self.checks > self.cancel_after:
            raise pipeline.Cancelled("cancelled")


SLOW = 0.15
pipeline._sample_eligible = lambda *a, **k: False  # every file takes the read path


def slow_dialogue(v, ctx=None, allow_whisper=True):
    time.sleep(SLOW)
    if "E03" in v.name:
        raise RuntimeError("unreadable stream")
    return {"cues": [(0.0, 1.0, f"dialogue from {v.name}")], "source": "test",
            "duration": 60.0, "scope": "full"}


pipeline.get_dialogue = slow_dialogue
s = get_settings()
s["transcribe_workers"] = 4

# Eight files at 0.15 s each: four at a time finishes well under serial time, and the
# results come back in file order with the path relative to the media root intact.
start = time.time()
read = pipeline._read_dialogues(FILES, ROOT, s, {}, {}, False, {}, Ctx())
elapsed = time.time() - start
serial = len(FILES) * SLOW
assert [v.name for v, _r, _d, _e in read] == [f.name for f in FILES], "order must not change"
assert read[0][1] == FILES[0].name
assert elapsed < serial * 0.65, f"{elapsed:.2f}s against a {serial:.2f}s serial baseline"

# The error belongs to its own file; the rest of the scan proceeds.
by_name = {v.name: (d, e) for v, _r, d, e in read}
assert by_name["The.X.Files.S01E03.mkv"][0] is None
assert by_name["The.X.Files.S01E03.mkv"][1] is not None
assert by_name["The.X.Files.S01E04.mkv"][0] is not None
assert all(e is None for v, _r, _d, e in read if "E03" not in v.name)

# One worker keeps the old behaviour: still correct, just no overlap to gain.
s["transcribe_workers"] = 1
start = time.time()
read = pipeline._read_dialogues(FILES, ROOT, s, {}, {}, False, {}, Ctx())
assert [v.name for v, _r, _d, _e in read] == [f.name for f in FILES]
assert time.time() - start > elapsed, "one worker should not be faster than four"

# Cancellation is checked between batches, so a pool cannot run on for the whole
# library after the user hits cancel: 16 files at 4 workers is two batches.
s["transcribe_workers"] = 4
many = FILES + [ROOT / f"The.X.Files.S02E{n:02d}.mkv" for n in range(1, 9)]
ctx = Ctx(cancel_after=1)
try:
    pipeline._read_dialogues(many, ROOT, s, {}, {}, False, {}, ctx)
    raise AssertionError("cancel was swallowed")
except pipeline.Cancelled:
    pass
assert ctx.checks == 2, f"cancel was checked {ctx.checks} times for two batches"

print("PASS")
