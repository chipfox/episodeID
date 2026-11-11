"""The plan says what settled each row, and what changed since the last plan.

A full read the matcher could not settle is decided by the AI's answer; a row that a
scan settled used to look exactly like one that had always been correct. The old
verdict is now kept on the row so the UI can say "this changed, and this decided it".
"""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_observability"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import planner  # noqa: E402
from app.pipeline import _mark_changes  # noqa: E402

root = W / "media" / "The X-Files"
(root / "Season 4").mkdir(parents=True, exist_ok=True)
series = {"path": str(root), "tvdb_id": 1, "options": {},
          "episodes": [{"season": 4, "episode": 12, "title": "Leonard Betts"},
                       {"season": 4, "episode": 14, "title": "Memento Mori"}]}
AI_FILE = "Season 4/The X-Files - S04E12 - Leonard Betts - Bluray-1080p.mkv"
PLAIN_FILE = "Season 4/The X-Files - S04E14 - Memento Mori - Bluray-1080p.mkv"
for rel in (AI_FILE, PLAIN_FILE):
    (root / rel).touch()

items = {it["source"]: it for it in planner.build_plan(series, {"files": [
    # A full read: the dialogue match was a tie (the twin's reference is this episode's
    # transcript), so the AI's answer is what settles it.
    {"rel": AI_FILE, "expected": ["S04E12"], "status": "LOW_CONFIDENCE", "duration": 2700,
     "segments": [{"start": 0, "end": 2700, "code": "S04E14", "score": 0.69,
                   "confidence": "low", "evidence": "whole",
                   "why": "references are near-identical",
                   "llm": {"code": "S04E12", "confidence": 0.95, "reason": "Leonard Betts"}}]},
    # Settled by the dialogue itself.
    {"rel": PLAIN_FILE, "expected": ["S04E14"], "status": "OK", "duration": 2700,
     "segments": [{"start": 0, "end": 2700, "code": "S04E14", "score": 0.5,
                   "confidence": "high", "evidence": "whole"}]},
]})}

ai_row, plain_row = items[AI_FILE], items[PLAIN_FILE]
print("AI row:", ai_row["kind"], ai_row["status"], ai_row["decided_by"],
      ai_row["decided_confidence"], "—", ai_row["reason"])
assert ai_row["kind"] == "ok", ai_row
assert ai_row["decided_by"] == "AI" and ai_row["decided_confidence"] == 0.95, ai_row
assert "(AI)" in ai_row["reason"], ai_row["reason"]
assert "previous" not in ai_row

assert plain_row["decided_by"] == "dialogue" and plain_row["decided_confidence"] is None, plain_row


class Ctx:
    def __init__(self):
        self.lines = []

    def log(self, line):
        self.lines.append(str(line))


# The scan that settles a review row marks it changed and says why in the log.
ctx = Ctx()
new = [dict(ai_row, previous=None), dict(plain_row)]
prev = [{"source": AI_FILE, "kind": "review", "status": "LOW_CONFIDENCE"},
        {"source": PLAIN_FILE, "kind": "ok", "status": "OK"}]
assert _mark_changes(new, prev, ctx) == 1
assert new[0]["previous"] == {"kind": "review", "status": "LOW_CONFIDENCE"}, new[0]
assert "previous" not in new[1], new[1]          # unchanged: no marker
assert any("Changed: " + AI_FILE in line for line in ctx.lines), ctx.lines
assert any("review (LOW_CONFIDENCE) → ok" in line for line in ctx.lines), ctx.lines
assert any("AI decided" in line for line in ctx.lines), ctx.lines

# A file the previous plan never saw is not "changed", and a join row (several files in
# one row) is left alone because its source is not one file.
new2 = [{"source": "Season 4/new.mkv", "kind": "ok", "status": "OK"}]
assert _mark_changes(new2, prev, Ctx()) == 0 and "previous" not in new2[0]
joined = [{"source": "Season 4/a.mkv", "sources": ["Season 4/a.mkv", "Season 4/b.mkv"],
           "kind": "join", "status": "OK"}]
assert _mark_changes(joined, prev, Ctx()) == 0 and "previous" not in joined[0]
print("PASS")
