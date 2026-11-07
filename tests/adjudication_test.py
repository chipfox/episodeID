"""A weak Laya guess only annotates the file; the claim always rides in the packet."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_adjudication_test"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"), DECISION_ENABLED="true", DECISION_MIN_CONFIDENCE="0.15",
                  DECISION_BASE_URL="http://127.0.0.1:9/v1", DECISION_MODEL="laya")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import adjudication  # noqa: E402


def file_result():
    return {"rel": "Show - S03E11.mkv", "scope": "sample", "duration": 600, "expected": ["S03E11"],
            "sample_text": "an artifact from Africa and a stigmatic colleague",
            "segments": [{"start": 0, "end": 600, "code": "S03E11", "score": 0.6,
                          "confidence": "provisional", "scope": "sample", "alternatives": []}],
            "whole": [{"code": "S03E11", "score": 0.6}]}


eps = {f"S03E{n:02d}": {"title": f"Episode {n}", "overview": "A case."} for n in range(9, 13)}

seen = {}


def fake_choose(low_conf=True):
    def choose(text, pool, instructions=None):
        seen["text"] = text
        seen["pool"] = pool
        seen["instructions"] = instructions
        return {"code": "S03E11" if low_conf else "S03E10",
                "confidence": 0.05 if low_conf else 0.40, "unknown": False,
                "probabilities": {"S03E11": 0.10, "S03E10": 0.05}, "model": "laya"}
    return choose


adjudication.decision.available = lambda: True

adjudication.decision.choose = fake_choose(low_conf=True)
low = file_result()
adjudication.adjudicate({"name": "Show"}, low, eps)
assert low["ai_code"] == "LOW_CONFIDENCE", low          # weak answers never apply
assert low["segments"][0]["code"] == "S03E11", low
assert "S03E11" in seen["instructions"], seen           # the claim tells the model what is checked
assert seen["pool"] and seen["pool"][0]["code"] == "S03E11", seen["pool"]
assert "/tmp" not in seen["text"] and "Show - S03E11" not in seen["text"]

adjudication.decision.choose = fake_choose(low_conf=False)
high = file_result()
adjudication.adjudicate({"name": "Show"}, high, eps)
assert high["ai_code"] == "S03E10", high                # strong answers are recorded…
assert high["ai_confidence"] == 0.40, high
assert high["segments"][0]["code"] == "S03E11", high    # …but the matcher segment stays untouched


# A candidate pool full of other episodes may not push the claim out of the packet.
whole = [{"code": f"S03E{n:02d}", "score": 0.9 - n / 100} for n in (12, 10, 9)]
crowded = {"rel": "Show - S03E11.mkv", "expected": ["S03E11"], "whole": whole, "segments": [],
           "sample_text": ""}
pool = adjudication.candidates(crowded, eps, 4)
assert any(candidate["code"] == "S03E11" for candidate in pool), pool

# Two copies of one transcript may not face each other as rivals.
pool = adjudication.candidates(crowded, eps, 4, blocked={"S03E12"})
assert all(candidate["code"] != "S03E12" for candidate in pool), pool
print("PASS")
