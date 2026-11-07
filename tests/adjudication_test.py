"""A weak Laya guess must not replace the matcher's suggestion."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_adjudication_test"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"), DECISION_ENABLED="true",
                  DECISION_BASE_URL="http://127.0.0.1:9/v1", DECISION_MODEL="laya")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import adjudication  # noqa: E402


def file_result():
    return {"rel": "Show - S03E11.mkv", "scope": "sample", "duration": 600,
            "sample_text": "an artifact from Africa and a stigmatic colleague",
            "segments": [{"start": 0, "end": 600, "code": "S03E11", "score": 0.6,
                          "confidence": "provisional", "scope": "sample", "alternatives": []}],
            "whole": [{"code": "S03E11", "score": 0.6}]}


eps = {f"S03E{n:02d}": {"title": f"Episode {n}", "overview": "A case."} for n in range(9, 13)}

adjudication.decision.available = lambda: True
adjudication.decision.choose = lambda text, pool: {
    "code": "S01E04", "confidence": 0.05, "unknown": False,
    "probabilities": {"S03E11": 0.10, "S01E04": 0.05}, "model": "laya"}

low = file_result()
adjudication.adjudicate({"name": "Show"}, low, eps)
assert low["segments"][0]["code"] == "S03E11", low
assert low["decision"]["accepted"] is False, low

adjudication.decision.choose = lambda text, pool: {
    "code": "S01E04", "confidence": 0.40, "unknown": False,
    "probabilities": {"S03E11": 0.10, "S01E04": 0.40}, "model": "laya"}
high = file_result()
adjudication.adjudicate({"name": "Show"}, high, eps)
assert high["segments"][0]["code"] == "S01E04", high
assert high["decision"]["accepted"] is True, high
print("PASS")
