"""Laya client: bounded packet, one typed choice, no prose trust."""
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_decision_client"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(
    MEDIA_ROOT=str(W / "media"),
    CONFIG_DIR=str(W / "config"),
    CACHE_DIR=str(W / "cache"),
    DECISION_ENABLED="true",
    DECISION_BASE_URL="http://laya.example.test/v1",
    DECISION_MODEL="laya",
    DECISION_MAX_CANDIDATES="4",
    DECISION_MAX_INPUT_CHARS="300",
    DECISION_TIMEOUT="7",
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import decision  # noqa: E402

calls = []


class Response:
    status_code = 200
    text = ""
    choice = "S01E02"

    def json(self):
        return {"model": "laya", "answers": [{
            "type": "choice", "name": "episode", "choice": self.choice,
            "confidence": 0.42,
            "probabilities": [
                {"value": "S01E01", "probability": 0.2},
                {"value": "S01E02", "probability": 0.42},
                {"value": "unknown", "probability": 0.38},
            ]}]}


def fake_post(url, *, json, timeout):
    calls.append({"url": url, "body": json, "timeout": timeout})
    return Response()


decision.httpx.post = fake_post
assert decision.available()

candidates = [{"code": f"S01E0{n}", "title": f"Episode {n}",
               "overview": f"Plot {n}", "score": 0.5 - n / 100} for n in range(1, 8)]
result = decision.choose("A long transcript " * 100, candidates)
print(result)
assert result["code"] == "S01E02", result
assert result["unknown"] is False
assert result["probabilities"]["S01E02"] == 0.42
assert len(calls) == 1
body = calls[0]["body"]
assert body["model"] == "laya"
assert calls[0]["url"] == "http://laya.example.test/v1/decisions"
assert len(body["questions"][0]["choices"]) == 5  # four candidates + unknown
assert len({c["value"] for c in body["questions"][0]["choices"]}) == 5
assert len(body["input"]) <= 300
assert calls[0]["timeout"] == 7

# Unknown is a real answer, not a forced candidate.
Response.choice = "unknown"
unknown = decision.choose("A transcript", candidates[:2])
assert unknown["code"] is None and unknown["unknown"] is True, unknown

# A malformed choice is an error, so the caller can escalate instead of guessing.
Response.choice = "S99E99"
try:
    decision.choose("A transcript", candidates[:2])
except decision.DecisionError:
    pass
else:
    raise AssertionError("invalid choice was accepted")
print("PASS")
