"""Settings renamed after the retired decision model still answer to their old names.

The deployment's stack.env exports DECISION_* variables; renaming the settings must not
silently change behaviour, so old env names and old settings.json keys keep working and
the new names win when both are present.
"""
import json
import os
import shutil
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_settings_alias"))
if W.exists():
    shutil.rmtree(W)
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"), CACHE_DIR=str(W / "cache"))
for name in ("SAMPLE_SECONDS", "DECISION_SAMPLE_SECONDS", "SAMPLING_FAST_PATH", "DECISION_FAST_PATH",
             "REFERENCE_DUPLICATE_SIMILARITY", "DECISION_CONFLICT_SIMILARITY"):
    os.environ.pop(name, None)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402

assert config.get_settings()["sample_seconds"] == 600, "default"

# An old env name still applies, and reports the new name as locked.
os.environ["DECISION_SAMPLE_SECONDS"] = "900"
assert config.get_settings()["sample_seconds"] == 900
assert "sample_seconds" in config.env_locked_keys()
assert "decision_sample_seconds" not in config.SCHEMA

# The new name wins when both are set.
os.environ["SAMPLE_SECONDS"] = "300"
assert config.get_settings()["sample_seconds"] == 300
os.environ.pop("SAMPLE_SECONDS")
os.environ.pop("DECISION_SAMPLE_SECONDS")

# An old settings.json key is folded onto the new key on read.
config.SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
config.SETTINGS_FILE.write_text(json.dumps({"decision_fast_path": True,
                                            "decision_conflict_similarity": 0.9}))
s = config.get_settings()
assert s["sampling_fast_path"] is True and s["reference_duplicate_similarity"] == 0.9

# Saving drops the old key, and a client still sending the old name is translated.
config.save_settings({"sampling_fast_path": False})
stored = json.loads(config.SETTINGS_FILE.read_text())
assert stored.get("sampling_fast_path") is False and "decision_fast_path" not in stored, stored
config.save_settings({"decision_sample_seconds": 450})
stored = json.loads(config.SETTINGS_FILE.read_text())
assert stored.get("sample_seconds") == 450 and "decision_sample_seconds" not in stored, stored
assert config.get_settings()["sample_seconds"] == 450
print("PASS (renamed settings keep their old env vars and stored keys working)")
