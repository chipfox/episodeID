"""Calibration gates and staleness for summary matching."""
import os
import sys
from pathlib import Path

W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_summary_cal"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.summary import (  # noqa: E402
    SummaryIndex, calibrate, calibration_stale, ensure_calibration)

N = 25
EPS = {f"S01E{i:02d}": {"season": 1, "episode": i, "title": f"Episode {i}",
                        "overview": f"The boys encounter the unique{i} thing."}
       for i in range(1, N + 1)}
REFS = {f"S01E{i:02d}": (f"dialogue about unique{i} and the thing " * 20)
        for i in range(1, N + 1)}

idx = SummaryIndex(EPS, REFS)
cal = calibrate(idx, REFS, EPS, target=0.80, min_refs=20, min_gated=8)
assert cal["available"], cal
assert cal["precision_lb"] >= 0.80, cal
assert cal["oof_lb"] >= 0.80, cal
assert cal["n_refs"] == N, cal
assert cal["refs_fp"] and cal["summaries_fp"]
print("gate:", {k: cal[k] for k in ("min_score", "min_margin", "precision",
                                   "precision_lb", "coverage", "oof_lb")})

# A series with too few references is not calibrated.
small = {"S01E01": EPS["S01E01"], "S01E02": EPS["S01E02"], "S01E03": EPS["S01E03"]}
small_refs = {c: REFS[c] for c in small}
cal_small = calibrate(SummaryIndex(small, small_refs), small_refs, small)
assert not cal_small["available"] and "need 20" in cal_small["reason"], cal_small

# Any reference or summary change invalidates the stored gate.
assert not calibration_stale(cal, REFS, EPS)
assert calibration_stale({**cal, "version": 1}, REFS, EPS)
changed_refs = dict(REFS)
changed_refs["S01E01"] += " an added line"
assert calibration_stale(cal, changed_refs, EPS)
changed_eps = {**EPS, "S01E01": {**EPS["S01E01"], "overview": "A different plot."}}
assert calibration_stale(cal, REFS, changed_eps)

# ensure_calibration reuses a valid gate and recalibrates a stale one.
series = {"episodes": list(EPS.values()), "options": {"summary_calibration": cal}}
settings = {"summary_target_precision": 0.80, "summary_min_refs": 20}
assert ensure_calibration(series, idx, REFS, settings) is cal
series2 = {"episodes": list(EPS.values()), "options": {}}
fresh = ensure_calibration(series2, idx, REFS, settings)
assert fresh["available"] and fresh.get("at"), fresh

print("summary_calibration_test: OK — 5 groups")
