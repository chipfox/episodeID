#!/usr/bin/env python3
"""Score a scan plan against an audited golden set.

    python tools/golden_check.py --plan http://10.0.0.151:8687/api/series/2/plan
    python tools/golden_check.py --plan saved-plan.json

The plan is what the app would do: renames are still suggestions a human has to
tick, so a wrong rename here is a wrong recommendation, not a moved file. Exit
status is 1 when the plan would mislead (a rename on a correct file, an OK on a
swapped one, or a change of some other kind). A review row on a file the golden
set marks undecidable is the right answer and never fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "xfiles.json"


def load_plan(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        from urllib.request import urlopen
        with urlopen(source, timeout=60) as r:
            data = json.load(r)
    else:
        data = json.loads(Path(source).read_text())
    if isinstance(data, dict) and "items" not in data and isinstance(data.get("plan"), dict):
        data = data["plan"]
    return data


def code_from_name(name: str) -> str | None:
    for token in name.replace("-", " ").split():
        if len(token) == 6 and token[0] == "S" and token[3] == "E" \
                and token[1:3].isdigit() and token[4:].isdigit():
            return token
    return None


def decide(plan: dict) -> dict[str, dict]:
    """The row that speaks for each file: where it would put it, and how."""
    out: dict[str, dict] = {}
    for it in plan.get("items") or []:
        rels = it.get("sources") or ([it["source"]] if it.get("source") else [])
        asserted = it.get("decided_by") in ("parts", "split files")
        evidence = sorted({sg.get("code") for sg in it.get("segments") or []
                           if sg.get("code")
                           and (sg.get("confidence") in ("high", "manual") or asserted)})
        for rel in rels:
            target = next((t["path"] for t in it.get("targets") or []), None)
            out[rel] = {"kind": it.get("kind"),
                        "code": code_from_name(Path(target).name) if target else None,
                        "reason": it.get("reason"),
                        "status": (it.get("status") or "").upper(),
                        "decided_by": it.get("decided_by"),
                        "expected": it.get("expected") or [],
                        "evidence": evidence}
    return out


def score(plan: dict, golden: dict) -> dict:
    """Score a plan. An `ok` row is judged by the row's own evidence: the high
    segments must name only the episodes its `expected` list declares (the planner
    derives that from the filename), otherwise it is a wrong OK. That is what
    catches a plan which marks everything OK without evidence."""
    decisions = decide(plan)
    swaps = golden.get("swaps") or {}
    undecidable = set(golden.get("undecidable") or [])
    out = {"files": len(decisions), "correct": 0, "deferred_correctly": [],
           "missed_swaps": [], "false_renames": [], "wrong_ok": [],
           "deferred_wrongly": [], "other_changes": [],
           "not_in_plan": [rel for rel in swaps if rel not in decisions]}
    for rel, d in decisions.items():
        kind = d["kind"]
        truth = swaps.get(rel)
        if truth:
            if kind == "rename" and d["code"] == truth:
                out["correct"] += 1
            elif kind in ("ok", "review"):
                out["wrong_ok"].append(rel)
            else:
                out["missed_swaps"].append(rel)
            continue
        if kind == "ok":
            # `ok` means two different things: "the evidence agrees with the name"
            # (status OK) and "leave this unsettled file alone" (a deferral). Judge
            # an assertion by its evidence, and a deferral like a review row.
            if d.get("status") and d["status"] != "OK" and d.get("decided_by") not in ("parts", "split files"):
                if rel in undecidable:
                    out["deferred_correctly"].append(rel)
                else:
                    out["deferred_wrongly"].append(rel)
                continue
            expected = set(d.get("expected") or [])
            evidence = set(d.get("evidence") or [])
            if not expected:
                out["wrong_ok"].append({"rel": rel, "why": "OK on a name with no episode"})
            elif not evidence or not evidence <= expected:
                out["wrong_ok"].append({
                    "rel": rel,
                    "why": f"OK but the evidence says {', '.join(sorted(evidence)) or 'nothing'}"
                           f" — name says {', '.join(sorted(expected))}"})
            else:
                out["correct"] += 1
        elif kind == "review":
            if rel in undecidable:
                out["deferred_correctly"].append(rel)
            else:
                out["deferred_wrongly"].append(rel)
        elif kind == "rename":
            out["false_renames"].append({"rel": rel, "to": d["code"]})
        else:
            out["other_changes"].append({"rel": rel, "kind": kind})
    out["accuracy"] = round(out["correct"] / max(1, len(decisions)), 3)
    return out


def report(r: dict) -> int:
    print(f"files: {r['files']}   correct: {r['correct']} ({r['accuracy']:.1%})")
    print(f"deferred to a human, correctly: {len(r['deferred_correctly'])}")
    for rel in r["deferred_correctly"]:
        print(f"  - {Path(rel).name}")
    for title, rows in (("WRONG OK - a swap would be left in place", r["wrong_ok"]),
                        ("FALSE RENAME - a correct file would be renamed", r["false_renames"]),
                        ("MISSED SWAP", r["missed_swaps"]),
                        ("DEFERRED, BUT THE ANSWER WAS KNOWN", r["deferred_wrongly"]),
                        ("OTHER CHANGE", r["other_changes"]),
                        ("NOT IN PLAN", r["not_in_plan"])):
        if not rows:
            continue
        print(f"{title}: {len(rows)}")
        for row in rows:
            if isinstance(row, dict):
                print(f"  - {Path(row['rel']).name} → {row.get('to') or row.get('kind')}")
            else:
                print(f"  - {Path(row).name}")
    failing = r["wrong_ok"] or r["false_renames"] or r["missed_swaps"] or r["other_changes"]
    return 1 if failing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="plan URL or a saved JSON file")
    ap.add_argument("--golden", default=str(GOLDEN))
    args = ap.parse_args()
    golden = json.loads(Path(args.golden).read_text())
    return report(score(load_plan(args.plan), golden))


if __name__ == "__main__":
    sys.exit(main())
