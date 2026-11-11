"""Read models behind the series and evidence views.

These were in `main.py` among the routes; they are here so the HTTP layer stays about
HTTP. Nothing in this module writes: `series_out` shapes a series row for JSON, and
`best_dialogue` only reads transcript caches.
"""
from __future__ import annotations

from pathlib import Path

from . import db
from .config import safe_media_path
from .references import _load_misses, ep_code, load_refs, load_swapped


def episode_match(s: dict) -> dict[str, dict]:
    """code -> the dialogue segment that matched that episode, best score first.

    The latest plan is the verdict a person is looking at (promoted evidence,
    title cards and overrides included); a series with no plan, or one older than
    its last scan, falls back to the scan's own segments. Each entry says how well
    the file matched (score, margin, confidence) and which file it was.
    """
    out: dict[str, dict] = {}

    def take(code: str, score, seg: dict, item: dict, source: str) -> None:
        if not code or score is None:
            return
        best = out.get(code)
        if best is not None and float(score) <= float(best["score"]):
            return
        margin = seg.get("margin")
        out[code] = {
            "score": round(float(score), 4), "confidence": seg.get("confidence"),
            "margin": round(float(margin), 4) if margin is not None else None,
            "start": seg.get("start"), "end": seg.get("end"),
            "scope": seg.get("scope") or item.get("scope"),
            "kind": item.get("kind"), "status": item.get("status"), "source": source,
        }

    plan = db.latest_plan_summary(s["id"])
    scan = db.latest_scan_summary(s["id"])
    if plan and (not scan or plan["created"] >= scan["created"]):
        for item in (db.latest_plan(s["id"]) or {}).get("items") or []:
            source = (item.get("sources") or [item.get("source")])[0] or ""
            for seg in item.get("segments") or []:
                take(str(seg.get("code") or ""), seg.get("score"), seg, item, source)
        return out
    for f in (db.latest_scan(s["id"]) or {}).get("results", {}).get("files") or []:
        source = f.get("rel") or ""
        for seg in f.get("segments") or []:
            take(str(seg.get("code") or ""), seg.get("score"), seg, f, source)
    return out


def series_out(s: dict, full: bool = False) -> dict:
    refs = load_refs(s["tvdb_id"])
    regular = [e for e in s["episodes"] if e["season"] > 0]
    out = {k: v for k, v in s.items() if k != "episodes"}
    out["episode_count"] = len(regular)
    out["reference_count"] = sum(1 for e in regular
                                 if ep_code(e["season"], e["episode"]) in refs)
    out["active_job"] = db.active_job(s["id"])
    scan = db.latest_scan_summary(s["id"])
    out["last_scan"] = {"created": scan["created"], "counts": scan["counts"]} if scan else None
    plan = db.latest_plan_summary(s["id"])
    # the plan is the verdict (title cards, overrides and applied changes included);
    # only shown when it was made from the latest scan or after it
    out["plan"] = plan if plan and (not scan or plan["created"] >= scan["created"]) else None
    from .autocheck import load_state
    ac = load_state(s["id"])
    pend = {k: v for k, v in (ac.get("pending") or {}).items() if k != "files"} or None
    root = Path(s["path"])
    flagged = [x for x in ac.get("imports") or [] if not x.get("ok") and (root / x["rel"]).exists()]
    out["file_check"] = {"checked_at": ac.get("checked_at"), "pending": pend, "imports": flagged}
    from . import integrity
    out["integrity"] = integrity.card_summary(s)
    from .executor import backup_videos
    try:
        out["backup"] = backup_videos(s)
    except OSError:
        out["backup"] = None
    if full:
        out["integrity_files"] = integrity.current(s)
        from .redownload import finished as _rd_finished
        from .redownload import load_state as _rd_state
        from .redownload import missing_episodes
        out["missing_count"] = len(missing_episodes(s))
        out["redownloads"] = sum(1 for r in (_rd_state(s["id"]).get("requested") or {}).values()
                                 if isinstance(r, dict)
                                 and r.get("status") in ("waiting", "stuck", "failed", "partly imported")
                                 and not _rd_finished(r))
        from .pipeline import reference_swaps
        last = db.latest_scan(s["id"]) if scan else None
        out["ref_swaps"] = reference_swaps((last or {}).get("results", {}).get("files") or [], refs,
                                           load_swapped(s["tvdb_id"]), (last or {}).get("created"))
        misses = _load_misses(s["tvdb_id"])
        matches = episode_match(s)
        out["episodes"] = [{
            **e, "code": ep_code(e["season"], e["episode"]),
            "ref": (refs.get(ep_code(e["season"], e["episode"])) or {}).get("source"),
            "ref_note": ref_note(refs.get(ep_code(e["season"], e["episode"])),
                                 ep_code(e["season"], e["episode"]), e.get("title")),
            "miss": misses.get(ep_code(e["season"], e["episode"])),
            "match": matches.get(ep_code(e["season"], e["episode"])),
        } for e in s["episodes"]]
    return out


def ref_note(ref: dict | None, code: str = "", title: str | None = None) -> str:
    n = ref_note_inner(ref, code, title)
    if ref and ref.get("swapped_from") and "swapped" not in n:
        n = (n + " · " if n else "") + f"swapped from {ref['swapped_from']}"
    return n


def ref_note_inner(ref: dict | None, code: str = "", title: str | None = None) -> str:
    if not ref:
        return ""
    if ref.get("source") == "opensubtitles":
        from .references import describe_conflict, is_low_trust
        n = f"via {ref.get('via')}"
        if is_low_trust(ref):
            n += f" — ⚠ {describe_conflict(ref, code, title)}"
        elif ref.get("verified"):
            n += f" — {ref['verified']}"
        return n
    if ref.get("source") == "fandom":
        return ref.get("page", "")
    return ""


def best_dialogue(series: dict, rel: str, entry: dict) -> tuple[str, str]:
    """The longest dialogue already on disk for this file, and which scope it is.

    `full` is a whole-file transcript, `sample` the scan's sample window read back
    from the dialogue cache, `excerpt` the head+tail clip an older scan stored in
    its results (the scan used to keep only ~1400 characters of the sample).
    Never transcribes: cached_dialogue() and cached_sample() only read caches.
    """
    from .media_text import cached_dialogue, cached_sample, dialogue_text
    text = str(entry.get("sample_text") or "")
    scope = "excerpt"
    try:
        video = safe_media_path(Path(series["path"]) / rel)
    except ValueError:
        return text, scope
    if not video.is_file():
        return text, scope
    start, end = entry.get("sample_start"), entry.get("sample_end")
    if start is not None and end is not None and float(end) > float(start):
        data = cached_sample(video, float(start), float(end) - float(start)) or {}
        sample = dialogue_text(data.get("cues") or [], limit=200_000)
        # ">=", not ">": a scan stores the whole sample window now, so the cached text
        # can be exactly as long as the stored one — it is still the sample, not a clip.
        if sample and len(sample.split()) >= len(text.split()):
            text, scope = sample, "sample"
    data = cached_dialogue(video) or {}
    full = dialogue_text(data.get("cues") or [], limit=400_000)
    if full and len(full.split()) > len(text.split()):
        text, scope = full, "full"
    return text, scope
