"""Job handlers: the long-running work behind each button in the UI."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from . import autocheck
from . import adjudication, db, decision, evidence, fandom, llm, titlecard, tvdb
from .config import get_settings
from .executor import (apply_plan, migrate_legacy_backup, sonarr_rescan, sonarr_series_type_note,
                       undo_apply)
from .jobs import Cancelled, handler
from .matcher import Matcher, classify, parse_filename_episodes
from .media_text import (cached_dialogue, detect_language, duration, get_dialogue, probe,
                         sample_dialogue)
from .planner import build_plan, list_videos
from .references import (SAME_EPISODE_SCORE, detect_fandom_wiki, ep_code, load_refs, norm_title,
                         refresh_references, renumber_references, repair_reference_conflicts,
                         verify_flagged, check_release_names)


def _series(ctx) -> dict:
    s = db.get_series(ctx.series_id)
    if s is None:
        raise RuntimeError("Series not found")
    return s


EPISODES_MAX_AGE_H = 12


def _ensure_episodes(series: dict, ctx, force: bool = False,
                     max_age_h: float | None = None) -> dict:
    """The series' TVDB episode list, fetched when missing, when asked, or when older
    than max_age_h (TVDB renumbers: episodes move into a new season, specials get
    numbers). Anything kept under an episode's old number moves with it."""
    if series["episodes"] and not force:
        if max_age_h is None:
            return series
        try:
            age = (datetime.now(timezone.utc) -
                   datetime.fromisoformat(series.get("episodes_updated") or "")).total_seconds() / 3600
        except ValueError:
            age = 1e9
        if age < max_age_h:
            return series
    ctx.status("Loading episode list from TVDB")
    try:
        eps = tvdb.series_episodes(series["tvdb_id"])
    except Exception as e:  # noqa: BLE001
        if series["episodes"]:
            ctx.log(f"TVDB: couldn't refresh the episode list ({e}); using the one from "
                    f"{(series.get('episodes_updated') or '')[:10]}.")
            return series
        raise
    # keep IMDb IDs we already looked up
    old = {(e["season"], e["episode"]): e for e in series["episodes"]}
    for e in eps:
        prev = old.get((e["season"], e["episode"]))
        if prev and prev.get("imdb_id") is not None and prev.get("title") == e.get("title"):
            e["imdb_id"] = prev["imdb_id"]
    moves = _renumbered(series["episodes"], eps)
    db.update_series(series["id"], episodes=eps, episodes_updated=db.now())
    if moves:
        titles = {ep_code(e["season"], e["episode"]): e.get("title") for e in eps}
        for o, n in sorted(moves.items(), key=lambda kv: kv[1]):
            ctx.log(f"TVDB: '{titles.get(n)}' is now {n} (was {o}).")
        moved = renumber_references(series["tvdb_id"], moves)
        _renumber_options(series["id"], moves)
        ctx.log(f"TVDB renumbered {len(moves)} episodes; {moved} references and any episodes "
                "you set by hand moved with them.")
    ctx.log(f"TVDB: {len(eps)} episodes in "
            f"{len({e['season'] for e in eps if e['season'] > 0})} seasons.")
    return db.get_series(series["id"])


def _renumbered(old_eps: list[dict], new_eps: list[dict]) -> dict[str, str]:
    """old code → new code for episodes whose (unique) title now has another number."""
    def by_title(eps):
        seen: dict[str, list[str]] = {}
        for e in eps:
            t = norm_title(e.get("title") or "")
            if t and not re.fullmatch(r"(tba|tbd|episode \d+)", t):
                seen.setdefault(t, []).append(ep_code(e["season"], e["episode"]))
        return {t: codes[0] for t, codes in seen.items() if len(codes) == 1}
    a, b = by_title(old_eps), by_title(new_eps)
    return {a[t]: b[t] for t in a.keys() & b.keys() if a[t] != b[t]}


def _renumber_options(series_id: int, moves: dict[str, str]) -> None:
    s = db.get_series(series_id)
    opts = dict(s.get("options") or {})
    m = lambda c: moves.get(c, c)  # noqa: E731
    if isinstance(opts.get("overrides"), dict):
        opts["overrides"] = {k: [m(c) for c in v] if isinstance(v, list) else v
                             for k, v in opts["overrides"].items()}
    for key in ("fandom_page_map", "fandom_title_overrides"):
        if isinstance(opts.get(key), dict):
            opts[key] = {m(k): v for k, v in opts[key].items()}
    st = opts.get("title_cards_status")
    if isinstance(st, dict) and isinstance(st.get("recurring"), dict):
        st["recurring"] = {m(k): v for k, v in st["recurring"].items()}
    db.update_series(series_id, options=opts)


@handler("refresh_episodes")
def job_refresh_episodes(ctx, params):
    series = _ensure_episodes(_series(ctx), ctx, force=True)
    return {"episodes": len(series["episodes"])}


def _find_wiki(series: dict, ctx, force: bool = False) -> dict:
    """Discover the series' Fandom wiki and/or map its transcript pages to episodes.
    Runs when no wiki is known yet, or when a wiki was typed in but its pages
    haven't been mapped. Returns the updated series."""
    opts = dict(series.get("options") or {})
    regular = sum(1 for e in series["episodes"] if e["season"] >= 1)
    wiki = (opts.get("fandom_wiki") or "").strip()
    if wiki and not force:
        if opts.get("fandom_page_map") is not None:
            return series
        ctx.status(f"Listing transcript pages on {wiki}")
        with fandom._client() as c:
            pages, how = fandom.list_transcript_pages(c, wiki)
        mapping = fandom.map_pages([e for e in series["episodes"] if e["season"] >= 1], pages)
        opts["fandom_page_map"] = mapping
        opts["fandom_discovery"] = {"wiki": wiki, "mapped": len(mapping), "total": regular,
                                    "found_via": how, "auto": False, "at": db.now()}
        ctx.log(f"{wiki}: {len(pages)} transcript pages ({how}); {len(mapping)} of {regular} "
                "episodes matched to a page by title.")
    elif force or not opts.get("fandom_checked"):
        ctx.status("Looking for a Fandom transcript wiki")
        ctx.log(f"Looking for a Fandom wiki for '{series['name']}'…")
        seen: list[str] = []
        found = fandom.discover(series["name"], series["episodes"], ctx, seen=seen)
        opts["fandom_checked"] = True
        if found:
            opts["fandom_wiki"] = found["wiki"]
            opts["fandom_page_map"] = found["page_map"]
            if found.get("pattern") and not opts.get("fandom_page_pattern"):
                opts["fandom_page_pattern"] = found["pattern"]
            opts["fandom_discovery"] = {"wiki": found["wiki"], "sitename": found["sitename"],
                                        "mapped": found["mapped"], "total": regular,
                                        "found_via": found["found_via"], "auto": True,
                                        "at": db.now()}
            ctx.log(f"Found {found['wiki']}.fandom.com ({found['sitename']}): {found['mapped']} of "
                    f"{regular} episodes have a transcript page — using them before OpenSubtitles.")
        else:
            # Wikis without a transcript category or searchable titles: try the classic layout.
            slug = detect_fandom_wiki(series["name"], series["episodes"], ctx, slugs=seen)
            if slug:
                opts["fandom_wiki"] = slug
                opts["fandom_page_map"] = {}
                opts["fandom_discovery"] = {"wiki": slug, "mapped": 0, "total": regular,
                                            "found_via": "{title}/Transcript pages", "auto": True,
                                            "at": db.now()}
                ctx.log(f"Found transcripts on {slug}.fandom.com ({{title}}/Transcript pages).")
            else:
                opts["fandom_discovery"] = {"wiki": None, "at": db.now()}
                ctx.log("No Fandom transcript wiki found for this series.")
    else:
        return series
    db.update_series(series["id"], options=opts)
    return db.get_series(series["id"])


@handler("fetch_refs")
def job_fetch_refs(ctx, params):
    series = _ensure_episodes(_series(ctx), ctx, max_age_h=EPISODES_MAX_AGE_H)
    before = dict((series.get("options") or {}).get("fandom_page_map") or {})
    series = _find_wiki(series, ctx, force=bool(params.get("find_wiki")))
    after = (series.get("options") or {}).get("fandom_page_map") or {}
    # Episodes that just gained a transcript page deserve another try even if they were
    # recorded as "not found" earlier.
    retry = [c for c in after if before.get(c) != after[c]]
    only = None
    if params.get("season") is not None:
        # "Fetch missing" for one season: its aired episodes without a reference,
        # including ones recorded as "not found" before (asked for, so try again).
        sn, have = int(params["season"]), load_refs(series["tvdb_id"])
        only = [ep_code(e["season"], e["episode"]) for e in series["episodes"]
                if e["season"] == sn and _aired(e) and ep_code(e["season"], e["episode"]) not in have]
        ctx.log(f"Season {sn}: fetching references for {len(only)} episode(s) without one.")
        retry += only
    ctx.status("Fetching references")
    result = refresh_references(series, ctx, force_codes=params.get("codes"),
                                retry_misses=bool(params.get("retry_misses")),
                                retry_codes=retry, only_codes=only)
    if get_settings()["reference_repair"]:
        ctx.status("Repairing references that share a transcript")
        fresh = db.get_series(series["id"]) or series
        eps_by_code = {ep_code(e["season"], e["episode"]): e for e in fresh["episodes"]}
        result["repairs"] = repair_reference_conflicts(
            fresh, ctx, story=lambda code: _story(eps_by_code, code))
    ctx.status("Cross-checking flagged OpenSubtitles references")
    result["names"] = check_release_names(db.get_series(series["id"]), ctx)
    result["verification"] = verify_flagged(db.get_series(series["id"]), ctx)
    db.update_series(series["id"], episodes=series["episodes"])  # cached IMDb ids
    return result


def _aired(e: dict) -> bool:
    """Aired already (or no date known)."""
    return not e.get("aired") or e["aired"] <= datetime.now(timezone.utc).date().isoformat()


def _sample_eligible(video: Path, expected: list[str], eps: dict,
                     runtime_minutes: float | None, settings: dict) -> bool:
    """Only single-episode files of normal runtime get the sample path.

    A file named for several episodes, or about twice one episode's runtime,
    is exactly where a 10-minute sample is least trustworthy: it must use the
    full matcher so splits and joins can be found.
    """
    if len(expected) != 1 or expected[0] not in eps:
        return False
    try:
        dur = duration(probe(video))
    except Exception:  # noqa: BLE001
        return False
    if dur <= 0:
        return False
    if dur <= float(settings["decision_sample_seconds"]) * 0.8:
        return False  # the file is short enough to read whole
    runtime = float(eps[expected[0]].get("runtime") or runtime_minutes or 0)
    if runtime and dur > runtime * 60 * 1.55:
        return False  # likely a multi-episode file; let the full matcher split it
    if not runtime and dur > 90 * 60:
        return False  # unknown runtime and unusually long: don't sample
    return True


def _runtime_minutes(eps: dict) -> float | None:
    values = sorted(float(e["runtime"]) for e in eps.values()
                    if e.get("runtime") and e["season"] > 0)
    return values[len(values) // 2] if values else None


def _sample_window_starts(full_duration: float, base_start: float, window: float,
                          max_windows: int) -> list[float]:
    """Non-adjacent sample starts, ordered by the places that most often resolve files.

    The first sample is deliberately not enough: ads, commentary, unrelated intros
    and quiet scenes all live near particular timestamps. Later windows therefore
    probe a few places before trying to split weak samples into segments.
    """
    full = float(full_duration or 0)
    window = max(1.0, float(window))
    max_windows = max(1, int(max_windows))
    starts = [float(base_start)]
    max_start = max(0.0, full - window) if full else None
    for offset in (2, 1, 4, 3, 5, 6, 7):
        if len(starts) >= max_windows:
            break
        start = float(base_start) + offset * window
        if max_start is not None:
            if start > max_start:
                start = max_start
            if start >= full:
                continue
        if any(abs(start - existing) < max(15.0, window * 0.05) for existing in starts):
            continue
        starts.append(start)
    return starts


def _candidate_from_top(top: list[dict], settings: dict, reference_count: int) -> dict | None:
    """Apply the same high-confidence boundary the matcher uses for full segments."""
    if not top:
        return None
    best = top[0]
    second = top[1] if len(top) > 1 else {"code": None, "score": 0.0}
    score = float(best.get("score") or 0)
    margin = score - float(second.get("score") or 0)
    if score < float(settings["min_score"]):
        return None
    if margin < float(settings["min_margin"]) or score < 1.4 * max(float(second.get("score") or 0), 1e-6):
        return None
    if reference_count < 5 and score < SAME_EPISODE_SCORE:
        return None
    return {"code": best.get("code"), "score": round(score, 4), "margin": round(margin, 4),
            "evidence": "whole", "alternatives": top[1:4]}


def _window_from_existing(f: dict, settings: dict, reference_count: int) -> dict:
    text = str(f.get("sample_text") or "")
    votes: list[dict] = []
    for segment in f.get("segments") or []:
        if segment.get("confidence") == "high" and segment.get("code"):
            votes.append({**segment, "evidence": segment.get("evidence") or "segment"})
    whole_candidate = _candidate_from_top(f.get("whole") or [], settings, reference_count)
    if whole_candidate:
        votes.append(whole_candidate)
    deduped: dict[str, dict] = {}
    for vote in votes:
        old = deduped.get(vote["code"])
        if old is None or float(vote.get("score") or 0) > float(old.get("score") or 0):
            deduped[vote["code"]] = vote
    language = detect_language([text], settings["subtitle_language"]) if len(text.split()) >= 40 else None
    return {"start": float(f.get("sample_start") or 0),
            "end": float(f.get("sample_end") or 0), "source": f.get("text_source") or "none",
            "words": len(text.split()), "has_text": bool(text.strip()), "language": language,
            "votes": list(deduped.values()), "whole": (f.get("whole") or [])[:5],
            "text": text}


def _scored_sample_window(video: Path, matcher: Matcher, start: float, duration: float,
                          expected: list[str], eps: dict, refs: dict, settings: dict,
                          ctx=None) -> dict:
    data = sample_dialogue(video, ctx, allow_whisper=True, start=start, duration=duration)
    cues = data.get("cues") or []
    text = adjudication.dialogue_text(cues, limit=20000)
    language = detect_language(cues, settings["subtitle_language"]) if len(text.split()) >= 40 else None
    if language:
        return {"start": float(data.get("sample_start", start)),
                "end": float(data.get("sample_end", start + duration)),
                "source": data.get("source") or "none", "words": len(text.split()),
                "has_text": bool(text.strip()), "language": language, "votes": [], "whole": [],
                "text": text}
    match_duration = float(data.get("duration") or duration)
    result = matcher.match_file(cues, match_duration)
    _discount_unreferenced(result.get("segments") or [], expected, match_duration, refs)
    result["segments"] = _story_parts(result.get("segments") or [], expected, eps, match_duration)
    votes: list[dict] = []
    for segment in result.get("segments") or []:
        if segment.get("confidence") == "high" and segment.get("code"):
            votes.append({**segment, "evidence": segment.get("evidence") or "segment"})
    whole_candidate = _candidate_from_top(result.get("whole") or [], settings, len(matcher.codes))
    if whole_candidate:
        votes.append(whole_candidate)
    deduped: dict[str, dict] = {}
    for vote in votes:
        old = deduped.get(vote["code"])
        if old is None or float(vote.get("score") or 0) > float(old.get("score") or 0):
            deduped[vote["code"]] = vote
    return {"start": float(data.get("sample_start", start)),
            "end": float(data.get("sample_end", start + duration)),
            "source": data.get("source") or "none", "words": len(text.split()),
            "has_text": bool(text.strip()), "language": None,
            "votes": list(deduped.values()), "whole": (result.get("whole") or [])[:5],
            "text": text}


def _aggregate_sample_evidence(expected: list[str], windows: list[dict], eps: dict,
                               settings: dict, required_mismatch: int,
                               conflicts: set[str] | None = None) -> dict:
    """Vote once per sample window, while treating story parts as non-opposing.

    ``conflicts`` lists episodes whose reference is a near-duplicate of the
    claimed one (one transcript filed under two numbers). Votes for them are
    neither agreement nor disagreement — the transcript genuinely matches both
    episodes — so they can neither confirm the filename nor propose a rename."""
    expected_story = set(_story(eps, expected[0])) if len(expected) == 1 and expected[0] in eps else set()
    conflicts = set(conflicts or ()) - expected_story - set(expected)
    expected_votes: list[dict] = []
    target_votes: dict[str, list[dict]] = {}
    story_codes: set[str] = set()
    conflict_codes: set[str] = set()
    language_counts: dict[str, int] = {}
    whole_by_code: dict[str, float] = {}
    texts: list[str] = []
    episode_like = False

    for window in windows:
        if window.get("text"):
            texts.append(f"[{int(window.get('start', 0)) // 60}–{int(window.get('end', 0)) // 60} min] "
                         f"{window['text']}")
        if window.get("language"):
            language_counts[window["language"]] = language_counts.get(window["language"], 0) + 1
        for top in window.get("whole") or []:
            code = top.get("code")
            if code:
                whole_by_code[code] = max(whole_by_code.get(code, 0.0), float(top.get("score") or 0))
        for vote in window.get("votes") or []:
            code = vote.get("code")
            if not code:
                continue
            episode_like = True
            if code in expected:
                expected_votes.append(vote)
            elif code in expected_story:
                story_codes.add(code)
            elif code in conflicts:
                conflict_codes.add(code)
            else:
                target_votes.setdefault(code, []).append(vote)

    text = "\n".join(texts)
    if len(text) > 12000:
        head = 8000
        text = f"{text[:head]}\n…\n{text[-3000:]}"
    whole = [{"code": code, "score": round(score, 4)}
             for code, score in sorted(whole_by_code.items(), key=lambda kv: -kv[1])[:8]]

    def score_of(votes: list[dict]) -> float:
        return max((float(v.get("score") or 0) for v in votes), default=0.0)

    def target_result(code: str) -> dict:
        votes = target_votes[code]
        top = max(votes, key=lambda v: float(v.get("score") or 0))
        return {"code": code, "votes": len(votes), "score": float(top.get("score") or 0),
                "evidence": top.get("evidence") or "sample", "alternatives": top.get("alternatives") or []}

    candidates = [{"code": code, "votes": len(votes), "score": round(score_of(votes), 4),
                   "evidence": "sample"} for code, votes in target_votes.items()]
    if expected_votes:
        candidates.insert(0, {"code": expected_votes[0].get("code"), "votes": len(expected_votes),
                              "score": round(score_of(expected_votes), 4), "evidence": "sample"})
    candidates.extend({"code": code, "votes": 0, "score": 0.0, "evidence": "story-part"}
                      for code in story_codes)
    candidates.extend({"code": code, "votes": 0, "score": 0.0, "evidence": "shared-reference"}
                      for code in sorted(conflict_codes))

    targets = sorted((target_result(code) for code in target_votes),
                     key=lambda item: (item["votes"], item["score"]), reverse=True)
    has_text = any(window.get("has_text") for window in windows)
    languages = sorted(language_counts, key=lambda code: -language_counts[code])
    all_windows_language = bool(languages) and sum(language_counts.values()) == len(windows)
    shared = ", ".join(sorted(conflict_codes))
    # A window whose best whole score clears min_score but never the margin gate still
    # heard recognisable episode dialogue — that is "too close to call", not "no dialogue".
    weak_match = any(score >= float(settings["min_score"]) for score in whole_by_code.values())

    if not has_text:
        return {"status": "NO_TEXT", "code": None, "note": "No sample window produced dialogue",
                "episode_like": False, "candidates": candidates, "whole": whole, "text": ""}
    if all_windows_language and not expected_votes and not target_votes:
        return {"status": "NO_MATCH", "code": None, "language": languages[0],
                "note": f"Audio appears to be {languages[0]}; English references cannot identify it",
                "episode_like": False, "candidates": candidates, "whole": whole, "text": text}
    if not expected_votes and not target_votes:
        if conflict_codes:
            return {"status": "LOW_CONFIDENCE", "code": None,
                    "note": f"Sample matches {shared}, whose reference is a near-duplicate of "
                            f"{expected[0] if expected else 'the filename'}'s — one transcript "
                            "cannot tell the two episodes apart",
                    "episode_like": True, "candidates": candidates, "whole": whole, "text": text}
        return {"status": "NO_MATCH", "code": None,
                "note": ("Sample windows agree on a story part, but not on one episode" if story_codes else
                         "Sample dialogue resembles several episodes, but no window favours one clearly"
                         if (episode_like or weak_match) else
                         "No sample window contained episode-like dialogue"),
                "episode_like": episode_like or bool(story_codes) or weak_match, "candidates": candidates,
                "whole": whole, "text": text}
    if expected_votes and not target_votes:
        if conflict_codes:
            return {"status": "LOW_CONFIDENCE", "code": None,
                    "note": f"Sample matches a transcript {expected_votes[0].get('code')} shares "
                            f"with {shared} — it cannot prove which of the two this file is",
                    "episode_like": True, "candidates": candidates, "whole": whole, "text": text}
        top = max(expected_votes, key=lambda v: float(v.get("score") or 0))
        return {"status": "OK", "code": expected_votes[0].get("code"),
                "score": float(top.get("score") or 0), "votes": len(expected_votes),
                "note": f"Sample windows confirm {expected_votes[0].get('code')}",
                "episode_like": True, "candidates": candidates, "whole": whole, "text": text}
    if expected_votes and target_votes:
        return {"status": "LOW_CONFIDENCE", "code": None,
                "note": "Sample windows support the filename and another episode",
                "episode_like": True, "candidates": candidates, "whole": whole, "text": text}
    best = targets[0] if targets else None
    if best and best["votes"] >= max(1, int(required_mismatch)):
        return {"status": "MISMATCH", "code": best["code"], "score": best["score"],
                "votes": best["votes"], "alternatives": best.get("alternatives") or [],
                "note": f"Multiple sample windows match {best['code']}",
                "episode_like": True, "candidates": candidates, "whole": whole, "text": text}
    return {"status": "LOW_CONFIDENCE", "code": None,
            "note": (f"{best['code']} is suggested once, but the filename needs more evidence"
                     if best else "Sample evidence is not strong enough"),
            "episode_like": True, "candidates": candidates, "whole": whole, "text": text}


def _apply_sample_aggregate(f: dict, agg: dict, window_seconds: float) -> None:
    status = agg["status"]
    f["status"] = status
    f["sample_text"] = agg.get("text") or ""
    f["whole"] = agg.get("whole") or []
    f["sample_candidates"] = agg.get("candidates") or []
    f["episode_like"] = bool(agg.get("episode_like"))
    f["note"] = agg.get("note") or f.get("note") or ""
    if agg.get("language"):
        f["language"] = agg["language"]
    if status in ("OK", "MISMATCH"):
        code = agg["code"]
        score = float(agg.get("score") or 0)
        runner_up = max((float(x.get("score") or 0) for x in agg.get("whole") or []
                         if x.get("code") != code), default=0.0)
        segment = {"start": 0.0, "end": float(f.get("file_duration") or f.get("duration") or window_seconds),
                   "code": code, "score": round(score, 4),
                   "margin": round(score - runner_up, 4),
                   "confidence": "high", "scope": "sample",
                   "alternatives": [x for x in agg.get("candidates") or [] if x.get("code") != code],
                   "words": len((agg.get("text") or "").split()),
                   "evidence": "sample-windows", "why": ""}
        f["segments"] = [segment]
        f["note"] = agg.get("note") or f"Sample windows support {code}"
    else:
        f["segments"] = []
    windows = len(f.get("sample_windows") or [])
    if windows:
        f["note"] += f" ({windows} sampled windows)"


def _sample_confirmation_pass(videos_by_rel: dict[str, Path], files: list[dict], matcher: Matcher,
                              eps: dict, refs: dict, settings: dict, ctx,
                              lex: evidence.Lexicon | None = None) -> None:
    base_start = float(settings["decision_sample_start"])
    window = float(settings["decision_sample_seconds"])
    max_windows = max(1, int(settings["decision_sample_max_windows"]))
    required = max(1, int(settings["decision_mismatch_windows"]))
    todo = [f for f in files if f.get("scope") == "sample" and f.get("status") != "OK"]
    if not todo:
        return
    ctx.log(f"Sample adjudication: rechecking {len(todo)} sampled file(s) with up to "
            f"{max_windows} windows")
    for n, f in enumerate(todo):
        ctx.check_cancel()
        ctx.progress(n / max(1, len(todo)), f"Sample evidence: {Path(f['rel']).name}")
        expected = f.get("expected") or []
        conflicts = (set(lex.conflicting(expected[0]))
                     if lex is not None and len(expected) == 1 else set())
        video = videos_by_rel.get(f["rel"])
        if video is None:
            continue
        initial_start = float(f.get("sample_start") or base_start)
        windows = [_window_from_existing(f, settings, len(matcher.codes))]
        for start in _sample_window_starts(f.get("file_duration") or 0, initial_start,
                                           window, max_windows):
            if any(abs(start - float(w.get("start", initial_start))) < max(15, window * 0.25)
                   for w in windows):
                continue
            try:
                windows.append(_scored_sample_window(video, matcher, start, window, expected,
                                                      eps, refs, settings, ctx))
            except Exception as e:  # noqa: BLE001 — sample expansion is optional evidence
                ctx.log(f"  couldn't sample {start / 60:.0f} min of {Path(f['rel']).name}: {e}")
            f["sample_windows"] = [{k: w.get(k) for k in
                                    ("start", "end", "source", "words", "has_text", "language", "votes", "whole")}
                                   for w in windows]
            agg = _aggregate_sample_evidence(expected, windows, eps, settings, required, conflicts)
            if agg["status"] in ("OK", "MISMATCH"):
                break
            if agg.get("language") and len(windows) >= min(2, max_windows):
                break
        f["sample_windows"] = [{k: w.get(k) for k in
                                ("start", "end", "source", "words", "has_text", "language", "votes", "whole")}
                               for w in windows]
        f["sample_start"] = min(float(w.get("start") or initial_start) for w in windows)
        f["sample_end"] = max(float(w.get("end") or initial_start + window) for w in windows)
        f["sample_window_seconds"] = window
        agg = _aggregate_sample_evidence(expected, windows, eps, settings, required, conflicts)
        _apply_sample_aggregate(f, agg, window)
        ctx.log(f"  {f['status']:<16} {Path(f['rel']).name}: {f['note']}")


@handler("scan")
def job_scan(ctx, params):
    s = get_settings()
    series = _ensure_episodes(_series(ctx), ctx, max_age_h=EPISODES_MAX_AGE_H)
    if params.get("imported"):
        ctx.log(f"Sonarr imported {len(params['only'])} file(s): fetching any missing references first.")
        try:
            job_fetch_refs(ctx, {"auto": True})  # what used to be a separate job per imported file
        except Cancelled:
            raise
        except Exception as e:  # noqa: BLE001 — scan with what there is
            ctx.log(f"Couldn't fetch references ({e}); scanning with the ones already here.")
        series = db.get_series(series["id"])
    root = Path(series["path"])
    opts = series.get("options") or {}
    include_specials = bool(opts.get("include_specials"))
    eps = {ep_code(e["season"], e["episode"]): e for e in series["episodes"]}
    note = sonarr_series_type_note(series)
    if note:
        ctx.log(note)

    refs = {c: r["text"] for c, r in load_refs(series["tvdb_id"]).items()
            if c in eps and (include_specials or eps[c]["season"] > 0)}
    ctx.log(f"References available for {len(refs)} of "
            f"{sum(1 for e in eps.values() if include_specials or e['season'] > 0)} episodes.")
    if not refs:
        ctx.log("No references yet — every file will need the AI fallback or a manual decision.")

    try:
        migrate_legacy_backup(series, ctx)
    except Exception as e:  # noqa: BLE001 — never block a scan on this
        ctx.log(f"Couldn't move the old backup folder out of the series: {e}")
    videos = list_videos(root, s["backup_folder"])
    if params.get("only"):
        only = set(params["only"])
        videos = [v for v in videos if str(v.relative_to(root)) in only]
    ctx.log(f"Scanning {len(videos)} video files in {root}")
    fast = bool(s["decision_fast_path"])
    runtime_min = _runtime_minutes(eps)
    if fast:
        ctx.log(f"Fast path: sampling the first {int(s['decision_sample_seconds']) // 60} min "
                "of single-episode files; multi-episode and long files use the full matcher.")
    # 1. Read every file's dialogue (cached unless new or replaced).
    read = []
    for i, v in enumerate(videos):
        ctx.check_cancel()
        rel = str(v.relative_to(root))
        ctx.progress(i / max(1, len(videos)) * 0.9, f"Reading {v.name}")
        expected = parse_filename_episodes(v.name)
        try:
            dlg = None
            if fast and _sample_eligible(v, expected, eps, runtime_min, s):
                dlg = sample_dialogue(v, ctx, allow_whisper=params.get("whisper", True),
                                      start=s["decision_sample_start"],
                                      duration=s["decision_sample_seconds"])
                if dlg.get("scope") == "sample" and dlg.get("cues"):
                    ctx.log(f"  sampled {rel.rsplit('/', 1)[-1]}: {dlg['source']} "
                            f"({dlg['duration'] / 60:.1f} min)")
                elif not dlg.get("cues"):
                    ctx.log(f"  sample found no text in {rel}; reading the whole file")
                    dlg = None
            if dlg is None:
                dlg = get_dialogue(v, ctx, allow_whisper=params.get("whisper", True))
            read.append((v, rel, dlg, None))
        except Exception as e:  # noqa: BLE001
            read.append((v, rel, None, e))
    # 2. Match. Word weights come from the references and every file's dialogue.
    background = [" ".join(c[2] for c in dlg["cues"]) for _v, _r, dlg, _e in read if dlg and dlg["cues"]]
    if params.get("only"):
        # A few files only: the rest of the library's dialogue still sets the word weights
        # (from the cache — files never read before aren't read now).
        done = {r for _v, r, _d, _e in read}
        for v in list_videos(root, s["backup_folder"]):
            if str(v.relative_to(root)) not in done:
                c = cached_dialogue(v)
                if c and c.get("cues"):
                    background.append(" ".join(x[2] for x in c["cues"]))
    matcher = Matcher(refs, s, background) if refs else None
    if matcher and len(refs) < 5:
        ctx.log(f"Only {len(refs)} reference(s): a dialogue match only counts when it's plainly that episode.")
    lex = None
    if refs and (fast or decision.available()):
        try:
            lex = evidence.Lexicon(eps, refs, story=lambda code: _story(eps, code),
                                   conflict_similarity=float(s["decision_conflict_similarity"]))
        except Exception as e:  # noqa: BLE001 — the scan works without it, just less precisely
            ctx.log(f"Reference check: couldn't compare references ({e}); continuing without it.")
    if lex and lex.conflict_pairs():
        pairs = lex.conflict_pairs()
        ctx.log(f"Reference check: {len(pairs)} pair(s) share one transcript, so they can neither "
                "confirm nor deny a filename — " +
                ", ".join(f"{'/'.join(p['codes'])} ({p['similarity']:.2f})" for p in pairs[:6]) +
                ("…" if len(pairs) > 6 else "") + ". 'Fetch references' looks for replacements.")
    files = []
    for i, (v, rel, dlg, err) in enumerate(read):
        ctx.check_cancel()
        ctx.progress(0.9 + 0.1 * i / max(1, len(read)), f"Matching {v.name}")
        expected = parse_filename_episodes(v.name)
        if err is not None:
            ctx.log(f"{rel}: could not read ({err})")
            files.append({"rel": rel, "size": v.stat().st_size, "duration": 0,
                          "text_source": "none", "expected": expected, "segments": [],
                          "whole": [], "status": "ERROR", "note": str(err)[:200]})
            continue
        res = matcher.match_file(dlg["cues"], dlg["duration"]) if matcher and dlg["cues"] \
            else {"segments": [], "whole": []}
        _discount_unreferenced(res["segments"], expected, dlg["duration"], refs)
        res["segments"] = _story_parts(res["segments"], expected, eps, dlg["duration"])
        sampled = dlg.get("scope") == "sample"
        if sampled:
            # Keep the matcher's real high/low result. A high sample match can
            # certify a single episode; a high mismatch is still only an
            # untracked suggestion. Weak samples go to review below.
            for sg in res["segments"]:
                sg["scope"] = "sample"
        status, note = classify(expected, res["segments"], dlg["source"])
        if sampled:
            if status == "OK":
                note = f"Sample of {dlg['duration'] / 60:.0f} min — matches the filename"
            elif status == "MISMATCH":
                note = f"Sample of {dlg['duration'] / 60:.0f} min — suggested match needs review"
            else:
                note = f"Sample of {dlg['duration'] / 60:.0f} min — weak sample evidence"
        files.append({"rel": rel, "size": v.stat().st_size, "duration": dlg["duration"],
                      "text_source": dlg["source"], "expected": expected,
                      "scope": dlg.get("scope") or "full",
                      "sample_start": dlg.get("sample_start"),
                      "sample_end": dlg.get("sample_end"),
                      "file_duration": dlg.get("file_duration"),
                      "sample_text": adjudication.dialogue_text(dlg["cues"]) if sampled else "",
                      "segments": res["segments"], "whole": res["whole"],
                      "status": status, "note": note})
        det = " + ".join(f"{sg['code']}{'' if sg['confidence'] == 'high' else '?'}"
                         for sg in res["segments"]) or "-"
        ctx.log(f"{status:<16} {rel}  →  {det}")
        for sg in res["segments"]:
            if sg.get("why"):
                ctx.log(f"{'':<16}   {sg['code']}? {sg['why']}")

    videos_by_rel = {rel: v for v, rel, *_ in read}
    if fast and s.get("decision_sample_expand") and matcher:
        _sample_confirmation_pass(videos_by_rel, files, matcher, eps, refs, s, ctx, lex)

    scanned = [f for f in files if f.get("scope") != "sample"]
    if scanned:
        _release_name_pass(series, scanned, ctx)
        _titlecard_pass(series, scanned, eps, ctx, reread=bool(params.get("reread_cards")))
        _title_card_swaps(scanned, ctx)
        for sw in reference_swaps(scanned, refs):
            ctx.log(f"References for {sw['codes'][0]} and {sw['codes'][1]} look swapped: title cards confirm each "
                    f"file's name, but each file's dialogue matched the other's reference ({sw['files'][0]}, "
                    f"{sw['files'][1]}). Swap them under Episodes & references.")

    if llm.available() and scanned:
        _llm_pass(series, scanned, eps, refs, ctx)

    if decision.available() and any(f.get("scope") == "sample" for f in files):
        _decision_pass(series, files, eps, ctx, lex)

    if params.get("only"):
        # Only some files were scanned: the rest keep what the last scan found (if they're
        # still there), so the plan still covers the whole series.
        prev = db.latest_scan(series["id"])
        mine = {f["rel"] for f in files}
        kept = [f for f in (prev["results"].get("files") or [] if prev else [])
                if f["rel"] not in mine and (root / f["rel"]).exists()]
        if kept:
            ctx.log(f"Kept the last scan's results for the other {len(kept)} files.")
        files = sorted(kept + files, key=lambda f: f["rel"])
    counts: dict[str, int] = {}
    for f in files:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    scan = {"files": files, "counts": counts, "reference_count": len(refs)}
    scan_id = db.add_scan(series["id"], scan)
    items = build_plan(db.get_series(series["id"]), db.latest_scan(series["id"])["results"])
    plan_id = db.add_plan(series["id"], scan_id, items)
    ctx.log("Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if params.get("imported"):
        autocheck.note_imports(series, [f for f in files if f["rel"] in set(params["only"])], ctx)
        if s["check_imports"]:
            from . import integrity
            try:
                integrity.check_paths(series, list(params["only"]), ctx)
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "Cancelled":
                    raise
                ctx.log(f"Couldn't check the new download(s) for damage: {e}")
    if params.get("only"):
        autocheck.record_files(series, [str(v.relative_to(root)) for v in videos])
        autocheck.check(db.get_series(series["id"]), submit=False)  # refresh the card's flag
    else:
        autocheck.record(series)
    return {"scan_id": scan_id, "plan_id": plan_id, "counts": counts, "files": [
        {"rel": f["rel"], "status": f["status"], "note": f.get("note", "")} for f in files
        if not params.get("only") or f["rel"] in set(params["only"])]}


# ------------------------------------------------------------- title cards

def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _merge_windows(ws: list) -> list:
    out: list = []
    for a, b in sorted(ws):
        if out and a <= out[-1][1] + 5:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _discount_unreferenced(segs: list[dict], expected: list[str], duration: float,
                           refs: dict) -> None:
    """Dialogue matched to an episode other than the filename's only counts against
    the filename when the filename's episode has a reference. Without one the
    matcher can only ever pick the closest other episode — and shows that reuse
    plots (PAW Patrol has several elephant rescues) make that look convincing."""
    for sg in segs:
        if not expected or sg["code"] in expected:
            continue
        pos = _expected_at(expected, (sg["start"] + sg["end"]) / 2, duration)
        cands = [pos] if pos else expected
        if any(c in refs for c in cands):
            continue  # the filename's episode was a candidate
        # A reference several times longer than this file's dialogue covers more than one
        # episode (one wiki transcript for a three-part story): matching it says nothing
        # about which of those episodes this is, however close the match.
        words = sum(int(x.get("words") or 0) for x in segs) or 0
        ref_words = len(str(refs.get(sg["code"]) or "").split())
        broad = words >= 200 and ref_words >= 1.8 * words
        if float(sg.get("score") or 0) >= SAME_EPISODE_SCORE and not broad:
            continue  # plainly that episode
        if broad:
            sg.update(confidence="low", no_ref_for=cands[0], broad_ref=True,
                      why=f"the reference for {sg['code']} is {ref_words / words:.1f}× as long as this file's "
                          f"dialogue — it likely covers several episodes, so it can't tell them apart; "
                          f"{' / '.join(cands)} (per the filename) has no reference of its own")
            continue
        sg.update(confidence="low", no_ref_for=cands[0],
                  why=f"{' / '.join(cands)} (per the filename) has no reference yet, so the "
                      f"dialogue could only match other episodes — {sg['code']} was the closest")


def _release_name_pass(series: dict, files: list[dict], ctx) -> None:
    """Sonarr remembers the release each file was imported from. When that name carries
    an episode title ("…S04E03.Dunder.Mifflin.Infinity.720p…"), it's evidence of its own:
    it settles a file the dialogue couldn't, or says which episode it probably is."""
    from . import sonarr
    from .references import name_check
    try:
        names = sonarr.release_names(series["tvdb_id"])
    except sonarr.SonarrError as e:
        ctx.log(f"Sonarr release names: {e}")
        return
    if not names:
        return
    used = 0
    for f in files:
        rel_names = names.get(f["rel"])
        if not rel_names or f.get("status") == "ERROR":
            continue
        exp = f.get("expected") or []
        checks = [name_check(rel_names, series["episodes"], c, series["name"]) for c in exp] or \
                 [name_check(rel_names, series["episodes"], "", series["name"])]
        if exp and all(v == "match" for v, _o, _n in checks):
            verdict, code, name = "match", None, checks[0][2]
        else:
            other = next(((o, n) for v, o, n in checks if v == "other"), None)
            if not other:
                continue
            verdict, code, name = "other", other[0], other[1]
        f["release"] = {"name": name, "verdict": verdict, "code": code}
        segs = f.get("segments") or []
        strong = [sg for sg in segs if sg.get("confidence") in ("high", "manual")]
        dur = float(f.get("duration") or 0)
        if verdict == "match":
            if any(sg.get("code") not in exp for sg in strong) or len(strong) == len(exp):
                continue  # the dialogue is sure of something else, or already sure of this
            step = dur / len(exp) if dur else 0
            f["segments"] = [{"start": round(i * step, 1), "end": round((i + 1) * step, 1), "code": c,
                              "score": 0, "confidence": "high", "alternatives": [], "evidence": "release name",
                              "why": f"Sonarr imported it from '{name}', which carries the title"}
                             for i, c in enumerate(exp)]
            f["status"], f["note"] = classify(exp, f["segments"], f.get("text_source") or "none")
            f["note"] = f"confirmed by its release name ({name})"
            used += 1
        elif not strong:
            # Another episode's title in the release name: worth a look, not a rename on its own.
            f["segments"] = [{"start": 0, "end": dur, "code": code, "score": 0, "confidence": "low",
                              "alternatives": [], "evidence": "release name",
                              "why": f"Sonarr imported it from '{name}', which names {code}"}]
            f["status"], _n = classify(exp, f["segments"], f.get("text_source") or "none")
            f["note"] = f"its release name ({name}) names {code}"
            used += 1
    if used:
        ctx.log(f"Sonarr release names: {used} file(s) settled or flagged by the release they were imported from.")


def _story(eps: dict, code: str) -> list[str]:
    """The episodes TVDB splits one story into, around `code`: neighbours in the same
    season with the same title apart from "(1)" / "(2)" / "Part 2" (The Office's
    "Dunder Mifflin Infinity (1)" and "(2)")."""
    from .planner import _MULTI_PART
    e = eps.get(code)
    if not e or not e.get("title"):
        return [code]
    base = _MULTI_PART.sub("", e["title"]).strip().lower()
    if base == e["title"].strip().lower():
        return [code]  # no part marker: a story of its own
    out = []
    for k in range(-4, 5):
        c = ep_code(e["season"], e["episode"] + k)
        x = eps.get(c)
        if x and _MULTI_PART.sub("", x.get("title") or "").strip().lower() == base:
            out.append(c)
    return sorted(out)


def _story_parts(segs: list[dict], expected: list[str], eps: dict, duration: float) -> list[dict]:
    """A story TVDB splits into parts ("… (1)", "… (2)") often has one subtitle or
    transcript for the whole hour, filed under one of the parts. Matched against it, a
    file holding the whole story reads as that one part from start to finish — so a
    correctly named S04E03E04 looked like S04E04 alone. When a match runs much longer
    than that part does on TVDB and the filename names parts of the same story, the
    match can't tell the parts apart: it confirms the filename's parts. When the file
    holds more parts than its name says, it goes to review with the parts named."""
    out = []
    for sg in segs:
        code = sg.get("code")
        e = eps.get(code) or {}
        rt = float(e.get("runtime") or 0) * 60
        span = float(sg.get("end", 0)) - float(sg.get("start", 0))
        story = _story(eps, code) if code else [code]
        named = [c for c in expected if c in story]
        if (sg.get("override") or len(story) < 2 or not named or not rt or span < 1.5 * rt
                or sg.get("confidence") not in ("high", "low")):
            out.append(sg)
            continue
        parts_here = min(len(story), max(1, round(span / rt)))
        title = _MULTI_PART_TITLE(e.get("title"))
        if parts_here <= len(named):
            # The filename's parts of this story, one after the other across the match.
            a, step = float(sg.get("start", 0)), span / len(named)
            for i, c in enumerate(named):
                out.append({**sg, "code": c, "start": round(a + i * step, 1), "end": round(a + (i + 1) * step, 1),
                            "confidence": "high", "story": story,
                            "why": f"one reference holds the whole of '{title}' (it matched {span / 60:.0f} min, "
                                   f"TVDB gives each part {rt / 60:.0f}) — so it confirms {', '.join(named)} "
                                   "(the filename) rather than telling the parts apart"})
        else:
            likely = story[:parts_here] if len(story) >= parts_here else story
            out.append({**sg, "code": named[0], "confidence": "low", "story": story,
                        "why": f"this runs {span / 60:.0f} min — about {parts_here} parts of '{title}' "
                               f"({', '.join(likely)}), but the filename names {', '.join(named)}; one reference "
                               "holds the whole story, so it can't tell the parts apart. If the file has them all, "
                               f"set it to {''.join([likely[0]] + [c[c.index('E'):] for c in likely[1:]])}"})
    return out


def _MULTI_PART_TITLE(t: str | None) -> str:
    from .planner import _MULTI_PART
    return _MULTI_PART.sub("", t or "").strip()


def _expected_at(expected: list[str], t: float, duration: float) -> str | None:
    """The episode the filename puts at time t: for "S11E38E39" the first half is
    S11E38 and the second S11E39."""
    if not expected:
        return None
    if len(expected) == 1:
        return expected[0]
    if duration <= 0:
        return None  # position unknown: callers fall back to all of the filename's episodes
    return expected[min(len(expected) - 1, int(t / (duration / len(expected))))]


def _resolve_partial(c: dict, sg: dict | None, expected: list[str]) -> str | None:
    """A partial read ("PUPS SAVE RYDER'S…", "MIGHTY PUPS") lists the titles it could
    be the start of. It can confirm the dialogue's match or the filename when either
    is one of those titles — it never picks an episode on its own."""
    cands = set(c.get("candidates") or [])
    if sg and sg.get("confidence") == "high" and sg.get("code") in cands:
        return sg["code"]                       # confirms a confident dialogue match
    for e in expected:
        if e in cands:
            return e                            # confirms the filename
    if sg:
        for code in [sg.get("code")] + [a["code"] for a in sg.get("alternatives") or []]:
            if code in cands:
                return code                     # agrees with a weaker dialogue candidate
    return None


def _clear_card(c: dict) -> bool:
    """A complete read of a title seen on at least two frames — not a glimpse or a
    card still animating in. Trusted over the filename when the dialogue is unclear."""
    return not c.get("partial") and bool(c.get("code")) and int(c.get("frames") or 1) >= 2


def _apply_title_cards(f: dict, cards: list[dict], runtimes: dict | None = None,
                       trust: bool = False) -> list[str]:
    """Merge title-card evidence into a scanned file's segments. Returns log notes.
    runtimes: episode code → minutes (TVDB), used to judge whether a stretch of
    weak dialogue fits inside a card-confirmed episode."""
    keep = ("code", "title", "text", "score", "time", "partial", "candidates", "frames")
    f["title_cards"] = [{k: c.get(k) for k in keep} for c in cards]
    if not cards:
        return []
    notes = []
    segs = f["segments"]
    expected = f.get("expected") or []
    dur = float(f.get("duration") or 0)
    if not segs:
        # No usable dialogue: the title cards alone identify the episode(s).
        usable = []
        for i, c in enumerate(f["title_cards"]):
            pos = _expected_at(expected, c["time"], dur)
            code = c["code"] if not c.get("partial") else \
                _resolve_partial(c, None, [pos] if pos else expected)
            if code:
                usable.append((code, c))
        for i, (code, c) in enumerate(usable):
            if i == 0:
                # Starts the file — unless the filename puts an earlier episode before it
                # (whose card wasn't read): then it starts around its card.
                later = code in expected and expected.index(code) > 0
                start = max(0.0, c["time"] - 60) if later else 0.0
            else:
                start = max(segs[-1]["start"] + 60, c["time"] - 5)
            if segs:
                segs[-1]["end"] = max(segs[-1]["start"] + 30, c["time"] - 90)
            segs.append({"start": round(start, 1), "end": round(dur, 1), "code": code,
                         "score": c["score"], "margin": 0, "confidence": "high",
                         "alternatives": [], "evidence": "title" if not c.get("partial")
                         else "title+filename", "why": "", "title_card": {**c, "code": code}})
        if usable:
            notes.append("identified from title card" + ("s" if len(usable) > 1 else ""))
    else:
        confirmed: set[str] = set()  # filename episodes a title card has confirmed
        originals = {id(sg): dict(sg) for sg in segs}  # what the dialogue said, before cards
        for c in f["title_cards"]:
            # The stretch of dialogue this card belongs to: one it falls inside (a card can
            # come up to 60 s before its dialogue starts, and 15 s after a stretch ends, as
            # dialogue is matched in overlapping windows). None → the card is in a stretch
            # where the dialogue matched nothing — at the start, between, or at the end.
            inside = [j for j, sg in enumerate(segs)
                      if sg["start"] - 60 <= c["time"] <= sg["end"] + 15]
            idx = max(inside) if inside else None
            sg = segs[idx] if idx is not None else None
            pos_expected = _expected_at(expected, c["time"], dur)
            if c.get("partial"):
                code = _resolve_partial(c, sg, [pos_expected] if pos_expected else expected)
                if not code:
                    continue  # incomplete read that matches nothing else: no evidence
                c = {**c, "code": code}
            if sg is None:
                # Typically an episode with no reference: the card starts a segment of its
                # own, filling the unmatched stretch.
                prev = max([j for j, x in enumerate(segs) if x["end"] <= c["time"]] or [-1])
                lo = segs[prev]["end"] if prev >= 0 else 0.0
                hi = segs[prev + 1]["start"] if prev + 1 < len(segs) else dur
                ok = c["code"] in expected or not expected or set(expected) <= confirmed \
                    or (trust and _clear_card(c))
                new = {"start": lo, "end": hi, "code": c["code"], "score": c["score"],
                       "margin": 0, "alternatives": [], "title_card": c,
                       "confidence": "high" if ok else "conflict",
                       "evidence": "title+filename" if c["code"] in expected else "title",
                       "why": "" if ok else f"title card reads '{c['text']}' ({c['code']}) where the "
                                            f"dialogue matched nothing; the filename says "
                                            f"{', '.join(expected)}"}
                segs.insert(prev + 1, new)
                if c["code"] in expected:
                    confirmed.add(c["code"])
                notes.append(f"title card '{c['text']}' → {c['code']} where the dialogue matched nothing")
                continue
            if sg.get("title_card"):
                first = sg["title_card"]
                lead = min(120.0, max(5.0, float(first["time"]) - float(sg["start"])))
                cut = float(c["time"]) - lead
                # The new episode runs to the next stretch of dialogue (or the end of the
                # file) — including any unmatched stretch right after this one.
                piece_end = segs[idx + 1]["start"] if idx + 1 < len(segs) else dur or sg["end"]
                if first.get("code") == c["code"] or cut - sg["start"] < 240 or piece_end - cut < 120:
                    continue  # same title again, or too close to the first card to be another episode
                # A second, different title inside one stretch of dialogue: the dialogue
                # matched one episode across both (e.g. neither has a reference), so the
                # card marks where the next episode starts. Split the segment there; the
                # new part starts from what the dialogue said, before the first card.
                o = originals.get(id(sg), sg)
                piece = {k: v for k, v in o.items() if k not in (
                    "title_card", "dialogue_code", "evidence", "title_conflict", "why", "no_ref_for")}
                piece.update(start=round(cut, 1), end=round(max(piece_end, sg["end"]), 1))
                if o.get("no_ref_for"):
                    piece["no_ref_for"] = pos_expected or o["no_ref_for"]
                sg["end"] = round(cut, 1)
                segs.insert(idx + 1, piece)
                idx, sg = idx + 1, piece
            old = sg["code"]
            # The card shows the title of a story TVDB splits into parts ("Diesel's Ghostly
            # Christmas" for "… (1)" and "… (2)"), the filename names those parts, and the
            # dialogue matched one part right across where the filename puts them all: one
            # reference holds the whole story. The card confirms the filename's parts.
            story = [e for e in expected if e in (c.get("candidates") or [])] if c.get("partial") else []
            if (len(story) >= 2 and old in story and sg.get("confidence") in ("high", "low")
                    and len({_expected_at(expected, t, dur) for t in (sg["start"] + 1, sg["end"] - 1)}
                            & set(story)) >= 2):
                a, step = float(sg["start"]), (float(sg["end"]) - float(sg["start"])) / len(story)
                why = (f"the title card reads the story's title ('{c['text']}'), the filename names its "
                       f"parts {', '.join(story)}, and the dialogue matched {old} right across them — one "
                       "reference likely holds the whole story, so it can't tell the parts apart")
                segs[idx:idx + 1] = [{**sg, "code": e, "start": round(a + i * step, 1),
                                      "end": round(a + (i + 1) * step, 1), "confidence": "high",
                                      "evidence": "title+filename", "dialogue_code": old, "story": story,
                                      "title_card": {**c, "code": e}, "why": why}
                                     for i, e in enumerate(story)]
                confirmed |= set(story)
                notes.append(f"title card '{c['text']}' names the story; confirms {', '.join(story)} (filename)")
                continue
            sg["title_card"] = c
            alts = {a["code"] for a in sg.get("alternatives") or []}
            if c["code"] in expected:
                confirmed.add(c["code"])
            if c["code"] == old:
                sg.update(confidence="high", evidence="dialogue+title", why="")
            elif c["code"] in expected:
                if old in expected:
                    why = (f"the dialogue matched {old} through this part as well; title card and "
                           f"filename both say {c['code']}")
                    notes.append(f"title card '{c['text']}' confirms {c['code']} (filename)")
                elif sg.get("no_ref_for") == c["code"]:
                    why = (f"no reference for {c['code']} yet, so the dialogue matched the closest "
                           f"other episode ({old}); title card and filename both say {c['code']}")
                    notes.append(f"title card '{c['text']}' confirms the filename ({c['code']} "
                                 f"has no reference; dialogue's closest was {old})")
                else:
                    why = (f"dialogue matched {old}; title card and filename both say "
                           f"{c['code']} — the reference for {old} may be wrong")
                    notes.append(f"title card '{c['text']}' confirms the filename; dialogue said {old} "
                                 f"(check the reference for {old})")
                sg.update(dialogue_code=old, code=c["code"], confidence="high",
                          evidence="title+filename", why=why)
            elif sg.get("confidence") != "high" and trust and _clear_card(c) \
                    and c["code"] not in alts and expected and not set(expected) <= confirmed:
                # The card decides: the dialogue is unclear and a clear card beats the name.
                sg.update(dialogue_code=old, code=c["code"], confidence="high", evidence="title",
                          why=f"title card '{c['text']}' ({c['code']}); the filename says "
                              f"{', '.join(expected)} and the dialogue is unclear ({old or '—'})")
                notes.append(f"title card '{c['text']}' says {c['code']}, not {', '.join(expected)} "
                             "(dialogue unclear)")
            elif sg.get("confidence") != "high" and (c["code"] in alts or not expected
                                                     or set(expected) <= confirmed):
                # Weak dialogue, and the card either is one of its runner-up matches or names
                # an extra episode in a file whose named episodes are all accounted for.
                sg.update(dialogue_code=old, code=c["code"], confidence="high",
                          evidence="title", why="")
                notes.append(f"weak dialogue match{' ' + old if old else ''} settled by title card")
            else:
                # The card alone disagrees with the filename and with what the dialogue
                # points to: don't let it decide by itself.
                sg.update(confidence="conflict", title_conflict=c["code"],
                          why=f"title card reads '{c['text']}' ({c['code']}) but "
                              + (f"dialogue matches {old}" if sg.get("confidence") == "high" else
                                 f"the filename says {', '.join(expected) or '—'} and the dialogue "
                                 f"is unclear ({old or '—'})"))
                notes.append(f"CONFLICT: title card says {c['code']}, dialogue {old or '—'}, "
                             f"filename {', '.join(expected) or '—'}")
        # A weak stretch of dialogue right after a card-confirmed episode, naming an
        # episode the filename doesn't, is part of that episode (as long as it fits in
        # the episode's runtime) — not a separate episode to cut out.
        absorbed: list[dict] = []
        for sg in segs:
            prev = absorbed[-1] if absorbed else None
            if (prev and prev.get("title_card") and prev.get("confidence") == "high"
                    and not sg.get("title_card") and sg.get("confidence") == "low"
                    and sg["code"] not in expected and sg["code"] != prev["code"]
                    and (len(expected) < 2 or _expected_at(
                        expected, (sg["start"] + sg["end"]) / 2, dur) in (None, prev["code"]))):
                rt_min = (runtimes or {}).get(prev["code"])
                if not rt_min or sg["end"] - prev["start"] <= rt_min * 60 * 1.4:
                    prev["end"] = sg["end"]
                    prev.setdefault("absorbed", []).append(
                        {"start": sg["start"], "end": sg["end"], "code": sg["code"],
                         "score": sg.get("score")})
                    notes.append(f"weak {sg['code']} match at {_fmt_t(sg['start'])}–{_fmt_t(sg['end'])} "
                                 f"kept with {prev['code']} (title card)")
                    continue
            absorbed.append(sg)
        merged: list[dict] = []
        for sg in absorbed:
            if merged and merged[-1]["code"] == sg["code"] and \
                    merged[-1].get("confidence") == sg.get("confidence"):
                merged[-1]["end"] = sg["end"]
            else:
                merged.append(sg)
        segs = merged
    f["segments"] = segs
    f.pop("hold", None)
    f["status"], note = classify(expected, segs, f.get("text_source") or "none")
    detected = {sg["code"] for sg in segs}
    if f["status"] == "MISMATCH" and expected and detected < set(expected):
        # Cards confirm part of the filename and nothing contradicts the rest: there is
        # just no evidence for the rest (no reference, card not found). Leave it alone.
        missing = [c for c in expected if c not in detected]
        f["status"], f["hold"] = "LOW_CONFIDENCE", True
        where = ", ".join(f"{_fmt_t(a)}–{_fmt_t(b)}" for a, b in f.get("title_scan") or [])
        notes.append(f"no evidence either way for {', '.join(missing)} — left as named"
                     + (f" (no title card read for it; looked at {where})" if where else ""))
    if notes and f["status"] != "LOW_CONFIDENCE":
        note = "; ".join(notes)
    elif notes and f["status"] == "LOW_CONFIDENCE" and not note.startswith("At least"):
        note = "; ".join(notes)
    f["note"] = note
    return notes


def reference_swaps(files: list[dict], refs: dict | None = None, swapped: dict | None = None,
                    scanned_at: str | None = None) -> list[dict]:
    """Pairs of references that hold each other's episode: a file confirmed as X (title
    card, with the filename or trusted) whose dialogue matched Y, and a file confirmed as
    Y whose dialogue matched X. A pair swapped since these files were scanned is left out
    (swapped: references.load_swapped; scanned_at: when the scan ran)."""
    seen: dict[tuple, str] = {}
    for f in files:
        for sg in f.get("segments") or []:
            d, c = sg.get("dialogue_code"), sg.get("code")
            if d and c and d != c and sg.get("confidence") == "high" and "title" in (sg.get("evidence") or ""):
                seen.setdefault((c, d), f["rel"])
    out = []
    for (x, y), rel in sorted(seen.items()):
        if x < y and (y, x) in seen:
            done = (swapped or {}).get(x) or {}
            if done.get("with") == y and (not scanned_at or done.get("at", "") >= scanned_at):
                continue  # swapped after this scan: the next scan shows whether it helped
            out.append({"codes": [x, y], "files": [rel, seen[(y, x)]]})
    return out


def _title_card_swaps(files: list[dict], ctx) -> None:
    """Two files whose title cards each name the other's episode have been swapped.
    One card contradicting its filename isn't trusted on its own (a misread would
    rename a good file), but two that point at each other are."""
    def conflict(f):
        segs = f.get("segments") or []
        if len(segs) != 1 or len(f.get("expected") or []) != 1:
            return None
        return segs[0].get("title_conflict") if segs[0].get("confidence") == "conflict" else None

    by_name = {}
    for f in files:
        if conflict(f):
            by_name.setdefault(f["expected"][0], []).append(f)
    done = set()
    for f in files:
        other_code = conflict(f)
        if not other_code or f["rel"] in done:
            continue
        partners = [g for g in by_name.get(other_code, [])
                    if g["rel"] not in done and conflict(g) == f["expected"][0]]
        if len(partners) != 1:
            continue  # none, or ambiguous (several files claim the same pair)
        g = partners[0]
        for a, b in ((f, g), (g, f)):
            sg = a["segments"][0]
            sg.update(code=sg["title_conflict"], confidence="high", evidence="title (swap)",
                      dialogue_code=sg.get("dialogue_code") or sg.get("code"),
                      why=f"the title cards in this file and in {Path(b['rel']).name} each name "
                          "the other's episode — the two files are swapped")
            sg.pop("title_conflict", None)
            a["status"], _ = classify(a["expected"], a["segments"], a.get("text_source") or "none")
            a["note"] = f"swapped with {Path(b['rel']).name} (title cards)"
            done.add(a["rel"])
        ctx.log(f"Title cards: {f['rel']} and {g['rel']} each show the other's title — swapped.")


def _titlecard_pass(series: dict, files: list[dict], eps: dict, ctx, reread: bool = False) -> None:
    s = get_settings()
    opts = dict(series.get("options") or {})
    mode = opts.get("title_cards", "auto")
    if not s["titlecard_enabled"] or mode == "off" or not files:
        return
    if not titlecard.available():
        ctx.log("Title cards: OCR engine not installed in this image — skipped.")
        return
    try:
        acc = titlecard.acceleration_status()
        ctx.log(f"Title cards: video decode on {acc['decode'].upper()}"
                + (f" ({acc['decode_note']})" if acc["decode_note"] else ""))
    except Exception:  # noqa: BLE001
        pass
    root = Path(series["path"])
    status0 = dict(opts.get("title_cards_status") or {})
    rec0 = status0.get("recurring")
    recurring: dict = dict(rec0) if isinstance(rec0, dict) and not reread else {}
    index = titlecard.TitleIndex(series["episodes"], bool(opts.get("include_specials")), recurring,
                                 titlecard.series_names(series))
    runtimes = sorted(e["runtime"] for e in eps.values() if e.get("runtime") and e["season"] > 0)
    runtime = runtimes[len(runtimes) // 2] if runtimes else None
    # How long one episode runs in these files, from the files named as two or more
    # episodes. TVDB's runtime can be missing, or count a 23-minute pair as one
    # "episode" length; then a single-named file holding two episodes (Rubble & Crew
    # "S02E02" with a second title card at 11:42) never gets its middle searched.
    per_ep = sorted(float(f["duration"]) / len(f["expected"]) for f in files
                    if len(f.get("expected") or []) >= 2 and float(f.get("duration") or 0) > 0)
    if len(per_ep) >= 2:
        own = per_ep[len(per_ep) // 2] / 60
        if runtime is None or own < runtime * 0.75:
            ctx.log(f"Title cards: an episode runs about {own:.1f} min in this series' files (from "
                    f"{len(per_ep)} multi-episode files"
                    + (f"; TVDB says {runtime:.0f}" if runtime else "") + ") — single-named files about "
                    f"twice that long are searched for a second title card too.")
            runtime = own
    rt = {c: e["runtime"] for c, e in eps.items() if e.get("runtime")}
    status = dict(opts.get("title_cards_status") or {})
    if not isinstance(status.get("recurring", {}), dict):
        status.pop("recurring", None)  # earlier format (no position): found again below
    force = 0.0
    if reread:
        # Re-read the videos instead of reusing cached OCR, and redo the auto test
        # and learned position, since both came from the old reads.
        force = time.time()
        if mode == "auto":
            status = {}
        else:
            for k in ("window", "recurring"):
                status.pop(k, None)
        ctx.log("Title cards: re-reading from the video files (cached OCR ignored).")
    scope_all = opts.get("title_cards_scope") == "all"
    trust = opts.get("title_cards_trust", True) is not False
    done: set[str] = set()
    hit_times: list[float] = []

    def run(f: dict, learned, keep_all: bool = False) -> list[dict]:
        searched: list = []
        cards = titlecard.detect(root / f["rel"], float(f.get("duration") or 0), f["segments"],
                                 index, learned, runtime, ctx, force, len(f.get("expected") or []),
                                 searched, f.get("expected") or [], keep_all)
        f["title_scan"] = _merge_windows(searched)
        for c in cards:
            starts = titlecard.episode_starts(float(f.get("duration") or 0), f["segments"], runtime,
                                              len(f.get("expected") or []))
            base = max([x for x in starts if x <= c["time"]] or [0.0])
            hit_times.append(c["time"] - base)
        return cards

    def learn() -> list | None:
        if len(hit_times) < 2:
            return status.get("window")
        ts = sorted(hit_times)
        med = ts[len(ts) // 2]
        return [max(0.0, round(med - 40)), round(med + 60)]

    def report(f, cards, notes):
        full = {c.get("code") for c in cards if not c.get("partial")} | {
            sg["code"] for sg in f.get("segments") or [] if sg.get("title_card")}
        looked = ", ".join(f"{_fmt_t(a)}–{_fmt_t(b)}" for a, b in f.get("title_scan") or [])
        short = len(full) < len(f.get("expected") or [1])
        where = f" (looked at {looked})" if looked and short and not any(
            "looked at" in n for n in notes) else ""
        if cards:
            desc = ", ".join(
                f"'{c['text']}' @{_fmt_t(c['time'])} → " + (c["code"] if not c.get("partial") else
                f"partial read, could be {', '.join((c.get('candidates') or [])[:4])}"
                + ("…" if len(c.get("candidates") or []) > 4 else "")) for c in cards)
            ctx.log(f"  title card {f['rel']}: {desc}{where}" + (f" — {'; '.join(notes)}" if notes else ""))
        elif looked:
            ctx.log(f"  title card {f['rel']}: none read (looked at {looked})")

    # Text from the opening titles can look like an episode title (Bluey's intro names
    # "BINGO" about 20 s into every episode, and "Bingo" is an episode). Check a sample
    # spread across the library: the same title read at the same point in several files
    # whose names say otherwise is that kind of text, and is ignored at that point.
    # Runs every scan; after the first time the sample's videos come from the cache.
    pool_all = sorted((f for f in files if f["status"] != "ERROR" and f.get("expected")),
                      key=lambda f: f["rel"])
    step = max(1, len(pool_all) // 12)
    sample = pool_all[::step][:12]
    if len(sample) >= 3:
        ctx.log(f"Title cards: checking {len(sample)} files across the library for text that "
                "appears in every episode (opening titles)…")
        found_at: dict[str, list] = {}   # code → [(rel, seconds into episode, text)]
        for n, f in enumerate(sample):
            ctx.check_cancel()
            ctx.progress(n / len(sample), f"Title cards (opening titles): {Path(f['rel']).name}")
            dur = float(f.get("duration") or 0)
            starts = titlecard.episode_starts(dur, f["segments"], runtime, len(f["expected"]))
            for c in run(f, None, keep_all=True):
                if not c.get("partial") and c.get("code") and c["code"] not in f["expected"]:
                    base = max([x for x in starts if x <= c["time"]] or [0.0])
                    found_at.setdefault(c["code"], []).append((f["rel"], c["time"] - base, c["text"]))
        new = {}
        for code, hits in found_at.items():
            if code in recurring:
                continue
            ts = sorted(h[1] for h in hits)
            med = ts[len(ts) // 2]
            near = {h[0] for h in hits if abs(h[1] - med) <= titlecard.RECUR_TOL}
            if len(near) >= 3 and len(near) >= len(sample) / 3:
                new[code] = (round(med, 1), len(near), sorted({h[2] for h in hits})[0])
        if new:
            for code, (med, _n, _t) in new.items():
                recurring[code] = med
            index = titlecard.TitleIndex(series["episodes"], bool(opts.get("include_specials")),
                                         recurring, titlecard.series_names(series))
            status.pop("window", None)  # learned from the opening titles' position
            for code, (med, n_files, txt) in sorted(new.items()):
                ctx.log(f"Title cards: '{txt}' shows in {n_files} of {len(sample)} files about "
                        f"{_fmt_t(med)} into the episode — opening titles, not the title of {code}. "
                        f"Ignored there; a {code} title card anywhere else still counts.")
        elif recurring:
            ctx.log("Title cards: opening-titles text already known: " + ", ".join(
                f"{code} at {_fmt_t(t)}" for code, t in sorted(recurring.items())) + ".")
        else:
            ctx.log("Title cards: no opening-titles text found.")
        hit_times.clear()
        if new or status.get("recurring") != recurring:
            status["recurring"] = recurring
            status.pop("recurring_checked", None)
            opts["title_cards_status"] = status
            db.update_series(series["id"], options=opts)

    if mode == "auto" and "has_cards" not in status:
        pool = [f for f in files if f["status"] == "OK"][:6]
        if len(pool) < 4:  # few confirmed files: top up with others so the test means something
            pool += [f for f in files if f["status"] not in ("OK", "ERROR")][:6 - len(pool)]
        ctx.log(f"Title cards: checking {len(pool)} files to see whether this series shows "
                "episode titles on screen…")
        hits = 0
        for n, f in enumerate(pool):
            ctx.check_cancel()
            ctx.progress(n / max(1, len(pool)), f"Title cards (probe): {Path(f['rel']).name}")
            confirmed = {sg["code"] for sg in f["segments"] if sg.get("confidence") == "high"}
            cards = run(f, None)
            # A confirmed file's card must agree with it; for unconfirmed files any
            # readable episode title counts as "this show has title cards".
            if cards and (f["status"] != "OK" or any(c["code"] in confirmed for c in cards)):
                hits += 1
            report(f, cards, _apply_title_cards(f, cards, rt, trust))
            done.add(f["rel"])
        has = bool(pool) and hits >= max(2, (len(pool) + 1) // 2)
        status = {"has_cards": has, "probed": len(pool), "hits": hits,
                  **{k: status[k] for k in ("recurring",) if k in status}}
        if has:
            status["window"] = learn()
        opts["title_cards_status"] = status
        db.update_series(series["id"], options=opts)
        if has:
            w = status.get("window")
            ctx.log(f"Title cards: found on {hits}/{len(pool)} probe files"
                    + (f" (usually {_fmt_t(w[0])}–{_fmt_t(w[1])} into an episode)" if w else "")
                    + " — using them.")
        else:
            ctx.log(f"Title cards: found on {hits}/{len(pool)} probe files — not used for this "
                    "series. (Force them on under Series options.)")
            return
    elif mode == "auto" and not status.get("has_cards"):
        return

    todo = [f for f in files if f["rel"] not in done and f["status"] != "ERROR"
            and (scope_all or f["status"] != "OK")]
    if not todo:
        return
    ctx.log(f"Title cards: checking {len(todo)} files"
            + ("" if scope_all else " the dialogue match didn't already confirm") + ".")
    use_vision = llm.vision_available()
    for n, f in enumerate(todo):
        ctx.check_cancel()
        ctx.progress(n / len(todo), f"Title cards: {Path(f['rel']).name}")
        cards = run(f, learn())
        if use_vision:
            cards = _vision_check(root / f["rel"], f, cards, index, series["name"], ctx)
        report(f, cards, _apply_title_cards(f, cards, rt, trust))
    if titlecard._accel.get("ocr_gpu_error") and not titlecard._accel.get("gpu_error_logged"):
        titlecard._accel["gpu_error_logged"] = True
        ctx.log(f"Title cards: GPU OCR failed, using the CPU instead — {titlecard._accel['ocr_gpu_error']}")
    if mode == "on" or status.get("has_cards"):
        w = learn()
        if w and w != status.get("window"):
            status["window"] = w
            opts = dict(db.get_series(series["id"])["options"] or {})
            opts["title_cards_status"] = status
            db.update_series(series["id"], options=opts)


def _vision_reason(f: dict, cards: list[dict], first_end: float) -> tuple[str | None, dict | None]:
    """Why the first episode's title card should go to the vision model, if it should:
    OCR read no card, only part of one, or a card that contradicts both the filename
    and the dialogue (the show's logo looking like a title…). Returns (reason, card)."""
    first = [c for c in cards if c["time"] < first_end]
    if not first:
        return "OCR read no title card", None
    full = [c for c in first if not c.get("partial")]
    if not full:
        return "OCR only read part of a title", None
    said = set(f.get("expected") or []) | {sg["code"] for sg in f.get("segments") or [] if sg.get("code")}
    if said and not any(c["code"] in said for c in full):
        c = max(full, key=lambda c: c.get("frames", 1))
        return f"OCR's card ({c['code']}) disagrees with both the filename and the dialogue", c
    return None, None


def _vision_check(video: Path, f: dict, cards: list[dict], index, series_name: str, ctx) -> list[dict]:
    """Ask the vision model about the first episode's card when OCR couldn't settle it.
    Its answer replaces OCR's read of that card only when it reads a whole title."""
    full_len = float(get_settings()["titlecard_scan_seconds"])
    reason, doubted = _vision_reason(f, cards, full_len)
    if not reason:
        return cards
    name = Path(f["rel"]).name
    windows = [w for w in f.get("title_scan") or [] if w[0] < full_len]
    frames = titlecard.vision_frames(video, windows)
    if not frames:
        ctx.log(f"  vision: {name} — {reason}, but no frame shows large text to send.")
        return cards
    try:
        text = llm.read_title_card([b for _, b in frames], series_name)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"  vision: {name} — model error: {e}")
        return cards
    r = index.read(text) if text else None
    times = ", ".join(_fmt_t(t) for t, _ in frames)
    if not text:
        ctx.log(f"  vision: {name} — {reason}; the model saw no episode title (frames at {times}).")
        return cards
    if not r or r.get("partial"):
        ctx.log(f"  vision: {name} — {reason}; the model read '{text}', which "
                + ("fits several titles" if r else "matches no episode title") + " — not used.")
        return cards
    if doubted and r["code"] == doubted["code"]:
        ctx.log(f"  vision: {name} — {reason}; the model also reads '{text}' → {r['code']}: keeping it.")
        return cards
    ctx.log(f"  vision: {name} — {reason}; the model read '{text}' → {r['code']}.")
    card = {**r, "text": f"{text} (vision)", "time": frames[0][0], "frames": 1, "vision": True}
    keep = [c for c in cards if c["time"] >= full_len]  # later episodes' cards in the file
    return sorted([card] + keep, key=lambda c: c["time"])


def _decision_pass(series, files, eps, ctx, lex=None):
    """Ask Laya only about sample-scoped files. Its choice stays provisional."""
    todo = [f for f in files if f.get("scope") == "sample"
            and f.get("status") not in ("OK", "MISMATCH") and not f.get("language")]
    if not todo:
        return
    ctx.log(f"Decision model: asking {get_settings()['decision_model']} about {len(todo)} "
            "sampled file(s) the matcher could not settle")
    for n, f in enumerate(todo):
        ctx.check_cancel()
        ctx.progress(n / len(todo), f"Decision: {Path(f['rel']).name}")
        verdict = adjudication.adjudicate(series, f, eps, ctx, lexicon=lex)
        if verdict is not None and ctx:
            ctx.log(f"  decision: {Path(f['rel']).name} → {verdict.get('code') or 'unknown'} "
                    f"({float(verdict.get('confidence') or 0):.0%})")


def _llm_pass(series, files, eps, refs, ctx):
    claimed = {sg["code"] for f in files for sg in f["segments"] if sg["confidence"] == "high"}
    no_ref = [c for c in eps if c not in refs and eps[c]["season"] > 0]
    todo = [f for f in files if f["text_source"] != "none" and
            (f["status"] in ("LOW_CONFIDENCE", "NO_MATCH"))]
    if not todo:
        return
    ctx.log(f"AI fallback: {len(todo)} files the dialogue match couldn't settle "
            f"(LOW_CONFIDENCE / NO_MATCH) → asking {get_settings()['llm_model']}")
    for n, f in enumerate(todo):
        ctx.check_cancel()
        ctx.progress(n / len(todo), f"AI: {Path(f['rel']).name}")
        if _llm_file(series, f, eps, claimed, no_ref, ctx) is False:
            return


def _llm_file(series, f, eps, claimed, no_ref, ctx, force: bool = False):
    """Ask the AI about one file's unsettled segments (force: every segment). Returns
    False if the AI endpoint failed."""
    dlg = get_dialogue(Path(series["path"]) / f["rel"], ctx)
    segs = f["segments"]
    if not segs:
        segs = [{"start": 0, "end": dlg["duration"], "code": None, "score": 0,
                 "margin": 0, "confidence": "none", "alternatives": []}]
        f["segments"] = segs
    for sg in segs:
        if sg["confidence"] == "high" and not force:
            continue
        cands: list[str] = []
        seasons = {c[:3] for c in f["expected"]}
        pool = (list(f["expected"]) + ([sg["code"]] if sg["code"] else []) +
                [a["code"] for a in sg.get("alternatives") or []] +
                [w["code"] for w in f.get("whole") or []] +
                [c for c in no_ref if c[:3] in seasons] + no_ref)
        for c in pool:
            if c in eps and (c not in claimed or c in f["expected"] or c == sg["code"]) and c not in cands:
                cands.append(c)
            if len(cands) >= 12:
                break
        if not cands:
            continue
        text = " ".join(c[2] for c in dlg["cues"] if c[1] > sg["start"] and c[0] < sg["end"])
        try:
            verdict = llm.judge(text, [{"code": c, "title": eps[c]["title"],
                                        "overview": eps[c].get("overview")} for c in cands],
                                series["name"])
        except Exception as e:  # noqa: BLE001
            ctx.log(f"  AI error: {e}")
            return False
        where = f" ({_fmt_t(sg['start'])}–{_fmt_t(sg['end'])})" if len(segs) > 1 else ""
        if verdict:
            sg["llm"] = verdict
            ctx.log(f"  {f['rel']}{where}: AI suggests {verdict['code']} "
                    f"({verdict['confidence']:.0%}) — {verdict['reason']}")
            if force and sg["confidence"] == "high" and verdict["code"] != sg["code"]:
                ctx.log(f"  (the dialogue match {sg['code']} is confident, so the plan keeps it; "
                        "use Set episode if the AI is right)")
        else:
            ctx.log(f"  {f['rel']}{where}: AI had no confident answer")
    return True


@handler("ai_check")
def job_ai_check(ctx, params):
    """Ask the AI about one file now (the Ask AI button)."""
    series = _series(ctx)
    if not llm.available():
        raise RuntimeError("The AI isn't set up (Settings → AI fallback).")
    scan = db.latest_scan(series["id"])
    if not scan:
        raise RuntimeError("Scan the series first.")
    files = [dict(f) for f in scan["results"]["files"]]
    f = next((x for x in files if x["rel"] == params.get("source")), None)
    if not f:
        raise RuntimeError("That file isn't in the last scan — scan again.")
    eps = {ep_code(e["season"], e["episode"]): e for e in series["episodes"]}
    refs = load_refs(series["tvdb_id"])
    claimed = {sg["code"] for g in files if g is not f for sg in g.get("segments") or [] if sg["confidence"] == "high"}
    no_ref = [c for c in eps if c not in refs and eps[c]["season"] > 0]
    f["segments"] = [dict(sg) for sg in f.get("segments") or []]
    ctx.log(f"Asking {get_settings()['llm_model']} about {f['rel']}")
    if _llm_file(series, f, eps, claimed, no_ref, ctx, force=True) is False:
        raise RuntimeError("The AI endpoint didn't answer — see the log.")
    scan_id = db.add_scan(series["id"], {**scan["results"], "files": files})
    old = db.latest_plan(series["id"])
    items = build_plan(db.get_series(series["id"]), db.get_scan(scan_id)["results"])
    key = lambda it: (it["kind"], tuple(it.get("sources") or [it["source"]]),  # noqa: E731
                      tuple(t["path"] for t in it.get("targets") or []))
    prev = {key(it): it for it in (old or {}).get("items") or []}
    for it in items:
        if key(it) in prev and it["source"] != f["rel"]:
            it["selected"] = prev[key(it)].get("selected", it["selected"])
    return {"plan_id": db.add_plan(series["id"], scan_id, items)}


@handler("replan")
def job_replan(ctx, params):
    series = _series(ctx)
    scan = db.latest_scan(series["id"])
    if not scan:
        raise RuntimeError("Scan the series first.")
    items = build_plan(series, scan["results"])
    plan_id = db.add_plan(series["id"], scan["id"], items)
    return {"plan_id": plan_id}


def _plan_after_apply(series: dict, plan: dict, ctx) -> int | None:
    """Carry the scan behind an applied plan forward to the files as they are now —
    renamed files under their new names, split pieces as files of their own, files
    moved to the backup folder gone — and make a new plan from it: what's left
    (unticked changes, reviews) plus the renamed files, now correct. No rescan needed:
    what each file contains was already worked out."""
    scan = db.get_scan(plan["scan_id"]) if plan.get("scan_id") else None
    if not scan:
        return None
    root = Path(series["path"])
    done = {rel: it for it in plan["items"]
            if it.get("selected") and it["kind"] in ("rename", "split", "aside", "join")
            for rel in (it.get("sources") or [it["source"]])}
    joined: set[int] = set()
    opts = dict(series.get("options") or {})
    overrides = dict(opts.get("overrides") or {})
    files = []
    for f in scan["results"]["files"]:
        it = done.get(f["rel"])
        if not it:
            if (root / f["rel"]).exists():
                files.append(f)
            continue
        ov = overrides.get(f["rel"])  # kept under the old name too, for Undo
        if it["kind"] == "aside":
            continue
        if it["kind"] == "join":
            if it["id"] in joined:
                continue  # one file for all the parts, made at the first part
            joined.add(it["id"])
            t = it["targets"][0]
            parts = {g["rel"]: g for g in scan["results"]["files"] if g["rel"] in it["sources"]}
            segs, off = [], 0.0
            for rel in it["sources"]:
                g = parts.get(rel) or {}
                segs += [{**sg, "start": float(sg.get("start", 0)) + off, "end": float(sg.get("end", 0)) + off}
                         for sg in g.get("segments") or [] if sg.get("code") in t["codes"]]
                off += float(g.get("duration") or 0)
            g = {**f, "rel": t["path"], "expected": list(t["codes"]), "duration": round(off, 1),
                 "segments": [{**segs[0], "start": 0.0, "end": round(off, 1)}] if segs else [],
                 "joined_from": it["sources"], "title_cards": []}
            files.append(_reclassified(g, root))
            continue
        if it["kind"] == "rename":
            t = it["targets"][0]
            g = {**f, "rel": t["path"], "expected": list(t.get("codes") or it.get("codes") or []),
                 "renamed_from": f["rel"]}
            if ov:
                overrides[t["path"]] = ov
            files.append(_reclassified(g, root))
            continue
        tg = sorted(it["targets"], key=lambda t: float(t.get("start") or 0))
        cu = it.get("cuts_used") or []
        dur = float(f.get("duration") or 0)
        bounds = [0.0] + [float(x) for x in cu] + [dur] if len(cu) == len(tg) - 1 and dur else None
        for n, t in enumerate(tg):  # split
            if t.get("duplicate"):
                continue
            a, b = (bounds[n], bounds[n + 1]) if bounds else (float(t.get("start") or 0), float(t.get("end") or 0))
            segs = [{**sg, "start": max(0.0, float(sg.get("start", 0)) - a),
                     "end": min(b, float(sg.get("end", 0))) - a}
                    for sg in f.get("segments") or [] if sg.get("code") in (t.get("codes") or [])
                    or (sg.get("llm") or {}).get("code") in (t.get("codes") or [])
                    or sg.get("override") in (t.get("codes") or [])]
            g = {**f, "rel": t["path"], "expected": list(t.get("codes") or []), "segments": segs[:1],
                 "duration": round(b - a, 1) if b > a else f.get("duration"), "split_from": f["rel"],
                 "title_cards": [c for c in f.get("title_cards") or [] if a - 60 <= c.get("time", 0) <= b]}
            if ov:
                overrides[t["path"]] = list(t.get("codes") or [])
            files.append(_reclassified(g, root))
    opts["overrides"] = overrides
    db.update_series(series["id"], options=opts)
    counts: dict[str, int] = {}
    for f in files:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    scan_id = db.add_scan(series["id"], {**scan["results"], "files": files, "counts": counts,
                                         "after_apply": plan["id"]})
    items = build_plan(db.get_series(series["id"]), db.get_scan(scan_id)["results"])
    plan_id = db.add_plan(series["id"], scan_id, items)
    left = sum(1 for i in items if i["kind"] in ("rename", "split", "aside", "join"))
    review = sum(1 for i in items if i["kind"] == "review")
    ctx.log(f"Plan updated: {left} change{'s' if left != 1 else ''} left, {review} to review.")
    return plan_id


def _reclassified(f: dict, root: Path) -> dict:
    f = dict(f)
    try:
        f["size"] = (root / f["rel"]).stat().st_size
    except OSError:
        pass
    f.pop("hold", None)
    f["status"], note = classify(f["expected"], f.get("segments") or [], f.get("text_source") or "none")
    if f.get("renamed_from"):
        f["note"] = f"renamed from {Path(f['renamed_from']).name}"
    elif f.get("split_from"):
        f["note"] = f"split from {Path(f['split_from']).name}"
    elif f.get("joined_from"):
        f["note"] = f"joined from {len(f['joined_from'])} parts"
    else:
        f["note"] = note
    return f


def _plan_after_undo(series: dict, ap: dict, ctx) -> None:
    """After an undo the files are back where the applied plan's scan found them:
    plan from that scan again."""
    plan = db.get_plan(ap["plan_id"]) if ap.get("plan_id") else None
    scan = db.get_scan(plan["scan_id"]) if plan and plan.get("scan_id") else None
    if not scan:
        return
    root = Path(series["path"])
    files = [f for f in scan["results"]["files"] if (root / f["rel"]).exists()]
    scan_id = db.add_scan(series["id"], {**scan["results"], "files": files, "after_undo": ap["id"]})
    items = build_plan(db.get_series(series["id"]), db.get_scan(scan_id)["results"])
    db.add_plan(series["id"], scan_id, items)
    ctx.log("Plan restored to what it was before that apply.")


@handler("apply")
def job_apply(ctx, params):
    series = _series(ctx)
    plan = db.get_plan(params["plan_id"])
    if plan is None or plan["series_id"] != series["id"]:
        raise RuntimeError("Plan not found")
    if plan["status"] != "draft":
        raise RuntimeError("This plan was already applied. Rescan to make a new plan.")
    result = apply_plan(series, plan, ctx)
    autocheck.record(series)
    if result.get("applied"):
        try:
            result["new_plan"] = _plan_after_apply(series, plan, ctx)
        except Exception as e:  # noqa: BLE001 — the files are already moved; just say so
            ctx.log(f"Couldn't update the plan ({e}); scan again to see what's left.")
        if get_settings()["sonarr_rescan_after_apply"]:
            sonarr_rescan(series, ctx)
    return result


@handler("undo")
def job_undo(ctx, params):
    series = _series(ctx)
    ap = db.get_apply(params["apply_id"])
    if ap is None or ap["series_id"] != series["id"]:
        raise RuntimeError("Apply not found")
    if ap["undone_at"]:
        raise RuntimeError("Already undone")
    result = undo_apply(series, ap, ctx)
    autocheck.record(series)
    try:
        _plan_after_undo(series, ap, ctx)
    except Exception as e:  # noqa: BLE001
        ctx.log(f"Couldn't update the plan ({e}); scan again to see what's left.")
    if get_settings()["sonarr_rescan_after_apply"]:
        sonarr_rescan(series, ctx)
    return result
