"""Find a plausible single-episode subtitle for human review, not auto-labeling."""
from .matcher import EP_RE, EP_X_RE, parse_filename_episodes
from .media_text import parse_subtitle_text
from .references import DUPLICATE_REFERENCE_SIMILARITY, _same_series, ep_code, name_check, reference_grams, subtitle_names
from .textsim import BIGRAM_FLOOR, bigrams


def find(client, series, episode):
    minutes = float(episode.get("runtime") or 0)
    if not minutes:
        return None
    code = ep_code(episode["season"], episode["episode"])
    hits = client.search(query=f"{series['name']} {episode['title']}", type="episode",
                         season_number=episode["season"], episode_number=episode["episode"],
                         parent_imdb_id=(series.get("imdb_id") or "").removeprefix("tt"))
    candidates = []
    for hit in hits:
        a = hit.get("attributes") or {}
        files = a.get("files") or []
        names = subtitle_names(hit)
        if (not _same_series(hit, series) or len(files) != 1 or not files[0].get("file_id")
                or a.get("machine_translated") or a.get("ai_translated")):
            continue
        parsed = [parse_filename_episodes(m.group(0)) for name in names
                  for parser in (EP_RE, EP_X_RE) for m in parser.finditer(name)]
        if [code] not in parsed or any(codes and codes != [code] for codes in parsed):
            continue
        checks = [name_check([name], series["episodes"], code, series["name"]) for name in names]
        if any(check[0] == "other" for check in checks):
            continue
        release = next((check[2] for check in checks if check[0] == "match"), "")
        if release:
            candidates.append((a.get("download_count") or 0, files[0]["file_id"], release))
    if not candidates:
        return None
    _, known = reference_grams(series["tvdb_id"])
    client.login()
    seen = set()
    for _, fid, release in sorted(candidates, reverse=True):
        if fid in seen:
            continue
        if len(seen) >= 3:
            break  # bound download quota spent by one search
        seen.add(fid)
        raw = client.download(fid)
        cues = parse_subtitle_text(raw, ".srt")
        if not cues or sum(len(c.text.split()) for c in cues) < 85:
            continue
        start, end = min(c.start for c in cues), max(c.end for c in cues)
        runtime = minutes * 60
        if start <= .22 * runtime and .60 * runtime <= end <= 1.18 * runtime and end - start >= .55 * runtime:
            grams = bigrams(" ".join(c.text for c in cues))
            # Shared text with a larger combined reference is expected. Reject
            # another code's near-identical transcript, not simple containment.
            if len(grams) >= BIGRAM_FLOOR and any(
                other != code and len(grams & g) >= DUPLICATE_REFERENCE_SIMILARITY * max(len(grams), len(g))
                for other, g in known.items()
            ):
                continue
            return {"release": release, "duration": round(end, 1), "text": raw}
    return None
