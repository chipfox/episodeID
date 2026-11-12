# Summary Match with Auto-Calibration and No-Subtitle Fallback — Implementation Plan

> **For Hermes:** Use the `subagent-driven-development` skill to execute this plan task-by-task, with a spec-compliance review and a code-quality review after each task.

**Goal:** Add a deterministic ASR-vs-episode-summary channel that (a) calibrates itself per series from the existing reference transcripts, and (b) automatically takes over when an episode has no reference transcript (“no subtitles”), without letting a summary alone cause an unsafe rename.

**Architecture:** A per-series `SummaryIndex` (TF-IDF over reference dialogue, query = episode title + cleaned TVDB overview) returns top-3 candidates with score, margin, and matched terms. A calibration pass turns the reference transcripts into a per-series precision/coverage gate. The scan stores summary evidence on each file; the planner treats it as a fallback only when the expected episode has no reference transcript, and only at the calibrated gate. Title cards and the 9B judge remain independent corroborators, not requirements.

**Tech Stack:** Python 3.12, scikit-learn `TfidfVectorizer`/`cosine_similarity` (already a dependency), the existing scan/plan/UI stack (FastAPI + vanilla JS), plain-script tests (`python tests/<name>_test.py`), no new database table.

---

## 1. What the data says

Measured on the live libraries before writing this plan:

| Signal | Beavis | X-Files |
|---|---:|---:|
| Episodes with `overview` | 226/226 | 225/225 |
| Episode reference transcripts | 140 (96 OpenSubtitles, 44 Fandom) | 218 |
| Episodes with no reference | 86 | 7 |
| TF-IDF + title summary retrieval, top-1 / top-3 | **75.0% / 85.0%** | **74.8% / 89.4%** |
| Same, real parakeet ASR (short single-episode files) | **85.7% top-1** (28 files) | not tested |
| TF-IDF + title, score ≥ 0.20 | 35% coverage @ 89.8% correct | 11.5% coverage @ 80% correct |
| BM25 | 25.0% top-1 | 32.1% top-1 |
| IDF term coverage | 17.1% top-1 | 44.5% top-1 |
| Title terms only | 17.1% top-1 | 0.5% top-1 |

Conclusions baked into this plan:

- **Use TF-IDF cosine, not BM25 or raw term coverage.** Summary-length normalisation matters.
- **Use `title + cleaned overview` as the query.** Beavis titles are often spoken; X-Files summaries carry the signal.
- **A summary is a candidate generator, not a certifier.** 75% top-1 is not a rename threshold.
- **Calibrate per series.** Beavis scores average 0.18, X-Files 0.13; a global 0.20 gate was ~100% precise on the Beavis ASR subset but only 80% on X-Files references.
- **Do not use laya.** It scored 7–9/18 vs 14/18 for the 9B, produced zero unique correct answers, and has a 512-token batch ceiling.
- **Strip the `videos:` block from Beavis overviews.** It is a music-video list, not plot.

## 2. What “no subtitles” means here

The app has two independent notions of “subtitles”:

| Concept | Field | Meaning |
|---|---|---|
| Subtitle text in the file | `file["text_source"]` | `embedded #n` / `sidecar name` = subtitle text exists; `remote …` / `whisper …` = ASR was used; `none` = no text |
| Reference transcript for the episode | `refs[code]` | OpenSubtitles/Fandom transcript the matcher compares against |

The **fallback trigger is the missing reference transcript**, because that is the thing whose absence otherwise forces a review. The summary channel runs on whatever the file's dialogue is — ASR, embedded, or sidecar — but a sidecar is never allowed to auto-apply.

| File text | Reference exists | Behavior |
|---|---|---|
| embedded/sidecar | yes | Existing reference matcher decides; summary is optional corroboration |
| embedded | no | **Summary fallback** on the embedded dialogue |
| ASR (no subtitles in file) | yes, matcher settled | Existing matcher decides; summary is optional corroboration |
| ASR (no subtitles in file) | yes, matcher unsettled | Summary is a **suggestion** only; never overrides a reference |
| ASR (no subtitles in file) | no | **Summary fallback** on the ASR transcript — the common case this feature targets |
| sidecar | no | Summary runs, but **review only**; sidecar is third-party text |
| `none` | any | No summary match; review / NO_TEXT |

For multi-episode files, evaluate each code/segment independently: a file can fall back for a part whose reference is missing while another part still uses its reference.

## 3. Calibration design

### 3.1 Build the index

For each series with at least one reference transcript:

1. Fit `TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)` on the **reference transcript texts**, one document per episode code.
2. Transform each query = `episode title + " " + clean_overview(overview)`.
3. Store `codes`, vectorizer, and summary matrix in memory for the scan/job. Do not persist the vectorizer; rebuild from refs (cheap: <0.3 s for 226 episodes).
4. If the series has no references, fit on the summaries instead, mark `calibration.available = False`, and use the conservative default gate (§3.4) for suggestions only.

### 3.2 Clean the summary

```python
_VIDEOS = re.compile(r"(?i)\bvideos?\s*:")

def clean_overview(text: str) -> str:
    """TVDB overview minus the music-video/credits tail; whitespace-normalised."""
    text = _VIDEOS.split(text or "", maxsplit=1)[0]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", text).strip()
```

Beavis S01E01: `"The boys go door-to-door … dominatrix.\nvideos:\nGuns N' Roses…"` becomes just the plot sentence. Do not strip the title.

### 3.3 Score

```python
q = vectorizer.transform([f"{ep['title']} {clean_overview(ep['overview'])}"])
sims = cosine_similarity(q, summary_matrix)[0]
order = sorted(range(len(codes)), key=lambda i: (-sims[i], codes[i]))
top1, top2 = order[0], order[1]
score, margin = float(sims[top1]), float(sims[top1] - sims[top2])
matched = top_idf_terms_in_window(...)  # for UI evidence only
```

For a 400-word ASR window, use the same query and take the max over windows; for a single-episode file, use the whole transcript. Windows are 400 words with 200-word overlap.

### 3.4 Calibrate from reference transcripts

Calibration answers: **at what score and margin was the summary channel right at least `target_precision` of the time on this series' own references?** It produces a suggestion gate at `target_precision` (default 0.95) and records the gate's lower bound; auto-applying a summary **alone** requires that lower bound to reach a stricter `summary_auto_precision` (default 0.99), otherwise a title card or quote-validated LLM verdict must agree.

```python
def calibrate(index, refs, codes, target=0.95, folds=5, min_refs=20, min_gated=8):
    usable = [c for c in codes if c in refs]
    if len(usable) < min_refs:
        return {"available": False, "reason": f"only {len(usable)} references"}
    rows = []
    for code in usable:
        scores = index.score(refs[code])
        true = scores[code]
        other = max(v for c, v in scores.items() if c != code)
        rows.append((true, true - other, max(scores, key=scores.get) == code))
    # Score every reference against all summaries; a gate is accepted only when
    # its Wilson 95% lower bound meets the target. A 5-fold out-of-fold pass is
    # reported as validation, not used to pick the point estimate.
    best = None
    for min_score in quantiles([r[0] for r in rows], 20):
        for min_margin in (0.0, 0.01, 0.02, 0.05, 0.08, 0.12):
            gated = [r for r in rows if r[0] >= min_score and r[1] >= min_margin]
            if len(gated) < min_gated:
                continue
            hit = sum(1 for r in gated if r[2])
            lb = wilson_lower(hit, len(gated), z=1.96)
            cov = len(gated) / len(rows)
            if lb >= target and (best is None or cov > best["coverage"]):
                best = {"min_score": min_score, "min_margin": min_margin,
                        "precision": hit / len(gated), "precision_lb": lb,
                        "coverage": cov, "gated": len(gated)}
    return {"available": best is not None, "target": target,
            "n_refs": len(rows), "refs_fp": refs_fingerprint(refs),
            "summaries_fp": summaries_fingerprint(...), **(best or {})}
```

Rules:

- **Wilson lower bound**, not the point estimate, so 20 references cannot produce a fake 100% gate.
- **5-fold out-of-fold validation** is reported alongside the chosen gate; the gate is accepted only if both the in-sample Wilson lower bound and the pooled out-of-fold lower bound meet the target. If no gate meets it, `available=false`.
- Store the gate in `series.options["summary_calibration"]`, including `at`, `n_refs`, `refs_fp`, `summaries_fp`, `min_score`, `min_margin`, `precision`, `precision_lb`, `coverage`, `gated`.
- **Auto-calibrate** when: (a) `job_scan` starts and the calibration is missing or stale; (b) `job_fetch_refs` finishes and the refs fingerprint changed; (c) `job_refresh_episodes` changes summaries; (d) the user presses “Recalibrate”. Staleness is a hash comparison, not a timer.
- If fewer than 20 references: `available=false`. Use the conservative default (`min_score=0.25`, `min_margin=0.05`) for suggestions only; auto-apply requires a title card or LLM agreement.
- Log one line per calibration: `Summary calibration: 140 refs, score >= 0.20 and margin >= 0.04, 98.1% LB precision at 35% coverage`.

### Implementation note — measured on the live libraries

The planned 0.95 suggestion gate was not reachable on either library. At a 0.80
Wilson lower-bound target:

- Beavis: 90 gated / 140 references, 88.9% precision, 80.7% LB, 64% coverage,
  70.8% out-of-fold LB.
- X-Files: 157 gated / 218 references, 90.4% precision, 84.8% LB, 72% coverage,
  84.8% out-of-fold LB.

The shipped default is therefore a 0.80 suggestion gate. A summary alone still
needs the 0.99 gate (which neither library reaches); the normal automatic path is
a passing suggestion gate plus a title card, and a high-scoring summary that agrees
with the file's existing name may settle an `ok` at 0.25 score / 0.02 margin —
never a rename. The calibration schema is versioned so a stale gate is rebuilt.

### 3.5 Automatic fallback rule

```python
def summary_mode(f, refs, eps) -> str:
    """'auto' = trustworthy fallback, 'suggest' = evidence only, 'off' = don't score."""
    source = str(f.get("text_source") or "")
    expected = [c for c in (f.get("expected") or []) if c in eps]
    if source == "none":
        return "off"                    # nothing to score
    if any(c not in refs for c in expected):
        if source.startswith("sidecar "):
            return "suggest"            # third-party text: never auto
        return "auto"                   # no reference transcript: the fallback
    # A reference exists. If the file has no subtitle track and the reference
    # matcher could not settle it, summary is a second opinion, not a replacement.
    if source.startswith(("remote ", "whisper ")) and \
            f.get("status") in ("LOW_CONFIDENCE", "NO_MATCH", "MISMATCH"):
        return "suggest"
    return "off"
```

Then:

```python
match = index.best(text_or_windows)
f["summary_match"] = {
    "code": match.code, "score": match.score, "margin": match.margin,
    "terms": match.terms, "calibrated": cal["available"],
    "passes": cal["available"] and match.score >= cal["min_score"]
              and match.margin >= cal["min_margin"],
    "auto": cal["available"] and cal.get("precision_lb", 0) >= s["summary_auto_precision"],
    "fallback": True, "mode": mode,
}
```

If the **expected code has no summary** (`expected[0] not in index.codes`), do not
auto-apply; store the suggestion and review. The summary can still identify a
different, summary-bearing episode.

Decision policy:

- `mode == "auto"`, `passes`, and `auto` (calibration lower bound ≥ `summary_auto_precision`) and code is in `eps`:
  - If `match.code == f["expected"][0]` → plan `ok`, `decided_by="summary"`.
  - If `match.code != expected` → plan `rename` to `match.code`, `decided_by="summary"`.
  - If the file is a multi-episode compilation, only auto when every expected code has a reference missing **and** every segment passes; otherwise review.
- `mode == "auto"`, `passes`, but `auto` is false: summary alone is not enough. Auto only if a title card agrees (`decided_by="summary+title"`) or a quote-validated LLM verdict agrees (`decided_by="summary+ai"`); otherwise review with the suggestion.
- `passes` is false → `review` with `suggested` and the score/margin/terms; title cards still run.
- No calibration (no refs / too few) → suggestion only; auto requires a title card or quote-validated LLM verdict agreeing.
- `mode == "suggest"` (sidecar, or ASR with a reference the matcher couldn't settle) → **review only**, even if the score is high.
- Sidecar text → always review, even if the score is high.
- A reference existing always beats summary evidence; summary is logged as corroboration only.

## 4. Fusion and title cards

- Run summary scoring **before** the title-card pass. It is milliseconds and can remove files from the expensive OCR queue:
  - Single-episode file, no reference, summary gate passes → `summary_confirmed`; skip title-card OCR for it.
  - Multi-episode compilation → keep title cards; summary windows corroborate each card and fill gaps where OCR read nothing.
- Summary may not override a strong reference match. If the reference says S03E04 at high confidence and the summary says S03E05 at 0.22, the reference wins and the conflict is noted.
- Optional LLM stage (Phase C): pass only the top 3 summaries and 2–3 excerpts around title cards/best windows; require a verbatim quote and allow `unknown`. It is a verifier, not the primary channel. If LM Studio is down, skip it; the deterministic gate remains.
- Do not use laya anywhere in this pipeline.

## 5. Tasks

### Task 1 — Add `clean_overview` and its test

**Files:** create `app/summary.py`, create `tests/summary_index_test.py`.

**Step 1:** Write the failing test (use the repo's standard test header from
`tests/parts_verify_test.py` — unique `EPID_TEST_DIR`, `MEDIA_ROOT`/`CONFIG_DIR`/
`CACHE_DIR`, and `sys.path` pointing at the checkout):

```python
import os, sys
from pathlib import Path
W = Path(os.environ.get("EPID_TEST_DIR", "/tmp/epid_summary"))
os.environ.update(MEDIA_ROOT=str(W / "media"), CONFIG_DIR=str(W / "config"),
                  CACHE_DIR=str(W / "cache"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.summary import clean_overview

def test_strips_videos_block():
    assert clean_overview("Plot line.\r\nvideos:\r\nBand - Song") == "Plot line."
def test_keeps_plot_and_collapses_space():
    assert clean_overview("A  plot.\n\nMore.") == "A plot. More."
```

Run: `EPID_TEST_DIR=/tmp/epid_summary python tests/summary_index_test.py`
Expected: FAIL — `ModuleNotFoundError`.

**Step 2:** Implement `clean_overview` exactly as §3.2; run the test, expect PASS.

**Step 3:** Commit: `git add app/summary.py tests/summary_index_test.py && git commit -m "feat: clean episode overviews for summary matching"`.

### Task 2 — `SummaryIndex` with TF-IDF and top-k

**Files:** `app/summary.py`, `tests/summary_index_test.py`.

**Step 1:** Add a fixture with 6 summaries and 4 reference transcripts where one summary shares distinctive words with one transcript; assert the correct code is top-1 and the terms include the distinctive word.

**Step 2:** Implement:

```python
@dataclass
class Match:
    code: str
    score: float
    margin: float
    terms: list[str]

class SummaryIndex:
    def __init__(self, episodes: dict, refs: dict[str, str]):
        # Queries cover every episode with a summary, including episodes that have
        # no reference: those are exactly the fallback targets. Only the vectorizer
        # is fitted on reference dialogue.
        self.codes = [c for c, e in episodes.items()
                      if clean_overview(e.get("overview"))]
        docs = [refs[c] for c in self.codes if c in refs] or \
               [self._query(episodes[c]) for c in self.codes]
        self.vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True,
                                          min_df=1)
        self.vectorizer.fit(docs)
        self.matrix = self.vectorizer.transform(
            [self._query(episodes[c]) for c in self.codes])
        self._episodes = episodes

    def _query(self, ep): return f"{ep['title']} {clean_overview(ep['overview'])}"

    def score(self, text: str) -> dict[str, float]:
        q = self.vectorizer.transform([text])
        sims = cosine_similarity(q, self.matrix)[0]
        return {c: float(sims[i]) for i, c in enumerate(self.codes)}

    def best(self, text: str) -> Match:
        scores = self.score(text)
        order = sorted(scores, key=lambda c: (-scores[c], c))
        top = order[0]
        return Match(top, scores[top],
                     scores[top] - scores[order[1]] if len(order) > 1 else 0.0,
                     self._terms(text, top))
```

`_terms` returns up to five highest-IDF query terms that occur in the transcript (for UI evidence).

**Step 3:** Test: correct code top-1; a transcript with no overlap produces low scores; an empty summary is skipped. Run the suite; commit `feat: add TF-IDF summary index`.

### Task 3 — Windowed scoring for compilations

**Files:** `app/summary.py`, `tests/summary_index_test.py`.

**Step 1:** Add `windows(text, size=400, step=200)` and a test that a long transcript with a distinctive plot word only in its second half still yields that term in the best window.
**Step 2:** `best_windows(text)` returns the best `Match` across windows, with the window start for evidence.
**Step 3:** Commit `feat: window summary scoring for multi-episode files`.

### Task 4 — Calibration data and Wilson gate

**Files:** `app/summary.py`, create `tests/summary_calibration_test.py`.

**Step 1:** Failing test: synthetic refs/summaries where the true scores are known; assert `calibrate()` returns `available`, `min_score`/`min_margin`, and `precision_lb >= target`; a series with 3 refs returns `available=False`.
**Step 2:** Implement `wilson_lower`, `refs_fingerprint`, `summaries_fingerprint`, and `calibrate` per §3.4 with 5-fold out-of-fold evaluation.
**Step 3:** Test: changing a reference text changes `refs_fp`; a stale calibration is detected; no gate can meet target → `available=False`.
**Step 4:** Commit `feat: calibrate summary matching from references`.

### Task 5 — Persist calibration and settings

**Files:** `app/config.py`, `app/main.py`, `tests/settings_alias_test.py`.

**Step 1:** Add settings:

```python
"summary_match_enabled": (True, "SUMMARY_MATCH_ENABLED", False),
"summary_min_refs": (20, "SUMMARY_MIN_REFS", False),
"summary_target_precision": (0.95, "SUMMARY_TARGET_PRECISION", False),
"summary_auto_precision": (0.99, "SUMMARY_AUTO_PRECISION", False),
"summary_default_min_score": (0.25, "SUMMARY_DEFAULT_MIN_SCORE", False),
"summary_default_min_margin": (0.05, "SUMMARY_DEFAULT_MIN_MARGIN", False),
```

**Step 2:** Add `summary_match` (auto/off) and `summary_calibration` to the series options PATCH whitelist; do not accept a calibration blob from the browser — it is server-written.
**Step 3:** Test defaults and env aliases; commit `feat: add summary matching settings`.

### Task 6 — Integrate into the scan

**Files:** `app/pipeline.py`, `tests/summary_fallback_test.py`.

**Step 1:** Failing test: a scan fixture with ASR text (`text_source="remote test"`), one expected code, **no reference** for it, a summary that shares distinctive terms, and a calibration whose lower bound meets `summary_auto_precision` → scan entry has `summary_match` with `auto=true`, plan `decided_by="summary"`. A second fixture with `text_source="sidecar …"` → review, no auto. A third with `text_source="none"` → review. A fourth with a 0.95 gate but no title card/LLM → review with a suggestion.
**Step 2:** In `job_scan`, after `refs` load and before the title-card pass:

```python
summary_index = summary.build_index(series, refs) if s["summary_match_enabled"] else None
cal = summary.ensure_calibration(series, summary_index, refs) if summary_index else None
...
for f in files:
    mode = summary.summary_mode(f, refs, eps) if summary_index else "off"
    if mode != "off":
        f["summary_match"] = summary.match_file(f, summary_index, cal, mode)
```

`match_file` uses the full transcript for a single expected code and `best_windows` for a compilation; it never reads or transcribes. For a multi-code file, call `summary_mode` per expected code/segment and store a match per segment; auto only for the segments whose reference is missing, and only when every auto segment passes the gate. A file with one auto segment and one suggestion segment goes to review.

**Step 3:** Auto-calibrate when `cal` is missing/stale; log `Summary calibration: …`.
**Step 4:** Store the gate and match in the scan entry; no new DB table.
**Step 5:** Commit `feat: add summary fallback to scans`.

### Task 7 — Planner evidence and decisions

**Files:** `app/planner.py`, `tools/golden_check.py`, `tests/summary_fallback_test.py`.

**Step 1:** Add `summary_match` to the plan `base` row and a review reason: `Summary suggests S07E05 (0.27, margin 0.06; terms: dolphin, show)`.
**Step 2:** Auto rules from §3.5: `ok`/`rename`, `decided_by="summary"` only when `mode="auto"`, `passes`, and `summary_match["auto"]` is true; `decided_by="summary+title"`/`"summary+ai"` when the title card or quote-validated LLM agrees; otherwise review with `suggested`.
**Step 3:** Golden: treat `decided_by` in `("summary", "summary+title", "summary+ai")` as an assertion only when `summary_match.passes`; otherwise it is a deferral. Add a fixture row.
**Step 4:** Commit `feat: use calibrated summary fallback in plans`.

### Task 8 — Title-card pre-filter

**Files:** `app/pipeline.py`, `app/summary.py`, `tests/summary_fallback_test.py`.

**Step 1:** Test: a single-episode file with `summary_match["auto"]` true is excluded from the title-card queue; a multi-episode file, or one relying on a title card for corroboration, is not.
**Step 2:** In `_titlecard_pass`, skip files with `summary_match["auto"]` and one expected code (or `summary_confirmed`), and log `Title cards: skipped N summary-confirmed files`. Keep title cards for files whose summary is only a suggestion, so they can provide the independent agreement.
**Step 3:** Commit `feat: skip title cards for summary-confirmed files`.

### Task 9 — UI

**Files:** `app/static/app.js`, `app/main.py`, `tests/` (JS syntax only).

**Step 1:** Series options fieldset **Summary match**:
  - Toggle `summary_match` (auto/off).
  - Calibration status: `140 references · score >= 0.20 · margin >= 0.04 · 98% LB precision · 35% coverage` or `Not calibrated (needs 20 references)`.
  - “Recalibrate” button → new job kind `summary_calibrate` or `POST /api/series/<id>/summary/calibrate`.
  - Hint: `Used automatically when an episode has no reference transcript. Sidecar text never auto-applies.`
**Step 2:** Plan row badge `Summary 0.27 S07E05` with tooltip terms and calibration; `decided_by` label `summary` / `summary+title`.
**Step 3:** Evidence dialog: show the episode summary, the transcript window, the top matched terms, score/margin, and whether the gate passed.
**Step 4:** Run `node --check app/static/app.js`; commit `feat: show summary evidence in the GUI`.

### Task 10 — Benchmark tool and acceptance

**Files:** create `tools/summary_bench.py` (adapt the scratch benchmark), `tests/summary_bench_test.py` (fixture-level, no live data).

**Step 1:** `python tools/summary_bench.py --series 1` prints top-1/top-3, MRR, score/margin gates, per-series calibration; `--asr` uses cached dialogue and filename labels.
**Step 2:** Acceptance: on Beavis, the calibrated gate’s out-of-fold precision lower bound ≥ target; on X-Files, top-1 stays ≥70%; no false OK on the golden set.
**Step 3:** Commit `tools: add summary matching benchmark`.

### Task 11 — End-to-end verification and deploy

**Step 1:** Run the focused suites:

```bash
EPID_TEST_DIR=/tmp/epid_summary python tests/summary_index_test.py
EPID_TEST_DIR=/tmp/epid_cal python tests/summary_calibration_test.py
EPID_TEST_DIR=/tmp/epid_fallback python tests/summary_fallback_test.py
EPID_TEST_DIR=/tmp/epid_golden python tests/golden_check_test.py
```

**Step 2:** `ruff check --select F,E9 app/` — same findings as the base revision.
**Step 3:** Build and deploy with the project’s exact-commit image. Compute the
variables locally; `$SHA`/`$BUILT` are expanded before the command reaches the
remote host:

```bash
SHA=$(git rev-parse --short HEAD)
BUILT=$(date -u +%FT%TZ)
git archive --format=tar HEAD | ssh root@10.0.0.151 \
  "rm -rf /tmp/epid-build && mkdir -p /tmp/epid-build && \
   tar -xf - -C /tmp/epid-build && \
   docker build -q --build-arg GIT_SHA=$SHA --build-arg BUILD_DATE=$BUILT \
   -t ghcr.io/ctznsniiips/episodeid:summary-$SHA /tmp/epid-build"
cd /home/hermes/workspace/stacks/episodeid
set -a; . ~/.hermes/.env; set +a
EPID_IMAGE=ghcr.io/ctznsniiips/episodeid:summary-$SHA python3 deploy.py --apply
```

**Step 4:** Verify `/api/status` build commit, the active image, and a targeted scan of a no-reference episode. Require the plan row to show `decided_by="summary"` and the golden gate to stay at 217/217 on X-Files.
**Step 5:** Commit the changelog entry with benchmark numbers and the deployment verification.

## 6. Rollout

- **Phase A (evidence only):** Tasks 1–5, 7 (review evidence), 9–10. Calibration runs and the UI shows it, but summary never changes a status. Deploy and measure coverage/precision on Beavis and X-Files.
- **Phase B (automatic fallback):** Tasks 6, 8 and the auto rules in Task 7. Enable auto only for no-reference, trusted-text files at the calibrated gate. Sidecars and uncalibrated series stay review-only. Re-run the golden gate before enabling by default.
- **Phase C (optional verifier):** top-3 summaries + quote-grounded 9B judge for files that fail the gate; no change to the deterministic fallback. Skip if LM Studio is unavailable.

Each phase is a separate commit/deploy; do not bundle the fallback with the evidence-only phase.

## 7. Acceptance criteria

- Auto-calibration runs on scan/fetch when refs/summaries change; the series options show `at`, `n_refs`, thresholds, precision lower bound, and coverage.
- A no-reference episode with trusted ASR/embedded text and a passing gate gets a summary decision (`ok` or `rename`) without manual review when the calibration lower bound is ≥ `summary_auto_precision`, or when a title card/LLM agrees; a failing gate gets a review with a ranked suggestion.
- Sidecar-only and `none` text never auto-apply.
- A reference transcript always outranks a summary; summary cannot override a high-confidence reference.
- The Beavis title-card queue shrinks by the number of summary-auto-confirmed single-episode files; multi-episode compilations still use title cards.
- Golden gate: X-Files 217/217, zero wrong OK; summary rows are assertions only with `passes=true`.
- No new database table; calibration lives in `series.options`, scan evidence in the scan JSON.
- No `laya` code or settings; no new model dependency.

## 8. Risks and open questions

- **Calibration overfit.** Mitigated by 5-fold out-of-fold evaluation and the Wilson lower bound; never trust a 20-reference point estimate.
- **ASR noise.** The 85.7% ASR result is 28 short single-episode files with filename labels, not compilations. Validate on the full ASR cache and spot-check before auto-renames.
- **Summary quality.** Beavis overviews include music videos; strip them. Generic or missing summaries produce low scores and reviews, not guesses.
- **Score drift per series.** Per-series calibration, never a global threshold; pooled defaults are suggestions only.
- **Sidecars.** Treated as third-party: summary may suggest, but never auto-applies.
- **Compilations.** Whole-file summary matching favors one episode; use windows and let title cards own boundaries.
- **No references at all.** No per-series calibration; use the conservative default and require a title card or LLM corroboration for any auto action.
- **“No subtitles” interpretation.** This plan treats a missing **reference transcript** as the fallback trigger, and a missing **file subtitle track** as the ASR case. If you meant “no embedded/sidecar text” exclusively, the trigger is still correct for the ASR case; the table in §2 covers both.
- **Open decision:** auto-rename on a summary-only fallback at the calibrated gate vs. review-only. The plan enables auto only for no-reference trusted text at the gate; if you want review-only, set `summary_match="off"` or add a `summary_auto` series option in Task 5.
