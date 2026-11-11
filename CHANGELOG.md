# Changelog

Tracks this fork's scan, matcher, and reference-quality changes. Format follows Keep a
Changelog; release image tags identify their source commit.

## [Unreleased] — 2026-11-11

### Removed
- **Dead code with no caller anywhere:** `media_text.has_image_subs()` and
  `IMAGE_SUB_CODECS` (nothing ever looked for image-based subtitles), `GET /api/guess`
  (the add-series dialog uses `/api/folders/*`, which share `guess_query`), and the
  `replan` job handler with its allowlist entry and UI label (plan rebuilding already
  happens in `plan_get` for stale versions and in `_plan_after_apply` after an apply).

### Changed
- **Settings whose names outlived their feature are renamed.** `decision_sample_*`,
  `decision_fast_path`, `decision_full_escalate`, `decision_mismatch_windows` and
  `decision_conflict_similarity` are now `sample_*`, `sampling_*` and
  `reference_duplicate_similarity`; `reference_conflict_similarity` is now
  `reference_twin_similarity`, which says what it measures. Old names keep working:
  stored `settings.json` keys are folded onto the new ones on read, old environment
  variables are still honoured (the live stack.env exports `DECISION_*`), the new name
  wins when both are set, and `env_locked_keys()` reports the new name. Pinned by
  `tests/settings_alias_test.py`.
- **`main.py` is about HTTP again.** The series/evidence read models (`series_out`,
  `episode_match`, `ref_note`, `best_dialogue`) moved to `app/views.py`; `main.py` went
  from 1,336 to 1,178 lines.

### Fixed
- **A complete sample was labelled "excerpt".** A scan stores the whole sample window
  now, so the cached sample and the stored text are the same length and the `>` check
  never fired; `best_dialogue` now treats an equal-length cache as the sample (`>=`) and
  calls it `excerpt` only when there is no sample cache at all.
- **A short audio track made a healthy file look damaged.** The integrity check decoded
  video and audio in one ffmpeg pass, and ffmpeg stops at the *shortest* mapped stream:
  a 30 s file with 18 s of audio reported "decoding stopped at 0:18 of 0:30 — the file
  may be cut short" and was classed `bad`. Video and audio are decoded separately now;
  video decides truncation, audio reports its own stop, and the audio/video length
  mismatch falls back to the decoded lengths when the container omits per-stream
  durations. `tests/check_files_test.py` pins the short-audio case as `warn`, not `bad`.
- **Reference repair markers never expired.** `_repairs.json` remembered "already
  repaired" and "don't retry, the replacement was the other part" with no link to the
  reference they described, so a reference that later changed stayed unrepairable for
  good. Each marker now stores a hash of the reference it was made for; markers with a
  different hash (and legacy markers with none) are dropped and the pair may be repaired
  again. Pinned by `tests/repair_marker_test.py`.
- **Two tests were asserting against the homelab, not their fixtures.** `auth_naming`
  now clears `SONARR_URL`/`SONARR_API_KEY` from the environment (the deployment exports
  them, and env beats stored settings, so the test was checking the real key), and the
  `cache_prune` fixture writes files of distinct sizes (the cache key is `size|mtime`,
  and equal sizes written in one clock tick collide on overlayfs).

### Added
- **History retention.** A scan stores ~0.5 MB of per-file results and nothing ever
  trimmed them, so the database grew with every scan (45 MB of scan/plan JSON at 66
  scans). After each scan the newest 20 scans and 20 plans of that series are kept, a
  scan still referenced by a surviving plan is never removed, finished jobs older than
  30 days are deleted (their log lines cascade), and the file is compacted when anything
  was pruned. `tests/db_retention_test.py`.
- **CI runs the tests.** `.github/workflows/tests.yml` installs ffmpeg and
  `requirements.txt` and runs the fast, deterministic suites on pull requests and on
  `main`; the docker workflow only ever built images, which is why three stale tests
  and the short-audio integrity bug went unnoticed.
- **README covers the transcript view and the sampling settings:** containment vs
  cosine, sample vs whole-file scope, "Read whole file", the per-episode match score,
  the `SAMPLE_*` / `SAMPLING_*` / `REFERENCE_*` variables and their `DECISION_*`
  aliases, and the bounded history.
- **"Read whole file" in the transcript dialog** for any sample-scope file, not only
  review rows — a full read is the like-for-like check on a sample verdict.
- **A `sample` chip on plan rows**, so a verdict from the first ten minutes is visible
  without opening the dialog.
- **`ai_check` survives a rename:** if the row's exact path is gone from the last scan
  it uses the same file name when that file is really on disk, and otherwise says the
  file was renamed or replaced instead of "isn't in the last scan".
- **The title-card vision fallback is documented where it is used.** The settings help
  says it is off by default, that OCR always runs first, that it needs a vision-capable
  model (`llm_vision_model`, or the text model if that is blank and can see images),
  and that the model only supplies title text; `llm.vision_available()`,
  `llm.read_title_card()` and `pipeline._vision_check()` carry the same contract.

## [Unreleased] — 2026-11-10

### Changed
- **The transcript view scores containment, not cosine.** "Does this reference contain the
  dialogue we heard" is a containment question: the share of the file's word pairs that
  appear in the reference. Cosine punished a 10-minute sample for being shorter than a
  full reference — on S04E12 the same reference read 36.1% (stored clip), 46.8% (real
  sample) and 88.0% (whole file) — while containment reads the same whether the
  reference is three times longer or ten. Measured over 217 files: the episode's own
  reference contains **0.70–0.94** of the file's word pairs (median 0.84), the closest
  unrelated reference **0.18** (p90 0.23). The score line names the share, the pair
  count and the reference size, colours it (≥60% green, ≥35% amber, below that red) and
  the closest-references list now ranks by containment. `textsim.bigrams()` is the one
  tokenizer, shared with the duplicate-reference guard (whose `overlap()` is the same
  measure read both ways); the evidence view's cosine path is gone — the cosine remains
  only where it answers a different question (reference-conflict detection and
  flagged-reference verification). Reference word pairs are cached per series and
  invalidated by each file's mtime and size (`reference_grams()`), so a repeat open
  costs 70–100 ms instead of ~600 ms.

### Fixed
- **A swapped reference pair can be repaired by fetching again.** The duplicate guard
  refused any fetched transcript that repeated another episode's reference — right for a
  wiki page that is really another episode's text (the Memento Mori case), but with two
  OpenSubtitles references swapped, each *correct* page equals the other reference's
  text, so both pages were refused and the swap could never be fixed by a fetch. Only
  **trusted** references veto a page now; a reference flagged with a numbering conflict
  is the thing the fetch exists to replace. Trusted/external disagreement is unaffected:
  the guard test's S04E14 page is still refused against S04E12's good transcript.
  `tests/reference_swap_test.py` had also been failing for an unrelated reason — it
  reads sidecar `.srt` files only and the deployed stack runs with
  `SIDECAR_SUBTITLES=false`, so it saw `NO_TEXT` and never reached the swap logic; it
  now sets the setting it depends on instead of inheriting the environment.
- **The transcript view compared a clipped excerpt, not the sample a scan matched.** A
  scan stored only `dialogue_text(dlg["cues"])`'s default 1,400-character head+tail clip
  as `sample_text` — 242 words of S04E12's 619-word sample window — and the evidence view
  scored that clip against the whole reference. Measured on S04E12 against the same
  reference: stored clip **36.1%**, the real 10-minute sample **46.8%**, a whole-file read
  **88.0%**. The view now reads the sample window back from the dialogue cache
  (`cached_sample()`, read-only), prefers a cached whole-file transcript above it, and
  falls back to the stored clip only when neither exists (labelled `excerpt`); new scans
  keep the whole sample (`limit=20_000`). The similarity line names both sides and their
  word counts and says that a sample is diluted by length, so 36% and 88% cannot be read
  as the same kind of number.
- **A fetched transcript that repeats another episode's reference is refused.** The
  X-Files wiki's `Memento Mori/Transcript` is the Leonard Betts transcript, and the
  fetch path would happily accept it — on an upgrade it would even *replace* the
  OpenSubtitles reference that fixed S04E14. `refresh_references` now compares each
  fetched page against the other references (`duplicate_reference`) and, on a match,
  keeps what it has, logs which episode the page really is, and marks the episode
  `fandom_tried` so the wiki is not asked again. The measure is **bigram containment**
  (share of the other reference's bigrams present in the page), not cosine: the page
  carries gallery/nav chrome that drops the cosine to 0.745 while still holding every
  bigram. Measured: bad page vs S04E12 **0.911**, next-closest different episode
  **0.232**, genuine two-parters 0.81–0.85; threshold 0.85 with a 400-bigram floor.
  Pinned by `tests/reference_duplicate_guard_test.py`.
- **S04E14's reference was S04E12's transcript.** The X-Files wiki's
  `Memento Mori/Transcript` page (revid 176572) really does contain the Leonard Betts
  transcript — it is not a scraper bug, so re-fetching from the wiki can never fix it.
  Replaced from OpenSubtitles (the upload's own file name carries "S04E14 - Memento
  Mori"): 3,261 words, Penny Northern present, Betts absent, 0.22 similarity to
  S04E12's reference. Both files then settle on dialogue alone — `S04E12 → S04E12`,
  `S04E14 → S04E14` — and the plan goes from 211 to **213 correct, 4 deferred**.
- **Plan rows say what changed and what settled them** (see Added): the scan records
  `previous` on rows whose kind/status moved, so a settled file cannot look as if it had
  always been correct.
- `tools/golden_check.py` counted **every** `kind == "ok"` row as correct, so a
  plan that marked all 211 files OK scored 97% and exited 0 — the one failure the
  gate existed to catch. An OK is now correct only when the row's own high-confidence
  segments name nothing outside its `expected` episodes; a bare or contradicting OK
  is a `wrong_ok` and fails the run. An `ok` row carrying an unsettled status is
  judged as a deferral instead (the plan leaves it alone), which is what the six
  undecidable files need. Verified against the live plan (211 correct, 6 deferred,
  exit 0), a single wrong OK (exit 1) and an all-OK plan (0%, exit 1).
- Evidence-gate promotions now create a high-confidence segment instead of only changing
  the status to `OK`. Without it, the planner treated S03E09, S04E01 and S05E10 as
  review rows despite the gate accepting them. The segment records the file-wide span,
  score and the window/word evidence that justified the promotion (`02a5c5d`).
- The decision model's `confidence` came straight from the endpoint's field of
  that name, but LM Studio's value contradicts its own probability list and
  drifts between identical replays (measured on one packet: 0.09 vs 0.20, with
  the chosen option's probability 0.31 vs 0.40) — so a file the model actually
  favoured at ~40% could be labelled `too weak 9%`. The chosen option's
  probability is now the confidence whenever the probability list covers it.
- The retired model's score badge is no longer part of the current UI; current rows show
  the evidence source and confidence only when those fields affect the decision.

### Added
- **Every episode shows its match score.** The Episodes & references table gains a
  Match column: the score of the segment that points at that episode, its confidence
  (`high` green, anything else amber), `sample` when it came from a sample window, and
  a tooltip with the margin, status and file behind it. Episodes with no file show a
  dash. The score comes from the latest plan — the verdict on screen, including
  promoted evidence, title cards and overrides — and falls back to the last scan's own
  segments for a series with no plan. Clicking a score opens that file's dialogue
  beside that episode's reference, so a number can be checked by reading. The
  transcript dialog's episode picker and reference line carry the same score.
  Pinned by `tests/episode_scores_test.py`.
- **Read the transcript and the reference, side by side, from a plan row.** Every row
  whose file was read gets a **Compare…** button: it opens the dialogue the scan
  actually judged (the sample it matched, or a whole-file transcript when one is
  cached — it never starts a transcription) beside one episode's reference text, with
  a picker for any episode that has a reference. Under the reference: the whole-text
  similarity, the words both texts use (names and places settle a match faster than a
  score does), and the six closest references as one-click buttons — so a doubtful row
  can be settled by reading instead of guessing. The **Episodes & references** table
  gets a **View** button per episode for the reference on its own. Served by
  `GET /api/series/<id>/evidence`; pinned by `tests/evidence_view_test.py`.
- **Plan rows say what settled them and what changed.** A scan that settles a row used to
  leave no trace: an `ok` row looked identical whether it had always been right or a
  full-file match had just decided it. Each row now carries `decided_by` and
  `decided_confidence` (`dialogue`, `AI`, `title card`, `release name`, `you`), the scan
  records the previous verdict on rows whose kind or status changed (`previous`, a
  `changed` chip, and a `Changed: …` line in the job log), and the kind badge reads
  `correct` instead of `ok`. A row the AI settled shows `AI 95%` next to it.
- **Audio longer than the upload cap is transcribed in parts and joined.** The remote
  ASR refuses any single request over `whisper_max_upload_seconds`, so the 87-minute
  joined two-parter (S09E19-20) was refused before encoding on every scan and stayed an
  `ERROR` row whose audio was never read. `transcribe()` now splits such a span into
  equal consecutive parts, transcribes each with its file-relative offset, and returns
  the cues joined in order; a span that exactly fits still goes up as one request.
  Measured on the real file: **2 uploads of 2,606 s, 326 cues, 74 s**, first cue at
  440 s and last at 5,137 s of 5,211 s.
- **Evidence gate: a weak absolute score no longer means "no idea"**
  (`evidence_gate`, default on; `gate_min_window_share` 0.30, `gate_min_windows` 4,
  `gate_min_words` 60). `min_score` is calibrated on whole transcripts, so a
  6.6-minute sample of the right episode can sit under it while most 90-second
  windows still vote for that episode: on the X-Files library 7 of the 11
  unsettled files had the correct episode at top-1 and were deferred anyway.
  A deferral is now accepted as `OK` only when the whole transcript's best
  episode *is* the file's single episode, at least 30% of the usable windows
  rank it first, at least four windows carry real dialogue and the file holds at
  least 60 words of it, no surviving segment names another episode, and the
  winning reference has no near-duplicate twin. Every other case keeps its
  deferral. Measured on the 217-file golden corpus: 206 → 209 correct decisions,
  false assertions 1 → 0.
- **Near-identical references can no longer produce a confident match**
  (`reference_conflict_similarity` 0.70). References are compared through the
  shared document view (`app/textsim.py`), not the matcher's phrase vectorizer:
  when the runner-up's reference is nearly the same text — a wiki transcript
  filed under two numbers (S04E12/S04E14 at 0.998), or both parts of a two-parter
  (S09E01/S09E02 at 0.840) — the segment is marked `low` with that reason stated,
  because a match on one of them cannot prove which was read.
- `tests/matcher_evidence_test.py` drives both behaviours through the real
  `match_file` → `classify` path: acceptance on broad thin evidence, refusal by
  each floor, the kill switch, the contradiction check, and the duplicate
  reference.
- **Automatic full re-read after a weak sample** (`decision_full_escalate`,
  default on). Files still `LOW_CONFIDENCE` / `NO_MATCH` after the sample
  windows and the decision model's first look are no longer parked for a human
  to press *Full-file match* on: the scan reads the whole file once (cached like
  any full read), re-matches every line, and asks the decision model again on
  the complete transcript. Whatever the full read still cannot settle goes to
  the AI fallback. Files whose reference is a near-duplicate of another
  episode's are skipped — no transcript length settles that pair — and a full
  read that turns up foreign-language dialogue flags the file instead.
  New `tests/full_escalation_test.py` drives the pass end to end.
- **Decision-model score on every plan row** (`decision_score_every_file`,
  default on). The decision model is now asked about every file with dialogue,
  not only the ones the matcher left unsettled, so each GUI row can show its
  pick and confidence — including `OK` rows, where the answer is an annotation
  that never changes the file's status.
- `adjudication.adjudicate()` gained `text=` (ask on a caller-supplied
  transcript instead of the file's own excerpt) and `quiet=` (no per-file log
  line, for the annotation-only scoring pass).

### Removed
- **The auxiliary decision model is out of the scan.** Measured against the 9B judge
  the app already runs on the same LM Studio server: 8–9 of 18 files correct
  versus 14, **zero unique correct answers** across the 217-file library (its one
  win, S09E01, the 9B already calls at 1.0), 126 disagreements with answers the
  matcher had right, and a confidence that does not separate right from wrong.
  Deleted: `app/decision.py`, `app/adjudication.py`, the decision pass, the plan's
  result field and the GUI badge, and nine `decision_*` settings with their
  `DECISION_*` env vars. Kept: the sample fast path (`decision_fast_path`,
  `decision_sample_*`, `decision_mismatch_windows`), the full re-read
  (`decision_full_escalate`), and `decision_conflict_similarity` for the reference
  check — the `decision_` prefix there is historical.
- `evidence.py` is now only the reference-conflict map (`ReferenceConflicts`,
  formerly `Lexicon`): the per-episode weights, descriptors, reference lines and
  packet selection existed for the decision model alone.
- `dialogue_text()` moved to `media_text` — the scan still stores a row's
  `sample_text` with it.

### Changed
- `Matcher.match_file()` returns an `evidence` block (`windows`, `words`,
  `rank_share`, `top1`, `top1_score`, `conflicted`, `twin`) — the window votes the
  segment gates used to discard. `classify()` takes it as an optional fourth
  argument, and every re-classification path passes the stored block, so a
  reloaded plan reaches the same status as a fresh scan. Scan entries and plan
  rows carry it; `PLAN_VERSION` 8 → 9 so stored draft plans rebuild.
- **The decision packet now carries the sample, not a handful of lines.** Two
  measured defects: `select_evidence()` kept at most four sentences and filled
  267 of the 1,403 characters available, and candidates were described by
  95-character truncated TVDB synopses while the file's text is dialogue. The
  selector now backfills to its budget (distinctive lines first, then the rest of
  the sample), candidates are described by their own reference lines with the
  synopsis as fallback (`Lexicon.reference_lines`), and the budget default rose
  800 → 1100. On the 18-file sample this moved the auxiliary model from 6–7 correct to 8–9 with
  refusals 8–9 → 6–7.
- **The budget is a token limit in disguise.** The deployed decision server
  rejects any input over its *physical batch* (512 tokens — `input (539 tokens)
  is too large to process`), so ~1100 characters of dialogue plus four candidate
  descriptions is the ceiling; 1,403 characters was measured at 11–15/18 while a
  larger batch was in effect. Raising `decision_max_input_chars` without raising
  the server's batch turns every call into a 500 and loses the annotation.
- `Lexicon` descriptors are built lazily now: with reference lines as the primary
  description, precomputing all 225 synopsis descriptors per scan was dead work.
- `select_evidence()` returns the sample instead of `""` when nothing in it is
  distinctive; the old empty string only existed to fall back to the raw sample,
  which the caller now receives directly (short line deleted).
- Every plan row once carried a model-result field (`{code, confidence, model}`) when
  the auxiliary model answered; that field and badge were retired with the feature.
  `PLAN_VERSION` was bumped so stored draft plans rebuilt from their scans.
- Scan logs gained the `Full re-read:` and `Decision model: scored …` lines;
  escalated rows say so in their note (`Full re-read after a weak sample — …`).
- Both new settings ship GUI-editable and are deliberately **not** in
  `stack.env` (env-provided keys lock the selector).

## [b96b211] — deployed 2026-11-09/10

### Changed
- Remote transcripts overlap instead of queueing behind the model lock
  (`b96b211`): 8 cold files 148 s → 72 s (2.06×); warm scans ~28 s.
- Deployment config: `WHISPER_REMOTE_MODEL` pinned to
  `parakeet-tdt-0.6b-v3.oss-align.q8_0` after the chip-pc `:8090` ASR switched
  its default from the 30B `salm` model (parakeet measured 164× realtime vs
  24.7× on the same clip; `salm` kept as rollback on the ASR host). The
  advertised ASR id follows the GGUF filename, so it changed with the model —
  consumers must re-read `GET /v1/models` after any model swap.
- Full-file matching stays a per-file operation on purpose, and inside an
  overlapping story pair (X-Files `S09E01/S09E02` at 0.84) the text winner
  flips with the transcript source — only the review row plus AI confirmation
  is a trustworthy signal there; the score alone is not.

## [earlier scan work] — 2026-11-07…09

- Fast path: 10-minute first sample, adaptive non-adjacent window expansion
  (up to 5 windows), multi-window voting before any rename is proposed.
- One language setting for dialogue, subtitles and metadata; foreign-language
  tracks and audio are detected, flagged and never matched blindly.
- Evidence selection for the decision model, reference-conflict quarantine,
  story-pair explanation, title-card OCR passes, elimination, joins, and the
  X-Files golden set (`tools/golden_check.py`) as the scanner regression gate.
