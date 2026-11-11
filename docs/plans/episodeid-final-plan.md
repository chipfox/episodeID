# episodeID — final improvement plan (2026-11-10)

Supersedes the earlier model-adjudication draft for the parts it covers. Every
number here was measured on the live X-Files library (217 files, 218 references,
213 with a stored model verdict) unless stated otherwise.

## Where the system stands

| component | status |
|---|---|
| deterministic matcher | 209/217 correct, 0 false assertions (evidence gate + conflict guard, `8c80d52`) |
| decision packet | fixed (`eb7f502`): backfills to budget, describes candidates by reference dialogue |
| auxiliary decision model (421M) | dominated; 8–9/18 on the 18-file sample, 0 unique correct answers in 217 files, hard 512-token batch ceiling |
| 9B judge | 14/18 on the same packets; already runs on deferred files; the GUI already renders its verdict (`sg.llm`) |
| golden gate | **cannot fail on a wrong OK** — `swaps` is empty and it counts every `kind == "ok"` as correct |

Decisions taken, with evidence:

- **The auxiliary decision model is removed from the decision path.** It offers one thing nothing else does — a
  score on rows the matcher already settled — and that badge is informative 40% of the
  time, refuses 52%, and is wrong 8%. Its one unique win (S09E01) is already produced by
  the 9B at confidence 1.0.
- **No fusion.** With the truth absent from the candidate list the 9B certified a wrong
  candidate at ≥0.8 confidence in 5 of 10 decoy trials; and the only file fusion could
  recover is an intentionally undecidable story pair (0.84 reference overlap).
- **No verifier framing, no forced choice.** Measured worse: sensitivity 0.33, and forced
  choice produced 9 wrong answers instead of 3.
- **No further packet growth.** LM Studio rejects inputs over its physical batch
  (512 tokens); 1,100 characters is the ceiling with four candidates.

## Phase 1 — remove auxiliary model (done: `95fbb29`)

**Deployed in `02a5c5d`**. `/api/status` reports that commit; retired model env vars are
absent from Portainer's stored env, the served bundle has no retired-model badge, and LM Studio
shows only the 9B judge loaded (the auxiliary model was unloaded, freeing 449 MB). The 9B AI annotation on
unsettled files is unchanged.

### Post-deploy acceptance (scan 75 / plan 62)

- Scan: **OK 211, NO_MATCH 4, LOW_CONFIDENCE 1, ERROR 1** (217 files).
- Hardened golden gate: **211 correct, 6 correctly deferred, 0 wrong OK**, exit 0.
- The three evidence-gate promotions now have actual high-confidence segments in the plan:
  S03E09 (11 windows / 174 words / 0.55 share), S04E01 (11 / 83 / 0.73), and S05E10
  (5 / 169 / 0.60). Without this segment materialization, the status was OK but the
  planner emitted a review row; `matcher.promote()` fixed that before this scan.
- Remaining six deferrals: S04E12/E14 (duplicate reference), S05E15 (Spanish),
  S09E01/E02 (overlapping two-parter), S09E19-20 (ASR upload limit). All are listed in
  the golden set as undecidable.
- auxiliary model unload verified: `lms ps` shows only `deepseek-v4-pro-qwen3.5-9b-uncensored-heretic-mtp`.

## Phase 2 — make the golden gate able to fail (done: `1e2b573`)

1. Delete `app/decision.py` and `app/adjudication.py`.
2. Move `dialogue_text()` to `app/media_text.py` (the scan still builds `sample_text`
   with it) and update its three call sites.
3. `app/pipeline.py`: delete `_decision_pass()`, the adjudicate re-ask inside
   `_full_escalation_pass()`, the `decision.available()` gates (≈ lines 683, 779) and the
   `adjudication`/`decision` imports. Keep the full re-read: escalation still re-matches
   the whole transcript and then hands the remainder to the AI pass.
4. `app/planner.py`: drop the retired model-result field and the `ai_code` branches in the
   sample-review reason. `_segment_choice()`'s `ai` strength comes from `seg["llm"]` and
   is untouched. `PLAN_VERSION` 9 → 10 so stored draft plans rebuild without the field.
5. `app/static/app.js`: remove the retired model's badge renderer and its render hook. Keep the existing
   `AI → S09E01 95%` badge (line ~543).
6. `app/main.py`: drop the `"decision"` / `"decision_evidence"` capability flags.
7. `app/config.py`: delete `decision_enabled`, `decision_base_url`, `decision_model`,
   `decision_timeout`, `decision_min_confidence`, `decision_max_candidates`,
   `decision_max_input_chars`, `decision_evidence`, `decision_score_every_file`.
   **Keep** `decision_fast_path`, `decision_sample_*`, `decision_mismatch_windows`,
   `decision_full_escalate` (they drive sampling, not auxiliary model) and
   `decision_conflict_similarity` (feeds `Lexicon`). *These were later renamed to
   `sampling_*`, `sample_*` and `reference_duplicate_similarity`; the old names still
   work as settings/env aliases.*
8. `app/evidence.py`: delete `_build_weights`, `weights`, `terms`, `_describe`,
   `descriptor`, `reference_lines`, `select_evidence` and their constants. Keep the
   conflict and story-pair half of `Lexicon` (`conflicting`, `story_pair_for`,
   `conflict_pairs`, `story_pair_conflicts`, `_find_conflicts`).
9. Tests: delete `adjudication_test.py` and `decision_test.py`; reduce `evidence_test.py`
   to the conflict/story-pair and aggregate assertions; update `full_escalation_test.py`
   and `sample_confirmation_test.py` to drop the auxiliary model stubs.
10. Deployment config: remove the nine `DECISION_*` variables from Portainer stack 190's
    env — including the `DECISION_MAX_INPUT_CHARS=800` pin that currently overrides the
    code default and locks the GUI selector.

Acceptance:

- `grep -rn "adjudication\|decision\." app/ | grep -v decision_fast_path` finds nothing.
- Full battery green except the two known environment failures (`auth_naming`,
  `cache_prune`); ruff findings identical to the branch base.
- Golden check no worse than today (expect ~216/217 after the gate).
- A scan log contains no `Decision model:` lines and still contains `AI fallback:` for
  files it cannot settle; the GUI has no retired-model badge and still shows AI badges.

Rollback: revert the commit. auxiliary model stays loaded on chip-pc until it is explicitly
unloaded; nothing about the model has to change for the rollback to work.

## Phase 2 — make the golden gate able to fail (done: 1e2b573)

The acceptance gate for everything else is currently blind: `tests/golden/xfiles.json`
has no per-file truth codes and `swaps` is empty, so `golden_check.py` counts any
`kind == "ok"` row as correct. A plan that marked all 211 files OK would pass.

1. Add a `codes` map (series-relative source → expected episode code(s)) built from the
   verified filenames.
2. In `score()`, for `kind == "ok"` require the row's high segment codes to equal the
   truth; count mismatches in `wrong_ok` and make them fail (exit 1).
3. Re-run against the live plan; expected 213–216 correct, zero `wrong_ok`.

Acceptance: a deliberately corrupted plan (one OK row pointed at the wrong episode)
fails the check; the live plan passes.

## Phase 3 — reference hygiene: investigated, no code change (2026-11-10)

The plan proposed quarantine, title verification and a per-part repair. Measured, the
first two have nothing to act on and the third is a source-data dead end:

- **Title verification: 0 findings.** Of the 218 references, 31 carry both an
  `os_title` and a TVDB title, and **none disagree** — the fetch path already
  verifies titles when it stores them. A load-time check would be a second
  implementation of a guarantee that already holds.
- **The duplicate pair is a wiki defect, not an app one.** S04E12/S04E14 are both
  `fandom` references at 0.998 similarity: one wiki transcript filed under two
  titles. `repair_reference_conflicts` has already tried OpenSubtitles for the pair
  and refused, because the replacements matched each other too (recorded in
  `_repairs.json`, net replacements 0 over four days).
- **What already works.** The pair is detected (`ReferenceConflicts`), the scan logs
  it, the plan row says one text cannot settle either episode, the matcher refuses to
  treat a match on one as high, and the full re-read skips it.

So the honest outcome is the one the golden set already encodes: those files stay
deferred with a stated reason. The actionable fix is manual — drop a distinct
per-episode transcript into the series' manual references directory (load_refs
prefers manual files), and the pair resolves on the next scan. No code is added for
a one-off data defect.

## Phase 4 — deploy and verify (deployed 2026-11-10)

- Portainer stack 190 updated via `deploy.py --apply` to the image built from
  commit `02a5c5d`; `/api/status` reported `build.commit: 02a5c5d` and `source: ci`. The running container was healthy, and
  SHA-256 prefixes for all nine changed application modules match the commit.
- The nine auxiliary model-only settings were removed from stored stack env by the earlier
  removal deploy. Seven `DECISION_*` variables intentionally remain: sampling,
  multi-window voting, full escalation and the reference-conflict threshold; the
  `decision_` prefix on these is historical. `MEDIA_LANGUAGE` remains unset.
- Full scan 75 completed: `OK 211, NO_MATCH 4, LOW_CONFIDENCE 1, ERROR 1`.
  Golden check: **211 correct, 6 correctly deferred, 0 wrong OK**, exit 0.
  S03E09, S04E01, S05E10 now have one high-confidence promoted segment each,
  with the evidence reason attached. Remaining deferrals are exactly the six
  listed by the golden set.
- auxiliary model unloaded by exact model identifier. Verified `lms ps`: only
  `deepseek-v4-pro-qwen3.5-9b-uncensored-heretic-mtp` remains loaded.
- Changed-module suites pass; ruff findings match the base. **Full sequential battery:
  52 of 56 pass, and all four failures (`auth_naming`, `cache_prune`, `check_files`,
  `reference_swap`) reproduce at the base commit `eb7f502`** — parity, not
  regression. The slow title-card suites (`titlecard_test` among them) pass uncapped.

## Follow-up — over-length audio is split instead of refused (`c6ccfe1`)

The last `ERROR` row existed only because the remote ASR refuses one request over
`whisper_max_upload_seconds`. `transcribe()` now splits such a span into equal parts,
transcribes each with its file-relative offset and joins the cues.

- Live proof before deploying: the 87-minute S09E19-20 file → **2 uploads of 2,606 s,
  326 cues, 74 s**, first cue 440 s, last 5,137 s of 5,211 s.
- After deploying, a scan of just that file: **ERROR → LOW_CONFIDENCE**, and the plan now
  shows two segments — `S09E19 0.535` and `S09E20 0.720` — each low solely because
  "references for S09E19 and S09E20 are near-identical — neither can prove which of them
  was read". The part cannot be picked from text alone (references overlap at 0.69), but
  the audio is read and the file's two parts are visible instead of invisible.
- Plan counts are now **OK 211, NO_MATCH 4, LOW_CONFIDENCE 2, ERROR 0**; the golden gate
  is unchanged at 211 correct / 6 correctly deferred / 0 wrong OK, exit 0.

## Follow-up — S04E12/S04E14 was a bad wiki page, not a matcher tie (`5bd95ca`)

S04E12 (Leonard Betts) sat at `LOW_CONFIDENCE` with a "review" row and a manual
full-file match that the UI could not explain. Three separate facts:

- The full read's own evidence: 53 windows / 3,239 words, top1 `S04E14 0.6932`,
  `conflicted: true, twin: S04E12` — a tie, not doubt.
- The file's own transcript says **"Leonard Betts" 31 times and "Memento" 0**.
- The X-Files wiki's `Memento Mori/Transcript` page (revid 176572) **is the Leonard
  Betts transcript** — 1.000 similarity to `Leonard Betts/Transcript`, 101/352 long
  lines verbatim, Penny Northern absent. It is the wiki's error, so re-fetching from
  the wiki cannot fix it; the pair was blocked in `_repairs.json` from 2026-11-07.

Fixed by replacing S04E14's reference from OpenSubtitles (its upload carries the
episode title; 3,261 words, Penny Northern present, Betts absent, 0.217 similarity
to S04E12). A scan of the two files then settles both **on dialogue alone**:
`S04E12 → S04E12` (0.316) and `S04E14 → S04E14` (0.360), no AI needed. Plan counts
**OK 213, NO_MATCH 2, LOW_CONFIDENCE 2 (ERROR 0)**; golden **213 correct, 4 correctly
deferred, 0 wrong OK**, exit 0. Backup of the bad reference:
`/config/S04E14.json.bad-fandom-*`.

Why the automatic full re-read skipped it: `_full_escalation_pass` only takes
sample-scope `LOW_CONFIDENCE`/`NO_MATCH` files **without** a `reference_conflict` —
no transcript length settles a pair whose references are the same text. The manual
"Full-file match" button sends `{only: [file], full: true}`, which bypasses the skip.

## Not doing (and why)

| rejected | evidence |
|---|---|
| auxiliary model corroboration fusion | 9B false-accepts a wrong candidate 5/10 at ≥0.8 when the truth is absent; the only recoverable file is an undecidable story pair |
| verification framing / forced choice | sensitivity 0.33; forced choice 9 wrong vs 3 |
| larger decision packets | server physical batch is 512 tokens |
| trigrams, stemming, ASR speller, BM25 | neutral or worse on 217 files (BM25 ranked identically) |
| title/overview augmentation | zero change; X-Files has no title cards |
| calibration job | no confirmed labels yet; needs a second series first |

## Deferred, not rejected

- **Calibration from confirmed renames** once a second series exists: refit
  `gate_min_window_share`, `gate_min_words` and the conflict threshold per series.
- **Passage-level matcher scoring**: it rescued two thin-sample files but lost three to
  the 1.4× ratio gate; re-test only after recalibration.
- **auxiliary model**: reconsider only if a decision server with a larger physical batch appears.
