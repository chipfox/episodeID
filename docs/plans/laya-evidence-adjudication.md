# Laya Evidence Adjudication Implementation Plan

> **For Hermes:** Use the subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Identify the episode in each video quickly and conservatively by using cheap deterministic evidence first, Laya only for ambiguous cases, and OpenSubtitles downloads only as a bounded last resort.

**Architecture:** EpisodeID builds one provenance-aware evidence bundle per file and uses the complete per-series reference index locally: reference transcripts, TVDB titles/overviews, Sonarr descriptions, wiki metadata, filename/release claims, title-card OCR, and cached embeddings. Deterministic lexical/semantic scoring combines those sources and narrows the candidate set; Laya makes one typed tie-break choice among at most 2–4 candidates using only compact, evidence-ranked excerpts. Sidecars remain diagnostic and untrusted, OpenSubtitles remains a bounded last resort, and no Laya result is ever auto-applied.

**Tech Stack:** Existing Python/FastAPI application, `httpx`, scikit-learn TF-IDF, remote salm ASR, LM Studio Decisions API (`POST /v1/decisions`), SQLite/JSON caches, existing OpenSubtitles and Sonarr clients.

---

## Confirmed live Laya contract

LM Studio at `http://10.0.0.11:1234` currently reports:

- model key: `laya`
- type: `decision`
- architecture: `modern-bert`
- quantization: `Q8_0`
- parameters: `421M`
- context: `8192`
- loaded parallelism: `4`

Use the Decisions API, not chat completions:

```json
{
  "model": "laya",
  "input": "compact evidence text",
  "questions": [
    {
      "type": "choice",
      "name": "episode",
      "instructions": "Choose the candidate best supported by the primary audio evidence. Choose unknown when evidence is insufficient.",
      "choices": [
        {"value": "S03E11", "description": "Revelations — religious case involving a stigmatic child."},
        {"value": "S03E12", "description": "War of the Coprophages — deadly cockroach case."},
        {"value": "unknown", "description": "No candidate is sufficiently supported."}
      ]
    }
  ]
}
```

The response contains `choice`, `probabilities`, and `confidence`; it contains no explanation. EpisodeID must construct the reason from recorded evidence. A live probe measured roughly 90–120 ms warm for a compact request, but an obvious keyword case produced only about 0.49 top probability. Therefore probability is a signal, not proof or a standalone threshold.

## Evidence policy

1. **Primary content evidence:** remote ASR from the video audio. A text subtitle stream may substitute only when its language is explicitly English or passes a conservative language-quality gate.
2. **Reference evidence:** manual/uploaded transcript and validated Fandom transcript, then cached reference text. Use excerpts selected by lexical overlap, not entire pages.
3. **Metadata:** episode title, TVDB overview, Sonarr episode description, filename claims, Sonarr release-name claims, and OCR title-card text.
4. **Sidecar subtitle:** always labeled `untrusted`; it may be a Bazarr download for another episode. It can reveal a conflict, but never becomes the primary identity source or an automatic rename proof.
5. **OpenSubtitles:** use search metadata before downloading; download at most one candidate subtitle per file per evidence-cache revision, compare it to ASR and references, and cache the result. A conflict remains `review`.
6. **Independence:** a Bazarr sidecar and an OpenSubtitles release may share the same subtitle lineage. They count as one subtitle source, not two independent votes.

## Fast decision flow

The normal path should be an **identify-first** pass. It must not pretend that a short sample proves every edge case; it should return a fast, reviewable identity result and escalate only when the file has properties that require deep verification.

```text
preload TVDB + Sonarr metadata + cached reference/metadata index once per series
        |
for each file:
  ffprobe duration/audio; parse filename and Sonarr claims
        |
  read sidecar locally only to record a possible conflict (never as primary evidence)
        |
  ASR the first 10 minutes of actual audio (or until the file ends)
        |
  rank candidates with compact deterministic scoring
  (reference transcript first; title+synopsis metadata if no transcript exists)
        |
  send 2–4 candidates + unknown + 3–6 distinctive ASR snippets to Laya
        |
  decisive, single-episode, no conflict ---------------------> FAST SUGGESTION
        |
  multi-episode, long/compound file, weak margin, sidecar conflict,
  non-English/empty sample, or filename disagreement ------------> DEEP PATH
        |
  sample a middle/late section, then full timed matching only if needed
        |
  OpenSubtitles download only after local evidence remains unresolved
        |
  persistent conflict ----------------------------------------> human review
```

**Current Laya runtime constraint:** LM Studio reports `physical_batch_size=512` for the loaded model. Requests containing a full 10-minute transcript plus several synopsis objects currently fail with HTTP 500 (`input ... is too large to process`). Keep the rendered decision packet below that limit by selecting a few distinctive snippets and only 2–4 candidates. Increasing the LM Studio physical batch may permit larger packets, but compact packets are preferable for latency and should remain the application default.

A small live probe found that first-10-minute ASR produced distinctive dialogue for four X-Files episodes in roughly 9–28 seconds after roughly 2.4 seconds of local encoding. That supports the fast path as a useful optimization. A separate compact-packet probe was not consistently correct across several cases, so Laya remains an adjudicator and suggestion source, not proof by itself.

### What the fast path may and may not claim

- It may say **“probable SxxEyy from bounded sample windows”** and put an unticked AI suggestion on the plan. An unresolved first sample is rechecked in additional non-adjacent windows before it is called a review.
- It may not claim that the entire file is verified when the file could contain multiple episodes, swapped parts, or an unexamined conflict.
- It may not auto-apply a rename solely because Laya's probability is high; thresholds must be calibrated against labeled files and deterministic evidence.
- A first 10-minute sample is often sufficient for ordinary single episodes because it contains plot-specific dialogue, but it is not guaranteed to contain a distinctive event: recaps, cold opens, generic procedural dialogue, credits, silence, or serialized setup can make it insufficient.
- TVDB is the canonical episode metadata source for this application. Sonarr descriptions can corroborate it; a wiki synopsis can fill a missing description; MovieDB/IMDb data is an optional fallback, not an additional independent vote. Deduplicate identical descriptions instead of sending all copies to Laya.
- A wiki **transcript** is materially stronger than a wiki synopsis. Use transcript/reference excerpts when available; use synopsis-only packets as candidate context, not as content proof.

Laya is not called for ordinary high-confidence deterministic files. It is called once per unresolved file, not once per segment and not once per candidate.

## Safety invariants (must not regress)

These are gates, not preferences. The existing full-file matcher is the only component that may produce trusted evidence for a rename, split, join, duplicate removal, or “correct” result.

1. **Sample evidence may certify only a clear single-episode match.** Each bounded sample window produces a high/low match vote. A `high` filename vote may produce `ok`; a rename requires the same alternate episode to win at least `DECISION_MISMATCH_WINDOWS` independent windows. Weak, sparse, story-part, non-target-language, or single-window evidence stays `review`. Sample evidence must never authorize a split, join, duplicate removal, or other multi-file action.
2. **The sample cannot claim the whole file.** Every sampled window records its start, end, transcript source, candidate scores, and evidence type; `sample_end` covers all sampled windows. `Matcher.match_file()` must receive each window's sampled duration, not the full file duration. A segment must never inherit `end = file.duration` merely because the file is longer than the sample.
3. **Multi-episode files require the full timeline.** The fast path must not produce `ok`, `rename`, `split`, `join`, or `aside` for a file whose filename claims multiple episodes, whose duration is materially longer than the TVDB runtime, or whose episode count cannot be established from the sample. Escalate to full-file matching.
4. **One episode in the filename is not proof of one episode in the file.** A sample-only `ok` is allowed only when the matcher is `high`, the filename claims exactly one episode, and the duration is consistent with a single episode. Any multi-episode or runtime mismatch escalates to the full-file path.
5. **Candidate cap must not hide the true answer.** Rank the complete reference index locally and union the filename/Sonarr claims before truncating. If the best supported candidate is outside the Laya pool, or no candidate clears the deterministic support floor, return review/`unknown`; never force Laya to choose among four weak candidates.
6. **Missing ASR is not `NO_TEXT`.** If remote or local ASR is unavailable and a validated embedded track or a diagnostic sidecar exists, the app must still identify what it can, marked provisional/untrusted. It must not discard usable evidence merely because the preferred ASR backend is down.
7. **Full-file behavior stays authoritative.** Existing `matcher.match_file()`, Viterbi segmentation, title-card arbitration, reference swaps, duplicate claims, `allow_splits`, Plan → Apply → Undo, and manual overrides must continue to pass their current tests. The fast path is additive; it must not replace those paths for ambiguous or multi-episode files.
8. **No auto-apply from any model.** Laya, Qwen, heuristic sample scoring, and release-name claims remain reviewable suggestions. Only existing trusted full-file evidence may pre-tick a rename.

Required regression tests before enabling the fast path:

```text
tests/sample_safety_test.py
  - high-scoring sample on a misnamed single episode → rename exists but selected=False
  - high-scoring sample that agrees with the filename → ok
  - sample segment end never exceeds sample_end, even when file duration is longer
  - filename S01E01-E02 with a first-10-minute match → review, not a rename that drops E02
  - filename S01E01 but runtime is roughly two episode runtimes → deep path, not OK
  - embedded English subtitle available while ASR is down → provisional identification, not NO_TEXT
  - Laya returns a candidate outside the four-candidate pool / unknown → review, no plan change
```

The first implementation is not complete until these tests pass alongside the existing synthetic scan → plan → apply → undo test.

## Implementation contract for the 10-minute fast path

The first sample is a separate evidence product; it must not overwrite or masquerade as a complete-file transcript.

- Add `DECISION_SAMPLE_SECONDS=600`, `DECISION_SAMPLE_START=0`, `DECISION_SAMPLE_EXPAND=true`, `DECISION_SAMPLE_MAX_WINDOWS=5`, and `DECISION_MISMATCH_WINDOWS=2` settings. Expansion starts on the first window plus non-adjacent windows at +20, +10, +40, and +30 minutes (clamped to the file) until enough evidence accumulates.
- Add a sampler API such as `transcribe(video, ctx=None, start=0, duration=None)` (or a clearly named wrapper) that applies `-ss`/`-t` to both local decoding and remote ffmpeg encoding.
- For remote ASR, translate returned cue times back to file-relative time by adding the sample start offset.
- Include file identity, sample start/duration, ASR backend/model/language/URL, and the evidence schema revision in the sample cache key.
- Keep the existing full-file `get_dialogue()` path available for deep verification and compatibility callers. Never reuse a full-file cache entry as a first-sample entry or vice versa.
- A short file is sampled to its actual duration. A file with no usable speech, an unsupported language, or a compound/multi-episode indication is escalated rather than declared identified.
- The first implementation must work with the current TF-IDF/reference index without requiring a new vector database. LM Studio embeddings may be added as a cached ranking signal behind a setting, but embeddings are not a prerequisite for the fast path.

The evaluation must compare first-10-minute results against labeled ground truth and record: correct suggestion rate, abstention/unknown rate, false-suggestion rate, deep-escalation rate, and p50/p95 time. Do not choose a production confidence threshold from Laya's raw probability alone.

## Internal packet shape

Create an internal, JSON-serializable `EvidenceBundle`; do not pass ad-hoc dictionaries between pipeline stages:

```python
@dataclass
class TextEvidence:
    kind: str                 # asr, embedded, sidecar, opensubtitles, reference
    source: str                # remote salm, embedded #3, Bazarr sidecar, ...
    trust: str                 # primary, conditional, untrusted, reference
    scope: str                 # full | sample
    language: str | None
    text: str
    cues: list[list[float | str]]
    metadata: dict[str, object]

@dataclass
class EpisodeCandidate:
    code: str
    title: str
    tvdb_overview: str
    sonarr_overview: str
    reference_excerpt: str
    scores: dict[str, float | str | bool]

@dataclass
class EvidenceBundle:
    file: dict[str, object]                 # rel path, size, mtime, duration
    claims: dict[str, object]               # filename, Sonarr, release names
    primary: TextEvidence | None
    auxiliary: list[TextEvidence]           # sidecar, embedded alternatives, OS fallback
    title_cards: list[dict]
    candidates: list[EpisodeCandidate]
    conflicts: list[str]
```

The bundle retains all local evidence for the UI/cache, but `decision.py` renders only bounded excerpts. The complete reference index is used locally for scoring and candidate retrieval; the rendered Laya input should normally stay below about 450 tokens: a few distinctive ASR snippets, 2–4 candidate-specific reference excerpts, concise deduplicated metadata, and a clearly marked sidecar excerpt only when it is relevant to a conflict.

Every plan segment must also carry its evidence scope. Sample segments remain labeled `scope: "sample"` and include their window/evidence metadata:

```json
{
  "start": 0,
  "end": 600,
  "code": "S03E11",
  "score": 0.42,
  "confidence": "high",
  "scope": "sample",
  "evidence": "first-10-minute ASR",
  "llm": {"code": "S03E11", "confidence": 0.73}
}
```

`confidence: "high"` is allowed only when the sample matcher itself reached `high`; it may produce `ok` when the filename agrees, and a mismatch is an unticked rename suggestion. `confidence: "provisional"` is mapped to an untrusted review in `planner._segment_choice()`. A full-file segment may use the existing trusted values (`high`, `manual`, `low`, `conflict`, `none`).

---

## Task 1: Add decision-model settings and a native client

**Files:**
- Modify: `app/config.py`
- Create: `app/decision.py`
- Create: `tests/laya_decision_test.py`

Add separate settings; do not reuse `LLM_BASE_URL` because the existing chat and title-card paths require `/chat/completions`:

```text
DECISION_ENABLED=true
DECISION_BASE_URL=http://10.0.0.11:1234/v1
DECISION_MODEL=laya
DECISION_TIMEOUT=30
DECISION_SAMPLE_SECONDS=600
DECISION_SAMPLE_START=0
DECISION_SAMPLE_EXPAND=true
DECISION_SAMPLE_MAX_WINDOWS=5
DECISION_MISMATCH_WINDOWS=2
DECISION_MIN_CONFIDENCE=0.15
DECISION_MAX_CANDIDATES=4
DECISION_MAX_INPUT_CHARS=800  # Laya's 512-token physical batch; keep a margin
DECISION_EMBEDDINGS_ENABLED=false
```

Implement `decision.available()`, `decision.choose(input_text, candidates)`, strict response validation, and a deterministic `reason()` helper. Require choice values to be candidate codes or `unknown`; reject malformed or out-of-pool responses. Preserve the returned probability map and model key for audit logs.

Test with a mocked `httpx.post`: assert the URL ends in `/decisions`, the body uses `choices: [{value, description}]`, malformed responses return `None`, and an out-of-pool choice cannot enter a plan.

Run:

```bash
python3 tests/laya_decision_test.py
```

Expected: `PASS`.

## Task 2: Separate evidence collection from primary dialogue selection

**Files:**
- Create: `app/evidence.py`
- Modify: `app/media_text.py`
- Create: `tests/evidence_packet_test.py`
- Create: `tests/source_selection_test.py`
- Create: `tests/sample_safety_test.py`

Add independent collectors for embedded text, sidecars, cached ASR, and sampled/full remote ASR. Add a cache-aware `sample_dialogue()` path that requests only the configured first 600 seconds and stores it under a sample-specific key; preserve full-file transcription for deep verification. Preserve the existing cache format while adding source metadata where possible.

`sample_dialogue()` must return `sample_start`, `sample_end`, and the actual cue extents. The sample scan path must pass `sample_end - sample_start` as the matcher duration and clip every resulting segment to that window; it must not pass the full file duration to `match_file()`. If sampling or ASR fails, keep the normal validated embedded/sidecar fallback and mark the result provisional rather than returning `NO_TEXT`.

Change the language gate so an untagged/`und` embedded stream is not silently accepted as English. An explicit `eng`/`en` stream may be primary; an untagged stream requires a conservative detector/quality gate, otherwise ASR is used. Sidecars are collected as auxiliary `untrusted` evidence and never short-circuit ASR in the identity path.

Keep `get_dialogue()` as a compatibility wrapper for existing callers, but make the scan pipeline use `collect_evidence()` and `primary_dialogue()` explicitly.

Test:

- explicit English embedded text is accepted;
- `und` Indonesian-like embedded text is not accepted as English primary evidence;
- a sidecar is present in the bundle but marked untrusted;
- changing ASR settings invalidates ASR cache, while adding a sidecar does not invalidate unrelated ASR data;
- a sample cache entry is distinct from a full-file entry and records the sample window;
- a failed ASR request falls back to validated embedded/sidecar evidence as provisional instead of `NO_TEXT`.

Run:

```bash
python3 tests/source_selection_test.py
python3 tests/evidence_packet_test.py
python3 tests/sidecar_subtitles_test.py
python3 tests/remote_transcription_test.py
```

## Task 3: Add candidate retrieval and bounded excerpts

**Files:**
- Modify: `app/matcher.py`
- Modify: `app/references.py`
- Modify: `app/sonarr.py`
- Create: `tests/candidate_packet_test.py`

Add pure helpers that:

1. rank all per-series reference texts using the existing TF-IDF model and, only when `DECISION_EMBEDDINGS_ENABLED=true`, cached embedding similarity;
2. combine audio-to-reference similarity with title/overview similarity, filename/Sonarr claims, title-card codes, and relevant alternatives;
3. retain source-level scores and provenance so duplicate TVDB/Sonarr/wiki metadata is not counted as independent evidence;
4. union the strongest deterministic candidates with explicit filename/Sonarr claims, then cap the Laya pool at 2–4 candidates plus `unknown`;
5. choose a few distinctive ASR windows and candidate-specific reference windows, not whole pages;
6. cap every excerpt by characters/tokens and remove repeated boilerplate;
7. fetch Sonarr episode descriptions once per series and precompute/cache reference embeddings once per reference revision;
8. return an escalation reason instead of a sample candidate when the filename claims multiple episodes, the duration is roughly 1.5× or more the TVDB runtime of a single episode, the filename has no episode code, or the top candidate's support is too weak to justify a Laya choice.

Do not lower `MIN_SCORE` or `MIN_MARGIN` to manufacture candidates. Candidate retrieval is allowed to be broad; final plan confidence remains conservative.

Test that a true filename candidate survives a TF-IDF/embedding top-k cutoff, the final Laya pool is capped at 2–4 plus `unknown`, a true candidate outside the pool causes escalation rather than a forced choice, source provenance is retained without double-counting duplicate metadata, excerpts stay below the packet budget, and generic show boilerplate is not the only reference text sent.

## Task 4: Render and adjudicate one Laya request per unresolved file

**Files:**
- Create: `app/adjudication.py`
- Modify: `app/planner.py`
- Modify: `app/pipeline.py`
- Create: `tests/adjudication_test.py`
- Modify: `tests/sample_safety_test.py`

Implement `adjudicate(bundle, matcher, ctx)` with this order:

1. return the existing full-file deterministic result when it is high-confidence and has no conflict;
2. build candidates and one compact packet;
3. call Laya once;
4. accept the result only as an AI suggestion when the choice is valid and not `unknown`;
5. attach probabilities, `engine: "laya"`, candidate scores, and a generated evidence reason;
6. leave the item unticked for apply, matching the current planner safety rule.

Use the existing `segments[*].llm` shape initially to minimize UI/planner churn, but add `engine`, `probabilities`, `scope: "sample"`, and `evidence_summary` fields. Set `confidence: "provisional"` for every sample-derived segment; do not use `high`. Update `planner._segment_choice()` so `provisional`/`scope == "sample"` is treated as an untrusted suggestion and the resulting item is always `selected: false`, even when the deterministic score is high. Do not ask Laya for free-form reasons. Do not send title-card images; Laya has no vision adapter. Send OCR text only.

If the file is not eligible for the fast path under the safety invariants (multi-episode filename, runtime mismatch, no filename code, weak candidate support, or source conflict), call the existing full-file matcher rather than adjudicating a sample.

The reason should be assembled from facts, for example:

```text
Laya suggests S03E11; ASR/reference score supports S03E11; filename says S03E11; sidecar conflicts and resembles S03E12; OpenSubtitles not consulted.
```

Test that a full-file high-confidence deterministic result makes zero decision requests, an ambiguous file makes exactly one, a malformed Laya result stays review, and an AI suggestion is never selected automatically. Also test the sample-safety cases: a high-scoring sample on a misnamed file produces an unticked rename (or review), sample extents never exceed the sample window, and a multi-episode filename never becomes a single rename from sample evidence.

## Task 5: Add bounded OpenSubtitles fallback

**Files:**
- Create: `app/opensubs_fallback.py`
- Modify: `app/references.py` only for shared parsing/helpers
- Create: `tests/opensubs_fallback_test.py`

The fallback runs only when Laya returns `unknown`/low margin or when trusted ASR, metadata, and sidecar evidence conflict.

Order:

1. reuse a cached fallback result;
2. search by exact file hash/size when supported;
3. otherwise search by series identity, season/episode candidates, and release name;
4. inspect subtitle release/file names before downloading;
5. download at most one best candidate subtitle for this file/evidence revision;
6. compare downloaded text to ASR and candidate references with the existing deterministic matcher;
7. add the result as `opensubtitles-fallback` evidence and keep the file in review if evidence still conflicts.

Reject machine-translated and AI-translated results, preserve subtitle ID/release/language/provenance, and never count a matching Bazarr sidecar plus its originating OpenSubtitles subtitle as independent confirmations.

Test that no OpenSubtitles request occurs for a decisive file, no download occurs for a decisive Laya result, one download occurs for an unresolved file, cached fallback prevents a repeat download, and conflict produces review rather than an automatic rename.

## Task 6: Integrate the pipeline without changing apply safety

**Files:**
- Modify: `app/pipeline.py`
- Modify: `app/main.py`
- Modify: `app/static/app.js`
- Modify: `tests/ai_check_test.py`

Replace `_llm_pass()` with a decision pass that operates on evidence bundles. Keep the existing chat LLM path for optional title-card vision and any future human-requested explanation; do not make Qwen a mandatory second request after every Laya decision.

Make the Ask-AI action use the same adjudication code as the scan, with `force=True` meaning “run the evidence adjudicator even if the normal scan stopped earlier,” not “ignore deterministic safety.” Display:

- primary evidence source;
- candidate and probability distribution;
- sidecar conflict badge;
- whether OpenSubtitles was consulted;
- a concise fact-based reason.

Add `/api/test/decision` and settings status showing that the configured endpoint is a native decision model. Preserve the plan's existing manual `APPLY` gate and AI unticked-by-default behavior.

## Task 7: Add a labeled evaluation harness before enabling automatic use

**Files:**
- Create: `tests/fixtures/decision_cases.json`
- Create: `tools/eval_decision.py`
- Create: `tests/decision_eval_test.py`

Seed fixtures from known cases:

- The X-Files S03E11/S03E12 reciprocal sidecar swap;
- untagged Indonesian embedded subtitles that previously preempted ASR;
- correctly named ordinary episodes;
- wrong filename with strong ASR/reference evidence;
- missing-reference episodes;
- multi-episode files and out-of-order pairs;
- title-card/filename/dialogue conflict;
- sidecar and OpenSubtitles sharing the same lineage.

The offline test uses mocked Laya/OpenSubtitles. The live harness uses the installed `laya` endpoint and reports accuracy, unknown rate, high-confidence precision, fallback rate, and p50/p95 latency. Choose production thresholds from this labeled set; do not assume Laya's probability is calibrated.

A result may be displayed as a high-confidence AI suggestion only when the calibrated threshold and deterministic evidence policy pass. No result is ever pre-selected for apply solely because it exceeds a probability threshold.

## Task 8: Optimize throughput after correctness is verified

**Files:**
- Modify: `app/transcribe.py`
- Modify: `app/pipeline.py`
- Modify: `app/jobs.py`
- Create/modify: `tests/parallel_scan_test.py`

Remove the global transcription lock from the remote backend; retain serialization only for local faster-whisper model access if required. Add a bounded worker pool with a default of four remote ASR requests, matching LM Studio Laya's current parallel setting and the ASR service's available slots. Keep one scan job per series and make cache writes atomic.

Do not run multiple full scans concurrently. Do not parallelize OpenSubtitles downloads without an explicit quota-safe limit. Measure end-to-end throughput, ASR queueing, Laya latency, and memory before changing the default above four.

## Execution units

The numbered sections below are implementation phases, not single subagent-sized tasks. Execute them as these ordered, reviewable units; each unit writes its tests first, runs the focused test command, then commits before the next unit:

1. Add decision settings and the pure Decisions API request/response validator (`app/config.py`, `app/decision.py`).
2. Add mocked client contract tests, including malformed and out-of-pool responses.
3. Add bounded local audio sampling and tests for `start`/`duration` ffmpeg arguments.
4. Add bounded remote audio sampling, cue-offset correction, and sample-specific cache keys.
5. Add evidence dataclasses and source collectors without changing plan behavior.
6. Add the explicit-language/quality gate and regression tests for untagged Indonesian subtitles.
7. Add candidate scoring/provenance aggregation over the existing reference index.
8. Add bounded ASR/reference excerpt selection and the 4-candidate packet renderer.
9. Add the pure adjudicator with mocked Laya responses and safety-state tests.
10. Add the planner trust gate for `provisional`/`scope: "sample"` segments, including the multi-episode and runtime-mismatch escalation tests.
11. Integrate adjudication into scan and Ask-AI while preserving the existing planner/apply flow.
12. Add OpenSubtitles fallback and lineage-aware cache tests.
13. Add API/UI status and decision-test plumbing.
14. Add the labeled offline/live evaluation harness; do not enable production thresholds before its report.
15. Only after correctness review, add bounded remote-ASR concurrency and benchmark it.

For every unit, perform spec review before code-quality review. Do not parallelize units that modify the same pipeline or planner files.

---

## Deployment and rollout

1. Implement and run the offline suite.
2. Run the live evaluation in shadow mode; log Laya decisions without changing plans.
3. Enable `DECISION_ENABLED=true` in review mode through Portainer's stack `Env` API, not by editing `stack.env` alone.
4. Verify the container environment, `/api/status`, `/api/test/decision`, and a known X-Files conflict.
5. Rescan a small selected set; inspect the plan and confirm AI rows remain unticked.
6. Expand to the 61 `NO_MATCH` files only after the known-case evaluation is satisfactory.
7. Keep OpenSubtitles fallback disabled for the first shadow run, then enable it with the one-download-per-file cache limit.

## Acceptance criteria

- Ordinary high-confidence files do not call Laya or download OpenSubtitles.
- Untagged embedded non-English subtitles cannot preempt remote ASR.
- Every Laya request uses `/v1/decisions`, model `laya`, bounded input, and typed choice objects.
- Candidate count is never greater than four episode candidates plus the explicit `unknown` option in a Laya request.
- Sidecars are visible as untrusted evidence and cannot directly authorize a rename.
- OpenSubtitles download is last resort, cached, quota-bounded, and provenance-aware.
- Laya suggestions are reviewable and never automatically selected for Apply.
- The original deterministic matcher, title-card logic, Sonarr release-name logic, Plan → Apply → Undo flow, and existing tests continue to work.
- Sample-derived segments carry `scope: "sample"` and retain their evidence strength. Weak samples remain `provisional`; a high filename vote may produce `ok`, while a high multi-window alternate may produce an unticked rename suggestion.
- No sample segment extends beyond the sampled windows, and no sample-only result produces a `split`, `join`, or `aside` for a multi-episode or runtime-mismatched file.
- A sample-only rename of a misidentified single episode is available for human review but starts unticked.
- When ASR is unavailable, validated embedded/diagnostic sidecar evidence still permits provisional identification instead of forcing `NO_TEXT`.
- A true candidate outside the bounded Laya pool, or an `unknown` result, escalates to review rather than forcing a choice.
- The live evaluation records accuracy and latency before any confidence threshold is treated as production-safe.
