# Fast follow-ups: close the four review rows

Ponytail plan — every item is the smallest version that works on the
existing machinery. Branch `feat/laya-fast-path`; no new abstractions, no
new deps. Target: The X-Files goes from 207/6/4 to 210/6/1 (only the
Spanish file stays parked unless item 4 lands).

## 1. Escalate to full-file check (biggest, easiest)

The scan job already accepts `{"only": [rel]}` (used by the Sonarr-import
path). Add one gate: `params["full"]` skips sample eligibility, so one file
gets the full-file matcher. UI: a `Full check` button on sample review rows
that POSTs `{"kind":"scan","params":{"only":[source],"full":true}}` through
the existing `/jobs` endpoint and refreshes, same as `Ask AI`.

~5 lines backend, ~10 lines UI. Resolves S09E01/E02 and S09E19-20 via total
overlap without changing scan economics.

## 2. Story-pair reference repair

`repair_reference_conflicts` skips same-story pairs; S09E01/E02 references
score 0.840 — just under the 0.85 conflict line. Include story pairs at a
fixed 0.6 floor (a constant, not a setting). Acceptance is unchanged:
OpenSubtitles replacements only stick when the two new references clearly
differ from each other.

## 3. Language-gate sidecar subtitles

Bazarr ships `*.en.srt` files that are Indonesian. Gate at the single choke
point `sidecar_subtitle()`: parse, run `detect_language`, return `None` when
the content isn't the target language. Every caller inherits the gate. Then
flip `SIDECAR_SUBTITLES=true` on the stack — a validated sidecar is a
free full-episode transcript, so full checks (item 1) need no ASR when one
exists.

## 4. Non-English audio escalation (best-effort)

For a file whose sample windows were language-tagged `X`: transcribe the
full file with `language=X` (salm handles Spanish), fetch the claimed
episode's OpenSubtitles subtitle in `X` (new `reference_in_language()` in
references.py, cached, quota-bounded), and match ASR against that one
reference. High overlap → OK; anything less → stay review with the reason.
On any OS failure, park it — never guess across languages.

## 5. ASR concurrency

The global transcription lock exists for the local Whisper model. The remote
backend is stateless HTTP: skip the lock there. Then prefetch the first
sample windows of all sample-eligible files with `ThreadPoolExecutor(4)`
(cache writes are per-file, ctx=None inside threads). Cold scans drop from
sum-of-files toward sum/4.

## 6. Golden set

`tests/golden_live.py` + `tests/golden_xfiles.json`: the 217 audited labels
(211 correct-by-name, 6 swap targets). GET the live plan, diff, print
PASS/FAIL. Run after every deploy so accuracy deltas are measured, not felt.

## Verification order

1. Unit checks per item (plain assert scripts that actually run).
2. Build `laya-followups-<sha>`, deploy to Portainer stack 190.
3. `fetch_refs` (item 2 repairs story pairs) → scan → escalate the three
   remaining reviews via the new button/endpoint → item 4 on S05E15.
4. `golden_live.py` prints the delta vs. the audit truth.
