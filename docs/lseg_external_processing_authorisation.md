# LSEG hosted-inference authorisation record

On 2026-08-04, the researcher explicitly confirmed that the applicable LSEG
licence permits sending the collected headline text to a third-party hosted
model for sentiment inference.

This record captures the researcher's authorisation; it is not an independent
legal opinion. Its scope is limited to the frozen 888,155-headline expanded LSEG
corpus and the following route:

- OpenRouter API;
- `google/gemma-4-26b-a4b-it`;
- DeepInfra only, FP8 only;
- zero-data-retention required;
- provider data collection denied;
- provider fallback disabled.

The permission does not change redistribution rules. Raw headlines, per-row
responses, and checkpoints remain on approved project storage, outside Git, and
must not be shared. Only aggregate, licence-safe results may be promoted.

## Execution record

The private 10-headline gate completed on 2026-08-04 with 10/10 successful
responses, all attributed to DeepInfra and recorded under the FP8-only request
contract. It consumed 4,120 prompt tokens and 263 completion tokens at a
reported cost of $0.00037782.

The full resumable run started on 2026-08-04 at 16:44:57 Europe/London with
concurrency 10. It resumed the same checkpoint, so the ten successful gate rows
were not repurchased.

- UCL host: `javelin-l.cs.ucl.ac.uk`
- tmux session: `lseg44-openrouter-gemma4`
- private checkpoint: `/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/openrouter/headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1.csv`
- private manifest: the checkpoint path plus `.manifest.json`
- initial private log: `/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/openrouter/full_run.log`

The API credential was passed through encrypted SSH to the process environment.
It was not written to UCL storage, embedded in the tmux command, or retained in
tmux's global environment.

## Concurrency acceleration record

On 2026-08-04 the concurrency-10 process was interrupted after a durable
checkpoint at 48,405 successful headlines. No completed row was removed. The
client was updated to obey a numeric OpenRouter `Retry-After` header before
retrying 429/503 responses; the request body and frozen scoring contract did not
change. The tested and deployed client SHA-256 was
`d2c8318daa25850812806ab9432935d546429ba37643e3da8348c6d7eb9e6da9`.

A frozen 2,000-attempt concurrency-20 gate then completed with 2,000 successes,
zero terminal API errors, and zero invalid outputs in 144.405 seconds (13.85
successful headlines/second). Reported gate cost was $0.07570327. This cleared
the predeclared continuation gates of at least five successes/second and under
3% terminal errors.

The full run resumed at concurrency 20 on 2026-08-04 at 19:53:20 Europe/London,
using the same checkpoint and `tmux` session name. Its private log is:

`/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/openrouter/full_run_c20.log`

That process was later stopped at a durable checkpoint of 88,055 successful
headlines. A second frozen 2,000-attempt gate at concurrency 25 completed with
2,000 successes, zero terminal API errors, zero invalid outputs, and no retries
in 89.133 seconds (22.44 successful headlines/second). Reported gate cost was
$0.07565025. This was 62.0% faster than the concurrency-20 gate on the same
route and request contract.

The full run resumed from 90,055 successful headlines at concurrency 25 on
2026-08-04; its manifest start time was 20:50:30 Europe/London. The first
durable 1,000-row batch added 1,000 successes with no new terminal errors. Its
private log is:

`/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/openrouter/full_run_c25.log`

One aborted relaunch between the gate and this full run was stopped before any
manifest or checkpoint activity. The post-stop audit confirmed that it added
no rows and left the 90,055-success checkpoint unchanged, so it is excluded
from timing, cost, and quality summaries.

## Completion record

The run and lower-concurrency cleanup passes completed on 2026-08-05. The final
manifest reports 888,155/888,155 unique successful headlines and zero remaining.
The append-only CSV has 889,103 rows: 888,155 successes, 919 historical API
errors, 27 historical invalid outputs, and two historical malformed responses.
Independent local validation found zero duplicate successes, zero invalid
successful probability vectors, and a total reported cost of $33.61534475.

Final private artifact:

`/cs/student/project_msc/2025/cf/pprender/runs/labels/lseg_us_sector_44_8m_headlines/openrouter/headline_scores_gemma4_26b_a4b_it_deepinfra_fp8_investor_headline_soft_label_v1.csv`

Its manifest is the same path plus `.manifest.json`. The final CSV SHA-256 is
`afb2d9c3029c4fba4fce3600218233a02d14051788f87313fd49c9da681c6d2f`.
Downstream work must select `status == "success"`; failed attempts remain only
as audit evidence.

## Backward-snapshot amendment: 2026-08-09

On 2026-08-09, the researcher explicitly authorised starting paid Gemma scoring
on the newly downloaded backward LSEG entries and reported adding $20 to the
OpenRouter balance. This amendment covers the immutable terminal-window
snapshot `snapshot_20260809_quota_stop` only. Its manifest records 11,543
completed company-date windows, 27,651 included pages, 607,097 unique scorable
normalized headlines, and population SHA-256
`9b314356592f766a69003b6c61b1bf16543a5dcde929d4c972a36c1075de860d`.

The model, prompt, provider, FP8, ZDR, data-collection denial, fallback, strict
probability-JSON, append-only checkpoint, and redistribution restrictions above
remain unchanged. The first paid chunk is capped at 300,000 attempts. At the
prior full-corpus realised mean of $0.000037848512 per attempted headline, its
planning estimate is $11.354553, leaving approximately $8.645447 of the
user-reported top-up for later snapshots and lower-concurrency cleanup. This is
an estimate rather than a spend guarantee; the checkpoint's reported API cost
is the authoritative run measure.

This amendment permits scoring and aggregate coverage/cost/quality monitoring
only. It does not permit prices, returns, partial strategy inference, model or
signal retuning, or describing a partial Gemma checkpoint as the completed
backward population. Later snapshots must reconcile successful rows by the
existing normalized-headline SHA-256 and retain all failed attempts for audit.

The bounded tranche completed on 2026-08-09 at 20:15:52 UTC. Its append-only
300,000 attempts comprise 299,987 unique successes and 13 API errors, for total
reported cost $11.30746476. Validation found zero duplicate successes, invalid
successful probability vectors, model/provider/FP8/prompt mismatches, or mixed
configuration hashes. The output SHA-256 is
`b85879da58716193243cd180a023e91dfabf142fb0f672f49a10ba206eb16e39`.
The status is `completed_partial`: 307,110 snapshot hashes remain unresolved,
and no further paid tranche is implied by this completion record.

## Second backward-snapshot tranche: 2026-08-09

At 22:19:48 UTC the researcher explicitly approved a second 200,000-attempt
tranche and its estimated $7.54 spend. It resumes the same append-only output,
snapshot population, configuration hash, model, prompt, DeepInfra FP8 route,
and privacy controls. The frozen proposal and approval record is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_gemma_tranche2_20260809.json`.

The run started at 22:21:06 UTC in local tmux session
`lseg33-backward-gemma-t2-20260809` with concurrency 25 and three retries. It
began from 299,987 unique successes and 307,110 unresolved hashes. It completed
at 01:39:02 UTC on 2026-08-10 after exactly 200,000 attempts: 199,998 successes
and two API errors. The append-only checkpoint therefore contains 499,985
unique successes, 15 preserved API errors, and 107,112 unresolved hashes. This
tranche cost $7.53824224; cumulative reported cost is $18.84570700. Validation
found zero duplicate successes, invalid successful probability vectors, route
mismatches, or output-checksum failures. Its output SHA-256 is
`2bd88cef837bd4e94372a6e7f9d4bcc519524e7b51968d317a439c09c2c149a6`.

## Final backward-snapshot tranche: 2026-08-10

At 10:10:43 UTC the researcher explicitly authorised Gemma scoring of all
107,112 unresolved hashes. A preflight aggregate credit check confirmed that
the estimated $4.04 spend was covered. The frozen approval record is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_gemma_final_tranche_20260810.json`.

The run started at 10:12:27 UTC in local tmux session
`lseg33-backward-gemma-final-20260810` with concurrency 25 and three retries,
using the same append-only output, population, model, prompt, DeepInfra FP8
route, and privacy controls. Its first verified checkpoint added 1,350 valid
successes with no new errors, duplicate successes, invalid probabilities, or
route mismatches. The pass later stopped with 605,129 unique successes and
1,968 unresolved hashes; every attempt remained in the append-only audit file.

## Lower-concurrency cleanup: 2026-08-10

At 14:45:16 UTC the researcher explicitly instructed: “do the cleanup over
1968”. The approval record is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_gemma_cleanup_20260810.json`.
The cleanup started at 14:47:35 UTC with concurrency 5 and five retries on all
1,968 unresolved hashes, using the identical model, prompt, DeepInfra FP8
route, privacy settings, population, and append-only output.

It completed at 15:18:03 UTC with 1,944 new successes, 18 API errors, and six
invalid outputs. The cumulative checkpoint contains 609,068 attempt rows and
607,073 unique successes from the 607,097-hash population, leaving 24 unresolved
(99.9960% coverage). Cumulative reported cost is $22.88230343. Validation found
zero duplicate successes, invalid successful probability vectors, route or
configuration mismatches, or output-checksum failures. The final output
SHA-256 is
`589ddbe857277fb46e3a81bc82a8b850b5192abebd7458777f0a3abb9078f596`.
Prices, returns, partial strategy inference, and retuning remain sealed.

## Second-account snapshot scoring: 2026-08-10

The researcher instructed: “Do the required gemini and finbert scoring on all
outstanding that are downloaded.” Here `gemini` is interpreted as the existing
Gemma 4 26B scorer; changing model families would break the frozen comparison.
The approval covers the 884,863-hash
`snapshot_20260810_second_account_quota_stop` population only. Exact-hash
reconciliation retains all 607,097 prior hashes, leaving 277,766 FinBERT scores
and 277,790 Gemma scores including the 24 earlier unresolved attempts.

The Gemma run retains model `google/gemma-4-26b-a4b-it`, DeepInfra-only FP8,
prompt hash `81596538d99b29b8`, ZDR, data-collection denial, no fallbacks, strict
probability JSON, and append-only failures. The authenticated credit preflight
reported $20.03 available versus an estimated $10.4364 spend. The frozen
approval is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_second_account_incremental_scoring_20260810.json`.
FinBERT remains pinned to ProsusAI/finbert revision
`4556d13015211d73dccd3fdd39d39232506f3e43` with cache-only inference. No price,
return, or strategy field is authorised by this amendment.

## Target-materiality pilot: approved and completed — 2026-08-12

The researcher authorised executing the bounded materiality-pilot plan with
“Go do this”. Return-blind preparation is complete: the immutable parent sample
contains 2,000 first-release, explicit-target, single-company, nontechnical
Reuters events across the same 33 companies. Selective local LSEG retrieval
found 1,445 events with substantive cleaned article bodies and 555 terminal
archival body failures. No price or return file was opened.

The proposed hosted payload is narrower than the prior full-corpus headline
scoring but contains more licensed content per item: the selected Reuters
headline, at most 6,000 characters of its cleaned article body, and at most five
strictly earlier same-company Reuters headlines used only for novelty. The
destination is OpenRouter, routed only to DeepInfra, using
`google/gemma-4-26b-a4b-it` with FP8, no fallback, zero-data-retention request,
`data_collection=deny`, reasoning disabled/excluded, temperature zero, a strict
six-field JSON schema, and the existing input/output price ceilings.

The researcher then supplied the exact required confirmation: “I approve
sending the selected Reuters headline and article-body text to OpenRouter,
routed only to DeepInfra, for Gemma 4 26B FP8 materiality scoring under the
stated no-retention/privacy controls.” The guarded scorer was invoked with
`--confirm-authorized-transfer`. A one-row probe passed, the concurrency-25
main pass preserved two invalid outputs and one API error, and lower-concurrency
cleanup recovered all three unresolved events without changing the prompt or
parser. The final private checkpoint has 1,445/1,445 unique valid successes
from 1,450 append-only attempts, no duplicate successes or route/schema
mismatches, and reported cost $0.18523289. Its score SHA-256 is
`c702a092ead572233b5854ab34d920eb2d1f8f7c82d72c98e25fe791d3fa04f1`.
The complete result is recorded in
`final_experiments/frozen_specs/lseg_target_materiality_pilot_scoring_result_20260812.json`.

This completed scoring does not loosen the scientific seal. Model scoring
opened no returns. A blinded 200-event primary human audit and 60-event
second-code subset now exist locally; both must be completed independently and
every frozen measurement gate must pass before Notebook 69 may permit a
separately implemented return analysis.
