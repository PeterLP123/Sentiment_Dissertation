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
