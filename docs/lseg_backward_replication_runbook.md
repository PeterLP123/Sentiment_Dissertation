# LSEG backward replication runbook

## Purpose

This arm applies the unchanged Notebook 36 Gemma-timing/FinBERT-ranking rule to
an earlier, non-overlapping LSEG interval. It is an external-time backward
replication, not a prospective test. A positive result can corroborate the
historical strategy; it cannot establish deployment readiness.

The binding contract is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_v1.json`.
The licensed headline files and all scorer outputs remain local and ignored.

## Frozen source population

- Dates: 2024-01-01 inclusive to 2025-10-26 exclusive.
- Calendar upper bound: 456 XNYS sessions.
- Universe: the exact frozen 33 companies, RICs, queries, and aliases.
- Source: every entitled English-language headline returned by those RIC
  queries; no Reuters-only restriction and no story-body retrieval.
- Deduplication: the existing normalized-headline SHA-256 identity.
- Daily retrieval budget: 9,500 of the user's 10,000 LSEG requests, retaining
  500 for entitlement checks and retries.

The researcher subsequently authorised use of the full 10,000-request daily
allowance. The frozen config retains its conservative 9,500 default so the
collection identity does not change. `--max-requests` is an operational-only
per-invocation override. On 2026-08-06, the 33-company entitlement check used
33 requests, so a 467-request top-up follows the original 9,500-request batch.
On later days with no new check, use `--max-requests 10000`. This amendment is
recorded in
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_operational_amendment_20260806.json`.

The acquisition config is
`configs/lseg_us_sector_33_backward_20240101_20251026_headlines.toml`.
Each invocation resumes from its checkpoint. It stops at the config's 9,500
default unless `--max-requests` supplies the separately recorded operational
cap. Do not change the config after the first retrieval.

The first quota day traversed companies first and reached complete AAPL/MSFT
coverage plus NVDA windows 1--141. From the next quota reset, the collector
traverses dates first: it fills every missing company for one date before moving
to the next date. Existing symbol/window/page checkpoints are reused unchanged.
This scheduling change makes an interrupted collection converge toward a
complete cross-sectional date prefix without changing the final population.
The exact transition is recorded in
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_collection_order_amendment_20260806.json`.

## Daily acquisition loop

Keep LSEG Workspace open and signed in. Use this entitlement check only when
the desktop session needs validation; its 33 requests count against that day's
allowance:

```bash
.venv/bin/sentiment-bench lseg-news-check \
  --config configs/lseg_us_sector_33_backward_20240101_20251026_headlines.toml \
  --all-companies
```

On a normal reset day with no entitlement check, run one full quota batch:

```bash
caffeinate -is .venv/bin/sentiment-bench fetch-lseg-news \
  --config configs/lseg_us_sector_33_backward_20240101_20251026_headlines.toml \
  --max-requests 10000
```

If the 33-company check ran that day, use `--max-requests 9967` instead.

The LSEG Data Library 2.1.1 asynchronous archive branch contains a malformed
`dict.update` call. The repository adapter bypasses only that broken branch by
placing `dateFrom`, `dateTo`, and `archive=true` directly in the SDK's extended
query parameters. `tests/test_lseg_source.py` protects this workaround; recent
headline retrieval is unchanged.

One full-quota batch is expected to take roughly two to three hours at the
frozen three-request-per-second pace. Day one shared its account-level quota
with an earlier 3,776-request prospective collection and therefore completed
only 1,469 backward windows. On 2026-08-07 the collector made 7,705 API
requests, including 12 retries, before the account-wide daily balance reached
zero; the quota window therefore already contained about 2,295 requests. The
resulting checkpoint has 4,829/21,912 complete company-day windows, 12,535
saved pages, and 112 complete 33-company dates through 2024-04-21. At the
observed complete-date density, allow roughly four further full-quota days.
These are planning estimates, not completion claims.

## Methodological stage gates

The researcher amended the collection-first rule on 2026-08-08 because of the
dissertation deadline. Local FinBERT preprocessing may now run incrementally on
immutable snapshots containing all and only terminal company-date page chains.
The model remains `ProsusAI/finbert` revision
`4556d13015211d73dccd3fdd39d39232506f3e43`, cache-only, with continuous score
`P(positive) - P(negative)`. Each snapshot, population hash, and output is kept
separate and resumable; the final completed population must still be reconciled
by unique successful normalized-headline hash. The amendment is recorded in
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_scoring_amendment_20260808.json`.

This does **not** open prices, returns, partial strategy construction, or model
selection. Gemma remains on the exact DeepInfra FP8 API contract and is run
separately by the researcher. Date-major scheduling continues to improve the
coherence of interim snapshots without turning them into final evaluation data.

The first authorised snapshot was materialized after the 2026-08-08 quota
stop. It contains 8,233 terminal company-date chains, 20,212 included page
checkpoints, and 471,107 unique scorable normalized headlines; the single
nonterminal frontier page was excluded. Its population SHA-256 is
`dcbd5ab498808f9839b63f7989e916f136f6636f76cf9d06d450a7929a099ad1`.
First-snapshot pinned FinBERT scoring was launched on UCL shared storage in tmux session
`lseg33-backward-finbert-20260808`, with output and its resumable manifest under
`runs/labels/lseg_us_sector_33_backward_20240101_20251026/incremental/snapshot_20260808_quota_stop/`.

The second authorised snapshot was materialized after the 2026-08-09 quota
stop. The return-blind audit records 11,543/21,912 terminal company-date
windows, 27,651 saved terminal pages, 329 complete 33-company dates, and a
2024-11-25 frontier at 18/33 companies, with zero pagination anomalies. The
snapshot contains 607,097 unique scorable normalized headlines (population
SHA-256
`9b314356592f766a69003b6c61b1bf16543a5dcde929d4c972a36c1075de860d`).
It retains all 471,107 prior FinBERT successes and adds 135,990 new hashes.
Cache-only UCL CUDA scoring completed with 607,097 FinBERT successes and the
same number of VADER companion rows: 1,214,194 successful rows in total, with
no failures, duplicate model-headline successes, invalid probabilities, or
metadata mismatches. The score file SHA-256 is
`7e4e53d2d0eca685b3bbd80a8aeff060b6bb1520175ffdfbeefe73ac0ae4e510`.
This is still preprocessing only; it does not open Gemma drift scoring,
prices, returns, or partial strategy inference.

On 2026-08-09 the researcher separately authorised paid Gemma preprocessing of
the same immutable snapshot after adding $20 to OpenRouter. The dated amendment
is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_gemma_scoring_amendment_20260809.json`.
It retains the exact Gemma 4 26B / DeepInfra FP8 / investor-prompt / ZDR
contract and caps the first tranche at 300,000 attempts, including a 10-row
smoke check. The smoke produced 10/10 valid successes for a reported
$0.00037509. The remaining 299,990-attempt pass completed at 21:15:52 BST in
local tmux session `lseg33-backward-gemma-20260809`. Across the whole tranche,
299,987 unique successes and 13 preserved API errors cost $11.30746476; there
are zero duplicate successes, invalid successful probabilities, or route
mismatches. The append-only output SHA-256 is
`b85879da58716193243cd180a023e91dfabf142fb0f672f49a10ba206eb16e39`.
Its per-pass manifests and logs remain ignored beside the immutable snapshot
inputs, and 307,110 hashes remain unresolved before any later paid pass.

The researcher explicitly approved a second 200,000-attempt tranche at
22:19:48 UTC on 2026-08-09, with an estimated $7.54 spend and about $1.15 of the
reported top-up retained. It resumed the same checkpoint at concurrency 25 in
tmux session `lseg33-backward-gemma-t2-20260809` and completed at 01:39:02 UTC
on 2026-08-10. Its 200,000 attempts produced 199,998 successes and two API
errors for $7.53824224, leaving 499,985 unique successes, 15 preserved API
errors, and 107,112 unresolved hashes. The append-only output SHA-256 is
`2bd88cef837bd4e94372a6e7f9d4bcc519524e7b51968d317a439c09c2c149a6`.

At 10:10:43 UTC on 2026-08-10 the researcher authorised a final attempt on all
107,112 unresolved hashes, with an estimated $4.04 spend. It started at
10:12:27 UTC at concurrency 25 in tmux session
`lseg33-backward-gemma-final-20260810`. The first verified checkpoint added
1,350 successes without new errors or validation failures. Its frozen approval
record is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_gemma_final_tranche_20260810.json`.

The concurrency-25 pass stopped with a 1,968-hash tail. At 14:45:16 UTC the
researcher explicitly authorised a lower-concurrency cleanup of that full
tail under the unchanged contract. The cleanup attempted all 1,968 hashes at
concurrency 5, adding 1,944 successes and preserving 18 API errors plus six
invalid outputs. The final append-only checkpoint has 607,073/607,097 unique
Gemma successes (99.9960%), 24 unresolved hashes, and $22.88230343 cumulative
reported cost. Validation records zero duplicate successes, invalid successful
probability vectors, route/configuration mismatches, or checksum failures; the
output SHA-256 is
`589ddbe857277fb46e3a81bc82a8b850b5192abebd7458777f0a3abb9078f596`.
The cleanup approval/completion record is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_incremental_gemma_cleanup_20260810.json`.

Collection continued separately through the second-account 2026-08-10 quota
stop. The current return-blind ledger has 17,967/21,912 terminal company-date
windows, 42,899 saved pages, 536 complete 33-company dates, and a 2025-06-20
frontier at 25/33, with zero pagination anomalies. Notebook 66 shades its
33×948 matrix by exact scoring state: 19,538 fully scored cells, 24 partial
cells touched by unresolved Gemma hashes, 7,777 downloaded/unscored cells, and
3,945 incomplete cells. One estimated full 10,000-request quota day remains.

The researcher then authorised FinBERT and Gemma preprocessing of every
terminal-window headline downloaded through that stop. The third immutable
snapshot contains 884,863 unique scorable normalized headlines (population
SHA-256
`a9349fdda44f3a9fc848554df08bb1de7e8830d5eb79a331201235ff342420b6`).
Hash reconciliation retains all 607,097 prior population hashes, adds 277,766,
and drops none. FinBERT therefore has 277,766 new hashes; Gemma has 277,790
pending because the 24 unresolved historical hashes are retried too. A
metadata audit found 3,193 inherited hashes whose target-snapshot metadata had
expanded or changed, so the FinBERT/VADER resume seed rewrites current
headline metadata while inheriting only the frozen probabilities. Its SHA-256
is `c51494ae00eef2000ae6d4672062c2e11e0cd7863c3f54986efb896120eccb0f`.

Gemma scoring started at 19:27:12 UTC on the unchanged Gemma 4 26B / DeepInfra
FP8 / investor-prompt / ZDR contract in local tmux session
`lseg33-backward-gemma-20260810`. FinBERT inputs, canonical seed, and launcher
are hash-verified on UCL shared storage. All 54 dated-inventory workers were
offline at the first launch check, so the one-shot local tmux watcher
`lseg33-finbert-ucl-watcher-20260810` checks every five minutes and launches
only after both the pool checker and remote launcher find a free GPU. Prices,
returns, partial strategy inference, and retuning remain sealed.

Refresh the aggregate-only three-stage status ledger with:

```bash
.venv/bin/python scripts/audit_lseg_backward_pipeline_status.py
```

This rewrites the ignored
`snapshot_20260810_second_account_quota_stop/pipeline_status.json` from terminal
LSEG checkpoint metadata, the aggregate-only FinBERT staging/completion record,
and both append-only Gemma attempt files. It audits coverage, hashes, route,
probability validity, duplicate successes, status counts, reported cost, and
remaining population without printing headlines or loading prices/returns.

After the collector records completion:

1. Complete and hash the immutable collection, normalize headlines, reconcile
   the incremental FinBERT snapshots, and freeze the complete unique scoring
   population.
2. Run the predeclared Gemma drift sample. It contains all 774 reference exact
   endpoints and 1,000 deterministic non-endpoint controls. If the model route,
   continuous agreement, direction, or endpoint-retention gates fail, stop
   without rescoring the new population or opening returns.
3. With separate user approval for paid licensed-text processing, score the
   complete population using the exact DeepInfra FP8 Gemma contract. Verify
   complete pinned-FinBERT coverage of the same final hashes from the
   incremental outputs, scoring only any remaining hashes.
4. Construct mapped firm-open signals, active dates, long/short counts, and
   selected names without prices. Require at least 120 complete sessions, 60
   active sessions, and 20 active sessions in each chronological half.
5. Only if every previous gate passes, load prices once and execute the frozen
   return replay. Report both gross and 10-bps net results; do not select or
   retune a variant.

### Final collection and drift-gate outcome (2026-08-11)

The complete terminal-window snapshot contains 21,912/21,912 company-dates,
53,904 page checkpoints, and 1,219,608 unique scorable normalized headlines.
Its population SHA-256 is
`68b987095715fc3ecca8d80056487f82b3e250094e2a6c3e986afa812cd449c7`.
The snapshot is marked `partial_collection=false`; no prices or returns were
loaded.

The frozen Gemma drift gate then rescored all 774 reference exact endpoints
plus 1,000 deterministic non-endpoint controls using the unchanged Gemma 4
26B / DeepInfra FP8 / prompt / no-fallback / ZDR contract. All 1,774 requests
succeeded for a reported $0.06719615. Five of six gates passed: Spearman was
0.9892, mean absolute difference 0.01654, direction agreement 0.9899, endpoint
opposite-direction count zero, and non-endpoint-to-endpoint rate 0.002.
Exact endpoint same-direction retention was 0.7881 against the predeclared
0.80 minimum, so the conjunctive drift gate failed.

The binding action is therefore to stop before full-population scoring or any
return opening. The approximately 334,762 pending Gemma hashes and the final
FinBERT delta were not launched. Do not weaken the 0.80 threshold or redefine
the exact endpoint after observing this result. The aggregate result is frozen
in
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_gemma_drift_result_20260811.json`;
licensed sample text and row-level scores remain local and ignored.

The researcher subsequently instructed the pipeline to complete both scorers.
This is recorded as a separate measurement-only override: it does not rescind
the failed drift gate, qualify the scores as the predeclared replication, or
open prices, returns, strategy replay, or retuning. Final-population
reconciliation found four prior normalized hashes superseded by the completed
story merge. The canonical seeds therefore retained 884,859 FinBERT hashes and
884,842 Gemma hashes, leaving 334,749 and 334,766 respectively. FinBERT
completed cache-only on UCL host `barbel-l` with 1,219,608/1,219,608 successes
and output SHA-256 `3800f4f7…`. Gemma's main pass left 21 API errors; the
lower-concurrency cleanup recovered all 21, finishing at
1,219,608/1,219,608 successes for $12.70683447 incremental reported cost and
output SHA-256 `09ea9ed4…`. The override record is
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_backward_full_scoring_override_20260811.json`.

## Return methodology

The primary strategy, timing map, strongest-event collision rule, exact Gemma
endpoint trigger, FinBERT ranking, 25% cap, cash handling, drift-aware turnover,
10-bps cost, next-open horizon, block bootstrap, factor model, half-stability,
and leave-out gates are inherited unchanged from the prospective contract.

Two non-promoting mechanism tests are predeclared as one BH-controlled family:

1. Circularly shift the complete Gemma event/count schedule while keeping the
   FinBERT ranks and portfolio mechanics fixed.
2. Randomize selected names within each session while preserving the Gemma
   date, long/short counts, cap, horizon, and costs.

These tests distinguish date/count timing from name selection. They cannot
promote a strategy that fails the primary cash, quality, factor, stability, or
dependence gates.

### Separately authorised post-stop replay (2026-08-12)

The researcher later authorised joining the fully scored backward block to the
original-33 portion of the recent LSEG corpus and running the unchanged chosen
strategy across the longer history. This is governed by
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_joined_long_window_v1.json`
and its return-blind completeness amendment, not by the failed original
replication contract. The 0.7881-versus-0.80 drift failure remains unchanged.

Notebook 67 validates a 1,967,554-headline cross-block population, reproduces
the recent Notebook 36 aggregate inputs exactly, and forces 11 incomplete
backward sessions to cash. The earlier block has gross/net Sharpe
0.691/−0.354, −3.43% net return, 6.60-bps/side break-even, zero-crossing cash
and sector-factor intervals, failed timing and name-selection tests, and only
1/33 company and 3/11 sector exclusions profitable. The 623-session joined
path remains positive at 0.467 net Sharpe and +6.51%, but it combines that
negative block with an already-opened strong recent block. The correct action
is to preserve the adverse temporal-robustness result without retuning. It is
not the preregistered replication, independent confirmation, or deployment
evidence.

## Historical pre-completion cost and timing estimate

At the time this estimate was recorded, the final headline count was unknown.
The then-current
terminal-window snapshot contains 884,863 unique scorable headlines. Reusing
607,073 valid Gemma successes leaves 277,790 attempts; at the observed prior
mean the run estimate is $10.4364 against $20.03 verified available credit.
The API's reported per-attempt cost, not this extrapolation, is authoritative.
The final full-population estimate must be recomputed after collection
completes; the one-shot audit and notebook refresh should then take three to
six hours.
