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
only 1,469 backward windows. At the observed archive density, allow roughly
eight further fresh-quota days, with later lower-volume companies potentially
improving that estimate. These are planning estimates, not completion claims.

## Methodological stage gates

No partial daily batch may be scored or joined to prices. While acquisition is
running, only request counts, window coverage, errors, pagination status, file
hashes, duplicate identities, and company coverage may be inspected.
Date-major scheduling improves the coherence of those interim coverage audits;
it does not relax the full-collection requirement before scoring or returns.

After the collector records completion:

1. Hash and audit the immutable collection, normalize headlines, and freeze the
   complete unique scoring population.
2. Run the predeclared Gemma drift sample. It contains all 774 reference exact
   endpoints and 1,000 deterministic non-endpoint controls. If the model route,
   continuous agreement, direction, or endpoint-retention gates fail, stop
   without rescoring the new population or opening returns.
3. With separate user approval for paid licensed-text processing, score the
   complete population using the exact DeepInfra FP8 Gemma contract. Score the
   same hashes locally with the pinned FinBERT revision.
4. Construct mapped firm-open signals, active dates, long/short counts, and
   selected names without prices. Require at least 120 complete sessions, 60
   active sessions, and 20 active sessions in each chronological half.
5. Only if every previous gate passes, load prices once and execute the frozen
   return replay. Report both gross and 10-bps net results; do not select or
   retune a variant.

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

## Costs and timing after collection

The final headline count is unknown until collection completes. Scaling the
existing same-universe corpus suggests roughly 2.3--3.0 million unique
headlines, approximately $90--$120 for Gemma at the realised earlier average.
That estimate must be recomputed from the frozen unique count before requesting
approval. Local FinBERT should take roughly two hours; Gemma scoring is likely
to take 36--60 hours; the one-shot audit and notebook refresh should take three
to six hours.
