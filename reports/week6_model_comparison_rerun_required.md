# Week 6 model comparison — scoring rerun required

**Status: provisional; do not present the current three-model comparison as a
complete-corpus result.**

Recorded on 15 July 2026 after auditing the score files used by
`week6_gemma_finbert_vader_20260715_final`.

## What happened

- A direct full-population audit of the frozen LSEG collection found `568,707`
  unique normalized headlines eligible for scoring (`710,625` raw rows before
  headline deduplication).
- The earlier Gemma run attempted `370,405` of them (`65.13%`). It successfully
  scored `85,961` headlines: `23.21%` of its attempted rows, but only `15.12%`
  of the complete population.
- Another `284,444` attempted headlines recorded `HTTP 402` API errors because
  the provider reported that payment/quota was required. A further `198,302`
  full-population headlines were never attempted by that run.
- FinBERT and VADER were subsequently scored on the same `85,961` successful
  Gemma headlines. This makes the existing comparison coverage-matched, but it
  does not make it a complete-corpus comparison.

The current model-comparison output under
`results/week6_model_comparison/week6_gemma_finbert_vader_20260715_final/`
therefore applies only to the completed hash-ordered subset. Preserve it as
provisional evidence; do not overwrite it.

## Required rerun

1. Recover and record the exact Gemma provider, model identity, prompt and
   inference settings used for the existing score file.
2. Restore sufficient provider quota, then resume the append-only Gemma scoring
   run. Successful rows must be retained and failed rows retried.
3. Require `568,707/568,707` successful unique Gemma headlines and zero
   unresolved failures before calling the scoring pass complete.
4. Rebuild FinBERT and VADER scores over that same complete headline population.
5. Rebuild the headline-value signals into a new immutable output directory.
6. Rerun `compare-week6-models` with a new run ID, including the same funded
   P&L assumptions and an explicit buy-and-hold benchmark.
7. Report complete score reconciliation, coverage, failures and hashes before
   interpreting differences between models.

If completing Gemma is not affordable, define a smaller stratified headline
sample before scoring any model, record the sampling seed and strata, and rerun
all three models on exactly that frozen sample. Do not describe the existing
quota-truncated subset as a designed sample.

## Two-model update — 15 July 2026

FinBERT and VADER have now both been scored locally over the complete `568,707`
headline population with zero failures. Their complete-population signals and
funded comparison were rebuilt under new immutable output names. See
`reports/week6_finbert_vader_full_population_test.md`.

This closes the coverage problem for a FinBERT-versus-VADER comparison only.
Gemma remains incomplete, so the earlier three-model ranking remains
provisional and must not be used to conclude that Gemma is intrinsically worse.
