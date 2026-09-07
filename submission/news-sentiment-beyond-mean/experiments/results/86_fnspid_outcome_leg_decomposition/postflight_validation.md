# Notebook 86 Postflight Validation

## Material passport

- Origin skill: academic-research-suite / experiment-agent
- Mode: post-review diagnostic, run and validate
- Executed: 2026-08-16 13:55 UTC
- Verification status: ANALYSED AND INDEPENDENTLY REPRODUCED
- Frozen specification: `fnspid_outcome_leg_decomposition_v1_20260816.json`

## Verdict

**CAUTION — the requested timing test resolves the review question by changing,
not strengthening, the headline interpretation.**

In development, the assigned-open-to-close coefficient is `-0.010291` (95% HAC(5)
CI `[-0.015960, -0.004622]`, BH q=`0.001121`), whereas assigned-close-to-next-open
is `+0.001548` (`[-0.004158, +0.007254]`, q=`0.594922`). The selected association
is therefore intraday in the assigned session, not predictive overnight.

The timing defect is not eliminated. Previous-close-to-assigned-open is `-0.006381`
(unadjusted p=`0.045719`, BH q=`0.068578`), and the separately requested
previous-open-to-assigned-open probe in Notebook 87 is strongly negative. Because
the checkpoint has no reliable row-level availability timestamps, these results
cannot establish whether a story preceded the price movement.

The evaluation intraday coefficient is `+0.003373`; its evaluation-minus-development
contrast is `+0.013664` (p=`0.020423`, BH q=`0.061269`). The full assigned-window
contrast remains `+0.014143` (p=`0.017284`). These are post-opening diagnostics,
not a repair of the original holdout or a new confirmation family.

## Other reconciliations

- The observed pooled 90th-minus-10th negative-share rank spread is `0.560560`,
  not the unattainable hypothetical `0.8`. The full-control translation is
  `-0.512374` return-rank percentile points.
- Prospective evaluation power was `35.4332%`; power recomputed with the realised
  evaluation HAC standard error is `26.5075%`. The realised standard error is
  `1.157583` times the projected value.
- The FNSPID HAR overlay maximum drawdown is `-16.3625%`, worse than the matched
  base's `-15.7300%`.
- Return-leg identities reconcile to at most `1.776e-15`; the recomposed abnormal
  return matches the committed panel.
- Story-count strata are reported separately and are not treated as commensurable
  full-range effect sizes.

## Reproducibility checks

- Notebook 86 completed with no error outputs.
- All nine CSV outputs were rerun from a fresh cache in
  `/private/tmp/news-sentiment-nb86.1QfgQj` and were byte-identical to the promoted
  outputs.
- The fresh manifest was canonically identical after excluding run-path and
  execution-time fields.
- Notebook 75 was not executed. Its SHA-256 remained
  `df0b6c5105158198f62c828842b3ff3e1051ccaf664e7a778c9c66e11cffc455`.
- The specification, runner, executed notebook, aggregate outputs and manifest are
  hashed in `promotion_manifest.json` and `manifest.json`.

## Claim boundary

This is a post-review diagnostic specified after the selected association and its
temporal non-replication were known. It supports an intraday placement statement
for the assigned window. It does not identify story arrival, causality, tradability
or independent replication, and it does not pool FNSPID with LSEG.
