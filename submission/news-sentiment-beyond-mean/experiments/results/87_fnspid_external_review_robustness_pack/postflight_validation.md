# Notebook 87 Postflight Validation

## Material passport

- Origin skill: academic-research-suite / experiment-agent
- Mode: post-review robustness pack, run and validate
- Executed: 2026-08-16 14:09 UTC
- Verification status: ANALYSED AND INDEPENDENTLY REPRODUCED
- Frozen specification: `fnspid_external_review_robustness_pack_v1_20260816.json`

## Verdict

**CAUTION — robustness survives, but timing remains unidentified and no result is
independent confirmation.**

The separately requested previous-open-to-assigned-open probe is `-0.019257`
(95% HAC(5) CI `[-0.025796, -0.012717]`, p=`7.868e-09`). Alongside Notebook 86's
intraday placement, this means the aggregate checkpoint is compatible with stale
assignment or earlier information as well as contemporaneous reaction. The paper
therefore withdraws the earlier next-open-specific predictive interpretation.

Trailing-beta adjustment does not remove the development result or era contrast.
The development coefficient is `-0.010543` (BH q=`0.000835`), or `-0.009505`
(q=`0.001182`) when trailing beta also enters as a control. The corresponding era
contrasts are `+0.018839` (q=`0.003022`) and `+0.016285` (q=`0.004919`). These are
robust conditional rank associations, not factor alphas or causal effects.

The full era contrast remains positive with HAC lags 0, 5, 21 and 42
(p=`0.01709`, `0.01728`, `0.02280`, `0.02008`). The regime-demeaned daily series
has first-order autocorrelation `0.00537`; Ljung--Box p-values at the checked lags
do not reject serial independence.

The FNSPID risk comparison contains 44 risk-off episodes. Every leave-one-episode-out
confidence interval remains above zero for overlay-minus-base cumulative return,
and excluding February--April 2020 also leaves a positive interval (p=`0.0034`).
This defeats a single-episode explanation but does not rescue the failed economic
gate: the overlay has worse cumulative return and maximum drawdown.

## Composition diagnostics

- FinBERT development class rates are 16.80% negative, 60.01% neutral and 23.20%
  positive; evaluation rates are 15.84%, 61.42% and 22.74%.
- Negative share equals zero on 74.93% of development firm-days and 76.85% of
  evaluation firm-days.
- Evaluation is more singleton-heavy: 56.53% versus 48.32%. A pure singleton
  threshold story therefore does not explain the sign reversal by itself.
- Average names per session fall from 226.2 to 203.8. This does not exhaust
  composition, classifier or mapping explanations.

## Reproducibility checks

- Notebook 87 completed with no error outputs.
- All fourteen CSV outputs were rerun from a fresh cache in
  `/private/tmp/news-sentiment-nb87.1ebBWQ` and were byte-identical to the promoted
  outputs.
- The fresh manifest was canonically identical after excluding run-path and
  execution-time fields.
- Notebook 75 was not executed and retained the same SHA-256.
- Repository validation subsequently passed 74 tests and reproduced all 27
  manuscript artifacts.

## Claim boundary

This pack responds to an external review after selection. It tests sensitivity to
timing, beta adjustment, HAC lag, dependence, episode concentration and observable
composition. It cannot create a pristine holdout, validate FinBERT labels against
humans, identify causal alpha or establish cross-source transfer.
