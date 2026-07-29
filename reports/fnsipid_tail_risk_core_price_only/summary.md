# FNSPID sentiment-conditioned tail risk - core result

- Variant: **price_only** — price repair `min_abs_return`, volatility refit `frozen_development`
  Relative to v1, this cell applies the minimum-absolute-return sensitivity to 4,015 candidate adjusted/raw-return gaps. Nothing else differs.
- Run mode: **full**
- Git commit: `unavailable` (dirty worktree: unavailable)
- Primary scorer: finbert, probability-based semantics; alpha = 0.025

## Question

After the initial price response, does firm-linked financial-news sentiment improve
one-day-ahead VaR and ES forecasts beyond price-based conditional volatility, news
arrival, and news volume?

## Answer

**No detectable semantic gain. This is a valid null.**

On 1,752 evaluation target dates covering 415,758 news-bearing firm-days across 561 firms, the paired FZ0 loss
difference `FZ0(M2) - FZ0(M1)` is **-0.000463937** with a 95% date-block
bootstrap interval of **[-0.00261967, +0.00129321]** (64.8% of bootstrap means below zero).
In relative terms mean loss changes by -0.0382%; a negative paired
difference (and a negative relative change) means semantics help.

## Evidenced

| Quantity | Value |
| --- | ---: |
| mean FZ0, M0, news-bearing origins | 1.225123 |
| mean FZ0, M1, news-bearing origins | 1.214724 |
| mean FZ0, M2, news-bearing origins | 1.214260 |
| mean FZ0, M2_intensity, news-bearing origins | 1.213831 |
| paired_fz0_M2_minus_M1 | -0.000463937 |
| paired_fz0_M2_minus_M2_intensity | +0.00042933 |
| paired_fz0_M1_minus_M0 | -0.0103995 |
| relative_fz0_improvement_pct_M2_vs_M1 | +0.0381928 |
| VaR hit rate, M0, full panel (nominal 0.025) | 0.03099 |
| VaR hit rate, M1, full panel (nominal 0.025) | 0.03069 |
| VaR hit rate, M2, full panel (nominal 0.025) | 0.03066 |
| VaR hit rate, M2_intensity, full panel (nominal 0.025) | 0.03065 |

Every model was scored on the same 958,461 evaluation rows. All forecasts satisfy
`ES < VaR < 0` in both standardised and return units, and all listed assertions passed.

**Calibration caveat.** Every model over-violates its nominal level. On the full evaluation
panel M1 breaches on 0.03069 of days against a nominal 0.025, and the 95%
date-block interval on the excess, [+0.00185, +0.01071], excludes zero.
8 of 8 DQ conditional-coverage and ES identification tests reject at 5%.
Evaluation-period standardised returns have standard deviation 1.112, so the
development-fitted frozen volatility filter under-predicts 2017-2023 volatility. The nested comparison
is therefore a relative ranking among models that are all somewhat under-conservative; it is not
a claim that any of them is correctly calibrated.

**Timing caveat.** The frozen upstream manifest records a mixed policy: 2,518,109 date-only/exact-midnight source rows use the strictly-next-session rule, while 5,660 precise-timestamp rows use a containing-or-next-session rule before windowing and deduplication. Original timestamps are absent from the completed checkpoint, so the stricter date-only rule cannot be re-verified per retained headline. All forecasts remain point-in-time at the mapped reaction-session close, but this is a documented deviation from a uniformly date-only design.

## Inference

Bounded reading: once the reaction-session shock, its magnitude, the conditional
volatility level, market state, news arrival and news volume are in the information
set, the additional signed-tone and intensity variables did not measurably sharpen
the one-day-ahead lower tail on this panel at this horizon and coarse timestamp
resolution. This is evidence of absence only at the precision the interval supports;
it does not show that news semantics are irrelevant to tail risk in general.

Scale-versus-tail diagnostic: with a frozen news-conditioned QLIKE volatility adjustment the
paired difference moves from -0.000463937 to -0.00123876 (95% interval [-0.0054343, +0.00126263]).
Reading: the baseline interval spans zero, so this diagnostic cannot identify a semantic increment for the scale adjustment to explain.

## Open limitations

- **Survivorship.** The linked price panel is the set of tickers present in the FNSPID
  archive; 37 of 574 cohort tickers end before 2023-12, so the sample is partly survivor
  conditioned. It is described as the available linked firm-price panel, not an S&P 500 panel.
- **Universe composition.** FNSPID links headlines by ticker, so the cohort mixes
  operating firms with exchange-traded funds (for example AGG, GLD, QQQ). No local
  metadata separates them, and they were not hand-removed, because an ad hoc curated
  exclusion would be a researcher degree of freedom.
- **Timestamp coarsening.** FNSPID dates are calendar dates. Headlines are pushed to the
  next session, which is conservative but discards intraday ordering and merges Friday,
  weekend and holiday news into one reaction session.
- **Recap content.** Price-recap headlines mechanically restate the move that already
  happened; the frozen regex flag is imperfect and the recap robustness check is a
  sensitivity, not a clean identification.
- **Corporate actions.** Only adjusted closes are available and no corporate-action
  metadata; residual unadjusted events would show up as artificial tail losses.
- **Shared dates.** Firm-days on the same calendar date are strongly dependent; the
  bootstrap addresses this but assumes weak dependence across date blocks.
- **Model misspecification.** One volatility filter and one tail parameterisation were
  pre-specified. Failing to reject a calibration test is not proof of correctness.

## Not claimed

- No causal claim, no deployable alpha, no claim about investor behaviour, and no claim
  of first use of sentiment in tail-risk forecasting. A null is a valid result.

## Reproduce

```bash
REPRO_DIR="reports/reproductions/fnsipid_tail_risk_core_price_only_$(date -u +%Y%m%dT%H%M%SZ)"
FNSPID_TAIL_RISK_RUN_MODE=full \
FNSPID_TAIL_RISK_VARIANT=price_only \
FNSPID_TAIL_RISK_OUTPUT_DIR="$REPRO_DIR" \
jupyter nbconvert \
  --to notebook \
  --execute notebooks/fnsipid_tail_risk_core.ipynb \
  --output "../$REPRO_DIR/fnsipid_tail_risk_core.executed.ipynb" \
  --ExecutePreprocessor.timeout=-1
```

A fresh output path is mandatory: the notebook fails closed rather than mixing a rerun
with files from an earlier bundle. Choose a new `REPRO_DIR` after any interrupted run.

This run used `FNSPID_TAIL_RISK_RUN_MODE=full` over all 567 eligible firms. Set it to `smoke` for the deterministic 12-firm engineering check.
