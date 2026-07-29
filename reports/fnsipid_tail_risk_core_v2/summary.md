# FNSPID sentiment-conditioned tail risk - core result

- Variant: **v2** — price repair `min_abs_return`, volatility refit `annual_expanding`
  Relative to v1, this cell repairs 4,015 adjusted-price rows whose adjustment factor stepped inconsistently and re-estimates the volatility filter before each evaluation year on an expanding window. Nothing else differs.
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
difference `FZ0(M2) - FZ0(M1)` is **-0.000510569** with a 95% date-block
bootstrap interval of **[-0.00266411, +0.00125345]** (67.1% of bootstrap means below zero).
In relative terms mean loss changes by -0.0425%; a negative paired
difference (and a negative relative change) means semantics help.

## Evidenced

| Quantity | Value |
| --- | ---: |
| mean FZ0, M0, news-bearing origins | 1.211890 |
| mean FZ0, M1, news-bearing origins | 1.202285 |
| mean FZ0, M2, news-bearing origins | 1.201775 |
| mean FZ0, M2_intensity, news-bearing origins | 1.201304 |
| paired_fz0_M2_minus_M1 | -0.000510569 |
| paired_fz0_M2_minus_M2_intensity | +0.000471201 |
| paired_fz0_M1_minus_M0 | -0.00960415 |
| relative_fz0_improvement_pct_M2_vs_M1 | +0.0424666 |
| VaR hit rate, M0, full panel (nominal 0.025) | 0.03008 |
| VaR hit rate, M1, full panel (nominal 0.025) | 0.02982 |
| VaR hit rate, M2, full panel (nominal 0.025) | 0.02982 |
| VaR hit rate, M2_intensity, full panel (nominal 0.025) | 0.02983 |

Every model was scored on the same 958,461 evaluation rows. All forecasts satisfy
`ES < VaR < 0` in both standardised and return units, and all listed assertions passed.

**Calibration caveat.** Every model over-violates its nominal level. On the full evaluation
panel M1 breaches on 0.02982 of days against a nominal 0.025, and the 95%
date-block interval on the excess, [+0.00097, +0.00999], excludes zero.
8 of 8 DQ conditional-coverage and ES identification tests reject at 5%.
Evaluation-period standardised returns have standard deviation 1.096, so the
annual-refitted point-in-time volatility filter under-predicts 2017-2023 volatility. The nested comparison
is therefore a relative ranking among models that are all somewhat under-conservative; it is not
a claim that any of them is correctly calibrated.

## Inference

Bounded reading: once the reaction-session shock, its magnitude, the conditional
volatility level, market state, news arrival and news volume are in the information
set, the additional signed-tone and intensity variables did not measurably sharpen
the one-day-ahead lower tail on this panel at this horizon and coarse timestamp
resolution. This is evidence of absence only at the precision the interval supports;
it does not show that news semantics are irrelevant to tail risk in general.

Scale-versus-tail diagnostic: with a frozen news-conditioned QLIKE volatility adjustment the
paired difference moves from -0.000510569 to -0.00136103 (95% interval [-0.0056253, +0.00117604]).
Reading: the gain largely survives, which is consistent with incremental lower-tail information.

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
REPRO_DIR="reports/reproductions/fnsipid_tail_risk_core_v2_$(date -u +%Y%m%dT%H%M%SZ)"
FNSPID_TAIL_RISK_RUN_MODE=full \
FNSPID_TAIL_RISK_VARIANT=v2 \
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
