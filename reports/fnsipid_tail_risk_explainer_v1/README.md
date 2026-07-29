# FNSPID tail-risk result — visual explainer

Generated from the frozen run `reports/fnsipid_tail_risk_core_v1/` (git `unavailable`, full mode, 561 firms).

This bundle re-estimates nothing. Every artifact it reads was hash-verified against the core run manifest, and the
one recomputed quantity (the bootstrap replicate distribution) is asserted equal to `bootstrap.csv` before plotting.

| # | Figure | What it shows |
| --: | --- | --- |
| 1 | **From 28.6 million raw news rows to 416 thousand scored news days**<br>[`figures/e01_funnel.png`](figures/e01_funnel.png) | Only 13 of 574 cohort firms are lost, and every exclusion has one deterministic reason. |
| 2 | **The leak-safe timing rule: news reacts, then we forecast the day after**<br>[`figures/e02_timing_rule.png`](figures/e02_timing_rule.png) | The exhaustive check over all 4,779 calendar dates in the window found 0 mappings that were not strictly forward. |
| 3 | **Three strictly nested information sets**<br>[`figures/e03_nested_models.png`](figures/e03_nested_models.png) | The comparison is M2 vs M1, not M2 vs nothing — the bar M2 has to clear is 'beyond arrival and volume'. |
| 4 | **Adding news arrival moves the loss. Adding sentiment barely moves it.**<br>[`figures/e04_loss_ladder.png`](figures/e04_loss_ladder.png) | On news days, arrival + volume buys -0.00972 of loss; semantics buys -0.000307 — about 3% as much. |
| 5 | **News arrival clears zero decisively. Sentiment does not come close.**<br>[`figures/e05_effect_sizes.png`](figures/e05_effect_sizes.png) | The semantic effect is 3.2% the size of the arrival effect, and its interval spans zero. |
| 6 | **The design can detect an effect — it detects one, just not from sentiment**<br>[`figures/e06_bootstrap_distributions.png`](figures/e06_bootstrap_distributions.png) | Arrival: 100.0% of replicates below zero. Sentiment: 60.1%. Signed tone: 36.9%. |
| 7 | **The semantic difference is one crisis, then a drift back toward zero**<br>[`figures/e07_where_it_comes_from.png`](figures/e07_where_it_comes_from.png) | The semantic curve is flat until the 2020 crash drops it to -8.1e-04, then drifts back to -3.1e-04. No persistent slope; the arrival curve, by contrast, declines throughout. |
| 8 | **Sentiment moves the tail through intensity, not direction — and only by a couple of percent**<br>[`figures/e08_how_much_do_models_disagree.png`](figures/e08_how_much_do_models_disagree.png) | Median absolute disagreement is 6.7 bp, or 2.16% of the VaR level. Against M1, M2 shifts VaR by +0.039 standardised units on neutral text, -0.065 on very adverse text and -0.089 on very positive text — so it is the confidence of the text that deepens the tail, not its direction, and the residual asymmetry runs slightly the other way. |
| 9 | **The price archive has adjustment discontinuities that manufacture fake tail losses**<br>[`figures/e09_price_adjustment_integrity.png`](figures/e09_price_adjustment_integrity.png) | 470 of 1,788,865 firm-days (0.026%) are flagged, and they breach at 66% against 3.0% on clean days. Common targets enter each model's nonlinear FZ0 loss differently, so they can move M2-vs-M1; the declared repair check moved the estimate but its interval still spanned zero. |
| 10 | **One company's tail forecasts through 2020: V**<br>[`figures/e10_illustrative_firm.png`](figures/e10_illustrative_firm.png) | V is the most news-covered retained operating company with a clean adjusted-price series. Both models widen sharply into the March 2020 crash, driven by the volatility filter rather than by text. |
| 11 | **All four models are equally under-conservative — the ranking survives, absolute accuracy does not**<br>[`figures/e11_calibration.png`](figures/e11_calibration.png) | Breach rate is about 3.06% against a nominal 2.5%. The M2-vs-M1 comparison is a paired ranking, so this bias cancels — but no model here is deployable as stated. |
| 12 | **The volatility filter, not the text, is doing the heavy lifting — and it is stretched**<br>[`figures/e12_volatility_filter.png`](figures/e12_volatility_filter.png) | Overall std(z) is 1.117, and 2020 reaches 1.40 — that is the main source of the over-breaching in the previous figure. |
| 13 | **The null is not an artifact of one arbitrary choice**<br>[`figures/e13_robustness_forest.png`](figures/e13_robustness_forest.png) | 7 of 8 rows cover zero. The only one that does not is the aggregation variant that discards cross-firm date dependence by construction. |

## Read-out

```text
EVIDENCED  (what the executed run found)

  1. News ARRIVAL and VOLUME improve one-day-ahead VaR/ES forecasts beyond price
     state alone.  M1 - M0 = -9.718e-03, 95% CI
     [-1.775e-02, -3.861e-03], 100.0% of bootstrap
     replicates below zero.

  2. News SENTIMENT adds nothing measurable on top of that.
     M2 - M1 = -3.067e-04, 95% CI
     [-2.249e-03, +1.316e-03] - the interval covers zero.
     The effect is 3.2% the size of the arrival effect.

  3. Signed TONE adds nothing beyond non-neutral intensity.
     M2 - M2_intensity = +2.738e-04, CI [-6.088e-04, +1.627e-03].

  4. The two models barely disagree.  Median absolute VaR difference is
     6.7 basis points, 2.16% of the forecast level.  Text moves
     the tail mostly through INTENSITY, not direction.  Against M1, M2 shifts VaR
     by +0.039 standardised units on neutral text but -0.065 on very
     adverse text and -0.089 on very positive text: confident text of either
     sign deepens the tail.  That is the same story the M2 vs M2_intensity
     ablation tells, seen at the level of the forecast itself.

  5. All four models breach too often (3.06% against a nominal 2.5%) and fail
     their conditional-coverage tests, equally.  Standardised evaluation returns
     have std 1.117, so the frozen 2011-2016 volatility filter
     under-predicts 2017-2023 volatility.

  6. DATA QUALITY.  470 firm-days (0.026%) carry an adjusted-price
     discontinuity larger than 5% that is not a dividend or split.  They breach
     at 66% versus 3.0% on clean days.  Common targets do
     not cancel from the nonlinear paired loss: the declared repair changed the M2-vs-M1
     estimate, but its 95% interval still spanned zero.  The artifacts also inflate the
     measured tail and are part of why every model over-breaches.

INFERENCE  (bounded interpretation)

  Once you know the size of the price shock, the current volatility level, the
  market state, that news arrived and how much of it arrived, the tone of that
  news carries no further information about tomorrow's lower tail - at this
  horizon, on this panel, at calendar-date timestamp resolution.

  The design is not blind: the same rows, the same bootstrap and the same loss
  detect the arrival effect at overwhelming confidence.  Attention is
  informative about tail risk; sentiment, conditional on attention, is not.

OPEN LIMITATIONS

  - Timestamp coarsening.  FNSPID gives calendar dates, so a whole session
    elapses before the forecast.  A null here says nothing about intraday.
  - Survivorship and universe.  37 of 574 tickers end before 2023-12, and the
    ticker-linked cohort mixes operating firms with ETFs.
  - Calibration.  No model here is correctly calibrated, so this is a relative
    ranking, not a validated risk system.
  - Price adjustment.  See item 6; a cleaned price source would be the single
    highest-value upgrade to this panel.
  - One specification.  One volatility filter, one tail parameterisation.

NOT CLAIMED

  No causality.  No tradeable edge.  No claim about investor behaviour.  No
  claim of novelty.  A null is a valid result.
```
