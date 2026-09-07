# Research design

## Recommended angle

**Working title:** *A Hard Negative-Story Threshold: Training-Period Association, Temporal Non-Replication and Economic Limits*

**Training-period question:** Is the hard negative-story share conditionally
associated with the assigned-session open-to-open abnormal-return rank beyond
rank-linear mean sentiment and news volume, and which component of the assigned
window contains the association?

**Testing-period question:** Does the unchanged baseline association,
conditional on mean sentiment and news volume, remain negative in the frozen
2020--2023 FNSPID testing period?

The contribution is not that “sentiment predicts returns.” It is the disciplined
separation of a selected training-period association from timing identification,
frozen baseline replication, economic scale and cross-source transfer. The
post-review decomposition places the association intraday and also finds a
previous-open effect. Missing row-level availability times therefore prevent a
predictive reading. The baseline does not replicate in 2020--2023; the era
contrast is consistent with instability in the measured pipeline association
without identifying its cause.

## Data regimes

| Regime | Role | Time use | Pooling rule |
|---|---|---|---|
| FNSPID, 570 priced firms and 715,546 news-bearing firm-days | Primary statistical spine and risk translation | Training through 2019-12-31; testing from 2020-01-01 | Never pool with LSEG |
| LSEG sector-33, backward block | Longer same-universe transfer check | Separate block | Never pool with recent LSEG or FNSPID |
| LSEG sector-33, recent block | Current Reuters portability and economic-value check | Separate block | Never pool with backward LSEG or FNSPID |

The chronological split is an honest training/testing partition, not a pristine untouched holdout: some earlier project design work observed the later years. Confirmatory language must therefore be avoided unless a new prospective sample is collected.

## Primary estimand

For each trading session, transform every variable to its average-tie percentile rank and subtract one half. Regress the transformed assigned-session open-to-next-open return on:

1. transformed mean continuous sentiment;
2. transformed negative-story share;
3. transformed `log(1 + story count)`;
4. in the price-path test, transformed lagged one-session return, lagged five-session return and 20-session volatility.

The daily coefficient on negative-story share is then averaged through time. Uncertainty uses HAC with five lags. Dates with fewer than ten complete firms or a rank-deficient design are excluded.

This estimates an incremental conditional association. It does not identify a causal effect of news.

Post-review timing tests hold the regressors fixed and replace the outcome with
assigned open-to-close, assigned close-to-next-open, previous-close-to-assigned-
open, and a separate previous-open-to-assigned-open probe. These tests diagnose
where the association is measured; they cannot reconstruct story arrival.

## Primary evidence

- Published conditional model: coefficient **-0.00831**, 95% HAC interval **[-0.01410, -0.00252]**, `p = 0.00491`, across 2,264 sessions and 512,149 complete firm-days.
- Full price-path controls: coefficient **-0.00914**, 95% HAC interval **[-0.01466, -0.00362]**, `p = 0.00118`.
- Assigned open-to-close: **-0.01029**, BH `q = 0.0011`; assigned close-to-next-open: **+0.00155**, `q = 0.595`.
- Previous-open-to-assigned-open probe: **-0.01926**, 95% HAC interval **[-0.02580, -0.01272]**, `p = 7.9e-9`. This makes assignment/timing error a live alternative explanation.
- The coefficient becoming slightly more negative after price controls argues against the result being only short-term reversal or lagged momentum in disguise.
- Frozen testing-period all-firm-day coefficient: **+0.00583**, 95% HAC interval **[-0.00426, +0.01593]**, BH `q = 0.515`, across 998 sessions and 203,393 firm-days.
- The predeclared testing-minus-training difference is **+0.01414**, 95% HAC interval **[+0.00250, +0.02579]**, `p = 0.0173`. This is consistent with a change in the measured baseline association across the two eras; it is not inferred from one period being significant and the other not.
- The contrast compares the baseline mean/count specification (**-0.00831**) across eras. The full price-path coefficient (**-0.00914**) is training-period robustness and is not the contrast's training-period input.
- Approximate prospective power for the stable training-period effect was **35.4%** at the first BH-2 hurdle. The observed testing-period HAC standard error was 15.8% larger than projected, implying **26.5% realised, ex-post power** and an MDE of **0.01588**. The failed family is not proof of an exact zero.
- The corrected full-control effect-size translation uses the observed tied-rank spread and is **-0.512 return-rank percentile points**, not the unattainable earlier 0.8-rank calculation.
- Model-free double sort: the pooled high-minus-low negative-share spread is **-0.618 bps per session**, but its 95% interval **[-1.510, 0.275]** crosses zero. This communicates the economic scale and uncertainty without relying on a regression.

## Multiplicity and nulls

The initial aggregation family compared nine rules under one inference regime and applied Benjamini-Hochberg correction. Negative-story share was the only corrected survivor. The repository also preserves the larger result ledger: 56 declared families, 264 reported tests or decisions, 27 BH survivors, and 36 entirely null families.

Nulls are part of the argument:

- learned trade/no-trade thresholds do not beat cash or the fixed rule;
- story-type, publisher and earnings conditioning do not produce a stable incremental rule;
- LSEG FinBERT conditional coefficients are negative but imprecise;
- recent tuned LSEG aggregate and firm-level risk rules fail their economic-value gates;
- prompt-engineered and structured-reaction scores fail the training-period selection gates.

Minimum detectable effects distinguish “evidence of no useful effect” from “not enough power.” For the two primary FinBERT LSEG transfer blocks, the detectable coefficient magnitudes were approximately 7.6 and 13.6 times the FNSPID estimate.

## Economic-value tests

Economic value is assessed after costs, turnover, matched-exposure comparators, tail metrics, block-bootstrap inference, half-sample checks and leave-one-company-out checks where applicable.

The boundary is clear:

- daily cross-sectional trading is too expensive; the best break-even cost in the aggregation comparison is about 0.625 bps per side against a charged 10 bps;
- the FNSPID mapped-session modifier has worse 2020--2023 maximum drawdown than HAR (**-16.36%** versus **-15.73%**), despite its squared-downside alignment result;
- the broad LSEG tuned overlay does not improve the frozen HAR strategy in the 167-session testing period;
- the firm-level LSEG rule cuts raw drawdown relative to an always-invested benchmark, but loses to exposure-matched controls, turns over roughly 100 times annually and fails every complete economic-value gate;
- prompt variants and the structured price-reaction score have negative gross and net performance in their training selection window.

## Risk-management interpretation

The defensible FNSPID risk claim is bounded. Aggregate negative-story pressure is
aligned with lower squared downside in one testing period when mapped directly
onto an independently motivated HAR volatility target. Current mapped-session
news sets the same session's modifier, so this is not an ex-ante timing test.
Circular shifts, 44 leave-one-episode-out checks and a February--April 2020
exclusion support the narrow alignment result. The return advantage over an
ex-post exposure-matched constant is imprecise, overall drawdown is worse, the
exact rule does not activate in recent LSEG, and tuned replacements fail
matched-comparator tests.

Write this as a regime-specific proof of concept and a portability limitation, not as a production risk signal.

## Claim ladder

1. **Strongest boundary:** the training-period association is measured intraday and before the assigned open as well as over the full assigned window; predictive timing is not identified.
2. **Temporal evidence:** the selected baseline fails a frozen replication, and the coefficient differs across eras. This is consistent with pipeline-level instability; its cause is not identified.
3. **Training-period evidence:** negative-story share is conditionally negative after mean, count and recent-price controls through 2019.
4. **Useful but descriptive:** the training-period economic magnitude is small and noisy in a model-free sort.
5. **Negative result:** realistic transaction costs prevent a daily directional strategy.
6. **Bounded secondary result:** FNSPID squared-downside alignment survives concentration checks, but the state is contemporaneous and overall drawdown is adverse.
7. **Portability boundary:** LSEG precision and economic gates are insufficient; prompt engineering does not rescue them.

These seven levels should remain separate throughout the manuscript.
