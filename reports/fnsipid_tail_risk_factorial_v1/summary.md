# FNSPID tail-risk 2×2 sensitivity attribution

## Verdict

The **annual expanding GJR-GARCH refit accounts for essentially all of the VaR hit-rate improvement** between v1 and v2. The minimum-absolute-return sensitivity over candidate adjusted/raw-return gaps improves standardised-return dispersion and ES identification, but does not improve the M1 VaR hit rate by itself.

Neither sensitivity changes the substantive result: news arrival and volume improve paired FZ0 loss beyond price state, while FinBERT semantics and signed tone do not add a detectable increment.

## Frozen design

| Cell | Candidate-gap sensitivity | Volatility-filter refit |
| --- | --- | --- |
| v1 | none | frozen 2011–2016 fit |
| price-only | minimum-absolute-return rule | frozen 2011–2016 fit |
| refit-only | none | annual expanding window ending before each evaluation year |
| v2 | minimum-absolute-return rule | annual expanding window ending before each evaluation year |

Every cell uses identical notebook source, helper and notebook-pair hashes, plus the same corpus, timing rule, split, model formulas, optimiser, bootstrap, seed and evaluation sample: 561 firms, 958,461 evaluation rows, 415,758 news-bearing rows and 1,758 target dates.

Timing limitation: the hashed upstream policy mapped 2,518,109 date-only/exact-midnight rows by **strictly next XNYS session** and 5,660 precise-timestamp rows by **XNYS session containing the UTC minute, otherwise next session** before windowing and deduplication. Original timestamps/type flags are absent from the completed checkpoint, so a uniformly date-only rule cannot be verified per retained event.

## Calibration attribution

| Cell | M1 VaR hit rate | M2 VaR hit rate | std$(z)$ | M1 ES residual | rejected coverage tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 3.0663% | 3.0647% | 1.1173 | 0.00462 | 8/8 |
| price-only | 3.0687% | 3.0655% | 1.1125 | 0.00440 | 8/8 |
| refit-only | 2.9793% | 2.9855% | 1.1013 | 0.00358 | 8/8 |
| v2 | 2.9819% | 2.9822% | 1.0962 | 0.00336 | 8/8 |

Relative to v1:

- Candidate-gap sensitivity alone changes the M1 hit rate by **+0.0024 percentage points**, changes std$(z)$ by -0.0049, and changes the ES residual by -4.7%. These rows are candidates, not proven errors, because authoritative corporate-action metadata are unavailable.
- Annual refitting changes the M1 hit rate by **-0.0870 percentage points**, closing 15.4% of the gap to nominal 2.5%; it changes std$(z)$ by -0.0160 and the ES residual by -22.5%.
- v2 changes the M1 hit rate by **-0.0844 percentage points**, changes std$(z)$ by -0.0211, and changes the ES residual by -27.4%.
- The 2×2 interactions are reported in `factorial_effects.csv`; interpretation remains bounded by absolute calibration tests.

Calibration is improved, not fixed: all eight conditional-coverage/calibration tests reject in every cell.

## Forecast-value conclusions

Paired FZ0 differences on news-bearing evaluation origins; 95% moving-block bootstrap over target dates, block length 20, 2,000 replications, seed 20260728:

| Cell | M1 − M0: news arrival | M2 − M1: semantics | M2 − M2-intensity: signed tone |
| --- | --- | --- | --- |
| v1 | −0.00972 [−0.01775, −0.00386] | −0.000307 [−0.002249, +0.001316] | +0.000274 [−0.000609, +0.001627] |
| price-only | −0.01040 [−0.01986, −0.00364] | −0.000464 [−0.002620, +0.001293] | +0.000429 [−0.000584, +0.002123] |
| refit-only | −0.00894 [−0.01714, −0.00295] | −0.000351 [−0.002284, +0.001284] | +0.000307 [−0.000584, +0.001699] |
| v2 | −0.00960 [−0.01930, −0.00275] | −0.000511 [−0.002664, +0.001253] | +0.000471 [−0.000551, +0.002203] |

Arrival excludes zero in all four cells; semantics and signed tone cover zero in all four. The conclusion is robust to both declared sensitivities separately and jointly.

## Execution and verification

All four cells ran concurrently from the same source bytes on UCL host `chub-l.cs.ucl.ac.uk` under Python 3.12.13. Input SHA-256 hashes match the frozen local FinBERT checkpoint and price archive. Every bundle reports `completed`, passes its notebook assertions (v1 35/35, price_only 35/35, refit_only 39/39, v2 39/39) and 10/10 gates, contains no executed-notebook errors, and verifies every manifest-listed output hash.

The workstation has an NVIDIA GeForce RTX 4070 Ti SUPER, but this notebook is a NumPy/SciPy/`arch` CPU workload with no CUDA path. The run record preserves observed GPU state at launch and completion; these notebooks did not allocate it.

## Files

- `calibration_comparison.csv` — four-cell calibration levels
- `factorial_effects.csv` — price-only, refit-only, combined and interaction differences from v1
- `contrast_comparison.csv` — frozen paired-loss contrasts and confidence intervals
- `ucl_run_record.txt` — remote host, source hashes and observed GPU state
- Source bundles: `reports/fnsipid_tail_risk_core_{v1,price_only,refit_only,v2}/`
