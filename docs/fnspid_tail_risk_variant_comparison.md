# FNSPID tail risk: v1 versus v2 (declared defect repair)

## Verdict

The completed 2×2 attribution shows that the **annual expanding GJR-GARCH
refit accounts for essentially all of the VaR hit-rate improvement** between v1
and v2. The adjusted-price repair improves standardised-return dispersion and ES
identification, but does not improve the VaR hit rate by itself.

Neither repair changes the headline result. News arrival still beats price state
decisively; news sentiment still adds nothing measurable on top of it. v1
remains the pre-registered result, while the other three cells are declared
defect-attribution checks.

## Material passport

- Origin: `notebooks/fnsipid_tail_risk_core.py`; variants `v1`,
  `price_only`, `refit_only`, and `v2`
- Origin date: 2026-07-28
- Verification status: `VERIFIED` — v1 and price-only passed 33/33
  assertions; refit-only and v2 passed 37/37; every cell passed 10/10 gates,
  contains zero notebook errors, and verifies every manifest-listed output hash
- Version label: `fnsipid_tail_risk_factorial_v1`
- Bundles: `reports/fnsipid_tail_risk_core_{v1,price_only,refit_only,v2}/`

## The 2×2 design

| Cell | Adjusted-price repair | Volatility-filter refit |
| --- | --- | --- |
| v1 | none | frozen 2011–2016 fit |
| price-only | minimum-absolute-return repair | frozen 2011–2016 fit |
| refit-only | none | annual expanding window |
| v2 | minimum-absolute-return repair | annual expanding window |

Everything else—corpus, timing rule, splits, model formulas, tail
parameterisation, optimiser, bootstrap and seed—is identical.

The price repair affected 4,015 rows across 307 of 561 firms (0.22% of
firm-days). Annual refitting made 3,927 attempts: 3,888 accepted and 39 carried
forward using the previous vintage of the same specification.

All four cells use exactly 561 firms, 958,461 evaluation rows, 415,758
news-bearing rows and 1,758 target dates.

## Calibration attribution

| Cell | M1 VaR hit rate | M2 VaR hit rate | std$(z)$ | M1 ES residual | rejected coverage tests |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 | 3.0663% | 3.0647% | 1.1173 | 0.00462 | 8/8 |
| price-only | 3.0687% | 3.0655% | 1.1125 | 0.00440 | 8/8 |
| refit-only | **2.9793%** | 2.9855% | 1.1013 | 0.00358 | 8/8 |
| v2 | 2.9819% | **2.9822%** | **1.0962** | **0.00336** | 8/8 |

Annual refitting closes 15.4% of the M1 breach-rate gap to the nominal 2.5%.
Price repair alone slightly worsens that hit rate by 0.0024 percentage points,
although it improves std$(z)$ and the ES residual. Those improvements are
approximately additive with refitting; the factorial interactions are small.

Calibration is better, not fixed. Every conditional-coverage/calibration test
still rejects in every cell. The remaining undercoverage is therefore not
attributable to either diagnosed defect.

## Forecast-value conclusions: unchanged

Paired FZ0 differences on news-bearing evaluation origins, with 95% moving-block
bootstrap intervals over target dates (block 20, 2,000 replications, seed
20260728):

| Cell | M1 − M0: arrival | M2 − M1: semantics | M2 − M2-intensity: tone |
| --- | --- | --- | --- |
| v1 | −0.00972 [−0.01775, −0.00386] | −0.000307 [−0.002249, +0.001316] | +0.000274 [−0.000609, +0.001627] |
| price-only | −0.01040 [−0.01986, −0.00364] | −0.000464 [−0.002620, +0.001293] | +0.000429 [−0.000584, +0.002123] |
| refit-only | −0.00894 [−0.01714, −0.00295] | −0.000351 [−0.002284, +0.001284] | +0.000307 [−0.000584, +0.001699] |
| v2 | −0.00960 [−0.01930, −0.00275] | −0.000511 [−0.002664, +0.001253] | +0.000471 [−0.000551, +0.002203] |

Arrival excludes zero in all four cells. Semantics and signed tone cover zero in
all four. The substantive result is robust to both defects separately and
jointly.

## UCL execution

All four cells ran concurrently from identical source bytes on UCL host
`chub-l.cs.ucl.ac.uk` under Python 3.12.13, with the frozen local input hashes
reproduced. v1 and price-only passed 33/33 assertions; refit-only and v2 passed
37/37; every cell passed 10/10 gates, contains no executed-notebook errors, and
verifies every manifest-listed output hash.

The workstation has an RTX 4070 Ti SUPER, but this notebook is NumPy/SciPy/`arch`
CPU code and has no CUDA path. Another process occupied the GPU throughout the
run; these notebooks did not allocate it.

## Reproduce

```bash
run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
for variant in v1 price_only refit_only v2; do
  repro_dir="reports/reproductions/fnsipid_tail_risk_core_${variant}_${run_stamp}"
  FNSPID_TAIL_RISK_RUN_MODE=full \
  FNSPID_TAIL_RISK_VARIANT="$variant" \
  FNSPID_TAIL_RISK_OUTPUT_DIR="$repro_dir" \
    jupyter nbconvert --to notebook --execute \
      notebooks/fnsipid_tail_risk_core.ipynb \
      --output "../${repro_dir}/fnsipid_tail_risk_core.executed.ipynb" \
      --ExecutePreprocessor.timeout=-1
done
```

`FNSPID_TAIL_RISK_PRICE_ARCHIVE` supplies a remote approved copy of the frozen
price archive without changing its manifest hash. `FNSPID_TAIL_RISK_OUTPUT_DIR`
redirects scratch runs so the frozen bundles are not overwritten.

Every `repro_dir` must be new. The notebook rejects a non-empty output directory
rather than mixing current-run files with a prior bundle.

Detailed tables and the UCL run record are in
`reports/fnsipid_tail_risk_factorial_v1/`.

## Limitations this does not touch

Unchanged from the v1 write-up: calendar-date timestamp coarsening, survivorship,
the ticker-linked universe mixing operating companies with funds, and the
absence of 22 of the 25 largest US firms from the inherited balanced-panel
cohort. Only the price-adjustment and frozen-filter items are addressed here.
