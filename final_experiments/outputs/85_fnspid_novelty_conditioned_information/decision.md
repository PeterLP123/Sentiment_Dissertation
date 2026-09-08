# Notebook 85 decision — supported

- Family: 3 declared tests, BH q=0.05. Survivors with the declared sign: 1 (T3).
- Block: FNSPID development only, through 2019-12-31. The evaluation block was not loaded.
- Measurement: 0.0946 of development stories are already-known at Jaccard 0.50; 53,367 of 512,153 firm-days are mixed.
- The repetition screen is uncalibrated: Notebook 02's blinded 180-story audit has no human labels, so an unrecognised paraphrase counts as new and the already-known leg is a lower bound.
- The contrast is identified on the busier end of the panel, because a mixed firm-day needs both legs present.

## Primary rows

```
test  sessions  estimate  hac_ci_low  hac_ci_high    hac_p     bh_q
  T1      1920 -0.015195   -0.039802     0.009412 0.226172 0.226172
  T2      1920 -0.042235   -0.082070    -0.002401 0.037700 0.056549
  T3      2263 -0.008815   -0.013872    -0.003758 0.000634 0.001902
```

## Declared story-count strata

```
stratum test  firm_days  sessions  estimate    ci_low   ci_high        p
 n >= 2   T1      53367      1920 -0.015195 -0.039802  0.009412 0.226172
 n >= 5   T1      17859       615 -0.027215 -0.110251  0.055821 0.520634
 n >= 2   T2      53367      1920 -0.042235 -0.082070 -0.002401 0.037700
 n >= 5   T2      17859       615 -0.045596 -0.142112  0.050920 0.354487
 n >= 2   T3     264683      2263 -0.008273 -0.014567 -0.001979 0.009987
 n >= 5   T3      48862      1880 -0.009781 -0.025166  0.005605 0.212793
```
