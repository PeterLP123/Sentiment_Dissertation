# Notebook 86 verdict — not alpha

- Development alpha gate: 1/5 conditions pass.
- Q5−Q1 already-known spread: -1.93 bps/session [-3.83, -0.10] against a 20-bps round trip.
- M2 price-path controls: coefficient -0.00774 (same sign: True).
- The 2020–2023 replay is retrospective and exploratory; the block was already opened by Notebook 75, and T3 was selected on development.

## Development books

```
                   book  sessions  mean_gross_bps  mean_net_bps  sharpe_gross  sharpe_net  breakeven_bps_per_side
     B1 raw known_share      2264        1.296196    -12.171554      1.040325   -9.753868                0.962445
B2 conditional residual      2263        0.746320    -14.433961      0.707404  -13.650725                0.491638
```

## Alpha gate

```
                                                condition      value  passes
                            net Sharpe > 0 at 10 bps/side -13.650725   False
                                break-even >= 10 bps/side   0.491638   False
                    bootstrap lower bound on mean net > 0 -15.131206   False
                            net Sharpe > 0 in both halves -14.556278   False
coefficient survives M2 price controls with the same sign  -0.007737    True
```

## Evaluation replay

```
                   book  sessions  mean_gross_bps  mean_net_bps  sharpe_gross  sharpe_net  breakeven_bps_per_side
     B1 raw known_share       998       -0.279768    -13.909060     -0.087668   -4.360531               -0.205269
B2 conditional residual       997       -0.192201    -15.418711     -0.061946   -4.969007               -0.126228
```

## Coefficient by block

```
      block  sessions  estimate    ci_low   ci_high        p
development      2263 -0.008815 -0.013872 -0.003758 0.000634
 evaluation       997  0.003366 -0.004609  0.011341 0.408125
```
