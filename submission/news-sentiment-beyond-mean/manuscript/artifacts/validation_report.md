# Derived-artifact validation report

Overall assessment: **ready to use with the stated caveats**.

- Notebook 75 was opened exactly once in the source repository and promoted
  without re-execution. Its all-firm-day testing-period coefficient is
  **+0.00583** and the predeclared
  testing-minus-training contrast is
  **+0.01414**.
- The prospective approximation gave
  **35.4%** power to recover
  the training-period baseline effect at the corrected two-test threshold. Realised precision
  implies **26.5%** ex-post
  power at the same threshold. The failed testing-period family is therefore not proof
  of a zero effect; the separate contrast is consistent with instability in the
  measured baseline pipeline association without identifying its cause.

## Inputs and grain

- All inputs are committed aggregate CSV files under `experiments/results/`.
- FNSPID and LSEG rows are kept separate. No row-level story text is read.
- Coefficients share shifted average-tie percentile-rank units, but FNSPID uses abnormal returns and
  LSEG uses raw returns. The cross-source figure states this difference.
- The FNSPID same-session news risk rule uses current-session news; only its
  normalisation baseline is lagged. Its circular shifts are schedule-alignment
  diagnostics, not evidence that the rule was observable before the return.
- The LSEG economic comparison uses one 167-session block and the
  symbol-matched constant selected in the training period as comparator.

## Calculation spot checks

- Full-control translation over the observed pooled tied-rank spread
  (0.56056):
  **-0.5124 percentile points**.
- Training-period timing coefficients: intraday
  **-0.01029** and post-close
  **+0.00155**. The separate
  previous-open probe is **-0.01926**.
- FNSPID 2020--2023 maximum drawdown: control
  **-15.73%**, overlay
  **-16.36%**.
- Charged cost / best break-even: **16.0047x**.
- LSEG MDE multiples: **7.6054x** in the earlier period and **13.5555x** in the later period.
- Under square-root precision scaling, those ratios imply approximately
  **57.8x** and
  **183.8x** as much
  time-series information for an FNSPID-sized MDE. This is an approximation,
  not a forecast of the exact sessions required.
- LSEG rule drawdown improvement versus fully invested: **1.6781 percentage points**.
- LSEG rule drawdown disadvantage versus matched constant: **0.6273 percentage points**.
- LSEG rule ending-wealth difference: **USD -30869.79 per USD 1m**.
- Conditional coefficient by same-day story count:
  **-0.00529** at n >= 2 and
  **-0.00183** at n >= 3, which retains
  **22.0%**
  of the full-sample estimate.
- The two largest LSEG earlier-period risk-off episodes supply
  **65.6%**
  of the block's squared-downside reduction.
- The first three economic components reconcile to the saved arithmetic
  difference within USD 1 per USD 1m (difference:
  **USD 0.61**).
  The figure uses an exact residual from their sum to compounded ending wealth.

## Required caveats

- The tied-rank translation is a linear interpretation of a rank coefficient,
  not a basis-point return forecast. The superseded hypothetical 0.8 move is not
  attainable on the observed regressor.
- MDE ratios describe power; they do not prove the LSEG coefficient is zero.
- The firm-level LSEG comparison is retrospective and selected on training-period
  data. It does not meet all registered economic-value criteria.
- The plots and tables summarise existing evidence and add arithmetic translations. They
  are not new model searches or confirmatory tests.
- Notebook 85 close-to-close, FF3 residual and soft-mass coefficients are a
  three-member post-hoc family frozen after the headline result. They are not
  confirmation of H1.
- Notebooks 86 and 87 are post-review diagnostics. The intraday and prior-open
  results require withdrawal of predictive-window specificity because story
  arrival cannot be reconstructed at row level.
