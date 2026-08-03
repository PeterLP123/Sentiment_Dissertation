# Invalidated and superseded runs

This ledger preserves negative and invalidated evidence instead of silently
overwriting it. Generated copies live under `final_experiments/outputs/`, which
is intentionally gitignored; this source-controlled note records why they must
not be cited.

## 2026-07-31 pre-rigour-repair snapshot

Generated snapshot:
`outputs/superseded/20260731_pre_rigour_repair/`

Status: **invalidated; do not quote its numerical conclusions.**

Reasons:

1. `04_surprise` indexed the market-model abnormal-return outcome at the
   backward return ending on the news session, although the declared estimand
   was the forward open(t) to open(t+1) return. Its initial-reaction control was
   shifted one additional session into the past.
2. The development strategy grid retained up to ten January 2020 return
   sessions to unwind December 2019 formations. The frozen strategy selection
   therefore read evaluation-period outcomes.
3. `03_aggregation` used sign books for its canonical portfolio table. This
   made non-negative signals such as dispersion and negative share one-sided,
   and collapsed sign-preserving rules onto identical books.
4. The reported “IC” pooled ranks across the full sample and merely clustered
   its standard error by date. It was not a daily cross-sectional IC.
5. `07_strategy_analysis` interpreted descriptive event-time paths as a real
   reversal without dependence-aware spread inference or correction across the
   displayed signal-by-lag family.
6. Transaction costs were labelled “per side” but multiplied by half-L1
   turnover only once, undercharging a gross-1 long/short book by a factor of
   two. The repaired convention charges `2 × turnover × per-side rate`.

The repaired run uses exact exchange-session returns, a strict split boundary,
daily cross-sectional Spearman IC with HAC inference, canonical dollar-neutral
cross-sectional rank books, and date-block/BH inference for event-time spreads.
It supersedes this snapshot only after all affected notebooks have rerun and
their outputs have passed the focused validation suite.

## 2026-08-03 pre-threshold-neutrality repair

Recoverable source state: `c32c74165dc803d70f9027cc74bff0666089b312`
(`final_experiments/10_thresholds.ipynb`).

Status: **invalidated; do not quote its evaluation portfolio results.**

The gate zeroed rejected names and then gross-normalised the surviving book as
a whole. If only one rank direction survived, the result was a directional
market bet rather than the declared dollar-neutral trade/no-trade strategy.
Mean evaluation net exposures ranged from −0.978 to +0.112, so the old net
Sharpes (fixed band −3.135; logistic −3.699; gradient boosted −1.938; MLP
−1.827; shuffled-label MLP −1.205) do not estimate the declared strategy.

The repaired implementation normalises surviving positive and negative legs
separately to +0.5 and −0.5 gross and sends one-sided selections flat. Notebook
10 was then rerun in full. This is one valid evaluation disclosure after one
void predecessor; it does not restore a pristine holdout.

## 2026-08-03 pre-liquidation and unmatched-activity threshold table

Recoverable source state: `ef9a40714c0d885d4e2a2b913160fa23113d4fd4`
(`final_experiments/10_thresholds.ipynb`).

Status: **superseded for accounting and comparability; preserve the historical
band but do not quote the old portfolio table.**

The neutrality-repaired threshold backtest still received the final interval's
return without paying to close its last held book. This understated costs,
especially for sparse gates. In addition, learned-gate cutoffs had to average
at least five active names and trade on at least half of validation sessions,
while the historical fixed-band sweep had no activity floor. Its selected band
`0.4` traded on only 2.98% of validation sessions, so it was not an activity-
matched comparator.

The repaired table charges final liquidation in every arm and retains band
`0.4` under the explicit label “historical fixed band.” A separate fixed-band
selection under the same activity floor chooses `0.0`. Applying that new rule
to the already-open evaluation block is an authorised iterative/retrospective
recomputation, not a pristine holdout result. Cash remains the deployment
comparator and is best in the recomputed table.
