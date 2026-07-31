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
