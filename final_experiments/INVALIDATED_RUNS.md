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

## 2026-08-05 inherited FinBERT metadata contract

Status: **superseded before any 44-company return result was produced.**

The exact-hash FinBERT export correctly preserves 888,155 reusable labels, but
some rows also preserve scorer-blind metadata from the older seed population:
4,222 company-association fields, 5,805 first timestamps, and 269 direct/context
flags differ from the current merged 44-company population. Reusing those
fields would misalign some inherited labels to companies or return sessions.

Notebook 12 records the discrepancy. Notebook 13 takes FinBERT label and score
probabilities by exact hash, but takes association, timestamp, and filter fields
from the current 44-company population enumerated by the completed OpenRouter
run. No previous expanded-corpus return result existed, so no numerical result
is invalidated; this entry prevents the stale metadata contract being revived.

## 2026-08-05 Notebook 14 pre-override-fix distance state

Generated snapshot:
`outputs/superseded/20260805_notebook14_pre_override_fix/`

Status: **distance h3/h5 arms invalidated; material arms unaffected.**

The first execution forward-filled `distance_signal` independently. When a new
selected material event had no cash-flow-distance phrase, pandas therefore
carried the older classified event through the newer event. That violated the
frozen rule that a new selected event overrides the prior state. The repaired
state builder starts a fresh segment at every selected material event; an
unknown-distance event now clears the older distance state. The full six-arm
notebook was rerun so its embedded plots, tables, manifest, and interpretation
all come from the repaired implementation. The superseded distance results must
not be quoted.

## 2026-08-05 Notebook 42 file-level price eligibility

Generated snapshot:
`outputs/42_fnspid_2010_preperiod_transfer/invalidated_v1_file_level_price_filter/`

Status: **invalidated; do not quote its performance results.**

The first backward-transfer execution treated a cohort symbol as eligible for
the 2010 sentiment cross-section whenever its FNSPID price file existed. The
canonical FNSPID panel instead requires a valid current adjusted open on each
firm-session before aggregation. The file-level shortcut admitted later IPOs
and other unavailable firm-sessions into the sentiment rank, then encountered
115 missing selected returns and forced 95 complete sessions to cash. Its 36
active-session gross/net Sharpes of −0.384/−2.475 do not estimate the frozen
construction and must not be used.

The repaired notebook applies the valid-current-open screen before computing
daily dispersion or ranks, without computing a return. It excludes 989
unavailable firm-sessions, passes the input-only gate with 59 active sessions,
and has zero missing selected returns after outcomes open. The repair was
chosen from panel-contract parity and return availability only, not return sign
or magnitude; the corrected negative result supersedes the invalid snapshot.
