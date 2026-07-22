# Recent-news 33-stock FinBERT robustness protocol

## Decision

Test whether the accepted one-session FinBERT event rule generalises from the
22-stock mid-cap Reuters cohort to the existing 33-stock, 11-sector Reuters/LSEG
cohort. Only the stock universe and observation window change. The FinBERT
revision, event eligibility, availability buffer, hard-label aggregation,
portfolio construction, holding period, transaction cost, cash rule and
inference settings remain identical to `recent-news-midcap-finbert-event-v1`.

This is a universe-robustness test, not a new parameter search. The 33-stock
sample has already been used for lexicon, FinBERT sign and rank-reversal work,
so it is marked `previously_explored`. No result from this cohort can be called
an untouched confirmation.

## Frozen hypothesis and acceptance rule

The hypothesis is that company-specific positive and negative FinBERT headline
labels contain a short-lived, same-direction return signal beyond the original
mid-cap universe. The run passes this robustness gate only if all of the
following hold at 10 basis points per dollar traded:

- net cumulative return and annualised Sharpe are positive in development;
- net cumulative return and annualised Sharpe are positive in evaluation;
- each chronological block contains at least 20 active sessions;
- approximate break-even cost is above 10 basis points per side in both blocks;
- all point-in-time, price-coverage, score-provenance and artifact-integrity
  checks pass; and
- deterministic replay validates and reuses the immutable result.

A positive combined Sharpe cannot rescue a negative chronological block.
Breadth and concentration are diagnostics, not post hoc acceptance filters.
They will be compared with the original strategy's 50% maximum single-name
absolute weight and 23 two-name active sessions.

## Frozen data and split

- News collection: `lseg_us_sector_33_6m`, covering 33 US stocks across 11
  sectors from 26 December 2025 to 26 June 2026.
- Price panel: 33 complete symbol histories from 14 November 2025 to 1 July
  2026, using split-adjusted, non-dividend-adjusted opens.
- Development/evaluation boundary: `2026-05-14`, retained from the earlier
  sector-33 work rather than selected after viewing this rule's results.
- Sample status: `previously_explored`.
- New LSEG requests authorised for this run: zero. Existing local news and
  prices are sufficient, preserving the 10,000-request allowance.

The older sector-33 score artifact used FinBERT revision
`7db323f79b751944bcfa66298ec06977e4518306`. It is not valid for an unchanged
replication of the accepted rule. Before opening the frozen backtest, every
unique company-matched headline will therefore be relabelled locally with the
accepted revision
`4556d13015211d73dccd3fdd39d39232506f3e43`. Scoring must be cache-only,
resumable, price-independent and hash-manifested. Licensed headlines and score
artifacts remain ignored and local.

## Frozen trading rule

1. Retain the earliest eligible occurrence for each `story_family × symbol`.
2. Require an explicit single-company target and remove market-price or
   technical-analysis headlines.
3. Map the pinned FinBERT headline argmax to `+1`, `0` or `-1`.
4. Add 15 minutes to recorded availability and use the first XNYS adjusted
   open strictly after that time.
5. Average hard labels by symbol and execution session; trade the sign.
6. Require at least one long and one short. Otherwise hold zero-return cash.
7. Allocate 50% gross equally to longs and 50% gross equally to shorts.
8. Hold for exactly one adjusted-open-to-adjusted-open session with no carry.
9. Charge 10 bps times `sum(abs(target_weight - current_weight))`, including
   final liquidation.

The machine-readable contract is
`configs/strategy_research/baselines/recent_news_sector33_finbert_event_v1.toml`.
No threshold, direction, minimum breadth, holding period, split date or cost
may change after this protocol is committed.

## Material passport

- Origin skill: academic-research-suite / experiment-agent planning
- Origin date: 2026-07-22
- Verification status: protocol frozen; experiment not yet run
- Version label: `recent_news_sector33_finbert_protocol_v1`
- Inherited materials: cleaned licensed Reuters/LSEG corpus, adjusted-open
  price panel, v1 baseline implementation and pinned local FinBERT snapshot
- New source materials: this protocol and the sector-33 TOML contract
- Planned generated materials: local score CSV and manifest, immutable
  aggregate backtest outputs, validation report and updated HTML explainer

## Required reporting

Report the outcome even if it fails. The handoff must include development,
evaluation and combined return/Sharpe/drawdown/turnover/cost metrics; activity
and concentration diagnostics; bootstrap uncertainty; the 11-class research
risk scan; deterministic replay evidence; exact hashes; and a direct comparison
with the accepted 22-stock result. No failed result may be overwritten or
silently re-tuned.

## Result fixed on 22 July 2026

The corrected immutable run resolved to
`recent-news-sector33-finbert-event-v1-4b9baebb01fa` and **failed** the
predeclared robustness gate. Development returned -9.159% net with -2.491
Sharpe; evaluation returned -5.035% net with -3.682 Sharpe. Approximate
break-even costs were 3.31 and -0.53 bps per side, both below the frozen 10 bps
charge. Activity and integrity gates passed, and a second identical command
validated/reused the result.

The result was preserved without retuning. See
`docs/recent_news_sector33_finbert_result.md` for the comprehensive metrics,
comparison with the retained mid-cap baseline, uncertainty, 11-class risk
audit, reproduction hashes and next gate.
