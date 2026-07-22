# FinBERT strategy search — 2026-07-22

## Decision

**No newly searched FinBERT construction improved on the existing one-session
mid-cap event baseline after realistic trading costs.** The rank-reversal and
persistent-signal search results are preserved nulls, not tuned trading rules.

The acceptable outcome for the day is therefore a clean, newly identified
artifact of an existing rule:
`recent-news-midcap-finbert-event-v1-dfdd88113a5f`. It returned +3.419% with
0.780 Sharpe in development and +4.004% with 0.593 Sharpe in evaluation after
10 bps per side. This sample was previously explored and its bootstrap interval
crosses zero, so it is retained as an exploratory positive-Sharpe baseline, not
as newly discovered or deployable alpha. The complete rules and audit are in
[`recent_news_midcap_finbert_strategy.md`](recent_news_midcap_finbert_strategy.md).

The completed FNSPID study still supports a narrower claim: FinBERT contains
information about next-session absolute abnormal returns and negative-tail
risk. That finding does not, by itself, establish an implementable directional
return strategy.

## Frozen LSEG rank-reversal diagnostic

The source-controlled v3 protocol is
[`lseg_sector33_finbert_rank_reversal_v3.toml`](../configs/strategy_research/baselines/lseg_sector33_finbert_rank_reversal_v3.toml).
It was specified after the level-signal result had been inspected and is
therefore explicitly marked as post-result, previously explored, and requiring
independent confirmation.

- Run ID: `lseg-sector33-finbert-rank-reversal-v3-bb0a6569afe8`.
- Construction: one-session-lagged, five-observation firm-level FinBERT rank;
  contrarian long bottom quintile and short top quintile.
- Evaluation: 32 active sessions from 2026-05-14, 10 bps per dollar traded.
- Gross cumulative return: -12.358%.
- Net cumulative return: -14.467%.
- Net Sharpe: -10.719.
- Maximum drawdown: -14.787%.
- Total turnover: 24.246.
- The same-direction one-session comparator had net Sharpe -4.103.
- The five-session block-bootstrap interval was strictly negative versus cash
  and versus the comparator.

The ignored local result is under
`results/strategy_research/finbert_strategy_v3/lseg-sector33-finbert-rank-reversal-v3-bb0a6569afe8/`.
It must not be retuned or overwritten.

## Mid-cap persistent-signal development gate

The separate `recent-news-finbert-persistence-v1` gate tested only sessions
before 2 January 2026 on the 22-stock mid-cap corpus. It declared three
absolute-score quantiles (0%, 50%, 70%) and four holding periods (1, 3, 5, 7
open-to-open sessions), for 12 candidates. Every portfolio required at least
two long and two short names, used 100% gross dollar-neutral exposure, and paid
10 bps per dollar traded.

No candidate passed the frozen requirement of at least 20 active days, positive
full-development Sharpe, and positive Sharpe in at least two of three
chronological subperiods. The least weak positive candidate used the median
absolute-score gate and a five-session hold, but produced only 0.201 net Sharpe,
0.436% cumulative net return, 22 active days, and one negative subperiod. Its
post-2026 and separate 33-stock evaluation periods were therefore not opened.

The ignored hash-pinned immutable grid is
`results/strategy_research/recent_news_finbert_persistence_v1/development_protocol_v2.json`.
The source-controlled runner deliberately refuses to evaluate a failed
development protocol.

## Full FNSPID evidence used for the redesign

The resumable local checkpoint contains 1,640,796 scored headlines, 574
coherent firms, 736,596 firm-sessions, and a 2011–2023 window. Finalisation
reused the completed checkpoint without rescoring.

The mean directional signal was marginal and had no Benjamini–Hochberg
significant horizon. The stronger result was risk-related: FinBERT negative
share predicted next-day absolute abnormal return, while positive FinBERT mean
was associated with lower negative-tail probability in the 2017–2023
evaluation regression. The incremental R-squared was small. These are forecast
relationships, not portfolio results.

The checkpoint and its licensed headline text remain ignored under
`reports/loop_moment2_finbert_20260719/` and must not be committed or
redistributed.

## Development-only portfolio diagnostics

These constructions were tested serially after observing earlier results.
They are recorded to prevent selective reporting and must not be described as
confirmatory evidence. Returns use adjusted-close prices, drifted portfolio
weights, final liquidation, and 10 bps per dollar traded.

| Construction | Gross result | Net result | Turnover/result note |
| --- | ---: | ---: | --- |
| Five-session trailing-news rank, rebalance every five sessions | Sharpe -0.309; cumulative -9.17% | Sharpe -2.135; cumulative -47.72% | Mean daily turnover 0.314 |
| Five overlapping one-day cohorts | Sharpe -0.254; cumulative -5.66% | Sharpe -3.143; cumulative -49.53% | Mean daily turnover 0.356 |
| Daily top/bottom FinBERT deciles, one-session hold | Sharpe 0.218; cumulative 11.18% | Sharpe -5.178; cumulative -95.70% | Mean daily turnover 1.849; the small gross edge is not economic |
| Monthly long-only top FinBERT quintile | — | Sharpe 0.587; cumulative 99.79%; max drawdown -39.37% | Mean daily turnover 0.0676 |
| Monthly long-only all news-eligible firms | — | Sharpe 0.666; cumulative 128.45%; max drawdown -41.44% | Mean daily turnover 0.0059; the unscreened comparator wins |

The positive monthly long-only result is mostly equity-market exposure. The
FinBERT screen reduced maximum drawdown by about 2.1 percentage points but
reduced Sharpe by 0.079 and cumulative return by about 28.7 percentage points
relative to the same investable universe.

A separate recent-news persistence grid also stopped at its development gate,
so its post-boundary evaluation remained sealed. None of 12 frozen
threshold/holding-period candidates met all requirements. The strongest
candidate with the required 20 active days used the median absolute-score gate
and a five-session hold: 22 active days, net Sharpe 0.201, cumulative net return
0.44%, but positive Sharpe in only one of three chronological subperiods. A
three-session version was positive in two subperiods but traded on only 11 days.

## Research implication

Use the one-session mid-cap event rule only as the current exploratory baseline;
do not present FinBERT as confirmed standalone deployable alpha from these
samples. A defensible next test is either a predeclared concentration cap on a
genuinely unseen future Reuters period, or an incremental risk overlay on an
already justified return signal. The current 2011–2023 FNSPID and 2025–2026
LSEG samples have now influenced strategy design and cannot provide that
confirmation.
