# Recent-news sentiment persistence v1 — frozen protocol

## Decision before the robustness run

This is the first strategy candidate in the 22 July search with positive net
Sharpe in both chronological blocks of its selection evidence. The exact rule
is frozen here before applying it to the separate 22-stock mid-cap corpus.

The rule is post-selected from an already explored 33-stock result. The mid-cap
corpus has also been used in earlier headline studies, so the next run is a
robustness check rather than pristine confirmation. A positive result may
support an exploratory "acceptable recent-data strategy" conclusion; it cannot
establish deployable alpha.

## Frozen rule

1. Use `headline/sentiment_all`, the existing deterministic rule-based headline
   score, and average it by company and news date.
2. Go long when the mean score is strictly greater than `+0.10`, short when it
   is strictly less than `-0.10`, and stay flat otherwise.
3. Map the day-level signal to the next observed trading-session close. The
   source table does not retain a usable intraday timestamp, so same-day
   execution is not allowed.
4. Hold the position for five trading sessions. A newer signal replaces the
   prior holding; unchanged positions do not generate turnover.
5. Allocate the starting capital equally across every stock in the frozen
   universe. No stock or low-correlation subset may replace the all-stock book
   after evaluation is seen.
6. Charge 10 bps per side on `abs(position_t - position_t-1)`. A direct long to
   short reversal therefore has turnover 2 and costs 20 bps of that stock's
   allocation. Force positions flat at split and sample boundaries.
7. Compute daily funded net returns, compounded sample return, annualised net
   Sharpe with 252 sessions and zero cash rate, volatility, drawdown, turnover,
   active days, and worst day.

## Selection evidence already observed

On the 33-stock Reuters corpus, with dates from December 2025 to July 2026:

| Period | Net sample return | Net Sharpe | Max drawdown | Active days |
| --- | ---: | ---: | ---: | ---: |
| Development | +1.110% | 1.328 | -1.230% | 85 |
| Evaluation | +0.757% | 1.371 | -0.907% | 40 |

These numbers motivated the locked rule and must not be counted as new
confirmation.

## One-shot robustness gate

The separate 22-stock corpus runs from June 2025 to July 2026. Only the frozen
threshold and holding period are supplied to the runner, so it has no parameter
choice. The first half is reported as an earlier block and the second half as a
later chronological evaluation.

Pass only if the all-stock evaluation portfolio has both positive net
cumulative return and positive net Sharpe at 10 bps per side. Report 5 and 20
bps sensitivity after the 10 bps verdict without changing the rule. Preserve a
failure and stop; do not change the threshold, hold, universe, or direction.

## Reproduction commands

```bash
.venv/bin/sentiment-bench analyze-week6-pnl \
  --signals Data/collections/lseg_us_midcap_22_1y/derived/content_quality_direct/daily_signals.csv \
  --prices Data/derived/prices/lseg_us_midcap_22_1y.csv \
  --scorer headline/sentiment_all \
  --run-id recent_news_sentiment_persistence_midcap22_10bps_20260722 \
  --development-fraction 0.50 \
  --thresholds 0.10 \
  --holding-periods 5 \
  --transaction-cost-bps-per-side 10 \
  --skip-low-correlation-portfolio
```

The skip flag disables only the optional development-selected stock subset. It
was added after the first runner attempt stopped because that diagnostic lacked
two sufficiently active mid-cap names. The all-stock signal, portfolio, split,
cost, and acceptance rule were not run or changed.

The same command is rerun under distinct immutable IDs at 5 and 20 bps only
after the 10 bps result is fixed. All LSEG news and price inputs already exist;
this protocol uses zero new LSEG requests.
