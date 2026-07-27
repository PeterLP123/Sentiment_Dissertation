# Recent-news mid-cap FinBERT event strategy

## Verdict

`recent-news-midcap-finbert-event-v1` is the acceptable strategy from the
22 July 2026 search. It is a sparse, market-neutral, one-session Reuters news
strategy for 22 US mid-cap stocks. On the frozen 10 basis-point-per-side cost
model it produced positive net return and positive Sharpe in both the earlier
development period and the later evaluation period.

For a quick plot-led read, open the
[results dashboard](recent_news_midcap_finbert_dashboard.html). For the full
rules, evidence, and limitations walkthrough, keep using the original
[interactive visual explainer](sentiment_trading_baseline_explainer.html).

This is an **exploratory research strategy**, not confirmed deployable alpha.
The sample was previously examined, only 37 evaluation sessions traded, the
95% block-bootstrap interval crosses zero, and a single stock can hold 50% of
NAV when a side has only one eligible name. Those constraints make it sensible
as a transparent positive-Sharpe baseline, but not ready for live capital.

## Results

The clean replay resolved to
`recent-news-midcap-finbert-event-v1-dfdd88113a5f`.

| Period | Dates | Sessions | Active | Gross return | Net return | Net Sharpe | Max drawdown | Break-even cost/side |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 2025-06-27 to 2025-12-31 | 130 | 25 | +8.014% | +3.419% | 0.780 | -4.965% | 18.18 bps |
| Evaluation | 2026-01-02 to 2026-06-26 | 121 | 37 | +11.466% | +4.004% | 0.593 | -4.899% | 16.52 bps |
| Combined | 2025-06-27 to 2026-06-26 | 251 | 62 | +20.399% | +7.560% | 0.638 | -6.028% | 17.16 bps |

The evaluation annualised mean net return was 9.42% at 15.89% annualised
volatility. It traded on 30.6% of evaluation sessions, and 56.8% of active
evaluation sessions were profitable. The 10 bps assumption is below the
evaluation break-even estimate of 16.52 bps per side, but the margin is not
large enough to ignore spread variation, impact, or borrow fees.

The five-session moving-block bootstrap estimated evaluation mean daily net
return at +0.0374%, with a 95% interval from -0.0886% to +0.1934%. The interval
is inconclusive. Positive realised Sharpe is observed; a statistically settled
edge is not.

## Working hypothesis

The rule is designed to capture short-lived cross-sectional underreaction to
company-specific news. FinBERT converts the headline into a directional
financial-polarity label; the strict target and technical-news screens remove
broad market commentary; next-open execution gives the publication time a
realistic buffer; and the one-session exit focuses the bet on immediate drift
rather than an untested long-lived sentiment state. Requiring simultaneous long
and short candidates removes most broad equity-market direction, while trading
only on qualifying sessions avoids forcing a weak score into a daily position.

That mechanism is a plausible explanation, not a causal result. The backtest
does not show whether delayed reaction, omitted risk factors, Reuters coverage,
or a few concentrated firm events generated the return. The rules “work” here
in the limited empirical sense that the same frozen construction remained
positive across the chronological split after the stated cost.

## Complete trading rules

### 1. Eligible news

Use the existing English Reuters/LSEG headline collection for the frozen
22-stock US mid-cap universe. For each Reuters story family and symbol, retain
the earliest point-in-time eligible occurrence. A headline must:

- explicitly concern the mapped company;
- concern only one target company;
- have a matching cached score and available adjusted-open price history; and
- not be a market-price or technical-analysis update.

The clean build began with 2,637 corpus records. Revision deduplication and
relevance filtering produced 1,187 point-in-time events; the final target and
technical screens retained 1,142. Licensed headline text stays local and is
not written to source-controlled reports.

### 2. Sentiment label

Score the headline only with `ProsusAI/finbert`, pinned to revision
`4556d13015211d73dccd3fdd39d39232506f3e43`. Take the largest of the model's
positive, neutral, and negative probabilities and map its hard label to
`+1`, `0`, or `-1`. The current artifact is cache-only: the replay does not
invoke a model or relabel any headline.

Across the 1,142 selected events, FinBERT labelled 430 positive, 515 neutral,
and 197 negative. VADER share-argmax was retained as a predeclared comparator,
but its labels never supplied both portfolio legs and therefore stayed in
cash; it is not part of the accepted rule.

### 3. Availability and execution

For each event, add a 15-minute processing buffer to the recorded publication
availability. Assign it to the first XNYS adjusted open strictly after that
buffer. This prevents a headline from trading at an already elapsed open.

If a symbol has multiple eligible headlines assigned to the same execution
session, average their hard labels. Trade the sign of that average:

- positive mean: long candidate;
- negative mean: short candidate; and
- zero mean: no position.

### 4. Portfolio construction

At each execution open:

1. Require at least one long and at least one short candidate. If either side
   is absent, hold zero-return cash for that session.
2. Set gross exposure to 100% and net exposure to 0%.
3. Allocate +50% of NAV equally across all long candidates.
4. Allocate -50% of NAV equally across all short candidates.
5. Hold for exactly one adjusted-open-to-adjusted-open session.
6. Exit at the next eligible XNYS open. Do not carry a position without a new
   eligible signal.

The backtest starts each chronological block at USD 1,000,000. The portfolio
was active on 62 of 251 sessions. Twenty-three active sessions contained only
two names, so the maximum observed single-name absolute weight was 50%. This
is the strategy's largest practical risk and must not be hidden by the
market-neutral label.

### 5. Costs and P&L

For every session, turnover is
`sum(abs(target_weight - current_weight))`. Charge 10 bps multiplied by this
gross traded weight on entries, exits, and reversals. Force and charge final
liquidation. Cash earns zero. Returns use split-adjusted opens; dividends are
not back-adjusted.

The model omits borrow availability and fees, financing, cash yield, taxes,
variable bid-ask spreads, market impact, and intraday liquidity. Because the
evaluation break-even estimate is only 16.52 bps per side, these omissions are
material rather than cosmetic.

## Data and material passport

| Material | Classification | Identity and use |
| --- | --- | --- |
| Reuters/LSEG corpus | Inherited local input | 22 stocks; news requested from 2025-06-26 to 2026-06-26; local licensed text |
| Adjusted-open panel | Inherited local input | 22 stocks, 5,830 rows, 2025-06-20 to 2026-07-10; split-adjusted, not dividend-adjusted |
| FinBERT score artifact | Inherited local input | 2,576 unique cached headlines; pinned model revision; no inference in this run |
| Frozen TOML config | New source artifact | Defines split, screens, signal, execution, portfolio, costs, inference seed, and output roots |
| Baseline implementation | Inherited code | Implementation hash `cf32cd29317ec042937533eb555c532fafc32335c510848f8ae6ef9f4c0db1b4` |
| Strategy results | Newly computed local output | Immutable run `recent-news-midcap-finbert-event-v1-dfdd88113a5f`; no licensed text in result artifacts |
| Interpretation | Analyst judgement | “Acceptable exploratory baseline” requires positive net return and Sharpe in both periods, not statistical confirmation |

All news and prices are post-2024. The 22 July search used **0 new LSEG
requests**, so the 10,000-request allowance remains available.

## Why this strategy was retained

The rule passed the protocol filed before its clean replay:

- development and evaluation net returns are positive;
- development and evaluation Sharpes are positive;
- each period has at least 20 active sessions;
- input, chronology, and output-integrity validation passed;
- the frozen cost is below the estimated break-even cost in both periods; and
- a second identical command validated and reused the immutable result.

The rule was not invented during this search. It is the best acceptable
existing mid-cap event rule, deliberately re-frozen and replayed under a new
strategy ID. That distinction avoids presenting reuse as discovery.

## Alternatives rejected on 22 July

| Candidate | Frozen evidence | Decision |
| --- | --- | --- |
| 33-stock lagged FinBERT rank reversal | Evaluation net -14.467%; Sharpe -10.719 | Reject and preserve null |
| Mid-cap persistent FinBERT grid | 0 of 12 candidates passed the development gate; best supported candidate Sharpe 0.201 but unstable across subperiods | Stop before evaluation |
| Five-session lexicon persistence | Mid-cap evaluation net -1.125%; Sharpe -1.325 at 10 bps/side | Reject; 5 bps (-0.498%, -0.582 Sharpe) and 20 bps (-2.367%, -2.786 Sharpe) also fail |
| This one-session mid-cap FinBERT event rule | Development Sharpe 0.780; evaluation Sharpe 0.593 at 10 bps/side | Retain as exploratory baseline |

These adverse results are part of the conclusion. No threshold, horizon,
universe, direction, or cost assumption was changed after the accepted clean
replay.

## Statistical and research-risk audit

The audit covers all 11 required fallacy classes:

| Risk | Assessment |
| --- | --- |
| Simpson's paradox | Not ruled out: aggregate performance can mask sector, stock, or time-block reversals. |
| Ecological fallacy | Avoided in the claim: positive portfolio performance is not claimed for every stock. |
| Berkson's paradox | Caution: conditioning on Reuters coverage, a fixed mid-cap cohort, and executable news may distort relationships. |
| Collider bias | Caution: explicit-target, single-target, and non-technical screens may condition on variables affected by news type and coverage. |
| Base-rate neglect | Avoided: activity, neutral labels, cash sessions, costs, and the inconclusive interval are reported. |
| Regression to the mean | No return-extreme selection rule is used, but the short sample leaves this possible in realised metrics. |
| Survivorship bias | Caution: the fixed 22-stock cohort contains firms with available recent histories and does not model delistings. |
| Multiple comparisons | High risk: several strategies were examined; failed candidates are reported, and this sample cannot be confirmatory. |
| Garden of forking paths | High risk: the broader research programme influenced which rule was retained, despite this run's frozen parameters. |
| Correlation versus causation | The result is observational predictive performance, not evidence that headline sentiment causes returns. |
| Reverse causality | Point-in-time next-open execution blocks direct look-ahead, but common information can affect both headlines and prices. |

Coverage is 11/11. The overall research-risk verdict is **caution**: no detected
timing leak or artifact-integrity failure, but selection, concentration,
sample reuse, and low power prevent a confirmatory claim.

## Reproduction and verification

From the repository root:

```bash
.venv/bin/sentiment-bench strategy baseline inspect \
  --config configs/strategy_research/baselines/recent_news_midcap_finbert_event_v1.toml

.venv/bin/sentiment-bench strategy baseline run \
  --config configs/strategy_research/baselines/recent_news_midcap_finbert_event_v1.toml
```

The first clean run completed the immutable artifact. The second invocation
reported `validated/reused`. Key SHA-256 identities were:

- config: `37fe64c48bebd6a25d869cf6af6f5b3425687f985aaaa7a0aa044bf33e4bb98c`;
- price panel: `2f067666238e33123e605dff40d0a8429892871807349fecf4b293261b44b832`;
- score artifact: `ee711b9d97d405996218429c5135c584d268a424d47909ca2d7d56d314d4915a`;
- metrics: `e010c21bf9ed430f8248e358243f9d22c3869cdf08e6cbc74ffb667813a2d08e`;
- daily P&L: `1bd11490c579bba9cde12ee22c17861916c125bc9c18cf24c16d84670708d4a6`;
- report: `4f48208f54b644b8b1693b001ea451a0bfdd7e7efa517394b3219ca74820e558`.

Generated results are intentionally ignored under
`results/strategy_research/baselines/recent-news-midcap-finbert-event-v1-dfdd88113a5f/`.
Source control retains the frozen config, protocol, explainer, runner, and
tests; the local manifest binds those sources to every generated file.

## Bottom line and next gate

Use this rule as the current recent-news positive-Sharpe research baseline. Do
not call it validated alpha and do not deploy it unchanged. The next legitimate
test is to collect a truly unseen future Reuters period, freeze a lower
single-name cap or minimum breadth rule before opening returns, add realistic
borrow/spread assumptions, and evaluate once without further tuning.
