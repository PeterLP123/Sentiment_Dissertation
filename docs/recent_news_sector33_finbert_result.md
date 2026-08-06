# Recent-news 33-stock FinBERT robustness result

## Verdict

The unchanged one-session FinBERT event rule **failed** on the existing
33-stock, 11-sector Reuters/LSEG cohort. It does not generalise from the
accepted 22-stock mid-cap sample on this evidence.

The run passed its data-integrity, point-in-time, activity and deterministic
replay checks, but failed every predeclared economic acceptance condition. At
10 basis points per dollar traded, net return and Sharpe were negative in both
chronological blocks and break-even costs were below the frozen cost. The
result is retained as a null; no threshold, horizon, direction, universe subset
or cost assumption was changed after returns were opened.

## Material passport

- Origin skill: academic-research-suite / experiment-agent run and validation
- Origin date: 2026-07-22
- Verification status: `VERIFIED` deterministic replay; statistical
  interpretation `CAUTION`
- Version label: `recent_news_sector33_finbert_result_v1`
- Frozen config:
  `configs/strategy_research/baselines/recent_news_sector33_finbert_event_v1.toml`
- Resolved run: `recent-news-sector33-finbert-event-v1-4b9baebb01fa`
- Data status: previously explored, post-2024 local Reuters/LSEG data
- New LSEG requests: zero
- Licensed material: headline text and model-score CSV remain local and ignored;
  source-controlled documents contain only aggregate results and identities

## Frozen result

| Period | Dates | Sessions | Active | Gross return | Net return | Net Sharpe | Max drawdown | Break-even cost/side |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Development | 2025-12-26 to 2026-05-13 | 95 | 92 | +4.570% | -9.159% | -2.491 | -9.765% | 3.31 bps |
| Evaluation | 2026-05-14 to 2026-06-26 | 30 | 30 | -0.334% | -5.035% | -3.682 | -5.655% | -0.53 bps |
| Combined | 2025-12-26 to 2026-06-26 | 125 | 122 | +4.222% | -13.733% | -2.814 | -14.680% | 2.33 bps |

Development incurred USD 133,226.67 of modelled transaction costs and averaged
1.481 gross traded weight per session. Evaluation incurred USD 46,235.91 and
averaged 1.610. Total modelled cost was USD 179,462.58 on the combined path.
Annualised evaluation turnover was 405.6 times the starting portfolio scale.

The development gross return was positive, but its 3.31 bps approximate
break-even cost was far below the frozen 10 bps charge. Evaluation was already
negative before costs, so lower trading costs cannot rescue that chronological
block. A positive combined gross return therefore does not establish a usable
strategy.

## Acceptance-gate decision

| Predeclared condition | Observed | Decision |
| --- | --- | --- |
| Positive development net return and Sharpe | -9.159%; -2.491 | Fail |
| Positive evaluation net return and Sharpe | -5.035%; -3.682 | Fail |
| At least 20 active sessions per block | 92; 30 | Pass |
| Break-even cost above 10 bps in both blocks | 3.31; -0.53 bps | Fail |
| Point-in-time, price, score and artifact integrity | All passed | Pass |
| Deterministic replay reuses immutable result | Validated/reused | Pass |

The overall robustness gate fails. Activity and software validity cannot offset
negative economic performance.

## Comparison with the retained 22-stock baseline

| Evaluation diagnostic | 22-stock mid-cap | 33-stock sector cohort |
| --- | ---: | ---: |
| Net cumulative return | +4.004% | -5.035% |
| Net Sharpe | +0.593 | -3.682 |
| Gross cumulative return | +11.466% | -0.334% |
| Average gross traded weight | 0.573 | 1.610 |
| Break-even cost/side | 16.52 bps | -0.53 bps |
| Maximum single-name absolute weight | 50% | 50% |
| Active evaluation sessions | 37 of 121 | 30 of 30 |

The windows and stock populations differ, so this is a robustness comparison,
not a controlled causal decomposition. Still, it gives two clear observations:

1. The positive after-cost mid-cap path did not transfer to the larger,
   more liquid sector cohort.
2. More stocks increased daily opportunity and typical breadth, but the
   one-session no-carry rule turned that activity into much higher turnover.

The 33-stock portfolio had no two-name active sessions: active sessions held
between three and fourteen names, with roughly ten typical names. However,
13 of 122 active sessions had a single name on at least one leg. Equal 50/50
leg allocation therefore still allowed a 50% position. A larger nominal
universe did not remove the concentrated-leg edge case.

## Statistical interpretation

Evaluation FinBERT minus zero-return cash had mean daily net return -0.169%.
The predeclared five-session moving-block 95% interval was -0.354% to +0.068%
over 2,000 replications. The point estimate is adverse, but the interval crosses
zero; its direction is statistically inconclusive rather than a settled proof
of a negative true mean. The realised net return and Sharpe nevertheless fail
the trading acceptance rule without relying on statistical significance.

Overall confidence is `CAUTION`: the implementation and deterministic replay
are solid, while the sample is previously explored, the evaluation block has
only 30 sessions, and the interval is wide.

## Research-risk and fallacy scan

Coverage: 11/11 required classes checked.

| Risk | Assessment |
| --- | --- |
| Simpson's paradox | Not ruled out: aggregate loss can mask sector, stock or subperiod reversals. No subgroup is promoted as a replacement strategy. |
| Ecological fallacy | Avoided: portfolio failure is not claimed to mean every stock or headline relation is negative. |
| Berkson's paradox | Caution: conditioning on LSEG coverage, an executable fixed cohort and strict target screens can distort associations. |
| Collider bias | Caution: single-target and non-technical screens may condition on news characteristics related to coverage and return response. |
| Base-rate neglect | Avoided: neutral labels, activity, cash behaviour, turnover, costs and uncertainty are reported. |
| Regression to the mean | No return-extreme stock selection was used, but the short realised window limits stability. |
| Survivorship bias | Caution: the fixed 33-stock cohort contains firms with complete recent data and does not model delistings. |
| Look-elsewhere effect | High risk in the wider programme; this exact universe test was frozen before opening its new-revision result and failed candidates remain reported. |
| Garden of forking paths | Caution: the broader research programme has examined several constructions. This failed arm is not retuned. |
| Correlation versus causation | No causal claim is made; the backtest measures an observational predictive association. |
| Reverse causality | Point-in-time next-open execution removes direct future-price leakage, but common information may affect both headlines and prices. |

## Reproducibility

From the repository root:

```bash
.venv/bin/sentiment-bench strategy baseline inspect \
  --config configs/strategy_research/baselines/recent_news_sector33_finbert_event_v1.toml

.venv/bin/sentiment-bench strategy baseline run \
  --config configs/strategy_research/baselines/recent_news_sector33_finbert_event_v1.toml
```

The first corrected-code invocation completed the immutable artifact. A second
identical invocation returned `validated/reused`. This is a deterministic
experiment, and the output hashes matched exactly.

Key SHA-256 identities:

- config: `4bb10982ff9cb37c4da5e1c3098dc02e022cda00fc06e3c6988f6394e7916719`;
- implementation: `580df81c885146dac51e821f01d2e558be6bd73ca18f636ca245bda9ad3b4c64`;
- corpus manifest: `fe3f68a0e6eab84954c41a6d891530797cbc45d53dafc1d06c5551c36732ad7c`;
- price panel: `026ab977e2d5a57ed11461e76c6e3c26240f740fadb99f14cbdda7e875a628b4`;
- score artifact: `a65d00650a72434eeb96a330a9dbe6ef9f00f685e6786f617c573d842706f752`;
- metrics: `ebfe03b4b33e5a4cb14e9212ed5f2360fb06844287207eb679724b6e40d64eb7`;
- daily P&L: `466e004819bfb0575954d32d3b23915ef26dcbb84008f94ec994a3a82d0c6c60`;
- report: `0f7fde877387d837c9411d808a65f8f53f8ccf046303cb9aef88e508abbc8d1f`.

## Conclusion and next gate

Keep the 22-stock mid-cap rule as a historical exploratory positive-Sharpe
comparator, not the current strategy base or universe-general alpha. Notebook
55 later reconciles it with the final evidence hierarchy: its local gate passes,
but the previously explored selection history, zero-crossing cash interval, and
50% maximum name weight prevent promotion. The unchanged rule failed on more
stocks, and the failure is too large to justify a cosmetic portfolio adjustment
on this already examined sample.

The next honest test should use a genuinely unseen future Reuters period. If a
broader production-oriented design is attempted, freeze a concentration cap,
minimum breadth, lower-turnover signal persistence or event threshold before
opening those future returns. This result does not authorise selecting a
winning subset of the 33 stocks after the fact.
