# Week 6 FinBERT/VADER full-population test

**Status:** completed exploratory two-model test. This supersedes the earlier
85,961-headline subset when discussing FinBERT versus VADER, but it does not
repair the incomplete Gemma arm of the three-model comparison.

## Question

How do FinBERT and VADER scores and funded trading results change when both are
run over every unique eligible headline in the frozen LSEG US Sector 33
collection?

## Population and scoring

- Raw LSEG headline rows: `710,625`.
- Unique normalized in-window company-matched headlines: `568,707`.
- Successful VADER scores: `568,707/568,707`.
- Successful FinBERT scores: `568,707/568,707`.
- Failures: `0`.
- FinBERT: `ProsusAI/finbert`, pinned revision
  `7db323f79b751944bcfa66298ec06977e4518306`.
- VADER: `nltk.sentiment.vader`.
- FinBERT device: Apple MPS; inference batch size `8`; checkpoint size `256`.
- Full score-file SHA-256:
  `55713b7b4c63ad542b055785c15cf2fa97b17314437fcf9e336aed7c9de5ab0f`.

The full score file contains licensed headline text and remains local and
ignored by Git.

## How the headline scores changed

| Model | Population | Headlines | Mean score | Standard deviation | Positive share | Negative share | Zero share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FinBERT | Previous subset | 85,961 | 0.10419 | 0.58199 | 67.44% | 32.56% | 0.00% |
| FinBERT | Full population | 568,707 | 0.10495 | 0.58272 | 67.41% | 32.59% | 0.00% |
| VADER | Previous subset | 85,961 | 0.05115 | 0.14671 | 38.52% | 13.53% | 47.95% |
| VADER | Full population | 568,707 | 0.05169 | 0.14738 | 38.76% | 13.58% | 47.66% |

The marginal headline-level distributions barely changed. VADER reproduced
the common subset exactly. FinBERT differed by at most `1.4e-5` on a common
headline because batched floating-point inference is not bit-identical, which
is immaterial relative to the score range.

The company-day signals changed much more because the subset contained a median
of only `11` scored headlines per company-day, versus `76` in the complete
population:

| Model | Previous covered company-days | Full covered company-days | Old/full daily-score correlation | Mean absolute daily-score change |
| --- | ---: | ---: | ---: | ---: |
| FinBERT | 5,774 | 5,985 | 0.664 | 0.1461 |
| VADER | 5,774 | 5,985 | 0.663 | 0.0386 |

This is why nearly unchanged headline distributions can still produce different
trading rules: the daily score is an average of a much larger and different set
of headlines.

## Frozen strategy changes

The development/evaluation split and funded assumptions were unchanged. The
evaluation begins `2026-05-02`, uses `£100,000`, and contains 41 observations.
Costs remain 10 basis points per side.

| Model | Population | Absolute gate | Holding period | Selected negative-correlation stocks |
| --- | --- | ---: | ---: | --- |
| FinBERT | Previous subset | 0.2868 (70th percentile) | 5 | COP, LIN, PLD |
| FinBERT | Full population | 0.1396 (50th percentile) | 5 | COP, GOOGL |
| VADER | Previous subset | 0.0000 | 7 | DUK, NVDA |
| VADER | Full population | 0.0000 | 1 | AMZN, XOM, EQIX |

## Evaluation results

| Model | Portfolio | Previous subset profit | Full-population profit | Full return | Full Sharpe | Full max drawdown |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VADER | All-stock | -£1,456.74 | -£659.57 | -0.660% | -0.369 | -3.938% |
| VADER | Negative-correlation | £1,496.67 | -£10,724.85 | -10.725% | -4.977 | -12.110% |
| FinBERT | All-stock | -£1,508.95 | -£3,322.73 | -3.323% | -3.645 | -3.542% |
| FinBERT | Negative-correlation | £224.72 | -£4,780.57 | -4.781% | -2.917 | -4.785% |
| — | 33-stock buy-and-hold | — | £951.35 | 0.951% | 0.621 | -2.649% |

The complete-population test removes the earlier positive correlation-portfolio
result. Buy-and-hold beats both full-population sentiment models on absolute and
risk-adjusted evaluation performance. The expanded VADER all-stock strategy is
less negative than its subset version, while FinBERT becomes more negative.

## Before-cost viewer diagnosis

The audited Week 6 viewer was rebuilt for both complete-population scorers. It
reconciles every decision-driving company-day to the raw LSEG associations and
the frozen model mean before displaying an execution.

| Model | Portfolio | Gross profit | Gross return | Gross Sharpe | Gross max drawdown | Cost impact |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| VADER | All-stock | £259.32 | 0.259% | 0.210 | -3.688% | £918.89 |
| VADER | Negative-correlation | -£10,122.73 | -10.123% | -4.718 | -11.694% | £602.12 |
| FinBERT | All-stock | -£1,803.94 | -1.804% | -1.960 | -2.475% | £1,518.80 |
| FinBERT | Negative-correlation | -£4,016.32 | -4.016% | -2.411 | -4.232% | £764.25 |
| — | 33-stock buy-and-hold | £1,153.01 | 1.153% | 0.740 | -2.649% | £201.66 |

VADER is the only sentiment strategy with a positive gross all-stock result,
but it is not a smooth or robust edge. Eighteen stocks made a gross profit and
15 lost money. GE (£933.81), LLY (£709.43), CAT (£555.63), HD (£502.94), and
TSLA (£402.79) offset large losses in NEM (-£907.51), COP (-£633.83), XOM
(-£585.80), NFLX (-£569.96), and CVX (-£384.56). Costs then turn the small
portfolio-level gross profit into a net loss.

FinBERT loses before costs across 20 of 33 stocks. Its strongest evaluation
winner is GE (£844.71 gross), but FCX (-£478.62), WMT (-£465.10), AMT
(-£394.11), EQIX (-£314.25), and PLD (-£306.46) outweigh the winners.

The negative-correlation portfolios fail because their selected constituents
do not have positive evaluation returns. VADER selected AMZN (-£354.57 gross),
XOM (-£585.80), and approximately flat EQIX (£0.59). FinBERT selected COP
(-£302.06) and approximately flat GOOGL (-£1.34). Negative correlation can
smooth profitable return streams; it cannot convert losing streams into a
profitable portfolio.

## Interpretation

The quota-truncated subset happened to preserve each model's overall score
distribution, but it did not preserve company-day signal composition, selected
rules, selected stocks or portfolio conclusions. The earlier positive VADER and
FinBERT correlation portfolios were therefore subset-sensitive and should not
be presented as robust evidence.

This remains an exploratory 41-observation evaluation on a previously examined
six-month corpus. Annualised metrics are extrapolations, and the result is not
causal evidence or investment advice.

## Local artifacts

- Scores: `Data/collections/lseg_us_sector_33_6m/derived/headline_scores_finbert_vader_full_20260715.csv`
- Signals: `Data/collections/lseg_us_sector_33_6m/derived/headline_value_finbert_vader_full_20260715/`
- Funded comparison: `results/week6_model_comparison/week6_finbert_vader_full_20260715/`
- VADER viewer: `results/week6_trade_explorer_finbert_vader_full_20260715/vader.html`
- FinBERT viewer: `results/week6_trade_explorer_finbert_vader_full_20260715/finbert.html`
