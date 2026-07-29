# Signal portfolio optimisation — `sentiment_signal_portfolio_net_h1_20260729`

> This is an exploratory, in-sample selection study on a previously explored cohort.
> It is not deployable alpha, investment advice, or causal evidence. Subsets and
> weights are chosen on the development block only; the evaluation block is
> reported once and never used to select.

## Frozen design

- Signals: 7 scorers, one daily return series each — `headline/actionable_sentiment_all`, `headline/sentiment_all`, `headline/sentiment_non_reuters`, `headline/sentiment_reuters`, `llm/finbert`, `llm/gemma-4-31b`, `llm/vader`.
- Return variant used for selection: **net**.
- Shared signal rule: threshold `0.0`, holding period `1` session(s). No per-scorer calibration, so the panel embeds no development grid search.
- Cost: `10.0` bps per side on `|change in position|`.
- Split: development before `2026-05-02`, evaluation on and after it.
- Sharpe: annualised over `252` periods at a `0.0` risk-free rate, sample volatility (`ddof=1`).
- Sessions dropped as structurally inactive: ['2025-12-26', '2026-05-01', '2026-06-29', '2026-06-30', '2026-07-01'].

## Signal correlations (development)

| Signal | actionable_sentiment_all | sentiment_all | sentiment_non_reuters | sentiment_reuters | finbert | gemma-4-31b | vader |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `headline/actionable_sentiment_all` | 1.00 | 0.83 | 0.82 | 0.48 | 0.47 | 0.54 | 0.39 |
| `headline/sentiment_all` | 0.83 | 1.00 | 0.99 | 0.50 | 0.68 | 0.66 | 0.59 |
| `headline/sentiment_non_reuters` | 0.82 | 0.99 | 1.00 | 0.50 | 0.68 | 0.65 | 0.59 |
| `headline/sentiment_reuters` | 0.48 | 0.50 | 0.50 | 1.00 | 0.45 | 0.40 | 0.47 |
| `llm/finbert` | 0.47 | 0.68 | 0.68 | 0.45 | 1.00 | 0.66 | 0.62 |
| `llm/gemma-4-31b` | 0.54 | 0.66 | 0.65 | 0.40 | 0.66 | 1.00 | 0.49 |
| `llm/vader` | 0.39 | 0.59 | 0.59 | 0.47 | 0.62 | 0.49 | 1.00 |

## Best equal-weighted subset by cardinality

| Signals | Size | Development Sharpe | Std. error | Evaluation Sharpe | Members |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | -0.583 | 1.722 | -1.166 | sentiment_all |
| 2 | 2 | -0.646 | 1.723 | -1.452 | sentiment_all, sentiment_non_reuters |
| 3 | 3 | -0.754 | 1.723 | -1.341 | sentiment_all, sentiment_non_reuters, vader |
| 4 | 4 | -1.054 | 1.724 | -1.329 | sentiment_all, sentiment_non_reuters, finbert, vader |
| 5 | 5 | -1.442 | 1.725 | -1.774 | sentiment_all, sentiment_non_reuters, finbert, gemma-4-31b, vader |
| 6 | 6 | -1.785 | 1.727 | -2.179 | sentiment_all, sentiment_non_reuters, sentiment_reuters, finbert, gemma-4-31b, vader |
| 7 | 7 | -2.063 | 1.729 | -2.494 | actionable_sentiment_all, sentiment_all, sentiment_non_reuters, sentiment_reuters, finbert, gemma-4-31b, vader |

## Part 1 and 2 — best two and three signals

- **Best pair** (development Sharpe -0.646): `headline/sentiment_all`, `headline/sentiment_non_reuters`.
- **Best triple** (development Sharpe -0.754): `headline/sentiment_all`, `headline/sentiment_non_reuters`, `llm/vader`.

## Part 3 — subset of unspecified size

The stopping rule is add/drop stationarity: adding any one further signal lowers
Sharpe, and removing any one held signal also lowers it. A singleton is vacuously
removal-stable because the empty average has no defined Sharpe.

- Subsets searched exhaustively: **127**.
- Add/drop-stable subsets found: **2**.
- Global development maximum: **sentiment_all** at Sharpe -0.583.

| Stable subset | Size | Development Sharpe | Best forgone addition | Best forgone removal |
| --- | --- | --- | --- | --- |
| sentiment_all | 1 | -0.583 | `headline/sentiment_non_reuters` -> -0.646 | n/a |
| vader | 1 | -0.735 | `headline/sentiment_all` -> -0.746 | n/a |

### Stepwise paths

- From the empty-to-best forward start: {sentiment_all} (terminal Sharpe -0.583).
- From the full set: {actionable_sentiment_all, sentiment_all, sentiment_non_reuters, sentiment_reuters, finbert, gemma-4-31b, vader} -> {sentiment_all, sentiment_non_reuters, sentiment_reuters, finbert, gemma-4-31b, vader} -> {sentiment_all, sentiment_non_reuters, finbert, gemma-4-31b, vader} -> {sentiment_all, sentiment_non_reuters, finbert, vader} -> {sentiment_all, sentiment_non_reuters, vader} -> {sentiment_all, sentiment_non_reuters} -> {sentiment_all} (terminal Sharpe -0.583).

## Part 4 — Sharpe-optimal weights for two signals (Lagrange)

Maximising `w'mu / sqrt(w'Sigma w)` subject to `1'w = 1` gives a **zero multiplier**:
the Sharpe ratio is scale invariant, so the budget constraint does not bind. The
stationarity condition collapses to `Sigma w` proportional to `mu`, hence
`w* = Sigma^-1 mu / (1' Sigma^-1 mu)`.

For the best pair `headline/sentiment_all` and `headline/sentiment_non_reuters`:

| Quantity | Value |
| --- | --- |
| Correlation | 0.9889 |
| Mean daily return (bps) | -1.843 / -2.286 |
| Daily volatility (bps) | 50.168 / 51.530 |
| Optimal weights | -10.2988 / 11.2988 |
| Closed form agrees with linear algebra | True |
| Lagrange multiplier | 0.000000 |
| `1' Sigma^-1 mu` | -6.2207 |
| Stationary point is a maximum | False |
| Sharpe: equal weight | -0.646 |
| Sharpe: better single signal | -0.583 |
| Sharpe: optimal weights | -1.036 |
| Gross leverage `|w_a| + |w_b|` | 21.598 |
| Covariance condition number | 178.5 |
| Grid check: best weight / Sharpe | 25.000 / 0.746 |
| Long-only weights | 1.0000 / 0.0000 |
| Sharpe: long-only | -0.583 |
| Evaluation Sharpe at frozen optimal weights | -3.907 |
| Evaluation Sharpe at frozen long-only weights | -1.166 |

`pair_weight_solutions.csv` carries the same solve for every pair. Rows where
`stationary_point_is_maximum` is false have `1' Sigma^-1 mu < 0`: there the
stationary point is the constrained *minimum* and the supremum is approached only
as leverage diverges, so no finite maximum exists.

## How much of the winner is selection luck?

- Lo (2002) standard error of the winning development Sharpe: 1.722.
- Moving-block bootstrap of the winning subset: mean -0.522, 95% interval [-3.957, 3.116], share at or below zero 0.620.
- Demeaned no-edge null, maximum over all 127 subsets across 2000 replications: mean 1.398, median 1.361, 95th percentile 4.078.
- Share of null replications at or above the observed winning Sharpe: **0.895**.

- Breakeven cost for the winning subset: 6.86 bps per side (charged rate 10.0 bps).

## Limitations

- Selection is in-sample by construction. The subset and weight searches maximise a
  development statistic, and the null distribution above shows how large that
  statistic becomes by chance alone when the best of many correlated candidates is
  taken. Treat the winning Sharpe as an upper bound, not an estimate.
- The signal universe is not independent. Several scorers are aggregations of the
  same underlying lexicon over nested headline populations, so their return series
  are near-duplicates. See the correlation table.
- The development and evaluation blocks are short. At these sample sizes the standard
  error of an annualised Sharpe is of the same order as the spread between candidate
  subsets, so rankings are not statistically separated.
- The cohort was already explored by earlier work in this repository. The evaluation
  block is leakage-controlled within this command but is not a pristine confirmatory
  holdout.
- Unconstrained Lagrange weights are unstable when signals are highly correlated: the
  covariance matrix is near-singular, and the solution takes large offsetting long and
  short positions that are unlikely to survive out of sample. The long-only variant is
  reported alongside for that reason.
- Returns are price returns on a fixed notional book with a zero risk-free rate. Borrow
  cost, financing, dividends, spread, market impact and capacity are not modelled.
- Weights are estimated from a sample covariance matrix without shrinkage, so that the
  Lagrange solution is reported exactly as derived. A shrinkage estimator would change
  the weights and is a separate design choice.
