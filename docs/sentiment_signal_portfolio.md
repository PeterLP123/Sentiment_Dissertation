# Sentiment Signal Portfolio

This page explains the `optimize-signal-portfolio` study: what it combines, the
four questions it answers, the Lagrange derivation behind the weighted solution,
and why the honest verdict on this universe is that combining these signals does
not help. For the command surface, options and defaults, see the
[CLI reference](cli_reference.md). The two frozen runs are
`results/signal_portfolio/sentiment_signal_portfolio_gross_h1_20260729/` and
`results/signal_portfolio/sentiment_signal_portfolio_net_h1_20260729/`.
The companion [findings notebook](../output/jupyter-notebook/sentiment_signal_portfolio_findings.ipynb)
verifies the recorded output hashes and visualises the frozen results.

> **Evidence status:** this is an **exploratory**, **in-sample selection** study on a
> cohort that earlier work in this repository has already examined. It is **not
> preregistered**, **not causal**, and **not deployable alpha or investment advice**.
> The study was **specified after the Week 6 results were observed**, and the loader
> refuses to run unless both of those facts are recorded in the manifest. Subsets and
> weights are chosen on the development block only; the evaluation block is reported
> once and never used to select. The null distribution reported below shows that the
> winning Sharpe should be read as an upper bound, not an estimate. A null result is
> valid evidence here and is retained rather than retuned away.

## What is combined

Seven scorers, one daily strategy return series each, built from
`Data/collections/lseg_us_sector_33_6m/derived/headline_value_models_20260715/daily_signals.csv`
against
`Data/collections/lseg_us_sector_33_6m/derived/headline_value_analysis_lseg_priced/sweep_ws/prices.csv`:

| Signal | Family |
| --- | --- |
| `headline/actionable_sentiment_all` | Lexicon aggregation, actionable headlines only |
| `headline/sentiment_all` | Lexicon aggregation, all headlines |
| `headline/sentiment_non_reuters` | Lexicon aggregation, non-Reuters headlines |
| `headline/sentiment_reuters` | Lexicon aggregation, Reuters headlines |
| `llm/finbert` | Local FinBERT classifier |
| `llm/gemma-4-31b` | Local Gemma classifier |
| `llm/vader` | VADER lexical comparator |

Each series is produced by the existing Week 6 accounting engine
(`sentiment_benchmark.week6_pnl`), so all seven share one price panel, one cost rate
of 10 bps per side charged on `|change in position|`, one session convention, and one
chronological split at `2026-05-02` from a `0.7` development fraction.

The point that matters methodologically is that **the signal rule is shared, not
calibrated**. Every scorer uses the same absolute threshold (`0.0`) and the same
holding period (`1` session). Calibrating a gate or a holding period per scorer would
embed a development grid search inside the panel before the combination search even
begins, and the combination result would then be a maximum over a maximum. The
published `compare-week6-models` runs deliberately do calibrate per scorer — that is
the right choice for comparing differently scaled model scores head to head, but it
makes those runs the more heavily searched, and therefore the more contaminated, of
the two designs. This study accepts a worse per-scorer rule in exchange for a cleaner
combination question.

**Session accounting.** The raw chronological split is 87 development and 41
evaluation sessions. The module filters five dates on which the summed
`active_stock_count` is zero: `2025-12-26` (the ragged first price session, with
only 7 of the 33 symbols priced), `2026-05-01` (the forced split-boundary
liquidation), and `2026-06-29`, `2026-06-30` and `2026-07-01` at the sample end.
Two fall in development and three in evaluation, leaving **85 development and 38
evaluation sessions**.

This is an activity-count filter, not a pure all-zero-return filter. All five
dates have zero gross return, but `2026-05-01` and `2026-06-29` carry liquidation
turnover for every scorer and therefore negative net returns before filtering;
the other three have zero turnover and zero net return. The frozen net run
therefore excludes those two exit-cost rows. Gross results are unaffected by this
distinction, while the reported net results and breakeven cost use the filtered
session convention.

## The four questions and their answers

All figures below are development-block annualised Sharpe ratios on the **gross**
selection surface unless stated otherwise, taken from the run's `summary.md` and
`manifest.json`.

### 1. Which two signals maximise the Sharpe of their equal-weighted average?

`headline/sentiment_all` and `headline/sentiment_non_reuters`, at **1.178**.

### 2. Which three?

Those two plus `llm/finbert`, at **1.104**. On the net surface the third member is
`llm/vader` instead of `llm/finbert`, at `-0.754` — the ranking of the marginal
addition is not stable across the two cost surfaces, which is itself a warning about
how thin the separation is.

### 3. Which subset of unspecified size is add/drop stable?

The stopping rule is add/drop stationarity: a subset qualifies when adding any one
further signal lowers its Sharpe **and** removing any one held signal also lowers it.
A singleton is vacuously removal-stable, because the average of the empty set has no
defined Sharpe.

All **127** non-empty subsets of the seven signals were enumerated exhaustively. The
best subset at each cardinality is:

| Size | Development Sharpe | Evaluation Sharpe | Members |
| --- | --- | --- | --- |
| 1 | 1.274 | 0.576 | sentiment_all |
| 2 | 1.178 | 0.344 | sentiment_all, sentiment_non_reuters |
| 3 | 1.104 | 0.626 | sentiment_all, sentiment_non_reuters, finbert |
| 4 | 1.055 | 0.641 | sentiment_all, sentiment_non_reuters, finbert, vader |
| 5 | 0.947 | 0.412 | sentiment_all, sentiment_non_reuters, finbert, gemma-4-31b, vader |
| 6 | 0.814 | 0.223 | sentiment_all, sentiment_non_reuters, sentiment_reuters, finbert, gemma-4-31b, vader |
| 7 | 0.630 | -0.038 | all seven |

Sharpe is **monotonically decreasing in subset size**, from 1.274 at `k = 1` down to
0.630 at `k = 7`. Because the best achievable Sharpe falls at every step, no subset of
size two or more can be add-stable relative to what it could drop, and the search
returns exactly **one** add/drop-stable subset: the singleton
**`{headline/sentiment_all}`** at **1.274**. Its best forgone addition is
`headline/sentiment_non_reuters`, which would take it to 1.178.

Backward stepwise from the full set confirms this rather than finding an interior
optimum. It walks all the way down, dropping one signal at a time — seven signals,
then six, five, four, three, two, and finally one — terminating at the same singleton
at 1.274. The forward path from the best single start never leaves it.

The answer to question 3 on this universe is therefore that **equal-weighted
averaging never helps**. The unspecified-size optimum is one signal, and every
combination is worse than its own best member.

### 4. What weights maximise the Sharpe of two signals under a sum-to-one budget?

Solved by Lagrange multipliers for all 21 pairs; see the derivation below. For the
best pair, the optimum is `w = [6.7677, -5.7677]`, a large offsetting long/short
position with a development Sharpe of **1.772** — which beats both the equal-weighted
1.178 and the best single signal's 1.274. That improvement is bought with gross
leverage of **12.535** on a covariance matrix whose condition number is **180.4**, and
it is not a result to trust: see [Is the winner real?](#is-the-winner-real) below.

## Why averaging fails here

Two reasons, both visible in the run artifacts.

**(a) The signals are near-duplicates.** Development pairwise correlations run from
**0.38 to 0.99**. Equal-weighted averaging reduces volatility only to the extent that
the components' errors are independent, and here they mostly are not.
`headline/sentiment_all` and `headline/sentiment_non_reuters` correlate at **0.99**
specifically because they are nested aggregations of the same lexicon over almost the
same headline population — the second is the first with the Reuters headlines removed,
and Reuters is a minority of that population. Averaging them is close to averaging a
series with itself, which cannot cut volatility but does move the mean.

**(b) Their Sharpes are widely dispersed.** The seven singleton Sharpes are:

| Signal | Development Sharpe (gross) |
| --- | --- |
| `headline/sentiment_all` | 1.274 |
| `headline/sentiment_non_reuters` | 1.077 |
| `llm/finbert` | 0.706 |
| `llm/vader` | 0.661 |
| `llm/gemma-4-31b` | 0.257 |
| `headline/sentiment_reuters` | -0.449 |
| `headline/actionable_sentiment_all` | -0.460 |

Two of the seven have negative gross Sharpe. Equal weighting is mean-blind: it moves
the portfolio mean toward the average of the members, which drags the strongest
signal's mean down at rate `1/k`, while the volatility reduction it buys is throttled
by the high correlations in (a). Diluting the mean faster than you cut the volatility
lowers the ratio, and combining (a) with (b) is exactly the regime where that happens
at every cardinality. The monotone decline in the table above is the arithmetic
consequence, not an anomaly.

## The Lagrange derivation

Let `mu` be the vector of mean daily returns, `Sigma` the covariance matrix of the
signal return series, and `w` the weight vector. The objective is the Sharpe ratio

```
S(w) = w'mu / sqrt(w'Sigma w)
```

subject to the budget constraint `1'w = 1`. The Lagrangian is

```
L(w, lambda) = w'mu / sqrt(w'Sigma w) - lambda (1'w - 1)
```

Differentiating, and writing `sigma_w = sqrt(w'Sigma w)`, stationarity in `w` requires

```
dL/dw = mu / sigma_w - (w'mu / sigma_w^3) Sigma w - lambda 1 = 0
```

### The multiplier is zero

`S(w)` is **homogeneous of degree zero**: for any scalar `c > 0`,

```
S(cw) = (cw)'mu / sqrt((cw)'Sigma(cw)) = c w'mu / (c sqrt(w'Sigma w)) = S(w)
```

The Sharpe ratio is scale invariant, so the whole ray through any candidate `w` gives
the same objective value. Euler's theorem for a degree-zero homogeneous function gives
`w' dS/dw = 0`. Contracting the stationarity condition with `w'` therefore leaves

```
0 = w' dS/dw = lambda (1'w) = lambda
```

so **`lambda = 0` exactly**. The interpretation is direct: the budget constraint costs
nothing, because rescaling to satisfy it never changes the objective. Both runs report
a multiplier of `0.000000`, as they must.

With `lambda = 0` the condition collapses to `Sigma w` proportional to `mu`, so
`w` is proportional to `Sigma^-1 mu`, and normalising by the budget gives

```
w* = Sigma^-1 mu / (1' Sigma^-1 mu)
```

For two signals with means `mu1, mu2`, volatilities `sigma1, sigma2` and correlation
`rho`, this has the closed form

```
w1 = (mu1 sigma2^2 - mu2 rho sigma1 sigma2)
     / (mu1 sigma2^2 + mu2 sigma1^2 - rho sigma1 sigma2 (mu1 + mu2))
```

with `w2 = 1 - w1`. The module computes both the linear-algebra and the closed-form
solution for every pair and records `closed_form_agrees` in
`pair_weight_solutions.csv`; it is `True` for all 21 pairs on both surfaces.

### The sign condition

The stationary point is only a maximum when the budget denominator is positive.

- **`1' Sigma^-1 mu > 0`** — normalising keeps the sign of the numerator, and `w*` is
  the constrained **maximum**.
- **`1' Sigma^-1 mu < 0`** — normalising flips the sign, and the same stationary point
  is the constrained **minimum**. On that ray the Sharpe of the budget-feasible
  solution has **no finite maximum**: the supremum is approached only as the weights
  diverge, taking gross leverage to infinity while `1'w = 1` is still satisfied. The
  brute-force grid check in `pair_weight_solutions.csv` shows exactly that behaviour,
  pinning at the grid boundary instead of at an interior optimum.

**10 of the 21 pairs** fall in the `< 0` case on the gross surface. Reporting an
"optimal weight" for those pairs without the sign test would be reporting the worst
feasible portfolio as the best.

### The observed solution

For the best gross pair, `headline/sentiment_all` (`mu = 4.028` bps daily,
`sigma = 50.194` bps) and `headline/sentiment_non_reuters` (`mu = 3.500` bps,
`sigma = 51.565` bps), with `rho = 0.9890`:

| Quantity | Value |
| --- | --- |
| Optimal weights | `6.7677` / `-5.7677` |
| Lagrange multiplier | `0.000000` |
| `1' Sigma^-1 mu` | `17.6101` (positive, so this is the maximum) |
| Sharpe: optimal weights | `1.772` |
| Sharpe: equal weight | `1.178` |
| Sharpe: better single signal | `1.274` |
| Gross leverage (sum of absolute weights) | `12.535` |
| Covariance condition number | `180.4` |
| Long-only KKT weights | `1.0000` / `0.0000` |
| Sharpe: long-only | `1.274` |

The optimiser has found a near-arbitrage between two series that correlate at 0.99: it
goes long 6.77 units of the higher-mean signal and short 5.77 of the lower-mean one, so
that the common component largely cancels and what remains is a levered bet on a mean
spread of `4.028 - 3.500 = 0.528` bps per day. With a condition number of 180.4 the
covariance matrix is close to singular in the direction that matters, so the weights
are estimated almost entirely from the sampling noise in that spread. No shrinkage is
applied, deliberately, so that the reported solution is exactly the one the derivation
gives.

Adding a no-short-selling constraint makes the point plainly. Under the long-only KKT
solution the weights **collapse to `[1, 0]`**: the entire position goes to the single
better signal and the Sharpe falls back to 1.274, the same value as the best singleton.
Once shorting is removed there is nothing for the combination to do.

## Is the winner real?

No — not on this evidence.

Selecting the maximum over 127 correlated candidates is a multiple-comparison
procedure, so the run reports a **demeaned moving-block bootstrap null of no edge**:
each signal's development returns are recentred to zero mean while the block structure
preserves the covariance between signals, and the *maximum Sharpe over all 127 subsets*
is recomputed on each of 2,000 replications with a five-session block length.

| Statistic | Gross surface | Net surface |
| --- | --- | --- |
| Observed winning development Sharpe | 1.274 | -0.583 |
| Null mean maximum-of-127 Sharpe | 1.411 | 1.398 |
| Null median | 1.372 | 1.361 |
| Null 95th percentile | 4.120 | 4.078 |
| **Share of null replications at or above observed** | **0.522** | **0.895** |

Read the gross column carefully. Under a null in which **no signal has any edge at
all**, the expected best-of-127 Sharpe is **1.411**, which is *higher* than the 1.274
actually observed, and **52.2%** of null replications match or beat the observed
winner. The selected subset's Sharpe is therefore **indistinguishable from
data-snooping noise**. It is what this search procedure produces on returns that have
been constructed to contain nothing.

The estimation noise on a single Sharpe points the same way. The Lo (2002) standard
error of an annualised Sharpe under IID returns is

```
SE = sqrt(252) * sqrt((1 + 0.5 * s^2) / n)
```

for a per-period Sharpe `s` over `n` observations. At `n = 85` development sessions
this is **1.725**; on the 38-session evaluation block it is roughly **2.48**. Both are
larger than the entire 0.644 spread between the best and worst cardinality in the table
above, so the ranking of candidate subsets is not statistically separated at all. The
moving-block bootstrap of the winning subset itself gives a 95% interval of
`[-2.151, 4.971]` around a mean of 1.314, with 23.9% of replications at or below zero.

## Cost verdict

The gross surface is the only one on which any of this is even nominally profitable.
On the **net** surface all seven signals have negative development Sharpe at
`h = 1` — from `-0.583` for `headline/sentiment_all` down to `-3.402` for
`headline/sentiment_reuters` — and the winning subset's breakeven cost is **6.86 bps
per side** against the **10 bps per side** charged. Turnover eats the entire gross
edge with about 3 bps to spare. A separate robustness check across holding periods
finds net Sharpe negative for all seven scorers at every holding period in
`{1, 3, 5, 7}`, so this is not an artefact of the one-session hold that both runs use;
each run here fixes a single holding period.

On the net surface the optimisation then **degenerates, for a genuine mathematical
reason rather than a bug**. When the mean return is negative, the Sharpe ratio
`mean / volatility` is a negative number divided by a positive one, so *reducing*
volatility makes it **more** negative. Diversification, whose whole purpose is to
reduce volatility, therefore actively hurts: the objective is maximised by taking the
least-bad mean and as much volatility as possible alongside it. The net run shows this
end to end — the global optimum is the singleton `{headline/sentiment_all}` at
`-0.583`, every pair is worse than its own best member (`-0.646` for the best pair,
against `-0.583` for the better of the two on its own), and the
Lagrange solve for that pair has `1' Sigma^-1 mu = -6.2207 < 0`, so the stationary
point at `-1.036` is the constrained **minimum** and the grid check runs off to the
boundary weight of `25.0`. Two subsets are add/drop stable on the net surface,
`{headline/sentiment_all}` at `-0.583` and `{llm/vader}` at `-0.735`, both singletons.
Ranking by Sharpe when means are negative is a ranking of the least-bad loss, and it
should be read as such.

## Reproduction

```bash
.venv/bin/sentiment-bench optimize-signal-portfolio \
  --signals Data/collections/lseg_us_sector_33_6m/derived/headline_value_models_20260715/daily_signals.csv \
  --prices Data/collections/lseg_us_sector_33_6m/derived/headline_value_analysis_lseg_priced/sweep_ws/prices.csv \
  --scorers headline/actionable_sentiment_all,headline/sentiment_all,headline/sentiment_non_reuters,headline/sentiment_reuters,llm/finbert,llm/gemma-4-31b,llm/vader \
  --run-id sentiment_signal_portfolio_gross_h1_20260729 \
  --return-variant gross --holding-period 1

.venv/bin/sentiment-bench optimize-signal-portfolio \
  --signals Data/collections/lseg_us_sector_33_6m/derived/headline_value_models_20260715/daily_signals.csv \
  --prices Data/collections/lseg_us_sector_33_6m/derived/headline_value_analysis_lseg_priced/sweep_ws/prices.csv \
  --scorers headline/actionable_sentiment_all,headline/sentiment_all,headline/sentiment_non_reuters,headline/sentiment_reuters,llm/finbert,llm/gemma-4-31b,llm/vader \
  --run-id sentiment_signal_portfolio_net_h1_20260729 \
  --return-variant net --holding-period 1
```

Both bootstraps are seeded (`--random-seed 20260729`), the input files are hashed into
each `manifest.json`, and the command refuses to overwrite an existing run directory.
Every other assumption takes the documented default.

## Limitations

These mirror the limitations block written into each run's own `summary.md`.

- Selection is in-sample by construction. The subset and weight searches maximise a
  development statistic, and the null distribution above shows how large that statistic
  becomes by chance alone when the best of many correlated candidates is taken. Treat
  the winning Sharpe as an upper bound, not an estimate.
- The signal universe is not independent. Several scorers are aggregations of the same
  underlying lexicon over nested headline populations, so their return series are
  near-duplicates.
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
