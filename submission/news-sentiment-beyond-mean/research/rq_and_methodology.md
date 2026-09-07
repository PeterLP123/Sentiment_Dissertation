# Research question and methodology blueprint

Status: final evidence synthesis after the one-shot testing-period opening, 14 August 2026.

## Research questions

> **Training period:** Does the share of negative stories in a firm's same-day news distribution
> contain incremental information about its assigned-window abnormal return after
> controlling for mean sentiment, news volume and recent price dynamics?

> **Temporal validation:** Does the unchanged baseline association, conditional
> on mean sentiment and news volume, remain negative in the frozen 2020--2023
> FNSPID testing period?

Economic magnitude, risk translation and LSEG portability remain ordered
validation objectives rather than additional headline questions.

## One-sentence answer supported by the current evidence

Negative-story share has a small conditional association in the FNSPID training period.
The full-control estimate is a training-period robustness result; the frozen
2020--2023 testing period instead repeats the baseline mean/count specification.
That coefficient is opposite-signed and imprecise, neither testing-period family
passes its correction, and the predeclared baseline contrast supports a change
in the measured pipeline association across eras. The evidence does not identify
why it changed, and the training-period association is too small for the tested
daily strategy to cover realistic costs.

## Organising hypotheses

These hypotheses organise the final account; they were not preregistered before
the aggregation family was examined and must not be described as confirmatory.

- **H1a (training-period association):** after conditioning on mean sentiment, story
  count and the recent price path, the daily coefficient on negative-story
  share is below zero.
- **H1b (frozen temporal replication):** the unchanged baseline all-firm-day and
  multi-story coefficients remain negative in 2020--2023 under their declared
  BH family. A separate HAC(5) contrast tests the era difference directly.
- **H2 (economic scale):** a model-free high-minus-low negative-share sort has
  the same direction as H1, but its magnitude and uncertainty must be assessed
  before any economic claim.
- **H3 (economic usability):** a directional implementation must cover turnover
  and a 10-basis-point per-side stock-trading cost to be called usable.
- **H4 (portability):** the sign, scale and practical action of the result must
  be examined separately in the backward and recent Reuters/LSEG blocks; the
  sources are never pooled.

## Ordered objectives

1. Compare nine pre-specified summaries of a firm's same-day story scores under
   one return definition and inference regime.
2. Estimate what negative-story share contributes beyond mean sentiment and
   story count, then add lagged one-day return, lagged five-day return and
   recent volatility to test the reversal/momentum confound.
3. translate rank-based evidence into basis points without a regression and
   compare its break-even cost with a realistic cost assumption.
4. Test a bounded secondary interpretation in which aggregate negative pressure
   scales an independently motivated volatility-managed portfolio.
5. Assess source and period portability in separate LSEG blocks, report minimum
   detectable effects for imprecise estimates, and preserve failed rule and
   prompt searches as results.

## FINER assessment

| Dimension | Assessment |
|---|---|
| Feasible | High. The curated repository contains the aggregate panel outputs, specifications, figures and tests needed to reproduce every promoted number without committing licensed text. |
| Interesting | High. Most pipelines must compress several same-company stories into one daily signal, yet that decision is often treated as implementation detail. |
| Novel | Moderate and bounded. Tone dispersion and negative-news asymmetry are established. The targeted search found few studies comparing multiple firm-day aggregation rules like-for-like and then joining statistical, cost and cross-source validation. This is not a claim that no prior study exists. |
| Ethical | Manageable. Licensed headlines and bodies remain local and unquoted; only aggregate outputs are committed. Model limitations and nulls are reported. |
| Relevant | High. The question connects natural-language processing, cross-sectional return measurement, multiple testing, transaction costs, volatility forecasting and model risk. |

## Data regimes

| Regime | Role | Boundary |
|---|---|---|
| FNSPID, 2011--2023 | Primary statistical panel and bounded risk translation | 715,546 news-bearing firm-days, 570 priced symbols, 3,262 sessions; the training period ends 31 December 2019. |
| LSEG sector-33 backward block | Same-universe historical transfer check | Separate Reuters source-period; never pooled. |
| LSEG sector-33 recent block | Current Reuters portability and economic check | Separate Reuters source-period; never pooled. |

The split is chronological but not a pristine holdout because earlier project
design work observed later-period outcomes. The training-period analysis is
exploratory. Notebook 75 nevertheless froze these specific estimands before
opening their testing-period outcomes, executed once, and preserved the adverse
result without retuning.

## Variables and estimand

For firm \(i\) and assigned trading session \(t\), let story \(j\) have
continuous sentiment \(s_{ijt}\) and a hard negative indicator
\(I(s_{ijt}\text{ is negative})\). The two central firm-day summaries are

\[
  \bar{s}_{it}=\frac{1}{n_{it}}\sum_{j=1}^{n_{it}}s_{ijt},
  \qquad
  q^-_{it}=\frac{1}{n_{it}}\sum_{j=1}^{n_{it}}
  I(s_{ijt}\text{ is negative}).
\]

On each eligible date, estimate a cross-sectional regression in centred
percentile ranks:

\[
 \widetilde{r}_{i,t+1}=\alpha_t+\beta_t\widetilde{q^-}_{it}
 +\gamma_t\widetilde{\bar{s}}_{it}
 +\delta_t\widetilde{\log(1+n_{it})}+\boldsymbol{\theta}_t'\mathbf{x}_{it}
 +\varepsilon_{it},
\]

where \(r_{i,t+1}\) is the next-open abnormal return and \(\mathbf{x}_{it}\)
contains lagged one-session return, lagged five-session return and 20-session
volatility in the full-control specification. The primary estimand is the time
mean \(\bar\beta=T^{-1}\sum_t\beta_t\). It is an incremental conditional
association, not a causal effect.

Dates with fewer than ten complete firms or a rank-deficient design are
excluded. The time-series standard error of \(\bar\beta\) uses a five-lag
heteroskedasticity-and-autocorrelation-consistent estimator.

## Aggregation comparison

The initial family applies the same daily cross-sectional Spearman information
coefficient, horizon and sample to nine summaries:

1. mean hard sentiment label;
2. mean continuous sentiment;
3. median continuous sentiment;
4. trimmed mean;
5. negative-story share;
6. score dispersion;
7. strongest-event score;
8. log-count-weighted mean; and
9. decayed sentiment state.

Benjamini--Hochberg correction is applied across this nine-rule family. The
direct regression above is then used to test whether the surviving rule merely
relabels the mean or the amount of coverage.

## Economic and risk validation

- **Model-free scale:** within each date, firms are bucketed first by mean
  sentiment and then by negative share. The pooled high-minus-low return spread
  is reported in basis points with HAC uncertainty.
- **Directional implementation:** returns are charged 10 basis points per side,
  including final liquidation. Break-even cost is the per-side cost that makes
  mean net return zero.
- **Risk translation:** a heterogeneous autoregressive volatility forecast and
  volatility target form a sentiment-free base. A frozen negative-pressure
  hysteresis state can only scale exposure down, at 2 basis points per side. In
  FNSPID, current mapped-session news sets the same session's modifier, so the
  result is a contemporaneous diagnostic rather than an ex-ante strategy test.
- **Validation:** comparisons use block-bootstrap uncertainty, matched-exposure
  controls where relevant, half-sample and leave-one-company-out checks where
  available, and circular shifts of the risk-off schedule as a schedule-alignment
  diagnostic. The FNSPID matched constant is fitted ex post; the LSEG controls
  are selected in training folds.
- **Null power:** saved standard errors are converted into two-sided 5% tests
  with 80% power minimum detectable effects; ratios to the FNSPID estimate make
  an imprecise LSEG null interpretable.

## Frozen temporal replication

Notebook 75 repeats the nine-rule IC family and the base conditional coefficient
on the 203,393 testing-period firm-days spanning 998 sessions. Family A applies BH
to the nine aggregator ICs. Family B applies BH to the unchanged all-firm-day
coefficient and a predeclared \(n\geq2\) coefficient. An outcome-blind
amendment added prospective power context and the separate stability model

\[
  \beta_t=\alpha+\Delta\mathbf{1}\{t\text{ is testing}\}+u_t,
\]

with HAC(5) uncertainty, where
\(\Delta=\bar\beta_{testing}-\bar\beta_{training}\). This avoids the
fallacy that significance in the training period and non-significance in the testing period
alone establishes a difference. No testing-period trading, retuning, alternative
horizon, threshold or extra stratum is permitted.

## Claim rules

- “Contains incremental information” is allowed only for the FNSPID
  training-period association with its conditioning set and interval; it must be
  paired with the frozen temporal non-replication.
- “Temporal instability” identifies a change in the measured baseline pipeline
  association. It does not identify a market regime effect unless composition,
  mapping, timing and classifier alternatives have been separated.
- “Economic value” requires a cost-covering return or a downside improvement
  against exposure-matched controls, with uncertainty and stability gates.
- “Portable” requires compatible scale and actionable behaviour in a separate
  data regime, not merely the same coefficient sign.
- “No evidence” must be paired with power or a minimum detectable effect when
  the estimate is imprecise.
- Notebook 75 may be cited only from the promoted one-shot record. It may not be
  rerun or extended on the testing period.

## Reproducibility record

The manuscript will map every promoted number to its notebook, frozen
specification, aggregate input hashes and random seed. Licensed FNSPID and LSEG
text is excluded from the repository and from the dissertation.
