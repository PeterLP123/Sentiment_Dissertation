# Devil's advocate checkpoint 1

Date: 13 August 2026.

Decision: **proceed, but only with a narrower claim and explicit unresolved
confounds.**

## Strongest objections

### 1. The headline was selected after seeing the aggregation family

Negative-story share became the dissertation focus because it was the only
Benjamini--Hochberg survivor. A reader could reasonably call the primary story
post-selection rather than confirmatory.

**Mitigation:** label the aggregation result and direct FNSPID tests as
exploratory; publish the complete nine-rule family and the wider null ledger;
reserve confirmation for the sealed FNSPID evaluation replication or a new
prospective LSEG period. Do not use “out of sample proof” language.

### 2. Negative share may proxy repeated coverage rather than information shape

Story count is controlled, but the current FNSPID checkpoint lacks reliable
publisher and story-family identifiers. Ten rewrites of one bad event can raise
negative share without supplying ten independent pieces of information.

**Mitigation:** state this as the main measurement limitation. The closest
future test should cluster near-duplicate stories by embeddings or Reuters story
chains before computing firm-day summaries. Until then, interpret negative
share as a property of the observed news flow, not a count of independent bad
events.

### 3. A statistically clean rank coefficient may have no practical meaning

The full-control coefficient is statistically detectable, yet the model-free
spread is only about -0.62 basis points per session with an interval crossing
zero. The best break-even trading cost is far below the 10-basis-point charge.

**Mitigation:** make the separation between information and implementation the
central result. Do not market the rule as alpha. Use the double sort and
break-even figure before discussing the risk overlay.

### 4. The risk result can look like a rescue attempt

Moving from an unprofitable directional signal to risk management after seeing
results may be interpreted as goalpost movement. The FNSPID overlay works in
one period and the exact recent LSEG rule never activates.

**Mitigation:** present risk translation as a bounded secondary case study on
top of an independently motivated HAR base. Lead with the mapped-session
schedule-shift result, state that it is not an ex-ante timing test, and
immediately pair it with the failed LSEG activation and tuned-rule gates. Never
call it a production risk-management signal.

### 5. The LSEG result is underpowered and cannot establish failure

Both LSEG conditional coefficients are negative, but their intervals are wide.
An examiner could reject the phrase “does not transfer” as an absence-of-
evidence error.

**Mitigation:** report the estimates, intervals and minimum detectable effect
ratios together. Say that portability is not established; do not say the true
effect is zero.

### 6. Model labels are not human-validated on the final corpora

FinBERT is domain adapted, but model accuracy on a benchmark is not proof that
its story labels are correct for FNSPID or Reuters headlines. A model artefact
could create the negative tail.

**Mitigation:** freeze the scorer to avoid selection on returns, describe its
training domain, and make a stratified human-label audit a priority for further
work. Prompt experiments remain a failed robustness arm, not validation of the
base labels.

## What would falsify the current interpretation?

- The full-control coefficient disappears after near-duplicate story families
  are collapsed.
- A prospectively sealed FNSPID or LSEG sample produces a coefficient near zero
  with enough power to detect the training-period estimate.
- Human labels show that the negative-share tail is dominated by systematic
  classifier errors.
- A matched no-news or random-story placebo reproduces the same coefficient.

## Required manuscript changes

1. Use one primary question about incremental information, not a trading claim.
2. Put selection history and story-family limitation in the introduction,
   methodology and conclusion.
3. Treat economic no-go and LSEG imprecision as results, not caveats buried at
   the end.
4. Use “not established” for portability and “regime-specific” for downside
   timing.
5. Give future work decision rules, not a generic list of possible models.
