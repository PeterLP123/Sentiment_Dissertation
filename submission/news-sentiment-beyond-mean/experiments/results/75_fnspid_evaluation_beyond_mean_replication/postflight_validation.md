# Notebook 75 Postflight Validation

## Material Passport

- Origin skill: academic-research-suite / experiment-agent
- Mode: run and validate
- Generated: 2026-08-14
- Verification status: ANALYZED
- Version label: one-shot FNSPID evaluation opening, parent v1 plus pre-outcome
  amendment v1

## Verdict

**CAUTION — internally valid temporal non-replication, not pristine confirmation.**

The declared analysis completed once with no notebook error output. All input and
output hashes match the run manifest. Saved daily coefficient series exactly
reproduce the registered stability contrast. No outcome-based retuning, extra
stratum, trading rule, horizon, or retry was performed.

The central development estimate did not replicate. On 203,393 evaluation
firm-days and 998 sessions, the all-firm-day conditional coefficient is
`+0.005833` (95% HAC(5) CI `[-0.004261, +0.015928]`, BH q=`0.5148`) and the
`n >= 2` estimate is `+0.002952` (`[-0.009901, +0.015806]`, q=`0.6526`). The
nine-aggregator family has zero BH survivors; negative share has IC `-0.003120`
(p=`0.3002`).

The predeclared evaluation-minus-development contrast is `+0.014143` (HAC(5)
SE `0.005941`, 95% CI `[+0.002499, +0.025788]`, p=`0.01728`) across 2,264
development and 998 evaluation sessions. This supports a change in the
coefficient across eras and a point-estimate sign reversal. It is a secondary
stability diagnostic and does not rescue either failed primary family.

The registered prospective approximation gave only 35.4% power to recover the
development effect at the first BH-2 hurdle and an approximate 80%-power MDE of
0.01372. That limits what a simple null could establish. Here, however, the
formal contrast adds evidence of temporal instability; it still does not prove
the evaluation coefficient is exactly zero.

## Reproducibility checks

- Parent specification, amendment, source, helper, and panel hashes passed before
  execution.
- One-shot guard refused any pre-existing output path.
- Notebook: 8 executed code cells, zero error outputs.
- Family A: 9 rows, 0 BH rejections.
- Family B: 2 rows, 0 expected-direction BH passes.
- Daily-series recomputation: exact match to the saved stability contrast.
- Focused helper tests: 3 passed.
- Ruff and `git diff --check`: passed.
- Independent rerun: deliberately not performed because the frozen governance
  permits only one evaluation opening. Reproducibility is therefore supported by
  deterministic code, saved series, and hashes rather than a second execution.

## Statistical-fallacy scan

1. **Simpson's paradox — addressed.** Development and evaluation are reported
   separately and the regime contrast is explicit; no pooled coefficient hides
   the reversal.
2. **Ecological fallacy — bounded.** The estimand is a firm-day cross-sectional
   rank relationship; no claim is made about an individual headline or investor.
3. **Berkson's bias — caution.** Results apply to news-bearing, price-linked
   firm-days and need not generalise to all firms or no-news days.
4. **Collider bias — caution.** Conditioning on news count and complete-price
   eligibility may select the sample; controls are fixed pre-outcome, but the
   design is observational.
5. **Base-rate neglect — not applicable.** This is not a diagnostic classifier.
6. **Regression to the mean — addressed.** Aggregator selection used development;
   the frozen evaluation opening provides the adverse out-of-era check.
7. **Survivorship bias — caution.** The priced-symbol cohort reflects available
   mappings and histories; conclusions are bounded to that panel.
8. **Multiple comparisons — addressed with limits.** BH covers the declared
   nine- and two-test families, while the separate one-member stability family is
   labelled. The broader repository's iterative search remains a limitation.
9. **Garden of forking paths — caution.** The Notebook 75 estimands and amendment
   were frozen before their outcomes and no retry occurred, but the evaluation
   era had been opened for other studies and earlier evidence motivated RQ-A.
10. **Correlation versus causation — addressed.** The result is predictive
    association/instability, not a causal effect of news sentiment.
11. **Reverse causality — reduced, not eliminated.** The outcome begins after the
    formation open, but date-grain news timing lacks point-in-time timestamps for
    every event, so residual timing ambiguity remains.

## Dissertation-safe interpretation

The defensible paper is a temporal non-replication study: a development-era
negative-share increment survives mean sentiment, news volume, and lagged-price
controls, but it does not transport to 2020–2023 and the estimated coefficient
changes sign. The contribution is the disciplined separation of development
detectability, temporal stability, cross-source portability, and trading value.
It is not evidence that negative share is universally useless, that sentiment
never matters, or that reversing the signal would work.
