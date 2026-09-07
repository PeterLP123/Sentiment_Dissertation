# Dissertation writing plan

Target: approximately 10,000 words in the main body, excluding the abstract,
references and appendices.

The abstract should be approximately 300 words. UCL describes it as short and
non-technical rather than imposing a numerical cap. It should still contain the
problem, design, main statistical estimate, economic limit and LSEG portability
boundary. The final chapter receives a deliberately generous 1,500-word budget
so that the dissertation can answer the question, explain practical meaning,
separate limitations from null results, and set out prioritised future work.

## One-sentence thesis

Hard negative-story share has a selected assigned-window association in the FNSPID
training period, but the association is intraday and also appears before the
assigned open, so predictive timing is unresolved. The frozen 2020--2023
baseline does not replicate, and scale, costs and cross-source evidence rule out
a portable economic claim.

## Suggested allocation

| Chapter | Words | Job |
|---|---:|---|
| Introduction | 900 | State the aggregation problem, question, contribution and bounded claim |
| Literature review | 1,900 | Financial text, negative-news asymmetry, aggregation, predictive inference, costs and volatility targeting |
| Data | 1,100 | FNSPID spine, separate LSEG blocks, session mapping, labels, split and limitations |
| Methodology | 2,100 | Nine aggregators, conditional estimand, outcome-leg and price/beta controls, HAC/BH, cost and risk tests |
| Results | 2,500 | Selected association, timing diagnosis, temporal test, scale, costs, risk, LSEG and prompt nulls |
| Conclusions and future directions | 1,500 | Answer the RQ, explain statistical and practical meaning, evaluate strengths and limitations, and prioritise prospective tests |

The six main-body allocations sum to 10,000 words. The abstract is additional.

## Abstract structure (about 300 words)

1. **Problem and question (about 70 words).** Explain why averaging multiple
   same-company stories may discard an informative negative tail, then separate
   the training-period full-control question from frozen baseline replication.
2. **Design (about 90 words).** State the FNSPID panel size, nine aggregation
   rules, direct conditional regression, recent-price controls, chronological
   split, multiplicity correction, costs and separate Reuters/LSEG transfer.
3. **Findings (about 100 words).** Give the training-period baseline coefficient,
   report the full-control coefficient separately as robustness, then give the
   frozen testing-period baseline coefficient, direct era contrast, power
   limitation, model-free scale and cost no-go.
4. **Conclusion (about 40 words).** Say plainly that the selected training-period
   association does not replicate temporally and is not a portable,
   cost-covering rule.

## Final-chapter structure (about 1,500 words)

1. **Direct answer and contribution (350--400 words).** Answer the primary
   question with the principal estimate, then distinguish the statistical,
   economic and portability conclusions.
2. **What the result means in practice (250--300 words).** Explain why the
   result is useful as evidence about information compression but does not
   justify a live trading or risk-management rule.
3. **Strengths and limitations (350--400 words).** Cover the common inference
   regime, multiplicity and costs, followed by the exploratory history,
   non-pristine split, machine labels, unobserved story novelty, source shift
   and low-powered LSEG blocks.
4. **Further work (350--450 words).** Prioritise a genuinely new prospective
   sample, human validation, explicit novelty/story-family controls and
   lower-turnover event horizons. State the expected decision from each test
   and avoid a generic wishlist.

## Results narrative

### 1. The mean loses information

Lead with the aggregation comparison and conditional coefficient. Explain negative share in plain English: two days can have the same average mood, but one can contain a concentrated pocket of bad news.

Use `fig_aggregation_ic.png`, then the conditional result from Notebook 71.

### 2. Timing prevents a predictive interpretation

Report the Notebook 86 intraday/post-close split and Notebook 87 previous-open
probe together. State that the intraday association and strong earlier-window
coefficient make contemporaneous arrival or assignment lag live alternatives.
Withdraw “specific to next-open” wording.

### 3. The result is not just a fixed-beta or yesterday's-price artefact

This is the most important robustness section. Show `fig_reversal_confound.png` and report the full-control coefficient of -0.00914 with its interval and p-value. Say that the price controls strengthen rather than erase the negative-share association.

### 4. Statistical value is not the same as economic value

Use `fig_conditional_double_sort.png` to translate the effect into basis points. Then use `fig_turnover_breakeven.png` to show why it is not tradable at the assumed costs. This apparent disappointment is a substantive finding, not a failed dissertation.

### 5. The frozen temporal replication fails

Report Notebook 75 before moving to cross-source evidence. State that it repeats
the baseline mean/count model rather than the training-period price-control model. The all-firm-day
testing-period coefficient is +0.00583 with an interval crossing zero and BH
`q=0.515`; the multi-story estimate also fails, and the nine-rule family has
no survivor. Report the predeclared testing-minus-training contrast
(+0.01414, `p=0.0173`), 35.4% prospective power and 26.5% realised-precision
power together. Describe this as
temporal non-replication and pipeline-level instability, not evidence of an
exact zero, an identified market regime mechanism or a reason to reverse the
signal.

### 6. Risk management works only as a bounded case study

Present the FNSPID HAR modifier, adverse overall maximum drawdown,
schedule-shift, leave-one-episode-out and crash-exclusion evidence together.
The mapped-session schedule is aligned with lower squared downside in one
period, but it is not an ex-ante timing test and must immediately be paired with
the failed recent LSEG activation and backward-transfer limits.

### 7. LSEG is the portability test, not a second chance to tune

Report the negative LSEG coefficient signs, their wide intervals and MDEs. Then show the tuned economic tests:

- aggregate LSEG overlay: no complete gate pass;
- firm-level overlay: lower raw drawdown but worse than matched-exposure controls and very high turnover;
- prompt variants: all selection gates fail;
- structured anticipated-reaction score: selection gate fails.

This section demonstrates research discipline: increasingly flexible attempts do not rescue a weak economic effect.

## Figures to prioritise in the main text

1. `fig_aggregation_ic.png`
2. `fig_reversal_confound.png`
3. `fig_conditional_double_sort.png`
4. `fig_turnover_breakeven.png`
5. `fig_har_sentiment_equity.png`
6. `fig_downside_placebo.png`
7. one combined LSEG economic-limit figure, preferably `fig_lseg33_economic_case_and_limit.png`

Move the detailed LSEG tuning, firm-level and prompt figures to an appendix unless the supervisor wants a practitioner-heavy results chapter.

## Tables to prioritise

- core conditional coefficients and price-path controls;
- assigned-window outcome-leg decomposition;
- final hypothesis and transfer verdicts;
- aggregation-family BH results;
- LSEG transfer coefficients plus MDE ratios;
- economic-value gate summary;
- consolidated null and provenance table in the appendix.

## Phrases to use

- “incremental conditional association” rather than “causal effect”;
- “training-period association that does not replicate” rather than “alpha”;
- “does not cover realistic costs” rather than “unprofitable forever”;
- “not established in LSEG because estimates are imprecise” rather than “does not transfer”;
- “temporal non-replication” rather than “no effect”;
- “regime-specific downside evidence” rather than “risk-management signal.”

## Phrases to avoid

- “proves”;
- “predicts” without stating sample and conditioning;
- “tradable” or “deployable”;
- “LSEG confirms”;
- “AI understands market reaction”;
- “no effect” when the MDE shows low power.

## Final checks before submission

- replace all drafting notes in `manuscript/main.tex`;
- add formulas for all nine aggregation rules;
- reconcile every number against `experiments/results/`;
- cite the exact data and model versions;
- state that the aggregation family was exploratory and that the chronological testing period was not pristine;
- include the AI-use statement required by the programme;
- compile from a clean checkout and inspect the PDF visually.
