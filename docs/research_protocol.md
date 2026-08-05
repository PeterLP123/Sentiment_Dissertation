# Research Protocol

Last updated: 2026-08-03

> **Current status.** This protocol governs the closing `final_experiments/` phase. The final research question has **not** been selected. The live [final experiments plan](../final_experiments/plan.html) records operational decisions and checklist status; this document records the scientific constraints that remain binding across every candidate RQ.

Operational guides:

- [Final experiments README](../final_experiments/README.md)
- [Dissertation execution plan](dissertation_execution_plan.md)
- [Documentation home](README.md)

## Purpose

The closing phase asks how financial-news sentiment should be measured at firm-day level before it is interpreted as a market signal. Earlier work established model competence and produced several trading, surprise, portfolio-combination, and tail-risk results. The remaining work concentrates on story selection, weighting, aggregation, and conditioning.

The design rule is:

> **Fix the measurement, not the trading rule.** FinBERT is the primary scorer. The final question concerns what to do with the many stories it scores.

A null is a complete result when the panel, split, inference, multiplicity control, and limitations are valid.

## Authority And Change Control

When documents conflict:

1. `final_experiments/plan.html` controls live decisions and workstream status.
2. This protocol controls scientific discipline and interpretation.
3. `dissertation_execution_plan.md` controls stage order and promotion gates.
4. Dated protocols/results remain immutable evidence for their own historical designs.

A changed primary RQ, data spine, split, event grain, timing rule, scorer identity, outcome, inference unit, or multiplicity family must be recorded before evaluation analysis and receives a new experiment identity. Completed result artifacts are never overwritten.

## Research Question Status

The “Beyond the Mean” cross-model-agreement question from the 12 July protocol is now a **standing fallback/default**, not the selected final RQ.

Gate F1 will choose one primary RQ and at most one secondary after the data panel and filtering/distribution EDA are complete, but before any evaluation-return comparison is opened.

| Candidate | Question | Current standing |
| --- | --- | --- |
| **RQ-A: Aggregation** | How should multiple same-day stories be aggregated into one firm-day sentiment signal, and does a distribution-aware rule add information about post-news abnormal returns beyond the mean? | Leading candidate: strongest fit to the observed instability of company-day means. |
| **RQ-B: Filtering** | Does conditioning on story novelty and publisher identity improve the return relevance of news sentiment? | FNSPID publisher/story-family fields remain unavailable. The separate expanded-LSEG/Gemma publisher arm is complete as a non-pooled retrospective null; zero of four IC tests survives BH and no arm is economically viable. |
| **RQ-C: Surprise** | Is firm-level sentiment surprise, net of market return and general sentiment level, informative beyond sentiment level itself? | Prior NO-GO; only viable with the wider panel and materially different demeaning design. |
| **RQ-D: Agreement** | Does cross-model agreement add out-of-sample information beyond mean sentiment and the initial price reaction? | Standing fallback/default; enabling PhraseBank/LSEG artifacts from the July 12 design were not completed. |
| **RQ-E: Thresholds** | Can a learned firm- or sector-conditioned trade/no-trade threshold outperform a fixed band? | Secondary only. The expanded LSEG map has four firms per sector but too little history for learned sector gates; the planned arm still needs a point-in-time FNSPID sector table. |

### Gate F1 rules

Before choosing:

- Workstream 1 panel and attrition evidence must be complete.
- Workstream 2 may inspect news/filter/score distributions, but not evaluation returns.
- The choice must state what evidence was visible, what remained hidden, and why the chosen RQ is feasible on the available fields.
- The primary estimand, horizon set, model family, inference unit, and multiplicity family must be frozen at the same gate.
- No third primary design is invented after evaluation results are opened.

## Evidence Already Settled

The closing phase does not re-run questions already answered unless a documented integrity defect appears.

- **Scorer competence:** pinned ProsusAI/FinBERT is the primary scorer. It reached 0.811 accuracy and 0.806 macro-F1 on the 5,947-row clean benchmark; hosted LLMs did not beat it.
- **VADER exclusion (researcher decision 2026-08-03):** prior share-argmax results collapse to neutral on Reuters headlines. Preserve those historical results, but do not spend further compute on VADER or carry it into new tables, agreement metrics, or return analyses.
- **Scorer combination:** the seven-signal subset search did not support ensembles; the best observed singleton was indistinguishable from a demeaned selection-luck null.
- **Company-day instability:** full-population scoring materially changed daily signals despite similar marginal headline-score distributions.
- **Surprise prior:** the 17 July Kalman-surprise pilot was a NO-GO; level beat surprise on out-of-sample R-squared and only 28 evaluation date clusters were available.
- **Second moment:** FinBERT negative share is a required candidate aggregator because it produced the strongest corrected text result in the repository for next-session absolute abnormal return.
- **Tail risk:** the four-cell VaR/ES factorial is closed. News arrival improved FZ0 loss; semantic tone did not add a clear increment.
- **Trading:** the 33-stock robustness run failed after costs/turnover. Profitability is not a completion criterion.

## Data Policy

### Primary spine: FNSPID

The primary closing-stage panel uses the coherent FNSPID 2011–2023 cohort.

Current evidence chain:

- revised Gate-1 timing cohort: 574 firms and approximately 1.64 million deduplicated firm events;
- Moment-2 FinBERT checkpoint: 1,640,796 scored events;
- final firm-day panel: 715,546 news-bearing firm-days, 570 priced symbols, 3,262 sessions;
- market benchmark: SPY;
- current earnings join: local LSEG results calendar.

Primary results may use only the verified cohort/checkpoint/panel chain. Before dissertation promotion, re-verify the large FNSPID archives against their recorded SHA-256 values.

Known unavailable fields on the current checkpoint grain:

- point-in-time `availability_timestamp`;
- publisher identity suitable for publisher weighting;
- `story_family_id` revision lineage;
- precise intraday timestamps for most events;
- explicit sector classification.

Do not impute these fields or make claims that depend on them. A raw zstd rescan or separate mapping artifact is required to add them.

### Robustness spine: LSEG

The LSEG sector-33 and midcap-22 corpora provide precise `version_created`, `source_code`, story-family identity, recent Reuters coverage, and partial body text. They remain licensed local data.

They are an out-of-regime robustness arm only:

- never pool FNSPID and LSEG rows;
- never treat one source regime as extra observations for the other;
- report separate estimates and explain differences in universe, window, source mix, and timing quality;
- do not widen the LSEG universe unless the chosen RQ specifically requires it and the request budget is frozen first.

### Labeled benchmark

`Data/derived/labeled/financial_sentiment_v2.csv` remains the default competence benchmark. Never edit it manually. `Data/data.csv` is legacy Kaggle material with 514 documented corrupt neutral duplicates and is not an ambiguity signal.

Benchmark competence supports scorer choice; it is not evidence of market predictiveness.

### Earnings calendar

The FNSPID cohort calendar was gathered from LSEG Workspace into `final_experiments/data/earnings/`.

Safe aggregate facts:

- 23,139 issuer-matched result events;
- 22,883 quarterly/FY/H subset rows;
- 490 of 574 cohort symbols with at least one kept quarterly event;
- 12,303 BMO, 9,764 AMC, 202 during-market, and 614 unknown-time rows.

CSV/JSON/checkpoint payloads are licensed and ignored. The current gather has recoverable ticker-resolution and issuer-title-filter gaps. Workstream 6 must validate calendar dates against earnings/guidance news and carry the coverage caveat.

## Frozen Development/Evaluation Split

Recorded 2026-07-30 for the FNSPID primary panel:

- **development:** `session_date <= 2019-12-31`;
- **evaluation:** `session_date >= 2020-01-01`.

There is no gap day. Development contains 512,153 current panel rows; evaluation contains 203,393.

The 2020–2023 block includes the COVID regime and is a stress-period evaluation. Earlier FNSPID work influenced the design, so use the phrase **chronological evaluation block**, never “pristine holdout” or “confirmatory sample.”

No fitting, threshold search, aggregation-rule selection, subgroup selection, or multiplicity-family change may use evaluation rows.

## Event Grain, Timing, And Returns

### Grain

The current panel grain is one `(symbol, session_date)` row where:

- at least one scored news event exists;
- a valid split-adjusted stock open exists;
- a valid SPY open exists for the same session.

Multiple stories are aggregated before inference. Raw event count is not independent sample size.

### FNSPID timing

News timing inherits the revised Gate-1 mapping already applied in the FinBERT checkpoint: date-only news maps to the first XNYS session strictly after the stated date. This supports next-open estimands but does not establish intraday availability.

### LSEG timing

For LSEG robustness, use `version_created` in UTC, convert to `America/New_York`, add the declared processing buffer, and trade only at the first eligible XNYS open. Do not align from `news_date` alone.

### Earnings timing

Current mapping:

- BMO: same XNYS session;
- AMC: next XNYS session;
- unknown time: conservative next session;
- during-market: current gather maps to same session, but this includes pre-news open-to-release trading and must be a declared sensitivity or exclusion.

### Returns

The current panel stores one-session split-adjusted open-to-open stock and SPY returns, with `ar_open_h1 = stock - SPY`. Dividends are not back-adjusted.

If Gate F1 selects a post-news CAR outcome, freeze the estimation window, initial-reaction treatment, and horizon set before evaluation. The event session must not be silently included in a claimed post-event outcome.

## Workstream Protocols

### W1: Consolidated panel

Status: completed for the current FNSPID checkpoint.

Required evidence:

- input inventory and source identities;
- attrition table from event checkpoint to priced firm-days;
- coverage by month, firm, and story-count band;
- frozen split labels;
- earnings join and coverage;
- explicit null/unavailable field list.

The panel is selected on news and price availability. Treat this as a collider/selection limitation, not a solved problem.

### W2: Filtering and within-day distribution EDA

Before Gate F1:

- describe firm-day `n`, mean, median, dispersion, skew, extrema, and positive/neutral/negative shares;
- condition those moments on story-count bands;
- separate direct target, contextual, technical/recap, repetition, and routine-reporting classes where fields permit;
- build a seeded 150–200-item human audit before claiming filter precision or event-type validity;
- do not choose publisher tiers from realised returns.

On FNSPID, publisher/story-family limitations may force publisher weighting and revision novelty to the LSEG robustness arm. State that rather than fabricating proxies.

The expanded LSEG corpus now also supports a local full-text cash-flow-distance
audit: 17,508 matched Reuters body rows / 16,861 unique headline hashes. The
frozen audit is 200 events with a 60-event independent double-code subset.
Coder sheets must remain blind to sentiment, machine strata, identity/date
metadata, and returns. Before any return model using the human labels is
specified or opened, quadratic-weighted Cohen's kappa must be at least 0.60,
its 95% event-bootstrap lower bound at least 0.40, and adjacent agreement at
least 0.80. Exact agreement and the full confusion matrix are reported without
an additional pass/fail threshold.

### W3: Aggregation

The incumbent is the sign of the mean hard label. The comparison family must include:

- mean hard label;
- mean continuous score;
- median;
- trimmed mean where `n` supports it;
- negative share;
- dispersion/IQR;
- strongest-event selection;
- attention/novelty weighting where measurable;
- decayed state.

All rules use the same eligible rows, split, target, and inference. Report predictive metrics such as rank correlation/information coefficient alongside any portfolio translation. Stratify by story-count band. If portfolio outcomes are included, report turnover, concentration, transaction costs, and break-even cost.

Declare the complete `rules × horizons` multiplicity family at Gate F1 and apply Benjamini–Hochberg or the frozen alternative.

### W4: Sentiment surprise

A valid retry must differ materially from the failed July pilot:

1. raw firm-day level;
2. minus a strictly lagged firm baseline;
3. minus the cross-sectional daily sentiment level;
4. optional sector-day demeaning only after a valid sector map exists.

Return adjustment uses a declared fitted market model or the frozen SPY-adjusted alternative. The initial reaction remains separate. Compare level-only, level-plus-surprise, and surprise-only models on out-of-sample loss; coefficients are secondary.

The report must cite the earlier NO-GO and disclose all invalidated prior runs.

### W5: Story type

Use the existing 12-type taxonomy only after a seeded human audit. Estimate type interactions in one pooled model rather than selecting 12 independent winners. Declare the `types × horizons` family before evaluation and correct it.

The honest default is no type weighting. A type-weighted aggregator enters W3 only if the interaction survives its frozen correction.

### W6: Earnings-date effect

Use the quarterly LSEG calendar as the default. Add `sessions_to_earnings` and freeze pre/event/post windows. Validate that `earnings_guidance` news clusters around recorded dates.

Required comparisons:

- inside versus outside the earnings window;
- exclusion robustness dropping all earnings-window firm-days;
- timing sensitivity for during-market/unknown rows;
- clear separation from post-earnings-announcement drift.

### Learned thresholds

Secondary only. Establish the fixed-band rule first. Compare logistic, gradient-boosted, and small MLP gates using development data only. Prefer sector-conditioned to stock-specific thresholds once a sector table exists. Include a label-shuffle/permutation control and report the learned threshold surface, not only performance. After row-level gating, normalise surviving long and short rank legs separately to 0.5 gross; a one-sided selection must go flat rather than become a directional market bet.

## Inference And Multiplicity

Every final analysis records:

- unit of observation and cluster/block unit;
- number of independent evaluation dates;
- random seed;
- block length and bootstrap replications where used;
- full candidate family and correction method;
- development-only model/threshold selection;
- effect sizes and intervals, not only p/q-values;
- all nulls, failed arms, attrition, and deviations.

Date-clustered or date-block uncertainty is primary because firms share market dates. Firm-level sensitivity and time-block stability are secondary. Raw row counts must never be presented as independent sample size.

## Artifacts And Promotion

Exploratory outputs stay under `final_experiments/outputs/<stage>/` and are ignored. Notebooks and thin helpers are source-controlled.

A result can be promoted into the dissertation only when:

1. its Gate F1 design record exists;
2. the data spine, split, scorer revision, estimand, and inference are frozen;
3. evaluation was not used for selection;
4. attrition and data-quality limitations are present;
5. outputs reproduce from the tracked notebook/helper plus recorded local inputs;
6. aggregate, licence-safe figures/tables are copied deliberately;
7. the run is registered in `experiments/manifest.toml` with code/data identities.

Do not hand-type final result values into the dissertation.

## Interpretation Limits

- Predictive/associational, not causal.
- FinBERT tone is not investor belief, order flow, or market positioning.
- Date-only news timing leaves residual intraday ambiguity.
- FNSPID and LSEG differ in universe, provider, timing, and source regime.
- Current FNSPID publisher and story-family fields are unavailable on the checkpoint grain.
- Earnings coverage has recoverable mapping/filter holes and does not include consensus surprises.
- Prices are split-adjusted but not dividend-adjusted.
- Selection on news and price availability can distort relationships.
- The evaluation block is previously explored.
- A backtest is not deployable alpha or investment advice.

## Superseded July 12 Design

The earlier “Beyond the Mean” protocol specified PhraseBank agreement validation plus a four-scorer LSEG held-out event study with 19/27/31 July gates. Those dates, planned commands, and artifact paths are no longer the active critical path.

The scientific question remains available as RQ-D. The implementation record is retained in [LSEG core analysis workflow](lseg_analysis_workflow.md) and the repository history; it must not be represented as completed evidence.
