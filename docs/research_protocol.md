# Research Protocol

Last updated: 2026-08-05

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
| **RQ-B: Filtering** | Does conditioning on story novelty and publisher identity improve the return relevance of news sentiment? | FNSPID publisher/story-family fields remain unavailable. Expanded-LSEG/Gemma publisher conditioning is a four-arm retrospective null. A separate exact-family first-release screen removes 0.936% of hashes but does not rescue the frozen continuous rule; its +0.935-bps/session change is uncertain and the filtered strategy is significantly below cash. |
| **RQ-C: Surprise** | Is firm-level sentiment surprise, net of market return and general sentiment level, informative beyond sentiment level itself? | Prior NO-GO; only viable with the wider panel and materially different demeaning design. |
| **RQ-D: Agreement** | Does cross-model agreement add out-of-sample information beyond mean sentiment and the initial price reaction? | Standing fallback/default; enabling PhraseBank/LSEG artifacts from the July 12 design were not completed. |
| **RQ-E: Thresholds** | Can a learned firm- or sector-conditioned trade/no-trade threshold outperform a fixed band? | Secondary only. The expanded LSEG map has four firms per sector but too little history for learned sector gates. A fixed, retrospective sector-neutral/hysteresis translation improves gross/cost efficiency but has no BH survivor or viable 10-bps arm; the planned learned arm still needs a point-in-time FNSPID sector table. |

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
- **Trading:** the 33-stock robustness run and the firm-level risk-brake retry fail after costs/turnover. A later aggregate negative-pressure SPY hysteresis arm is descriptively better than long SPY but misses its frozen inference/risk gate. Profitability is not a completion criterion.

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

The separate expanded-LSEG Notebook 17 portfolio-translation family is
retrospective evidence, not a Gate-F1-selected W3 arm. Its best fixed rule,
sector extremes with top-half/bottom-half retention, has gross Sharpe 1.393 and
break-even 4.053 bps/side but trades on only 57 of 167 sessions, has gross
block-bootstrap p=0.226, and remains negative at 10 bps/side. Preserve it as a
turnover/coverage diagnostic and do not tune the opened window again.

Notebook 18 applies that exact frozen rule to paired FinBERT and Gemma scores.
The Gemma-minus-FinBERT gross mean difference is +1.034 bps/day, but its
five-session block-bootstrap interval [−4.863, +7.153] spans zero (p=0.731)
and changes sign across chronological halves. Moreover, the frozen
`strongest_event` rule returns missing when opposing stories tie at the maximum
absolute score. Discrete Gemma outputs create this tie on 3,507/7,348
firm-opens (47.73%), versus none for FinBERT, so the strategy comparison also
changes the opportunity set. Any future cross-scorer strongest-event design
must freeze a tie policy and either match activity or treat abstention as an
explicit component of the estimand. Do not describe the current result as a
clean scorer-quality comparison or validated alpha.

Notebooks 19–25 test sentiment only as a risk-sizing overlay around an
independently specified exposure. The firm-level momentum brakes are a clean
null: sparse exits and re-entries add cost without improving evaluation return.
The aggregate market-pressure hysteresis arm is the first economically
interesting follow-on: at 2 bps/side it raises the previously opened
2020–2023 SPY Sharpe from 0.568 to 0.783 and reduces maximum drawdown from
−32.05% to −28.01%. It does **not** pass the frozen gate: the paired net-return
interval spans zero, the corrected downside test fails, the drawdown reduction
is too small, and yearly results reverse in 2022. A post-result circular-shift
Sharpe p=0.051 is hypothesis-generating only; excluding 2020 leaves hysteresis
with Sharpe 0.418 versus 0.617 for long SPY. Preserve the exact 1.5/0.5
hysteresis rule as a candidate for genuinely new data; do not tune it on the
FNSPID or opened LSEG windows.

Notebook 21 establishes the sentiment-free HAR 10% volatility target as the
most useful independent base on this block: net Sharpe 0.673 and maximum
drawdown −15.73%, versus 0.568 and −32.05% for long SPY, at the cost of lower
mean exposure and total return. Adding aggregate pressure inside the variance
forecast does not improve QLIKE reliably and does not pass either strategy
gate. Notebook 22 then closes the surprise-to-trading bridge using development
only: all 12 fixed surprise arms fail at 10 bps/side, and the fully demeaned and
firm-demeaned signals have identical within-day ranks.

Notebook 23 rejects a standard one-session-lagged 200-session trend layer: it
reduces HAR net Sharpe to 0.431. The unchanged sentiment hysteresis repairs that
weak path to 0.677 but still does not beat HAR alone in total return or maximum
drawdown. Notebook 24 therefore tests the direct frozen product, HAR × aggregate
hysteresis. It has the strongest descriptive gross/net Sharpes in the chain
(0.878/0.835) and lower downside-squared loss than HAR (p=0.0148), but its paired
return interval spans zero, maximum drawdown is slightly worse, and its mean
advantage reverses over 2021–2023. Neither promotion gate passes. This is a
bounded risk-timing hypothesis for genuinely new data, not permission to tune
either component on the opened evaluation block.

Notebook 25 tests the same product over 2013–2019 with 84 expanding HAR fits;
each fit admits only variance targets ending strictly before its refit date.
The overlay again lowers downside-squared loss (p=0.0052), raises net Sharpe
from 1.114 to 1.164, and improves maximum drawdown from −12.53% to −8.57%.
It also lowers paired mean return by 0.134 bps/day, its interval spans zero,
total return falls from +96.91% to +93.13%, and the Sharpe gain misses the 0.10
gate. This older-date corroboration narrows the claim further: aggregate
sentiment may be a downside-risk modifier, but the current evidence does not
support persistent excess return.

Notebook 26 supplies a separate, non-pooled LSEG/Gemma mechanism check. The
unchanged aggregate rule cannot be judged honestly on the eight-month window:
its 126-session warm-up leaves 41 causal z-score sessions and only one risk-off
trigger, below the frozen 60-session/five-trigger feasibility floor. A semantic
firm-level alternative cuts a fixed-share basket holding to 25% for one interval
only when Gemma `strongest_event` is exactly −1.0. At 10 bps/side this raises
gross/net Sharpe from 1.500/1.472 to 1.614/1.526 and lowers downside-squared
loss after two-outcome BH (p=0.0088), but paired net improvement is only +0.148
bps/day [−0.405, +0.678] (p=0.588), and the Sharpe/drawdown promotion thresholds
are missed. Because Notebook 13 already opened this window and motivated the
`strongest_event` follow-up, the result is corroborating downside-mechanism
evidence, not fresh validation or alpha.

Notebook 27 asks whether the 25% sentiment state should be an absolute cap on
HAR rather than a multiplier. This is one discrete operator audit, not an
exposure sweep. The cap marginally improves product Sharpe before 2020
(1.176 versus 1.164) but reduces it in 2020–2023 (0.769 versus 0.835); paired
cap-minus-product return intervals span zero in both eras. It still reduces
downside-squared loss versus HAR after the declared four-test-per-outcome BH in
both eras (p=0.0064 and 0.0030) and has higher Sharpe than HAR in all four
reported subperiods. The frozen operator-selection gate fails, so the cap does
not replace the product. The defensible claim is narrower: the binary risk
state has repeated downside relevance, while these opened samples do not
identify how aggressively exposure should be cut. Do not test more composition
operators on these windows.

Notebook 28 tests whether the selected Gemma `strongest_event` arm can survive
costs as a sparse long-only rotation instead of a daily rank book. Exact −1
holdings are cut to 25%, and released current-open notional is transferred
equally to exact +1 firms when available. The rotation has the best descriptive
LSEG gross/net Sharpe at 10 bps/side (1.692/1.560) and its incremental gross
break-even over the cash brake is 23.46 bps/side. However, rotation-minus-brake
net mean is only +0.261 bps/day [−0.238, +0.807] (p=0.323), downside does not
improve, and the half-sample Sharpe difference changes sign. All gates fail.
Treat this as evidence that sparse reallocation can control turnover, not that
Gemma supplies validated incremental alpha; most performance remains the long
44-company basket.

Notebook 29 removes that basket and freezes a gross-1 dollar-neutral exact
extrema spread: 50% long exact +1 firms, 50% short exact −1 firms, one
open-to-open interval, and cash unless both signs exist. On 80/167 active
sessions it reaches gross/net Sharpe 2.558/1.185, +10.74% net return, and
18.52-bps gross break-even after charging 10 bps/side. The actual gross mean
beats 4,999 count-matched random firm assignments (one-sided p=0.0094), so the
predeclared mechanism gate passes. The stricter claims do not: paired net mean
is +6.49 bps/session [−4.52, +16.95] (p=0.238), first-half net Sharpe is −0.378
versus 2.912 in the second, and 66/80 active sessions have a one-name leg. All
44 leave-one-company-out replays remain profitable, but Goldman Sachs supplies
51.5% of gross profit. The two-name-per-side diagnostic is attractive at 10
bps (net Sharpe 2.088; +9.14%; break-even 44.26 bps) but has only 14 active
sessions. Treat this as the first strong, cost-surviving Gemma directional
mechanism; the opened window, uncertainty, temporal reversal, and concentration
rule out validated alpha or deployment. Do not tune it. The next valid test is
an independently extended time period under the exact frozen rule.

Notebook 30 audits Notebook 29's predeclared 25% single-name quality cap without
changing the selected firms, active sessions, horizon, or cost assumption. The
cap binds on 66/80 active sessions, scales both legs symmetrically to available
name capacity, and leaves unused gross in cash. At 10 bps/side, net Sharpe rises
from 1.185 to 1.697, maximum drawdown improves from −7.85% to −4.97%, mean
turnover falls 40.7%, and break-even rises from 18.52 to 23.24 bps/side while
net return remains +10.20%. Its standalone quality gate passes, but the stricter
claims do not: capped-minus-uncapped return is −0.514 bps/session
[−5.483, +4.291] (p=0.832), and capped-versus-cash return is +5.974 bps/session
[−0.973, +13.217] (p=0.100). Paired downside-squared loss improves after the
two-test BH correction (p=0.012), both half-sample Sharpes are non-negative, and
all 44 leave-one-company-out paths remain positive. Freeze it as a credible
risk-controlled extension candidate, not proven cap superiority or alpha. The
next valid test remains an independently extended time period under this exact
frozen capped rule.

Notebook 31 tests whether Notebook 30's result survives exact sector-dollar
neutrality. It subtracts each frozen four-company sector's mean target and
scales down only when the gross/name constraints require it. The local earnings
calendar has no overlap with the 2025–2026 LSEG panel, and exact within-sector
opposing signs supply only 9 sessions, so neither alternative passes the
return-blind 60-session feasibility floor. Demeaning reduces maximum sector
weight from 50% to zero and equal-weight-market beta from 0.100 to 0.031, but
net Sharpe/return fall from 1.697/+10.20% to −1.857/−6.78%. The paired return
loss is −10.118 bps/session [−15.491, −4.902] (BH p=0.0004). A post-result
arithmetic decomposition assigns 87.3% of the loss to removed gross return and
12.7% to added cost. This is a decisive factor-control null: the positive
sample result is not within-sector stock selection.

Notebook 32 then tests the complementary between-sector projection once,
explicitly because Notebook 31 exposed that component. Every capped target is
replaced by its equal-weight four-company sector mean; the strategy is never
scaled up. At 10 bps/side, its gross/net Sharpe is 3.805/1.868, net return
+6.42%, maximum drawdown −3.80%, break-even 19.36 bps/side, and maximum name
weight 12.5%. Actual sector placement beats 4,999 count-matched random firm
assignments (one-sided p=0.001), paired downside improves after BH (p=0.0094),
and all company/sector exclusion paths remain positive. However, projected-
minus-capped return is −2.198 bps/session [−7.604, +3.127] (p=0.418), the cash
interval crosses zero (p=0.113), and half Sharpes are −0.119/3.410. Freeze this
as a strong between-sector mechanism and one secondary arm for new dates,
while Notebook 30 remains the primary candidate because policy selection
failed. This is not validated alpha. Do not blend or resize the components on
this window.

Notebook 33 replays those two frozen rules on the immutable original 33-company
universe, defined before return access as the first three members of every
sector. Capped and projected gross/net Sharpes are 3.344/2.046 and 4.183/2.421;
net returns are +10.42% and +8.07%, break-even costs are 25.37 and 23.25
bps/side, and both chronological halves are positive. Both universe-quality
gates and the sector-placement randomisation gate pass. Neither alpha gate
passes after treating the two cash comparisons as one BH family; the projected
cash comparison is only nominally significant (p=0.0346). The fourth-member
addition cohort has only 3/167 eligible sessions and no multi-name sectors, so
it is stopped before returns. This is an opened-window universe-robustness
result, not an independent holdout, and must not trigger retuning.

Notebook 34 fixes Notebook 33's Gemma activity dates, daily +1/−1 counts, cap,
horizon, and costs, then substitutes FinBERT's within-session rank for name
selection. FinBERT has no exact +1/−1 strongest-event scores, so it cannot
produce the sparse trigger and the comparison is deliberately diagnostic.
FinBERT capped is descriptively stronger (net Sharpe 2.271 versus 2.046), while
Gemma projected is stronger (2.421 versus 1.630). Gemma-minus-FinBERT paired
return intervals span zero for capped [−9.277,+7.271] and projected
[−4.160,+7.153] bps/session; both selection gates fail and the scorer ordering
reverses across halves. This supports a new-date hypothesis that Gemma timing
and FinBERT ranking may be separable roles, but no hybrid, blend, or regime
switch may be fitted on the opened window.

Notebook 35 holds that hybrid construction fixed and circularly shifts the full
Gemma eligibility/count schedule through all 166 non-zero alignments. The null
therefore preserves event frequency, count distribution and clustering while
breaking alignment with FinBERT ranks and next-open returns. Observed capped
net mean is 6.802 bps/session versus 0.217 across shifts (98.19th percentile;
one-sided p=0.0240); projected is 3.213 versus −1.277 (97.59th percentile;
p=0.0299). Both net-mean timing tests survive their two-rule BH family, and both
rules retain break-even above 10 bps/side plus positive half-sample Sharpes.
Both timing gates pass. The permitted claim is that Gemma's exact-event schedule
contains unusual when-to-trade information for this retrospective hybrid. The
circular-stationarity and endpoint-wrap assumptions, opened sample, and
post-result construction chain still prohibit an alpha or deployment claim.

Notebook 36 is the direct audit of the one capped hybrid retained from that
chain. It admits no parameter sweep: Gemma fixes exact-event dates and counts,
FinBERT ranks names, the universe is the original 33, the cap is 25%, the
horizon is one session, and costs are 10 bps/side with final liquidation. A
9,999-replication five-session block comparison with cash gives +6.802
bps/session [+0.414,+13.371] (p=0.039), gross/net Sharpe is 3.700/2.271,
break-even is 25.24 bps/side, both chronological halves are positive, and all
rebuilt company/sector exclusions remain positive. The retrospective cash and
dependence gates pass. The predeclared HAC(5) market-plus-ten-sector-spread
intercept is +4.823 bps/session [−0.436,+10.083] (p=0.072), so the stricter
factor-alpha gate fails. The permitted claim is a leading opened-window
strategy candidate with evidence against single-company and single-sector
dependence. It is not validated alpha; only new-date or separately frozen
external evidence can promote it.

Notebook 37 falsifies portability of the exact numeric endpoint used to define
that event schedule. On 85,960 identical hashes, the older partial Cerebras
Gemma 31B checkpoint emits 10,172 exact ±1 scores versus 95 for the complete
DeepInfra Gemma 26B checkpoint, a 107.1-fold difference. Identical firm-open
aggregation gives only 5 shared eligible dates out of 85 in the union
(Jaccard 0.059); positive/negative daily-count correlations are 0.035/0.016.
Matched-coverage 26B has only six eligible sessions and therefore fails the
pre-return 60-session floor. The permitted 31B-trigger/FinBERT-rank capped
stress test has gross/net Sharpe 1.624/−0.543, −2.73% net return, break-even
7.50 bps/side, negative half-sample Sharpes, and all-shift timing p=0.204.
Both transfer and timing gates fail. The exact equality is a calibration and
quantisation property, not a scorer-independent semantic class. Do not reuse
it as the confirmatory event definition: a calibration-aware semantic rule
must be frozen using return-independent evidence before any new-date replay.

Notebook 38 tests one such return-independent measurement rule using the public
1,000-row validation checkpoint. The lowest directional threshold satisfying
the calibration requirements is 0.65, and the held-out audit precision is
82.9% positive and 83.0% negative. The measurement gate passes, but the frozen
binary firm-open construction does not: 5,472 of 5,511 original-33 firm-opens
with a selected direction contain both positive and negative selected headlines
(99.29%), so collision abstention leaves only two active sessions. Gross/net
Sharpe is −0.755/−1.013, cash p=0.577, all-shift timing p=0.743, and the
sector-spanning alpha p=0.428; all return and dependence gates fail. This is a
structural NO-GO for independent binary per-headline thresholds, not a failure
of public sentiment calibration. Any later confirmatory rule must preserve a
continuous confidence or directional margin at firm-open grain and be frozen
for genuinely new dates. Do not choose that resolution rule on these opened
returns.

Notebook 39 supplies the required continuous alternative without loading any
price or return column. Across 85,960 hash-matched headlines, 31B and 26B signed
scores have Spearman 0.953 and 89.2% sign agreement. Mean signed score at
firm-open has Spearman 0.983, 93.9% sign agreement, median daily rank
correlation 0.977, and top-two/bottom-two Jaccard 0.720/0.864. The predeclared
scale-free schedule—current cross-sectional q90−q10 spread at or above its
strictly prior trailing-60 75th percentile after 40 observations—has 22 active
dates per model, 17 shared (Jaccard 0.630), and shared-date name Jaccard
0.647/0.922. Mean continuous is the first reducer in the frozen simplicity
order to pass every portability gate; strongest event fails both measurement
and schedule gates. Applying the rule input-only to all 888,155 26B labels
produces 40 active sessions among 127 post-warm-up sessions in the complete
167-session rectangle. Therefore freeze mean continuous, adaptive dispersion,
top-two/bottom-two names, 25% cap, one-session hold and 10-bps/side cost for a
genuinely new-date replay. Do not evaluate or modify this rule on the already
opened return window. The result supports measurement portability and strategy
feasibility, not alpha.

Notebook 40 supplies a post-research cross-regime economic stress test rather
than pristine confirmation. Before reading FNSPID returns, the rule is
translated to all priced news-bearing FNSPID names with relative breadth
`ceil(n × 2/33)` and a two-name-per-leg minimum; reducer, dispersion schedule,
cap, gross exposure, horizon, accounting, and 10-bps/side cost remain fixed.
Because only 9/33 original LSEG names overlap and FNSPID carries FinBERT scores,
this tests construction transfer, not same-scorer or same-universe replication.
The evaluation gross Sharpe is 1.056 with +27.29% gross return, but break-even is
4.904 bps/side and 10-bps net Sharpe/return are −1.103/−23.27%. Net mean is
significantly below cash (−2.585 bps/session, [−4.662, −0.410], p=0.0185), both
evaluation halves are negative after cost, and development gross Sharpe is only
0.187 with 0.532-bps break-even. Conditional name selection is nominally
promising (p=0.035) but fails the declared two-test BH family (q=0.070); timing
and factor-alpha gates also fail. The permitted claim is that the continuous
sentiment rank contains diversified gross evaluation-era information across a
broad panel, while turnover, costs, and cross-era instability prevent deployment.
No breadth, threshold, horizon, or cost assumption may now be tuned on these
outcomes. Any next rule must be frozen from return-free signal persistence and
turnover diagnostics, then tested on genuinely new dates.

Notebook 41 executes that input-only diagnostic across the complete LSEG/Gemma
schedule and both FNSPID eras. Its sole predeclared candidate retains the
episode-opening basket through consecutive active sessions and exits when the
dispersion gate turns off. Although target turnover falls 32.35%–43.30%, the
opening names do not remain the current extrema: continuation long/short
Jaccard is 0.275/0.157 for LSEG/Gemma and only 0.024–0.035 in FNSPID. All
persistence gates fail. Median episode length is one, and entry/exit alone
implies 11.03–12.84 bps of target-only cost per active session at the frozen
cost. The candidate is rejected without reading a return. This closes the
obvious holding-period shortcut: a future slower signal must have a separately
motivated semantic state and be frozen on input evidence before new-date
returns, rather than retaining stale daily sentiment ranks.

Notebook 42 adds a one-shot backward external-time stress test without changing
Notebook 39's rule. A separate local checkpoint contains 41,019 deduplicated
2009-Q4/2010 FNSPID headlines scored with the cached FinBERT snapshot; late
2009 supplies return-free warm-up only. The established valid-current-open
firm-day screen leaves 24,263 aggregate firm-sessions / 537 symbols. The frozen
input gate clears with 59 active 2010 sessions (30/29 by half), after which all
selected returns are available. Gross/net Sharpe is −1.141/−3.397 and
break-even is −4.802 bps/side. Net mean is significantly below cash at −6.806
bps/session [−11.094, −2.819] (p=0.0011); both halves, both BH-controlled
mechanism tests, and the factor-alpha gate fail. The first execution is
invalidated because file-level price eligibility admitted later IPOs and forced
95 sessions to cash; only the corrected firm-day-eligible result may be quoted.

This is not a pristine holdout because the cohort is conditioned on later
2011–2023 coverage, but it is sufficient to reject a stable cross-era reading
of Notebook 40's positive 2020–2023 gross result. Do not reverse sentiment,
change breadth, or retune the dispersion threshold on 2010.

Notebook 43 then tests the remaining LSEG revision-weighting concern directly.
It reconstructs terminal-suffix `story_id` families, keeps the earliest
timestamped release of each family without using a return, and applies Notebook
39's exact original-33 continuous Gemma construction. The filter removes 8,314
later-revision-only hashes (0.936%), changes 1,938 firm-opens and eight decision
dates, and raises active dates from 40 to 43. All-headline versus
first-release-only gross Sharpe is −1.737/−1.473; net Sharpe is
−2.635/−2.502 and net return is −18.13%/−16.82%. The filtered-minus-baseline
effect is +0.935 bps/session [−0.652,+2.619] with BH q=0.262, while the filtered
rule is significantly below cash at −10.786 bps/session (BH q=0.0366). Both
halves and both frozen gates fail.

Because the LSEG return window was already opened, Notebook 43 is bounded
retrospective evidence rather than confirmation. It nevertheless rejects
story-family overweighting as the explanation for the failed economics and
provides direct LSEG/Gemma evidence that Notebook 39 is a portable measurement
construction, not historical alpha. Do not tune another collapse or
revision-weighting policy. Notebook 47's later first-to-current delta test asks
a distinct information question and remains bounded by the same opened-window
restriction. Any new-date replay must retain the exact rule and be framed
against this negative prior evidence.

Notebook 44 is the aggregate-only strategy evidence synthesis. It loads source
CSVs/manifests, never row-level prices, returns, positions, headlines, or
licensed text. Its evidence tier is an ordinal governance aid rather than a
statistical weight: independent price-only benchmark; frozen chronological or
external-time test; cross-period stress test; bounded retrospective audit; and
post-result diagnostic. Sharpe ratios across different universes, dates, costs,
and selection histories are not pooled or treated as directly comparable.

A sentiment rule is eligible for promotion only if its paired net-return
interval is favourable and excludes zero, its effect is temporally stable, its
local predeclared gate passes, and the evidence tier is at least chronological
or external-time. After the Notebook 50 refresh, one of eleven paired sentiment
or construction comparisons
in Notebook 44 clears the interval condition and six show lower downside loss
under their source conventions. The sole favourable interval is Notebook 47's
bounded retrospective revision-delta comparison with a matched latest-level
construction; it is not a cash-alpha result and does not clear the remaining
promotion conditions. Those p-values are dependent and are not meta-analysed.
Retain the sentiment-free HAR volatility target as the
conventional research benchmark, but not as a time-stable or
deployment-qualified strategy. Keep the exact HAR × sentiment product frozen
only for genuinely new dates and stop historical strategy tuning. This is a
Gate F1 evidence input, not a selection of the final RQ; the human novelty-audit
condition remains binding.

Notebook 45 is the single permitted post-result diagnostic of the repeated
downside result. It enumerates every circular placement of the complete frozen
risk-state schedule in three disjoint regimes: the 2013–2019 walk-forward HAR
product, the 2020–2023 HAR product, and the 2025–2026 LSEG/Gemma exact-negative
firm brake. Shifts preserve state frequency, duration, and cost accounting; the
LSEG shifts move the full 44-company flag row together, retaining firm identity
and cross-sectional concentration. The primary estimand is the mean reduction
in net downside-squared return relative to the local base. One-sided exact
shift p-values form a three-test BH family; mean net return is a separate
three-test family. No shift, threshold, severity, or strategy is selected.

Only the 2020–2023 downside result is unusually timed after BH (exact p=0.0100,
q=0.0301). The 2013–2019 schedule is in the 95.5th percentile but misses BH
(p=0.0455, q=0.0682), and the LSEG/Gemma brake does not beat generic placements
(p=q=0.3653). All return-timing gates fail. Therefore the cross-regime timing
gate fails: describe the evidence as one-period aggregate market-risk timing,
not a general semantic timing mechanism or Gemma alpha. Later price rows are
not a prospective sample because the local LSEG news boundary remains
2026-06-26. Do not open them without matching new news and a frozen prospective
population.

Notebook 46 is the external-time transfer of the independent base, not a new
fit. It applies Notebook 21's unchanged 2011–2019 HAR coefficients and QLIKE
scale to the 167-session 2025–2026 LSEG S&P 500 window. The forecast-transfer
gate requires the circular-block interval for naive-minus-HAR QLIKE improvement
to exclude zero. The base-quality gate also requires positive net performance,
annualised volatility no greater than 12.5%, no-worse drawdown than the index,
and nonnegative Sharpe in both chronological halves. A constant mean-matched
exposure arm separates volatility scaling from forecast timing.

The optional Gemma arm is fixed without return fitting as
`1 − 0.75 × exact-negative firms / 44`, the equal-notional market aggregate of
Notebook 26's exact firm brake. Its return and downside increments versus HAR
form one two-test BH family; comparison with a constant mean-matched multiplier
is secondary mechanism evidence. The LSEG company-return family was opened in
earlier notebooks, so no sentiment result here can establish alpha.

HAR produces 9.74% annualised volatility and −7.77% maximum drawdown, but net
Sharpe is 0.240 versus 0.940 for constant matched exposure. Its paired timing
effect is −2.731 bps/session [−4.394,−1.360], and the QLIKE improvement interval
includes zero; both external-time base gates fail. The Gemma modifier adds
+0.013 bps/session and +0.007 Sharpe. Downside reduction versus HAR passes the
two-test family (BH q=0.0244), but drawdown improves only 1.4% and the downside
advantage is absent versus a constant multiplier (p=0.8218). Record this as
mechanical de-risking, not semantic timing or alpha. No deployment-qualified
base remains.

Notebook 47 is a bounded retrospective story-revision test, not a new-date
validation. For each Reuters story family, it computes current minus first
Gemma signed sentiment and keeps only companies attached to both endpoints.
The current revision maps to the first open at least 15 minutes later; the
latest transition within each family/company/open is retained, and transitions
are averaged within firm-open. On the immutable original 33, the frozen
one-session gross-1 spread is long positive revision deltas and short negative
deltas, with equal legs, a 25% name cap, unused capacity in cash, final
liquidation, and 10 bps/side. A matched comparator holds the same dates, firms,
long/short counts, and gross exposure while ranking latest Gemma score levels.

The hash-only reconstruction yields 8,654 eligible first-to-current
transitions, 1,606 nonzero firm-open updates, and 165 both-sign active sessions.
Revision delta has gross/net Sharpe 1.523/−1.866 and 4.49-bps/side break-even.
Its paired net mean versus cash is −8.268 bps/session [−17.899,+1.200]
(BH q=0.0914), but it beats the matched latest-level construction by +15.620
bps/session [+2.683,+28.725] (BH q=0.0368). Gross Sharpe is positive in both
chronological halves; net Sharpe is negative in both. Record incremental
revision information relative to score level, but do not call it cash alpha,
factor alpha, or a deployable strategy.

Notebook 48 tests the single parameter-free implementation implication from
Notebook 47: carry each firm's last nonzero revision sign until the next
nonzero update. It does not select an expiry, decay, threshold, lookback,
holding period, or cost. The input gate requires at least 120 active sessions,
50 per half, at least 60% target-turnover reduction, median state age no more
than five sessions, p90 age no more than 20, and the unchanged 25% cap. The
strategy and Notebook 47 event arm form a two-comparison BH family versus cash
and versus each other; factor, temporal-half, and company/sector leave-one-out
checks remain binding.

The input gate passes: turnover falls 72.3% to 0.416 per session, median/p90
age is 2/18 sessions, and all 167 sessions are active. Gross/net Sharpe is
1.459/−0.331, net return is −1.39%, and break-even is 8.15 bps/side. Net Sharpe
is positive at 1–5 bps/side but not at the frozen 10 bps. The persistent-minus-
event effect is +7.498 bps/session [−0.006,+15.165] (BH q=0.1064), the
cash interval includes zero, the factor intercept is negative, 10-bps half
Sharpes are +0.434/−1.074, and only 2/33 company and 2/11 sector exclusions are
positive. Every promotion gate fails. Preserve the turnover and cost frontier
as useful implementation evidence and stop persistence tuning on this opened
window.

Notebook 49 is the final permitted semantic restriction of the revision signal.
It is frozen input-first and selects only families satisfying the scale-free
condition `initial_score × current_score < 0`; zero endpoints abstain. The
current revised sign sets direction, multiple flip families are averaged within
firm-open, and the original-33 one-session equal-leg 25%-cap construction and
10-bps/side cost remain unchanged. There is no magnitude threshold, duration,
or alternative sign definition. The broad Notebook 47 revision arm must
reproduce exactly. Net comparisons with cash and the broad arm form one BH
family; a conditional random-name test preserves every active date, leg count,
feasible gross, and same-day revision-update universe.

The return-free gate passes with 467 transition associations, 390 firm-opens
across 32 companies, and 74 both-sign sessions split 46/28 by half. The strict
subset has gross/net Sharpe 0.256/−1.646, net return −7.17%, and only
1.34-bps/side break-even. Its cash difference is −4.366 bps/session
[−9.428,+0.388] (BH q=0.161); its difference from broad revision delta is
+3.902 bps/session [−5.289,+13.007] (BH q=0.409). The conditional random-name
one-sided p-value is 0.4227, gross half Sharpes are +0.452/−0.008, and the
sector-spanning HAC intercept is −5.524 bps/session
[−10.658,−0.390] (p=0.0350). Every promotion gate fails. Record that dramatic
tone reversals do not explain Notebook 47's incremental gross information and
stop all further revision filtering on this opened window.

Notebook 50 is a separate, conventional risk-allocation audit of the leading
Notebook 36 LSEG/Gemma hybrid; it is not another sentiment threshold or event
filter. Before reading the candidate's forward returns, it freezes the exact
Gemma event schedule/counts and FinBERT-selected names, then applies the
repository's existing inverse-volatility projector using a strictly lagged
20-session open-return standard deviation, five-observation minimum, 50-bps
floor, 25% name cap, exact dollar neutrality, no added leverage, and unused
capacity in cash. The one-session horizon, final liquidation, and 10-bps/side
cost remain unchanged. Paired candidate-minus-cash and candidate-minus-equal-
weight net-return tests form one two-comparison BH family; both chronological
halves, the Notebook 36 factor control, and rebuilt company/sector exclusions
remain binding.

The return-blind gate passes with all 70 event sessions retained (21/49 by
half), complete selected-name volatilities, an 11.7% target-turnover reduction,
and exact cap/neutrality compliance. Realised annualised net volatility falls
from 7.55% to 6.47% and total turnover from 74.56 to 65.86. The inverse-
volatility arm retains gross/net Sharpe 3.691/2.207, +9.77% net return, and
24.37-bps/side break-even, but the equal-weight hybrid remains slightly
stronger at 3.700/2.271 and +11.82%. The paired difference is −1.136
bps/session [−2.679,+0.216] (BH q=0.123); cash BH q=0.111 and the
sector-spanning alpha interval [−0.656,+8.712] bps/session (p=0.0919) also
fail. Both half Sharpes and every company/sector exclusion remain positive.
Record a lower-risk/lower-turnover implementation trade-off, not policy
superiority or validated alpha. The equal-weight hybrid remains the leading
same-window candidate; any confirmation still requires genuinely new dates.
The exact forward contract is now source-controlled at
`final_experiments/frozen_specs/lseg_gemma_finbert_hybrid_prospective_v1.json`.
It excludes the opened window, requires at least 120 complete calendar sessions,
60 primary active sessions, and 20 active sessions per chronological half, and
fails closed on scorer, prompt, provider, privacy, timing, or universe drift.

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
- Cash is assigned zero return in the risk-overlay notebooks rather than a historical risk-free rate.
- Selection on news and price availability can distort relationships.
- The evaluation block is previously explored.
- A backtest is not deployable alpha or investment advice.

## Superseded July 12 Design

The earlier “Beyond the Mean” protocol specified PhraseBank agreement validation plus a four-scorer LSEG held-out event study with 19/27/31 July gates. Those dates, planned commands, and artifact paths are no longer the active critical path.

The scientific question remains available as RQ-D. The implementation record is retained in [LSEG core analysis workflow](lseg_analysis_workflow.md) and the repository history; it must not be represented as completed evidence.
