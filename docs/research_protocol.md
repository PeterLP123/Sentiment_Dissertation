# Research Protocol

Last updated: 2026-08-14

> **Current status.** This protocol governs the closing `final_experiments/` phase. Gate F1 selected the aggregation/temporal-stability research question on 2026-08-14, before Notebook 75 opened its declared outcomes. Notebook 75 then recorded temporal non-replication. The live [final experiments plan](../final_experiments/plan.html) records operational decisions and checklist status; this document records the scientific constraints and result boundaries that remain binding.

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

Gate F1 was taken on 2026-08-14 under the researcher's direct instruction, before
Notebook 75 opened any of its evaluation statistics. The selected primary RQ is:

> **Does negative-news share add predictive information for next-session
> abnormal stock returns beyond mean continuous sentiment and news volume, and
> is that increment stable across time?**

No separate secondary RQ was selected. The “Beyond the Mean” cross-model-
agreement question from the 12 July protocol is retired as a standing fallback,
not silently relabelled as the completed question. The Gate F1 evidence boundary
and dropped candidates are recorded in
`final_experiments/outputs/02_filters_and_distribution/decision.md`.

| Candidate | Question | Current standing |
| --- | --- | --- |
| **RQ-A: Aggregation** | How should multiple same-day stories be aggregated into one firm-day sentiment signal, and does a distribution-aware rule add information about post-news abnormal returns beyond the mean? | **Selected and completed as a temporal-stability question.** Development supports a negative-share increment, but Notebook 75 records zero evaluation BH survivors and a positive evaluation-minus-development contrast with an interval above zero. The paper result is temporal non-replication, not stable superiority. |
| **RQ-B: Filtering** | Does conditioning on story novelty and publisher identity improve the return relevance of news sentiment? | Not selected. FNSPID publisher/story-family fields remain unavailable, the human audit is incomplete, and retrospective LSEG filter arms are null. |
| **RQ-C: Surprise** | Is firm-level sentiment surprise, net of market return and general sentiment level, informative beyond sentiment level itself? | Not selected. Prior NO-GO and later evaluation exposure make it a weaker final design. |
| **RQ-D: Agreement** | Does cross-model agreement add out-of-sample information beyond mean sentiment and the initial price reaction? | Not selected. Enabling PhraseBank/LSEG model-panel and human-agreement artifacts from the July 12 design were not completed. |
| **RQ-E: Thresholds** | Can a learned firm- or sector-conditioned trade/no-trade threshold outperform a fixed band? | Not selected. Existing policies fail, a stable base signal is absent, and the planned arm still lacks a point-in-time FNSPID sector table. |

### Gate F1 record and rules

The following rules governed the 2026-08-14 choice:

- Workstream 1 panel and attrition evidence must be complete.
- Workstream 2 may inspect news/filter/score distributions, but not evaluation returns.
- The choice must state what evidence was visible, what remained hidden, and why the chosen RQ is feasible on the available fields.
- The primary estimand, horizon set, model family, inference unit, and multiplicity family must be frozen at the same gate.
- No replacement primary design is invented after the adverse evaluation result.

The visible evidence included the development aggregation/conditional results,
the development-only diagnostics and price-path control, the non-pooled LSEG
portability arms, the model-free magnitude and precision audits, and the wider
null strategy/prompt evidence through Notebook 84. Every Notebook 75 statistic
remained hidden. Its parent design froze the BH-9 and BH-2 families; the
outcome-blind 2026-08-14 amendment added only a separate HAC(5) temporal contrast,
prospective power context, provenance clarification and a fail-closed one-shot
guard. No evaluation trading or additional estimand was added.

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
an additional pass/fail threshold. Because `NA` is not ordinal, kappa excludes
pairs containing `NA` and at least 48/60 pairs must be jointly numeric. Exact
and adjacent agreement retain all 60 pairs (two `NA` labels agree; mixed
numeric/`NA` does not). The 95% interval uses 9,999 event resamples with seed
`20260819`; undefined kappa replicates are omitted. These details were frozen
before coding.

Notebook 16 now binds coder IDs/headlines/bodies with immutable semantic hashes;
do not rerun it after coding starts. Notebook 62 validates those identities and
executes the reliability gate only after both sheets are complete. Its current
aggregate state is A 0/200 and B 0/60, so no reliability metric, machine proxy,
sentiment score, or return has been opened.

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
or external-time. After the Notebook 65 refresh, one of fifteen paired sentiment
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

Notebook 55 is the aggregate-only reconciliation of the previously frozen
22-company LSEG mid-cap FinBERT event strategy. It reads only the committed TOML
plus the local immutable run manifest, period metrics, block-bootstrap summary,
and per-stock aggregate contributions. It must not read corpus, scores, prices,
daily P&L, positions, session construction, or headline text; it launches no
scorer and constructs no return. The canonical parsed-config identity and every
loaded result hash must match the run manifest. Preserve both decisions: the
legacy local gate passes because development/evaluation net return and Sharpe
are positive with at least 20 active sessions and break-even above 10 bps/side;
the final promotion gate fails because the run is previously explored tier-2
evidence, its evaluation cash interval is [−8.86,+19.34] bps/session, and it is
not eligible temporal confirmation. Maximum observed name weight is 50%. Add
the result to Notebook 44 only as a historical LSEG/FinBERT strategy-shaped
comparator. It is not the current base, a Gemma comparison, or permission to
retune the rule. LSEG/FNSPID and the distinct LSEG regimes remain non-pooled.

Notebook 56 is the aggregate-only lineage and coverage audit for every frozen
local LSEG/FinBERT strategy-research run. It reads only run manifests and
aggregate metrics, validates every declared metrics hash, and groups exact
payload duplicates before comparing declared strategy families. It must not
load corpus, scores, prices, daily returns, positions, or text, and it constructs
no signal or return. The inventory contains ten run directories, five exact
metrics payloads, and four declared families; three repeated payload groups are
artifacts, not replications. A family clears the descriptive floor only when its
primary arm has positive development and evaluation net return and Sharpe with
at least 20 active sessions in each period. Only `midcap_event_v1` clears that
floor, and Notebook 55 already captures its canonical run. The only other
positive evaluation arm has net Sharpe 0.545 on five active sessions after
development Sharpe −1.456. Therefore no distinct adequately covered positive
local base is omitted. This is a coverage control, not new evidence or
permission to promote or retune a strategy; the non-pooling rule remains binding.

Notebook 57 is the aggregate-only candidate-base and implementation-frontier
audit. It may read only existing aggregate result/gate/capacity tables, manifests,
and the already-frozen prospective contract. It must not load a price, return
path, position, score, headline, or licensed-text row; construct a signal or
return; pool regimes; rank candidates solely by Sharpe; reselect the prospective
primary; or alter the contract. Four roles are fixed: Notebook 21 HAR is the
research benchmark, Notebook 36's equal-weight Gemma-event-timing/FinBERT-name
ranking hybrid is the prospective primary, Notebook 50 inverse-volatility is a
non-promoting secondary risk diagnostic, and Notebook 55 is a historical
comparator. The primary's opened-window gross/net Sharpe is 3.700/2.271 at 10
bps/side, its 5-bps scenario net Sharpe is 2.995, break-even is 25.24 bps/side,
both half Sharpes are positive, and its cash interval is [+0.41,+13.37]
bps/session. These are reasons to spend one prospective test, not validation:
the sector-factor interval [−0.44,+10.08] crosses zero and the candidate was
selected through an iterative opened-window chain. Under the frozen Y=1 impact
scenario, net Sharpe falls to 1.679 at $1m, 0.413 at $10m, and −0.632 at $25m;
ten $25m orders exceed 5% ADV. Preserve the unchanged candidate and capacity
scenarios until genuinely new dates satisfy the frozen population gates;
Notebook 58's sampling-uncertainty result governs any capacity claim.

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
not a general semantic timing mechanism or Gemma alpha. The first new-news batch
now extends strictly beyond 2026-06-26, but Notebooks 53–54 prove that its closed
interval can contain only 27 post-cutoff XNYS sessions. This is below the frozen
120-session minimum, so it must not be joined to later cached prices or opened
for scorer, return, or performance analysis.

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

Notebook 51 is a frozen implementation-capacity diagnostic of that unchanged
equal-weight hybrid, not another signal or allocation search. Its source-
controlled specification precedes outcome calculation and fixes a 20-session
lagged dollar-volume and open-return-volatility estimator (five-observation
minimum), 10-bps/side fixed cost, square-root impact
`Y × sigma × sqrt(order notional / ADV)`, primary `Y=1`, sensitivity `Y=0.5`,
a `$1m, $2m, $5m, $10m, $25m, $50m, $100m, $250m, $500m, $1bn` grid, and a
hard 5%-of-lagged-full-day-ADV order ceiling. Full-day ADV is explicitly an
optimistic proxy for execution specified at the open.

The zero-impact replay must reproduce the validated drift-aware ledger and
Notebook 36 within `1e-12`; every nonzero rebalance or terminal order must have
finite positive lagged ADV and finite non-negative lagged volatility. The
reported capacity is the largest predeclared primary-grid point with positive
total net return, positive net Sharpe, and every order at or below 5% ADV. All
grid points and the frozen $1m/$10m references are reported even when they fail.

Both input gates pass: the largest identity error is `2.22e-16`, and all 419
nonzero orders have valid lagged liquidity. Under primary `Y=1`, $1m has net
Sharpe 1.679 and +8.53% total return; $10m has net Sharpe 0.413 and +1.89%,
with maximum participation 2.30%. At $25m, net Sharpe is −0.632, total return
is −3.29%, and ten orders breach 5% ADV—eight DUK and two PLD—with maximum
participation 5.52%. The optimistic grid capacity bound is therefore $10m.
This is an implementation upper bound and failure anatomy, not deployment
qualification, independent confirmation, or a promotion effect. A separately
frozen liquidity-aware sizing audit may diagnose the concentrated bottleneck,
but cannot revise the alpha conclusion or replace the prospective new-date gate.

Notebook 58 is the pre-outcome-frozen sampling-uncertainty audit of Notebook
51's unchanged primary `Y=1` capacity curve. It uses the identical full AUM grid,
10-bps fixed cost, impact model, and 5%-ADV ceiling. The circular moving-block
bootstrap is fixed at five sessions, 9,999 replications, and seed `20260815`,
with identical resample indices across AUM. A grid point is uncertainty-qualified
only when the 95% interval for mean daily net return is strictly positive, at
least 95% of bootstrapped terminal returns are positive, observed net Sharpe is
strictly positive in both chronological halves, and every order remains within
5% lagged full-day ADV. The reported capacity is the largest passing predeclared
grid point; interpolation is forbidden.

The replay reproduces Notebook 51's metrics within `1.78e-15` and the shared
bootstrap helper exactly. No AUM passes. At $1m, point net Sharpe is 1.679 but
its bootstrap Sharpe interval is [−0.534,+3.431], mean-net interval is
[−1.29,+11.64] bps/session, positive-terminal probability is 93.48%, and half
Sharpes are 3.052/0.016. At $10m, point Sharpe is 0.413, mean-net interval is
[−5.20,+8.02], positive-terminal probability is 62.53%, and half Sharpes are
2.408/−2.011. Thus $10m remains only an optimistic point-estimate scenario
bound. This does not invalidate Notebook 51's arithmetic, alter the candidate,
or license historical retuning; it prevents an uncertainty-qualified capacity
or deployment claim before the genuinely new-date replay.

Notebook 59 is a pre-outcome-frozen accounting decomposition of Notebook 36's
unchanged open-to-open gross path, not a change in holding period. For each
stock and entry session it uses start-open-normalized entry-open-to-close and
close-to-next-open contributions whose sum must reproduce the open-to-open
return within `1e-12`. The two component means form one 9,999-replication,
five-session circular-block family with BH correction. Costs stay separate and
long/short pieces are descriptive. The identity error is `7.46e-17`. Intraday
contributes +6.465 bps/session [+1.420,+11.555] with BH q=0.0242 and 57.38% of
total gross sum; overnight contributes +4.802 [+0.113,+10.141] with q=0.0611.
The long leg supplies 97.43% of total arithmetic gross contribution with
all-session/half gross Sharpes 4.344/4.254/4.856; the short leg supplies 2.57%
with 0.163/0.387/0.040. Do not turn either component or leg into a new exit,
direction, or strategy rule on this opened window.

Notebook 60 is the single frozen follow-up to that long-leg concentration. It
keeps every Gemma date/count, FinBERT-selected long name, positive target weight,
original-33 membership, and open-to-open interval fixed. The broad control
spreads each session's total long exposure equally across all 33 companies; the
sector control preserves each session's sector-level long exposure and spreads
it equally across the sector's three original members. Only the two paired gross
mean differences are tested, using 9,999 five-session circular-block resamples
and BH correction. No control ledger, cost, turnover, net return, or new traded
strategy is constructed. The selected long leg beats matched market exposure by
+8.362 bps/session [+3.407,+13.780] (BH q=0.0038) and matched sector exposure by
+3.662 [+0.952,+6.623] (q=0.0121); both differences are positive in both
descriptive chronological halves. Record this as useful post-result evidence
that the opened-window long result contains within-sector name-selection value.
It does not identify causal sentiment information, override Notebook 36's
zero-crossing sector-factor alpha interval, alter the prospective contract, or
permit historical retuning or promotion.

Notebook 61 is the final permitted historical session-component follow-up. Its
pre-outcome contract preserves Notebook 60's exact selected long and same-sector
control weights and decomposes their gross difference into start-open-normalized
intraday and overnight pieces. The two component means form one 9,999-replication,
five-session circular-block family with BH correction. Their sum must reproduce
Notebook 60 within `1e-12`; no component cost, net return, alternative exit, or
traded strategy may be constructed. The identity error is `5.79e-17`. Intraday
selected-minus-sector is +1.454 bps/session [−0.324,+3.399], BH q=0.1264;
overnight is +2.208 [−0.006,+4.845], q=0.1264. Neither gate passes. Intraday is
positive in both descriptive halves, but overnight changes from +4.747 to
−0.362 bps/session. The Notebook 60 total remains useful gross name-selection
evidence, but it has no statistically isolated or temporally stable session
component. Close historical intraday/overnight, direction, threshold, and
allocation decomposition; carry only the unchanged open-to-open prospective
candidate forward.

Notebook 63 is one final frozen characteristic-confound audit of Notebook 60's
unchanged selected-long path, not another control portfolio. It jointly uses
strictly lagged 1/5-session returns, 20-session volatility, and log ADV; the
pooled no-intercept coefficients are fitted on nonselected names after
sector-date demeaning and refitted inside each of 9,999 five-session circular-
block resamples. The raw +3.662-bps/session sector difference is reproduced
within `1.74e-18`. The four controls explain −0.073 bps/session, leaving a
+3.735-bps residual [+0.867,+6.746], p=0.0086, positive in both descriptive
halves (+5.335/+2.116). Record this as evidence that the opened-window
name-selection result is not a simple proxy for those characteristics. Do not
construct or cost a residual strategy, add controls, alter the prospective
contract, or describe the result as independent confirmation or validated
alpha. That no-strategy boundary governed Notebook 63 itself; the researcher's
subsequent instruction to continue strategy iteration authorized exactly the
two pre-outcome-frozen Notebook 64 translations below, without reopening a
wider hedge or scaling search.

Notebook 64 is a user-authorized iterative retrospective economic translation
of Notebook 60's two gross controls. It changes no signal date, eligibility,
Gemma count, FinBERT-selected long name, selected-long weight, cap, horizon, or
price convention. The market residual subtracts the equal-weight 33-name
control; the sector residual subtracts equal-weight three-name controls inside
each sector. Signal and hedge weights are netted by name before the validated
drift-aware ledger charges 10 bps/side and final liquidation. The four primary
comparisons (two versus cash and two versus the unchanged Notebook 36
incumbent) share one BH family; the two sector-spanning HAC(5) intercepts share
a second family. No scaling up is permitted.

The market residual has net Sharpe 2.332, +8.70% net return, −1.85% maximum
drawdown, 55.20 total turnover, and 25.30-bps/side break-even, compared with
2.271, +11.82%, −2.90%, 74.56, and 25.24 for the incumbent. It remains positive
in both halves, but its +5.06-bps/session cash comparison [+0.29,+10.40] has BH
q=0.0966, its +4.23-bps sector-factor intercept has q=0.0970, and its mean net
return is −1.75 bps/session below the incumbent [−5.65,+1.99]. The sector
residual has net Sharpe 0.818 and a −1.74-bps/session second-half mean. Both
frozen classifications are `nonviable`. Record the market hedge as descriptive
risk-efficiency evidence, not superior return evidence; reject the sector
hedge. Neither result changes the prospective contract, validates alpha, or
permits another historical hedge, scaling, or cost search.

Notebook 65 is one further user-authorized, pre-outcome-frozen retrospective
information-specificity test, not independent confirmation. The local LSEG
corpus has zero populated `subjects` or `entities` rows, so no Reuters topic
taxonomy is invented. V1 required explicit-target, single-company,
non-market-price/technical headlines for both scorers; it stopped before return
construction when this left one FinBERT strongest-event firm-open rank missing.
No v1 portfolio outcome was opened. V2 was then frozen before outcomes: apply
the same return-blind filter only to Gemma's timing/count leg and retain
Notebook 36's complete unfiltered FinBERT ranks exactly. No filter sweep,
fallback, threshold, hedge, scaling, allocation, or horizon search is allowed.

The filter raises defined Gemma strongest-event firm-opens from 3,841 to 4,826,
but active both-sign sessions collapse from 70 to 22. At 10 bps/side the
filtered hybrid has gross/net Sharpe 1.508/0.881, +3.09% net return, 23.01 total
turnover, and 23.62-bps/side break-even. Its +1.88-bps/session cash comparison
[−2.64,+7.01] has BH q=0.4449; its −4.93-bps/session difference from the
incumbent [−10.82,+0.59] has q=0.1754; and its sector-spanning alpha is +1.00
[−2.46,+4.46], p=0.5723. Both half means are positive, but the cash,
incumbent-improvement, factor, and 60-active-session gates fail. Record the
`nonviable` result as a specificity-versus-breadth mechanism null. It changes no
prospective contract and closes further historical metadata-filter variants.

Notebook 52 performs that single permitted input-first sizing audit. For an
entry from flat, minimising the square-root objective within an unchanged leg
gives weights proportional to `ADV / max(sigma, 0.005)^2`; deterministic water
filling caps any name at 25% and redistributes the remaining original leg
budget. The rule cannot change Gemma event dates/counts, FinBERT-selected
names/signs, gross or net exposure, horizon, cost, or source regime. Its
predeclared input gate requires complete lagged inputs, lower aggregate
from-flat impact proxy, and target-path L1 turnover no greater than the equal-
weight hybrid. If any condition fails, modified returns must not be constructed.

All 70 event sessions, selected names/signs, leg budgets, neutrality, cap, and
liquidity inputs pass. The impact proxy falls 6.08%, but target-path turnover
rises 0.54%, from 74.458 to 74.858. The turnover condition therefore fails;
`returns_constructed` remains false, conditional result and inference tables
remain empty, and no modified next-open return is inspected. Record an input-
only NO-GO. Do not tune smoothing, shrinkage, refresh frequency, or another
liquidity objective on the opened window. The equal-weight hybrid remains the
only prospective candidate under its existing new-date contract.

Notebook 53 applies that contract's population gate before any scorer or return.
The first prospective acquisition interval (2026-06-26 inclusive to 2026-08-06
exclusive) was frozen and committed before retrieval for the exact original 33,
all entitled English-language headline sources, and no story bodies. Its
completed manifest records 162,123 source story/headline rows, 3,772 pages, 3,776 requests,
four recovered retries, zero failures, and zero pagination anomalies. A
metadata-only audit finds 162,049 headlines strictly after the cutoff, 74
exactly at it, none before it, and all 33 symbols. No licensed text is emitted.

The closed interval has an upper bound of 27 eligible XNYS sessions versus 120
required. Therefore active-session gates are not evaluated, local FinBERT and
paid Gemma scoring are not launched, no price or return is loaded, and no
performance inference is run. The earliest possible 120th post-cutoff XNYS
session is 2026-12-16. Continue immutable accumulation through at least that
date, then apply the remaining 60-active-session and 20-per-half gates before
authorising the one-shot replay. This stop is prospective protocol compliance,
not strategy evidence for or against alpha.

Notebook 54 makes the prospective population append-only without
opening the scoring or return stages. Its source-controlled registry records each
ordered acquisition-spec path and hash; the reusable auditor rejects parent,
config, source, interval, universe, story-body, timestamp, or pagination drift and
requires batch intervals to be contiguous. Licensed JSONL is streamed locally;
only counts and population hashes are emitted. The first batch has 162,123 source
story rows, 130,742 unique normalized headlines in the retrieved file, and
130,692 strictly post-cutoff eligible normalized hashes from 162,049 eligible
rows. Eleven rows normalize empty. Thus the eventual scoring population is
19.35% smaller than scoring every eligible source story. At the prior realised
Gemma average this would be about $4.95, but that is a planning estimate rather
than a quote or authorisation. The calendar gate remains 27/120; FinBERT, paid
Gemma, prices, returns, active-session gates, and performance inference remain
sealed.

### Post-drift-stop same-33 long-window sensitivity

Notebook 67 is a separately frozen researcher-authorised sensitivity, not the
original backward replication. The 2024-01-01 to 2025-10-26 same-33 scorer
population is joined to only the original-33 associations in the 2025-10-26 to
2026-06-26 corpus. Normalized-headline hashes are deduplicated across blocks by
retaining the earliest timestamp and its attached scores and unioning same-33
company associations. This produces 1,967,554 unique headlines from 1,983,998
block rows, with 16,444 cross-block overlaps. The recent aggregate panels must
reproduce Notebook 13's firm-open keys, strongest-event scores, and returns to
`1e-12` before any long-window result is accepted.

The Notebook 36 strategy is unchanged: exact Gemma +1/−1 strongest-event
values determine whether and how many names trade, continuous FinBERT
strongest-event scores rank the long and short names, both legs are equal,
single-name weights are capped at 25%, the hold is one open-to-next-open
interval, and the drift-aware ledger charges 10 bps per traded side plus final
liquidation. A return-blind completeness audit found 11 backward firm-open
sessions with fewer than all 33 companies. Under the amendment frozen before
any strategy result was observed, those dates remain in the common price
calendar but are forced to cash; they are never dropped, imputed, carried
forward, or used in circular schedule shifts.

Report the backward block, already-opened recent block, and full joined path
separately. Primary backward diagnostics are cash with a 9,999-replication
five-session block bootstrap, market and sector-spanning HAC-5 intercepts,
chronological halves, the fixed cost grid, every non-zero circular shift of the
complete Gemma schedule, 4,999 conditional random-name assignments, and
rebuilt leave-one-company and leave-one-sector-out replays. The backward result
is gross/net Sharpe 0.691/−0.354, −3.43% net return, and 6.60-bps/side
break-even over 456 sessions, of which 445 are complete and 94 active. Cash and
factor intervals cross zero, both mechanism tests fail BH, and only 1/33
company and 3/11 sector exclusions remain profitable. The 623-session joined
path has gross/net Sharpe 1.639/0.467 and +6.51% net return, but this is
descriptive because it includes the already-opened strong recent block.

The frozen hosted-Gemma endpoint-retention gate remains failed at 0.7881 versus
0.80. Therefore Notebook 67 cannot qualify as the preregistered replication,
independent confirmation, prospective evidence, or permission to retune. Its
valid role is an adverse earlier-time robustness result that materially weakens
the historical strategy case while leaving the post-2026-06-26 prospective
contract unchanged.

### Frozen continuous-rule same-source temporal replay

Notebook 70 is a bounded retrospective test of the separate Notebook 39
continuous construction. It does not reselect an aggregator or tune a trading
rule. The frozen rule uses mean continuous Gemma sentiment per company-open,
current cross-sectional q90−q10 dispersion at or above the strictly prior
trailing-60 75th percentile after 40 complete observations, top-two/bottom-two
selection, a 25% cap, gross-one dollar neutrality, one open-to-next-open hold,
drift-aware turnover, final liquidation and 10 bps per side. Eleven incomplete
backward sessions remain in the calendar as cash and do not enter threshold
history. The backward and already-opened blocks are separate estimands; the
joined path is descriptive only.

The backward block has gross/net Sharpe 0.308/−1.159, −15.20% net return and a
2.09-bps/side break-even. Its five-session block-bootstrap cash interval is
[−7.71,+0.90] bps/session. The opened block has gross/net Sharpe
−1.409/−2.391, −17.45% net return and −14.02-bps/side break-even. Its cash
interval is [−20.79,−2.48] bps/session and remains negative after BH correction
across the two source-block cash tests (q=0.033); its sector-spanning HAC-5
interval [−19.29,−1.32] is also wholly negative. Every half in both blocks is
net-negative, and the joined 623-session path loses 30.00% net.

Across 16,444 duplicate normalized headlines, Gemma scores remain highly rank
repeatable (Spearman 0.989) but are exactly equal only 88.46% of the time. This
supports a narrow dissertation distinction: measurement repeatability does not
imply temporal or economic portability. It is not independent confirmation,
validated alpha, a reversal signal, or permission to tune the threshold,
breadth, costs or holding period. The distinct materiality 2026 confirmation is
not opened by this replay.

### Conditional negative-share information beyond mean sentiment

Notebook 71 closes a specific gap in the original Workstream-3 evidence. The
earlier nine-rule screen showed that negative-story share was the sole corrected
development survivor, but it did not directly estimate whether that summary
added information after controlling for the mean. Before execution, Notebook 71
froze a daily cross-sectional regression of centred percentile-ranked next-open
return on centred ranks of mean continuous sentiment, negative-story share and
log one plus story count. Dates with fewer than ten complete firms or a rank-
deficient design are excluded and counted. The coefficient of interest is the
negative-share rank; inference is an intercept-only HAC(5) regression over the
daily coefficient series.

On 2,264 FNSPID development sessions and 512,149 complete firm-days, the mean
negative-share coefficient is −0.008310 with 95% HAC interval
[−0.014100,−0.002520] and p=0.004907. The frozen expected direction and interval
gate pass. This supports the bounded statement that, within the FNSPID
development sample, the distribution of same-day story tone contains
incremental return-ranking information that the average tone and story volume
do not capture.

The LSEG arm remains separate and uses raw next-open return because a compatible
abnormal-return model is unavailable. Pinned-FinBERT estimates are negative in
both the backward and recent blocks (−0.028611 and −0.015727), but their 95%
intervals [−0.072826,+0.015603] and [−0.094534,+0.063080] cross zero; neither
passes the declared two-block BH family. Gemma estimates are sensitivity only
and cannot rescue the primary scorer. All LSEG negative-share rank books are
net-negative after 10 bps per side. Therefore the result is not universal
portability, validated alpha, or permission to pool source regimes.

Gate F1 selected the temporal-stability version of this question before the
FNSPID evaluation estimands were opened. Notebook 75 then executed its parent
freeze once, with an outcome-blind amendment adding power context, a separate
evaluation-minus-development HAC(5) contrast, exact provenance and a fail-closed
output guard. On 203,393 evaluation firm-days and 998 sessions, the all-firm-day
conditional coefficient is +0.005833 with 95% interval
[−0.004261,+0.015928] and BH q=0.5148. The predeclared n≥2 coefficient is
+0.002952 [−0.009901,+0.015806], q=0.6526. The nine-aggregator evaluation
family has zero BH survivors; negative share has IC −0.003120 (p=0.3002).

The separately predeclared evaluation-minus-development contrast is +0.014143
with 95% interval [+0.002499,+0.025788] and p=0.01728. This is a temporal
sign-changing shift in the estimated coefficient, not merely a difference in
which era crosses a significance threshold. The pre-run normal approximation
gave only 35.4% power to recover the development effect at the first BH-2 hurdle
and an approximate 80%-power MDE of 0.01372, so the evaluation null is not proof
of an exact zero. Together, the frozen families and contrast establish temporal
non-replication. They do not license reverse-sign trading, a new threshold,
another horizon, further stratification, or replacement of the selected RQ.

### Exact negative-pressure HAR transfer to LSEG

Notebook 72 tests whether the FNSPID aggregate risk mechanism transfers without
changing its design. It uses pinned-FinBERT headline-count-weighted negative
share across complete 33-company LSEG sessions, a current observation minus
strictly prior 252-session mean divided by strictly prior standard deviation,
at least 126 prior observations, 1.5/0.5 hysteresis entry/exit thresholds, and a
0.25 risk-off multiplier. This state multiplies Notebook 46's unchanged frozen
HAR exposure; the ledger charges 2 bps per side and compares both HAR and a
matched constant exposure.

The recent-window z-score never reaches the frozen entry threshold: its maximum
is 1.348955 and risk-off occurs on 0/167 sessions. HAR, overlay and matched
control are therefore identical (net Sharpe 0.239561; total net return 1.241%;
maximum drawdown −7.770%). All return, downside, timing and cross-source gates
fail. This is a clean mechanical non-transfer. It is not evidence about the
counterfactual performance of a lower threshold, and the opened LSEG window
cannot be used to choose one.

Notebook 73 adds one separately frozen historical extension without changing
that rule. Commit `12086d5` fixes the `.SP500` LSEG instrument, 2023-01-01 to
2025-10-27 acquisition range, backward decision window, sentiment rule, HAR
coefficients, accounting, inference and gates before the older index returns
are retrieved. The hash-manifested export contains 707 opens and covers every
one of the 456 backward decision sessions through its next open. The input-only
audit reproduces 53 risk-off sessions across 18 entries.

At 2 bps per side, frozen HAR / pressure overlay / matched-constant net Sharpes
are 1.291 / 1.520 / 1.291. The overlay-minus-HAR mean is +0.144 bps/session with
a 95% circular-block interval [−1.295,+1.688] (p=0.8486); the first and second
half effects are +0.858 and −0.571 bps/session. HAR-minus-overlay downside-
squared loss is positive after the two-test BH correction (q=0.0336), and the
actual downside schedule beats all circular placements after its separate
two-test correction (q=0.0482). However, the matched-constant downside interval
crosses zero, maximum drawdown improves only 3.63% against the frozen 10% gate,
and the return lower bound is worse than the frozen −1-bp/session tolerance.
The timing gate therefore passes while the return, useful-risk and full cross-
source gates fail. Report partial historical downside-timing portability, not
alpha, universal transfer or permission to alter the 1.5/0.5 thresholds.

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

## Bounded LSEG Firm-Specific Materiality Pilot — 2026-08-12

This is a candidate Gate-F1 measurement path, not a selected RQ. Notebook 68
freezes 2,000 unique first-release Reuters company-sessions from the joined
same-33 history without opening prices or returns. The supported window is
2024-01-02 to 2026-06-25. Selective story retrieval fixes a body-available
population of 1,445 events (72.25%); the remaining 555 have terminal archival
shell, short-body, unavailable-story, or not-found outcomes and are not
replaced. Generalisation is therefore conditional on substantive LSEG body
availability, whose company, block, polarity and time attrition must be shown.

The frozen joint construct is target-company direction/severity, firm-specific
economic materiality, novelty, valuation horizon, target specificity and
evidence sufficiency. A 200-event blinded primary audit and 60-event second-code
subset precede returns. Primary gates are inter-coder quadratic-weighted kappa
at least 0.60, model-human kappa at least 0.40, severe ordinal disagreement at
most 0.10, high-materiality precision at least 0.70, non-neutral direction-sign
precision at least 0.70, and target-specific F1 at least 0.80. A failure stops
the return stage rather than licensing prompt tuning or resampling.

Hosted scoring received the exact payload/destination confirmation and is
complete: 1,445/1,445 valid DeepInfra/Gemma annotations from 1,450 append-only
attempts, with four invalid outputs and one API error preserved before cleanup,
for reported cost $0.18523289. No price or return file was opened during that
measurement stage. The blinded coder-A 200 and coder-B 60 sheets now exist with
labels empty, so the human reliability gate remains unpassed.

On 2026-08-12 the researcher explicitly waived that human gate for one bounded
**model-only exploratory development analysis only**. This deviation does not
pass or replace the measurement gate. Notebook 69 therefore admits the 1,200
body-available model-scored events with entry sessions through 2025-12-31,
hash-checks the complete LSEG price export while parsing numeric price fields
only before 2026-01-01, and rejects any return whose end session reaches 2026.
It opens zero confirmation returns. The frozen eight-test family covers h1/h5
incremental clustered coefficients, expanding chronological OOS loss and IC,
and fixed sparse h1/h5 10-bps strategies. Zero tests survive BH. The h5
direction×materiality×novelty translation is descriptively positive (gross/net
Sharpe 0.741/0.230, +1.86% net return, 14.48-bps/side break-even) and positive
in both calendar years, but it does not distinguish cash. Its favourable paired
FinBERT comparison (4.55 bps/session; BH q=0.031) was frozen only after the h5
summary was visible and is recorded as outcome-selected post-primary evidence.
The separate post-primary reaction-magnitude family also has zero BH survivors.
These findings may support a dissertation about the distinction between richer
semantic representation, implementation economics and predictive alpha; they
cannot be presented as human-validated measurement, confirmation or alpha.
The 2026 return period remains sealed.

## Bounded LSEG Prompt-Engineering Search — 2026-08-13

Notebook 83 is a user-authorised exception to the closed hosted-model scope. It
tests five fixed target-company prompt formulations on a return-blind population
of 9,317 first-release Reuters company-session events from the same 33 firms.
The only external payload is the current headline, target company/name and at
most five strictly earlier same-company Reuters headlines. Requests are routed
through OpenRouter to DeepInfra-only Gemma 4 26B FP8 with fallback disabled,
zero-data-retention requested and provider data collection denied. Append-only
reported cost is $2.97091685, below the $19 operational and $20 authorised caps.
The postflight account endpoint shows a $3.00574212 usage delta and
$18.529134759 remaining. Its $0.03482527 excess over the attempt ledger is left
unattributed because the account total can include other activity or billing
reconciliation.

Each prompt has at least 99% strict-success coverage. Their strict intersection
is 9,195/9,317 events (98.69%) across all 33 firms. The original 99% common-
sample gate is preserved as a pre-return v1 NO-GO. Before any price or return
access, a disclosed v2 feasibility amendment fixes a round 98% floor and keeps
only the unchanged strict five-prompt intersection; no invalid label is coerced,
imputed or retried further. This deviation is an outcome-blind data-quality
decision, not preregistration.

Selection uses 2024 only. Each prompt forms the fixed strongest-positive versus
strongest-negative sparse portfolio at its declared one- or five-session
horizon, pays 10 bps per side and is compared with existing Gemma and FinBERT on
the exact same formation dates. All five prompt portfolios have negative gross
and net means. The least negative, delayed-reaction h5, earns -0.99 gross and
-4.01 net bps per session, net Sharpe -1.10 and -9.96% ending return. Zero
prompts pass the economic/stability gate, so no 2025 prompt portfolio or
comparison is computed and 2026 remains sealed.

The first successful notebook execution nevertheless attached event-level 2025
returns before selection, although it computed and displayed no 2025 portfolio,
comparison or statistic. That test-boundary incident is hash-recorded. The
corrected final execution attaches 2024 events first and materialises zero 2025
event return when selection is empty. The result is therefore a transparent
retrospective prompt-engineering null, not a pristine one-shot confirmation and
not proof that no possible prompt can add value.

## Structured Anticipated-Reaction Score — 2026-08-13

Notebook 84 is the researcher's final requested prompt sensitivity after seeing
Notebook 83. It removes the novelty question because a model reading one current
headline cannot establish whether the information is new. The hosted request
also omits every earlier headline. From the current Reuters headline and named
target only, Gemma must separately report target specificity, better/worse/
unclear versus expectations, temporary/persistent/unclear implications, down/
flat-or-unclear/up first tradable reaction, and a final score on the fixed
−1, −0.75, ..., +1 grid. The score is directional reaction sentiment, not a
percentage-return forecast.

The prompt, schema, h1 sparse portfolio, 10-bps cost and 2024 selection gate are
fixed before prices. DeepInfra shared-pool overload creates 79 API and six
transport failures in the first pass. A single low-concurrency cleanup leaves
3,241/3,242 strict 2024 scores (99.97%) and one uncoerced invalid output across
all 33 companies. Append-only reported cost is $0.18951202. The postflight
account delta is $0.19209614, leaving $18.337038619; its $0.00258412 excess is
not attributed. No earlier headline is sent and no fallback provider is used.

Chronological hosted scoring is tightened before returns: score 2024 first,
request 2025 scores only if selection passes, and never score 2026 for this
retrospective arm. The 2024 h1 book forms on 234/251 sessions. Its gross mean is
already negative at −2.59 bps/session; mean net is −19.67 bps/session, gross/net
Sharpe −0.287/−2.178, net ending return −40.53%, drawdown −45.08%, and
annualised turnover 215.16. It is positive and beats each same-date scorer only
in Q4. Selection fails, so 2025 and 2026 remain unscored and no 2025 event
return is materialised.

The mechanism is useful despite the null: explicit reaction wording produces a
nonzero score on 52.14% of strict events and many more h1 formation sessions,
but the additional directional activity is wrong before costs and especially
damaging after turnover. Do not reverse the sign or tune wording, score grid,
neutral policy, breadth, horizon, cost or portfolio on the opened year. This
rejects the specified reaction-score construction, not all possible prompts.

## Superseded July 12 Design

The earlier “Beyond the Mean” protocol specified PhraseBank agreement validation plus a four-scorer LSEG held-out event study with 19/27/31 July gates. Those dates, planned commands, and artifact paths are no longer the active critical path.

The scientific question remains available as RQ-D. The implementation record is retained in [LSEG core analysis workflow](lseg_analysis_workflow.md) and the repository history; it must not be represented as completed evidence.
