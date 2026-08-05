# Final Experiments

Notebook-first closing programme for the dissertation. This directory is intentionally lighter than `src/sentiment_benchmark`: rapid analysis, plots, and interpretable tables take priority over new CLI commands or production-style abstractions.

- Live decisions and checklist: [`plan.html`](plan.html)
- Scientific guardrails: [`../docs/research_protocol.md`](../docs/research_protocol.md)
- Stage plan: [`../docs/dissertation_execution_plan.md`](../docs/dissertation_execution_plan.md)

## Current Decisions

| Decision | Frozen/current value |
| --- | --- |
| Primary spine | FNSPID 2011–2023 |
| Robustness spine | LSEG sector-33, midcap-22, and the expanded 44-company/8-month all-source corpus; each remains separate and is never pooled with FNSPID |
| Firm-day panel | 715,546 news-bearing rows, 570 priced symbols, 3,262 sessions |
| Development block | `session_date <= 2019-12-31` |
| Evaluation block | `session_date >= 2020-01-01` |
| Primary scorer | Pinned ProsusAI/FinBERT checkpoint already used in the Moment-2 study |
| Final RQ | Not chosen; Gate F1 follows filtering/distribution EDA |
| Closed scope | New VaR/ES work, scorer bake-offs, and multi-scorer ensembles |

The split is a **chronological evaluation block**, not a pristine holdout. The samples have influenced earlier design work.

## Workstream Order

1. **Inventory and panel**: completed. Data paths, primary spine, earnings calendar, panel, attrition, and coverage plots exist.
2. **Filtering and distribution EDA**: executable EDA and the provisional machine-class arms are complete. Publisher fields are unavailable and the 180-row human audit is still unlabelled, so the validation exit condition is not met.
3. **Exploratory secondary arms**: story type, earnings, pooled learned thresholds, and all three non-pooled LSEG regimes have now been run. See [`EXPLORATORY_EXPERIMENT_LEDGER.md`](EXPLORATORY_EXPERIMENT_LEDGER.md).
   The expanded LSEG follow-on now has complete pinned-FinBERT and OpenRouter
   Gemma 4 26B coverage. Notebook 12 audits full-population scorer agreement;
   Notebook 13 preserves the resulting null return experiment. Notebook 14
   then falsifies a separately frozen sparse novel-material-event and
   machine-proxy cash-flow-distance strategy family. Notebook 15 closes the
   expanded-corpus publisher family as another clean null. Notebook 16 uses
   the newly matched Reuters bodies to prepare a blinded 200-event human
   cash-flow-distance audit; its labels remain outstanding. Notebook 17 then
   tests fixed sector-neutral and rank-hysteresis translations of Notebook
   13's strongest gross arm. Notebook 18 applies the exact frozen hysteresis
   rule to both scorers and shows that Gemma's apparent activity advantage is
   partly a score-resolution-induced abstention policy, not clean evidence of
   scorer superiority. Notebooks 19–25 then test the one remaining defensible
   strategy use of the corrected FNSPID negative-share relationship: risk
   sizing around an independent return exposure. The firm-level momentum brake
   is a clean null; the aggregate SPY hysteresis arm is economically interesting
   but misses its predeclared statistical/risk gate. A sentiment-free HAR
   volatility target is the strongest independent base (net Sharpe 0.673,
   maximum drawdown −15.73%); sentiment does not improve its variance forecast,
   and the development-only surprise strategy family is erased by costs. A
   standard 200-session trend layer weakens the base. Applying the unchanged
   aggregate hysteresis state directly to HAR produces the best descriptive
   risk-adjusted arm (net/gross Sharpe 0.835/0.878), but the paired return
   interval spans zero, drawdown is slightly worse, and the advantage reverses
   after 2020. A prior-only 2013–2019 walk-forward audit independently
   corroborates the downside mechanism: sentiment improves net Sharpe from
   1.114 to 1.164 and drawdown from −12.53% to −8.57%, with downside-squared
   p=0.0052, but slightly lowers mean and total return. Notebook 26 then keeps
   the expanded LSEG/Gemma data in the strategy chain without pooling regimes:
   a maximally-negative one-session firm brake raises the 44-stock fixed-share
   basket's gross/net Sharpe from 1.500/1.472 to 1.614/1.526 at 10 bps/side and
   lowers downside-squared loss (BH p=0.0088), but the paired return gain is
   only +0.148 bps/day [−0.405, +0.678] and neither promotion gate passes. The
   unchanged market rule has only 41 causal z-score sessions and one trigger,
   so it is recorded as infeasible rather than shortened. Notebook 27 then
   tests one untuned interpretation change across both FNSPID eras: using the
   same 25% sentiment state as an absolute HAR cap rather than a multiplier.
   The cap beats the product before 2020 (net Sharpe 1.176 versus 1.164) but
   loses over 2020–2023 (0.769 versus 0.835), so its frozen operator-selection
   gate fails. It nevertheless beats HAR Sharpe in all four reported
   subperiods and lowers downside-squared loss versus HAR in both full eras.
   This narrows the finding to a robust risk-state mechanism whose optimal
   exposure severity is unidentified. Notebook 28 then attempts a genuinely
   different LSEG/Gemma translation: exact −1 holdings are cut to 25% and the
   released notional is transferred to exact +1 firms. At 10 bps/side it is
   the strongest descriptive LSEG strategy so far (gross/net Sharpe
   1.692/1.560; +10.76% net return), and its incremental gross break-even
   versus the cash brake is 23.46 bps/side. But paired intervals span zero,
   downside does not improve, and rotation underperforms the brake in the first
   temporal half. Its translation, alpha, and useful-risk gates all fail.
   Notebook 29 then removes the long basket entirely: on 80 sessions with both
   exact signs, it holds a gross-1 dollar-neutral +1/−1 spread for one interval.
   At 10 bps/side, gross/net Sharpe is 2.558/1.185, net return is +10.74%, and
   gross break-even is 18.52 bps/side. Exact labels beat 4,999 count-matched
   random firm assignments (one-sided p=0.0094), so the frozen mechanism gate
   passes. The net mean interval still spans zero (p=0.238), first-half net
   Sharpe is −0.378 versus 2.912 in the second, 66/80 active sessions have a
   one-name leg, and Goldman Sachs supplies 51.5% of gross profit. All 44
   leave-one-company-out replays remain profitable, while a two-name-per-side
   diagnostic has net Sharpe 2.088 but only 14 active sessions. Preserve this as
   the strongest cost-surviving Gemma mechanism and seek independent time
   extension; the alpha and strategy-quality gates fail and no result is
   promoted. Notebook 30 then applies Notebook 29's already-declared 25%
   single-name quality cap without changing signs or active sessions. The cap
   raises net Sharpe to 1.697, cuts maximum drawdown to −4.97%, lowers mean
   turnover by 40.7%, and raises break-even cost to 23.24 bps/side while
   retaining +10.20% net return. The standalone quality gate passes and all 44
   leave-one-company-out replays remain positive, but capped-minus-uncapped
   mean return is −0.514 bps/session [−5.483, +4.291] (p=0.832), and the
   capped-vs-cash interval still spans zero (p=0.100). Freeze this as the first
   decent risk-controlled LSEG/Gemma candidate, not proven policy superiority
   or alpha, for independent time extension. Notebook 31 then removes every
   sector-dollar exposure by demeaning the capped targets within the frozen
   four-company sectors. The strategy collapses to −1.857 net Sharpe and −6.78%
   net return; sector-neutral-minus-capped return is −10.118 bps/session
   [−15.491, −4.902] (p=0.0004). A post-result decomposition attributes 87.3%
   of that gap to removed gross return, not added costs, establishing that the
   signal is between-sector rather than within-sector. Notebook 32 tests that
   complementary projection once, spreading each sector target equally across
   four names. At 10 bps/side it achieves gross/net Sharpe 3.805/1.868, +6.42%
   net return, −3.80% drawdown, 19.36-bps break-even, and a 12.5% maximum name
   weight. Actual sector placement beats 4,999 count-matched random assignments
   (one-sided p=0.001), paired downside improves after BH (p=0.0094), and all
   company/sector exclusions stay profitable. But its return intervals cross
   zero and first-half net Sharpe is −0.119, so only the sector-placement
   mechanism gate passes. Keep Notebook 30 as the primary new-date rule and
   freeze the exact projection as one secondary mechanism arm; do not blend or
   tune it on this window. Notebook 33 then removes all 11 later breadth
   additions while leaving both rules unchanged. On the immutable original 33,
   capped gross/net Sharpe is 3.344/2.046 with +10.42% net return, and projected
   gross/net Sharpe is 4.183/2.421 with +8.07%; both rules pass the frozen
   universe-quality gate and have positive first/second-half net Sharpes. The
   sector-placement null replicates (one-sided p=0.001), and all company/sector
   exclusions remain profitable. Neither multiplicity-controlled alpha gate
   passes: projected-versus-cash is nominally positive (p=0.0346) but not after
   BH across the two rules. The add-11 cohort has only 3/167 eligible signal
   sessions and cannot form multi-name sectors. This is strong same-window
   universe robustness, not independent validation. Notebook 34 then freezes
   those 70 Gemma event dates and per-day long/short counts but substitutes
   FinBERT's continuous within-day rank for name selection. FinBERT has no exact
   +1/−1 strongest-event values, so it cannot create the trigger itself. Its
   capped rank substitute is nevertheless descriptively stronger than Gemma
   (net Sharpe 2.271 versus 2.046; +11.82% versus +10.42%), while Gemma remains
   stronger after sector projection (2.421 versus 1.630). Paired intervals span
   zero and the ordering reverses across halves; both scorer-selection gates
   fail. Treat this as a scorer-specificity null and a new-date hypothesis that
   Gemma may time sparse events while FinBERT ranks names—not permission to
   switch, blend, or fit a scorer regime on this window. Notebook 35 tests that
   timing hypothesis against all 166 non-zero circular shifts of the complete
   Gemma event/count schedule. Both timing gates pass after BH across the capped
   and projected rules. The capped hybrid's observed 6.802 bps/session net mean
   exceeds the 0.217-bps shift mean by 6.585 bps (98.19th shifted percentile;
   one-sided p=0.0240); projected is 3.213 versus −1.277, a 4.490-bps advantage
   (97.59th percentile; p=0.0299). Break-even remains 25.24/18.41 bps/side and
   both halves are positive. This is the strongest current mechanism evidence:
   Gemma contributes unusual event timing while FinBERT can rank names. It is
   still a retrospective circular-shift result, not validated alpha. Notebook
   36 then audits the single capped hybrid directly, without a parameter sweep.
   At 10 bps/side it has gross/net Sharpe 3.700/2.271, +11.82% net return,
   −2.90% drawdown, and 25.24-bps break-even. Its cash-relative net mean is
   +6.802 bps/session with a five-session block interval of [+0.414,+13.371]
   and p=0.039, so the frozen retrospective cash gate passes; both halves and
   all 33 company/11 sector exclusions remain positive. The stricter HAC(5)
   market-plus-sector-spread alpha is +4.823 bps/session
   [−0.436,+10.083] (p=0.072), so the factor-alpha gate fails. Retain this as
   the leading LSEG/Gemma strategy candidate and seek a genuinely new-date
   replay; do not call it validated alpha or tune these 167 sessions again.
   Notebook 37 then stress-tests whether the exact numeric ±1 trigger itself
   transfers across Gemma implementations. On the same 85,960 hashes, the
   older Cerebras Gemma 31B checkpoint emits 10,172 exact endpoints versus 95
   for DeepInfra Gemma 26B (107.1×). After identical aggregation, eligible-date
   Jaccard is 0.059 and positive/negative count correlations are 0.035/0.016;
   matched-coverage 26B has only 6 eligible sessions and stops before returns.
   The permitted 31B hybrid has gross/net Sharpe 1.624/−0.543, −2.73% net
   return, 7.50-bps break-even, negative Sharpe in both halves, and timing
   p=0.204. Both transfer and timing gates fail. Notebook 36 remains the leading
   same-window candidate, but exact endpoint equality is now rejected as a
   portable semantic definition. Any new-data replay must first freeze a
   calibration-aware event rule without using these returns.
   Notebook 38 performs that return-independent calibration on the 1,000-row
   public validation checkpoint. The first threshold whose calibration-side
   Wilson lower bound exceeds 0.75 in both directions is 0.65; held-out audit
   precision is 82.9% positive and 83.0% negative. Applying the rule to the
   complete 888,155-headline LSEG/Gemma checkpoint reveals an input-design
   failure before any useful strategy inference: 5,472 of 5,511 original-33
   firm-opens with a selected direction contain both positive and negative
   selected headlines (99.29%). The frozen binary tie-abstention rule therefore
   trades only 2/167 sessions and has gross/net Sharpe −0.755/−1.013, −0.58%
   net return, cash p=0.577, timing p=0.743, and sector-spanning alpha p=0.428.
   All strategy gates fail. Preserve this as a structural NO-GO: public label
   calibration succeeds, but binary per-headline thresholds cannot resolve a
   high-volume firm-open. A future rule must retain continuous confidence or
   margin and must be validated on genuinely new dates, not retuned here.
   Notebook 39 builds that base without loading any return. On the same 85,960
   hashes, 31B-versus-26B headline-score Spearman is 0.953 with 89.2% sign
   agreement. Mean signed score at firm-open reaches Spearman 0.983, 93.9% sign
   agreement, median daily rank correlation 0.977, and top-two/bottom-two name
   Jaccard 0.720/0.864. A scale-free trailing-dispersion schedule has 22 active
   dates per model, 17 shared (Jaccard 0.630), with shared-date name Jaccard
   0.647/0.922. Every frozen portability gate passes for mean continuous;
   `strongest_event` fails both gates. On the complete 888,155-headline 26B
   input, the frozen rule yields 40/127 active post-warm-up sessions across the
   167-session rectangle, without reading returns. This is a portable,
   feasible strategy specification—not alpha evidence. Notebook 43 later gives
   it one bounded retrospective LSEG replay; only genuinely new dates can supply
   independent performance evidence.
   Notebook 40 then translates that frozen construction into the broad FNSPID
   panel without pooling regimes: FinBERT mean-continuous scores, all available
   priced news firms, and relative breadth `ceil(n × 2/33)`, with every other
   schedule, cap, horizon, accounting, and 10-bps cost choice unchanged. Across
   998 evaluation sessions it earns gross Sharpe 1.056 and +27.29% gross return,
   but net Sharpe is −1.103, net return is −23.27%, and break-even is only 4.904
   bps/side. Net mean is significantly below cash (−2.585 bps/session,
   [−4.662, −0.410], p=0.0185); both evaluation halves are negative after cost.
   Conditional name selection is nominally informative (+2.528 bps/session
   versus matched random names, p=0.035) but misses the two-test BH gate
   (q=0.070), while timing fails (p=0.208). Development gross Sharpe is only
   0.187 with 0.532-bps break-even. Preserve this as a diversified gross-signal
   result and a clean deployment NO-GO, not scorer equivalence or validated
   alpha. Any lower-turnover redesign must be frozen from input-only persistence
   diagnostics and tested on new dates rather than tuned to these returns.
   Notebook 41 performs that return-free audit in both regimes. A single
   episode-hold candidate—keep the opening basket through consecutive active
   sessions—reduces target turnover by 32.35% in LSEG/Gemma and 34.50%/43.30%
   in FNSPID development/evaluation. It fails the semantic-persistence gate:
   continuation long/short Jaccard is only 0.275/0.157 for LSEG/Gemma and about
   0.02–0.03 in both FNSPID eras. After eliminating every active-to-active
   rebalance, entry/exit alone still costs 11.03–12.84 bps per active session at
   the frozen assumption. Reject “hold the first basket longer” without opening
   a return. The next valid economic test remains the exact Notebook 39 rule on
   genuinely new LSEG/Gemma dates.
   Notebook 42 adds one backward external-time stress test without retuning the
   construction. It locally scores 41,019 deduplicated 2009-Q4/2010 FNSPID
   headlines with FinBERT while preserving Notebook 39's LSEG/Gemma-derived
   continuous rule. The return-free screen clears with 59 active 2010 sessions
   (30/29 by half). After the established valid-current-open firm-day filter,
   all selected returns are available, but gross/net Sharpe is
   −1.141/−3.397, gross/net return is −5.53%/−15.87%, and break-even is
   −4.802 bps/side. Net mean is significantly below cash at −6.806
   bps/session [−11.094, −2.819] (p=0.0011); both halves, both mechanism tests,
   and the factor-alpha gate fail. This falsifies a stable cross-era alpha
   interpretation of Notebook 40's positive later-period gross result. The
   file-level-price-filter predecessor is preserved as invalid because it
   admitted later IPOs and forced 95 sessions to cash; it is not quoted.
   Notebook 43 then uses a previously unused LSEG field to test the remaining
   repetition concern directly. Collapsing terminal-suffix `story_id` families
   to their earliest release removes 8,314 later-revision-only hashes (0.936%),
   changes 1,938 firm-opens and eight frozen decision dates, but does not rescue
   the strategy. The unchanged all-headline LSEG/Gemma rule has gross/net Sharpe
   −1.737/−2.635 and −18.13% net return; first-release-only improves those to
   −1.473/−2.502 and −16.82%, but the +0.935-bps/session paired difference is
   uncertain (95% interval [−0.652,+2.619], BH q=0.262). The filtered rule is
   significantly below cash (−10.786 bps/session, BH q=0.0366), has negative
   gross and net Sharpe in both halves, and fails every economic gate. This
   closes story-family collapse and supplies direct LSEG/Gemma evidence that
   Notebook 39's portable measurement construction is not historical alpha.
   Notebook 44 then consolidates the aggregate-only strategy evidence without
   loading any price, return, position, or text rows. Its governance screen
   retains the sentiment-free HAR target as the sole working base (net Sharpe
   0.673, +29.85% net return, −15.73% drawdown). Across six paired sentiment
   comparisons, zero favourable return interval excludes zero, although five
   report lower downside loss. The continuous-rule stress table is net-negative
   at 10 bps/side in all five regime/split rows. This freezes the useful result
   as a cross-regime downside-risk mechanism plus a directional-alpha null; it
   is an input to Gate F1, not the RQ decision.
   Notebook 45 then asks whether the lower downside loss is unusually well
   timed or merely mechanical de-risking. It enumerates every circular placement
   of the exact frozen state in three disjoint regimes. Only FNSPID 2020–2023
   passes the three-test downside family (99.1st percentile; exact p=0.0100,
   BH q=0.0301). FNSPID 2013–2019 is suggestive but misses BH (p=0.0455,
   q=0.0682), while the LSEG/Gemma firm brake is ordinary (p=q=0.3653).
   Zero return-timing tests pass. The cross-regime mechanism gate therefore
   fails: retain one-period market-risk timing evidence, not a general semantic
   timing or Gemma-alpha claim. Later price rows are not used because no local
   LSEG news exists after 2026-06-26.
4. **Gate F1**: next formal decision. Choose one primary RQ and at most one secondary after resolving or explicitly waiving the audit blocker.
   `05_interpretation` assembles the evidence ledger that feeds this decision.
5. **Promotion**: register the accepted run and export aggregate, licence-safe figures/tables to the dissertation.

## Files

| Path | Role | Status |
| --- | --- | --- |
| `00_data_inventory.ipynb` | Inventories FNSPID/LSEG assets, profiles the earnings gather, and records the spine decision evidence. | Completed |
| `01_panel.ipynb` | Builds the FNSPID firm-day panel and writes attrition/coverage evidence. | Completed |
| `lib/panel.py` | Thin reusable panel builder and frozen split constants. | Active |
| `data/earnings/README.md` | Earnings-calendar schema, session mapping, and quality caveats. | Tracked |
| `data/earnings/license_record.md` | LSEG source, access date, requested fields, redistribution boundary, and safe aggregate counts. | Tracked |
| `data/earnings/*.csv`, `*.json`, `checkpoints/` | Licensed LSEG results-calendar payloads. | Ignored/local |
| `outputs/00_data_inventory/` | Generated inventory tables and figures. | Ignored/local |
| `outputs/01_panel/` | Generated panel, manifest, attrition, schema, and coverage figures. | Ignored/local |
| `02_filters_and_distribution.ipynb` | Within-day score EDA plus novelty heuristic mix and audit template. | Active |
| `lib/distribution.py` | Firm-day score moments and n-bin mass helpers. | Active |
| `lib/novelty.py` | Strictly-earlier first-mention / near-dup features and separate market-recap flag. | Active |
| `outputs/02_filters_and_distribution/` | Moments, repetition-screen mix, blinded audit plus weight key (gitignored). | Ignored/local |
| `03_aggregation.ipynb` | Nine aggregator signals; development IC vs `ar_open_h1`, BH, n-bin facet, turnover/break-even. | Active |
| `lib/aggregators.py` | Same-day aggregation rules, decayed state, mean daily cross-sectional Spearman IC with HAC inference. | Active |
| `lib/evaluate.py` | Portfolio translation, turnover, break-even, block bootstrap. | Active |
| `outputs/03_aggregation/` | Aggregator panel, IC tables, portfolio arm tables and figures. | Ignored/local |
| `04_surprise.ipynb` | Demeaning stack, market-model AR, nested OOS surprise horse race against a control-only nest. | Active |
| `lib/surprise.py` | Firm/CS demeaning, trailing market model, OOS R² comparison with block-bootstrap differentials. | Active |
| `outputs/04_surprise/` | Surprise panel, decomposition, horse-race tables/figures. | Ignored/local |
| `05_interpretation.ipynb` | Rule redundancy, dispersion-vs-volume conditioning, significance-vs-economics, Gate F1 evidence ledger. | Active |
| `outputs/05_interpretation/` | Distinctness, fixed-n IC, position-mode comparison, OOS-vs-control figures and ledger. | Ignored/local |
| `06_strategy.ipynb` | Strategy backtest: horizon × breadth sweep on development, frozen research spec, explicit trade-versus-cash deployment gate, corrected evaluation run plus disclosed void predecessor, equity/drawdown/cost curve. | Active |
| `lib/strategy.py` | Overlapping-tranche backtest on dense returns, development sweep, frozen-spec runner. | Active |
| `outputs/06_strategy/` | Sweep, `frozen_spec.json`, daily series, equity/drawdown, per-year and cost-curve figures. | Ignored/local |
| `07_strategy_analysis.ipynb` | Strategy-null diagnostics plus Q5−Q1 date-block inference and BH over the displayed signal×lag family; overlays the authorised recomputed cash policy while retaining the historical traded arm. | Active |
| `08_story_type.ipynb` | Provisional three-way story classes and one pooled 12-type interaction model; writes a blinded human-audit sheet and conditional type-weighting gate. | Executed; human validation blocked |
| `lib/event_types.py` | Vectorised ordered taxonomy, candidate story classes, audit sampling, and date-clustered pooled type slopes. | Active |
| `09_earnings.ipynb` | Exchange-session earnings distance, pre/event/post interactions, timing sensitivity, taxonomy/calendar validation, and W3 exclusion robustness. | Executed |
| `lib/earnings.py` | Frozen calendar mapping, signed event-time windows, and date-clustered interaction inference. | Active |
| `10_thresholds.ipynb` | Fixed-band, logistic, gradient-boosted, MLP, and label-shuffle gates with chronological fit/validation/evaluation, matched activity constraints for future comparisons, final-liquidation costs, and bilateral dollar-neutral normalisation. | Executed; sector input blocked |
| `lib/thresholds.py` | Gate features, model definitions, dollar-neutral gated portfolios, cutoff selection, costs, and date-block intervals. | Active |
| `11_lseg_robustness.ipynb` | LSEG publisher inventory, ex-ante Reuters weighting, and separate sector-33/midcap-22 IC families. | Executed |
| `lib/lseg_robustness.py` | Publisher aggregation, strict next-open mapping, source weighting, and HAC IC tables. | Active |
| `12_lseg_44_labelling.ipynb` | Expanded 44-company corpus contract, exact-hash FinBERT inheritance, full-population OpenRouter Gemma 4 26B audit, and separate human-validation gate. VADER is excluded downstream; the prompt is frozen as `investor_headline_soft_label_v1` (`81596538d99b29b8`). | Executed; human validation blocked |
| `lib/lseg_labelling.py` | Provenance-checks reusable exact-hash labels before resumable expanded-corpus scoring. | Active |
| `13_lseg_44_gemma_robustness.ipynb` | Retrospective non-pooled 44-company return arm: precise timestamp +15-minute mapping, two scorers × nine aggregators, fixed costs, block bootstrap, and metadata-filter sensitivity. | Executed; clean null |
| `lib/lseg_expanded.py` | Validates append-only success rows, canonicalises current-population metadata, verifies the two LSEG price exports, maps timestamps to eligible opens, and evaluates the fixed expanded-corpus families. | Active |
| `14_lseg_sparse_event_strategy.ipynb` | Retrospective falsification of two sparse Gemma signals × 1/3/5-session state: novel material-event selection and the same event attenuated by a headline-only cash-flow-distance proxy. Includes six-arm BH, fixed costs, bootstrap uncertainty, temporal stability, and the Notebook 13 magnitude reference. | Executed; clean null |
| `lib/sparse_events.py` | Strict-prior indexed novelty, provisional material-event and cash-flow-distance features, strongest-event selection, exchange-session state carry, inactive-day closure, and fixed six-arm evaluation. Headline text is removed from returned frames. | Active; machine rules unvalidated |
| `15_lseg_gemma_publisher_conditioning.ipynb` | Frozen expanded-LSEG publisher family: all-source, Reuters-only, non-Reuters-only, and Reuters-2× Gemma h1 signals with source-specific timestamps, four-test BH, costs, and block-bootstrap uncertainty. | Executed; clean null |
| `lib/lseg_publishers.py` | Validates the merged source-code corpus, resolves only `NS:RTRS`, builds source-specific event timestamps, and evaluates the fixed four-arm family without exposing headline text. | Active |
| `16_lseg_cash_flow_distance_audit.ipynb` | Validates the Reuters body corpus against successful Gemma hashes and prepares a return-blind 200-event audit plus an independently ordered 60-event double-code subset. Saved outputs show aggregate counts only. | Executed; audit prepared, labels outstanding |
| `lib/cash_flow_audit.py` | Hash-validates the licensed full-text source, chooses one deterministic representative per headline, makes the fixed stratified sample, and writes leak-free local coder sheets and a private key. | Active |
| `data/cash_flow_distance_codebook.md` | Frozen 0–3/NA construct definitions, coding fields, arbitration rule, and reliability-before-returns gate. | Tracked |
| `17_lseg_sector_portfolio_translation.ipynb` | Four return-blind h1 translations of Gemma `strongest_event`: global rank, within-sector rank, sector extremes, and top-half/bottom-half hysteresis. Reports activity, turnover, break-even, cost curves, block-bootstrap/BH inference, temporal halves, and a labelled matched-day diagnostic. The refreshed notebook records the later finding that discrete opposing Gemma extrema induce abstention. | Executed; improved gross/cost efficiency but clean 10-bps null |
| `18_lseg_sector_hysteresis_scorer_robustness.ipynb` | Applies the exact frozen sector-hysteresis rule to paired FinBERT and Gemma scores on the same 7,348 firm-opens. Reports paired block-bootstrap inference, costs, activity overlap, temporal halves, score agreement, and the strongest-event resolution mechanism. | Executed; no robust scorer advantage and no viable 10-bps arm |
| `lib/sector_portfolios.py` | Frozen 11-sector × four-company map, equal-sector dollar-neutral weights, extreme retention, liquidation accounting, block-mean inference, and fixed-weight cost repricing. | Active |
| `19_fnspid_sentiment_risk_overlay.ipynb` | Applies two sparse negative-news brakes to a predeclared monthly 12-minus-1 price-momentum book, with split-safe dense returns, 10-bps costs, paired inference, risk/alpha gates, cost curves, and yearly diagnostics. | Executed; clean null |
| `20_fnspid_market_sentiment_risk_overlay.ipynb` | Uses lagged market-wide FNSPID negative pressure to cut SPY exposure for one day or under fixed hysteresis. Reports 2-bps costs, paired net/downside inference, non-inferiority and risk gates, cost/year stability, and a labelled post-result timing attribution. | Executed; promising descriptive hysteresis, gate not passed |
| `21_fnspid_sentiment_volatility_target.ipynb` | Fits development-only sentiment-free and sentiment-augmented HAR variance models, evaluates paired QLIKE, and translates both through the same unlevered 10% SPY volatility target. | Executed; HAR is a useful independent base, sentiment increment fails |
| `22_fnspid_surprise_strategy_translation.ipynb` | Development-only 12-arm economic translation of the W4 surprise panel across fixed 1/5/10-session holds and two breadths. Evaluation remains unopened by this notebook. | Executed; clean cost null, cash gate |
| `23_fnspid_trend_volatility_sentiment_base.ipynb` | Tests a standard one-session-lagged 200-session trend filter alone and with HAR, then applies the unchanged aggregate sentiment hysteresis as an incremental modifier. | Executed; trend base fails, sentiment repairs it but does not beat HAR alone |
| `24_fnspid_har_sentiment_hysteresis.ipynb` | Applies the exact frozen Notebook 20 hysteresis state directly to Notebook 21's sentiment-free HAR volatility target, with common accounting, matched-exposure attribution, costs, inference, and temporal stability. | Executed; strongest descriptive Sharpe, alpha/risk gates not passed |
| `25_fnspid_har_sentiment_walkforward.ipynb` | Reconstructs the sentiment-free HAR base with expanding prior-target-only fits every 21 sessions, then applies the exact aggregate hysteresis state over 2013–2019. | Executed; downside mechanism corroborated, return/promotion gates not passed |
| `26_lseg_gemma_sparse_risk_brakes.ipynb` | Non-pooled LSEG/Gemma check on a fixed-share 44-company basket: unchanged aggregate hysteresis feasibility plus a sparse one-session brake for maximally-negative Gemma firm events, with 10-bps costs, temporal halves, paired inference, and explicit selection disclosure. | Executed; modest downside evidence, alpha/useful-risk gates not passed |
| `27_fnspid_har_sentiment_composition.ipynb` | Reconstructs the exact Notebook 24/25 sentiment state and compares the existing HAR multiplier with one absolute-25%-cap interpretation across both opened eras, including multiplicity, costs, exposure mechanics, four subperiods, and labelled stability diagnostics. | Executed; cap does not dominate product, all promotion gates fail |
| `28_lseg_gemma_sparse_extreme_rotation.ipynb` | Sparse long-only LSEG/Gemma translation: cut exact −1 holdings to 25% and transfer released notional to exact +1 firms, with a fixed-share basket and cash-brake comparators, 10-bps costs, paired/BH inference, cost curves, and temporal halves. | Executed; strongest descriptive LSEG arm, all gates fail |
| `lib/fixed_share.py` | Tested equal-dollar fixed-share construction, sparse brake/rotation targets, exact cash and traded-notional accounting, cost-funded entry, final liquidation, and path summaries for LSEG basket translations. | Active |
| `29_lseg_gemma_exact_extrema_spread.ipynb` | Gross-1 dollar-neutral exact +1/−1 Gemma spread with cash on one-sided days, drift-aware ledger accounting, costs, block and conditional random-label inference, temporal halves, concentration, two-name, attribution, and leave-one-company-out diagnostics. | Executed; mechanism gate passes, alpha/strategy-quality gates fail |
| `30_lseg_gemma_capped_extrema_spread.ipynb` | Applies the predeclared 25% single-name quality cap to Notebook 29's unchanged exact-extrema spread, with paired return/downside inference, costs, temporal halves, concentration, attribution, and leave-one-company-out diagnostics. | Executed; standalone quality gate passes, policy-selection/alpha gates fail |
| `31_lseg_gemma_sector_neutral_extrema_spread.ipynb` | Demeans Notebook 30's capped targets inside the frozen four-company sectors, recording infeasible earnings/same-sector-match alternatives, exact sector control, costs, paired inference, temporal halves, factor exposure, and company/sector exclusions. | Executed; decisive factor-control null, all gates fail |
| `32_lseg_gemma_between_sector_extrema.ipynb` | Projects capped exact-event targets onto equal-weight sector baskets, with a conditional random-label sector-placement test, costs, paired inference, temporal halves, factor exposure, attribution, and company/sector exclusions. | Executed; sector-placement mechanism gate passes, policy/quality/alpha gates fail |
| `33_lseg_gemma_original33_universe_robustness.ipynb` | Replays the frozen capped and between-sector Gemma rules after removing all 11 breadth additions, with an add-11 signal-only stop, 44-versus-33 comparison, inference, costs, temporal halves, attribution, and exclusions. | Executed; both universe-quality gates and the sector-placement mechanism gate pass; alpha gates fail |
| `34_lseg_gemma_finbert_matched_count_substitution.ipynb` | Holds the original-33 Gemma event dates, per-day long/short counts, cap, horizon, and costs fixed while substituting FinBERT within-day name ranks for both capped and projected rules. | Executed; no scorer-selection gate passes; FinBERT capped is a promising but retrospective hybrid diagnostic |
| `35_lseg_gemma_event_timing_circular_shift.ipynb` | Compares the frozen Gemma-event/FinBERT-rank hybrid with every non-zero circular shift of the full event/count schedule, preserving frequency, counts, clustering, cap, horizon, and costs. | Executed; both BH-controlled timing gates pass; opened-window mechanism evidence only |
| `36_lseg_gemma_finbert_hybrid_alpha_audit.ipynb` | Audits the single capped Gemma-timing/FinBERT-ranking hybrid against cash with five-session block inference, static market/sector-spanning HAC controls, costs, chronological halves, and rebuilt company/sector exclusions. | Executed; retrospective cash gate passes, stricter factor-alpha gate fails; leading candidate for new-date validation |
| `37_gemma_calibration_transfer.ipynb` | Hash-matched measurement falsification of the exact ±1 trigger across older Cerebras Gemma 31B and complete DeepInfra Gemma 26B, followed only where feasible by the unchanged FinBERT-rank capped strategy and timing null. | Executed; exact endpoint trigger is not calibration-portable; 31B strategy and timing gates fail |
| `38_gemma_calibrated_semantic_event_hybrid.ipynb` | Public-label-only calibration of a Gemma directional-confidence threshold, followed by a frozen original-33 binary collision-abstention schedule and the unchanged FinBERT-rank capped audit. | Executed; public precision gate passes, but 99.29% opposing-direction collision leaves only two active sessions; all return gates fail |
| `39_gemma_continuous_portability.ipynb` | Return-free cross-Gemma audit of continuous firm-open reducers, daily ranks, selected-name overlap, a scale-free trailing-dispersion event schedule, and complete-checkpoint deployment feasibility. | Executed; mean continuous passes every portability gate and freezes a 40-session input-only future strategy schedule; no returns evaluated |
| `40_fnspid_lseg_gemma_continuous_rule_transfer.ipynb` | Post-research cross-regime replay of Notebook 39's continuous adaptive-dispersion construction on the broad FNSPID/FinBERT panel, with relative-breadth translation, isolated split accounting, costs, mechanism nulls, factor alpha, stability, and concentration. | Executed; diversified gross signal in evaluation, but weak development, 4.904-bps break-even, negative 10-bps net performance, and all deployment/mechanism/factor gates fail |
| `41_input_only_turnover_persistence.ipynb` | Return-free LSEG/Gemma and FNSPID audit of turnover sources, active-episode structure, and one frozen episode-opening-basket hold rule. | Executed; turnover falls 32%–43%, but selected-name persistence fails in every regime and entry/exit cost remains structurally high; candidate rejected before returns |
| `42_fnspid_2010_preperiod_transfer.ipynb` | Backward external-time replay of Notebook 39's unchanged LSEG/Gemma-derived continuous construction on locally scored 2009-Q4/2010 FNSPID headlines, with current-open eligibility parity, input-only feasibility, costs, mechanism nulls, factor control, halves, and concentration. | Executed; 59 fully tradable active sessions, gross/net Sharpe −1.141/−3.397, all replication gates fail; stable cross-era alpha rejected |
| `lib/preperiod_finbert.py` | Resumable local-only extraction and FinBERT scoring for the separate pre-2011 checkpoint, reusing the frozen FNSPID timing and event-identity contract while exposing only aggregate firm-session signals. | Active; 41,019 headlines scored locally, licensed text remains ignored |
| `43_lseg_gemma_story_family_first_release.ipynb` | Bounded retrospective comparison of Notebook 39's exact continuous Gemma rule before and after return-independent first-release-per-LSEG-story-family filtering, with an input-only gate, exact ledger accounting, costs, two-test BH inference, halves, and plots. | Executed; 8,314 later-revision hashes removed, but gross/net Sharpe remains −1.473/−2.502 and both improvement/quality gates fail |
| `44_strategy_evidence_synthesis.ipynb` | Aggregate-only evidence registry, paired sentiment-overlay scorecard, and cross-regime continuous-rule stress table with an ordinal evidence-quality policy that prevents retrospective Sharpe from overriding independence, stability, and inference. | Executed; HAR retained as sole working base, 0/6 sentiment return intervals exclude zero, no sentiment strategy promoted |
| `45_sentiment_downside_timing_placebo.ipynb` | Exact all-shift timing placebo for the frozen HAR sentiment products in 2013–2019 and 2020–2023 plus the separate LSEG/Gemma exact-negative firm brake, preserving schedule structure and accounting. | Executed; 2020–2023 downside timing passes BH, older FNSPID and LSEG/Gemma do not; cross-regime and return gates fail |
| `lib/lseg_story_families.py` | Hash-only reconstruction of terminal-suffix LSEG story families and earliest releases with source-manifest verification; licensed headline text is never returned. | Active; used by Notebook 43 |
| `lib/sparse_spread.py` | Tested exact-extrema equal-leg targets, symmetric capacity capping, within-sector demeaning, equal-weight sector projection, fail-closed exposure audits, and compact conversion of the validated drift-aware ledger. | Active |
| `lib/risk_overlay.py` | Leakage-safe monthly momentum, negative-risk flags, aggregate pressure state, fixed exposure rules, dense adjusted-open returns, explicit cost accounting, and paired block-bootstrap helpers for Notebooks 19–40. | Active |
| `lib/volatility_target.py` | Causal HAR variance features, development-only and expanding walk-forward fitting/calibration, unlevered volatility targeting, lagged price-trend construction, exposure composition, costs, and paired circular-block inference for Notebooks 21–25 and 27. | Active |
| `lib/openrouter_validation.py` | Frozen, resumable public-benchmark gate for OpenRouter Gemma 4 26B pinned to DeepInfra with ZDR, no fallback, strict probability JSON, and cost/quality metrics. It refuses to use LSEG inputs by construction. | Gate passed: 1,000/1,000 coverage; accuracy 0.808; macro-F1 0.813 |
| `../scripts/run_openrouter_gemma4_validation.py` | Spend-gated command for preparing, executing, or explicitly retrying the 1,000-row OpenRouter validation. See [`../docs/openrouter_gemma4_validation.md`](../docs/openrouter_gemma4_validation.md). | Executed 2026-08-04; evidence ignored/local |
| `lib/openrouter_lseg.py` | Bounded, append-only and resumable licensed-corpus scorer. Freezes the 888,155-headline population, exact input hashes, provider/privacy/FP8 contract, prompt hash, and researcher permission; failed attempts remain auditable and are retried without duplicating successes. | Completed 2026-08-05: 888,155/888,155 unique successes; $33.6153 |
| `../scripts/run_openrouter_lseg_scoring.py` | Explicit-authorisation entrypoint for the full LSEG scorer. The private output and log paths are recorded in [`../docs/lseg_external_processing_authorisation.md`](../docs/lseg_external_processing_authorisation.md). | Executed; private append-only output remains off Git |
| `outputs/12_lseg_44_labelling/` | Licence-safe aggregate coverage, label-distribution, and LLM-design tables/figures. | Ignored/local |
| `outputs/13_lseg_44_gemma_robustness/` | Aggregate scorer/return tables, firm-open panels, daily portfolios, figures, and manifest; no headline text. | Ignored/local |
| `outputs/14_lseg_sparse_event_strategy/` | Aggregate eligibility, six-arm return tables, daily portfolios, figures, and manifest; no headline text. | Ignored/local |
| `outputs/15_lseg_gemma_publisher_conditioning/` | Aggregate source inventory, four-arm result tables, daily portfolios, figures, and manifest; no headline text. | Ignored/local |
| `outputs/16_lseg_cash_flow_distance_audit/` | Licensed coder worksheets, private sampling key, aggregate sampling summary/plot, and manifest. Never commit this directory. | Ignored/local |
| `outputs/17_lseg_sector_portfolio_translation/` | Aggregate four-arm results, activity-matched diagnostic, daily portfolios, cost curve, figures, and manifest; no headline text. | Ignored/local |
| `outputs/18_lseg_sector_hysteresis_scorer_robustness/` | Aggregate paired-scorer results, cost curve, activity and resolution diagnostics, daily portfolios, figures, and manifest; no headline text. | Ignored/local |
| `outputs/19_fnspid_sentiment_risk_overlay/` | Aggregate firm-level brake results, paired inference, activity, cost/year diagnostics, daily paths, figures, and manifest; no headline text. | Ignored/local |
| `outputs/20_fnspid_market_sentiment_risk_overlay/` | Aggregate SPY exposure results, paired net/downside inference, gates, timing attribution, daily pressure/path data, figures, and manifest; no headline text. | Ignored/local |
| `outputs/21_fnspid_sentiment_volatility_target/` | Aggregate variance-forecast, strategy, inference, stability, cost, figure, and manifest evidence; no headline text. | Ignored/local |
| `outputs/22_fnspid_surprise_strategy_translation/` | Development-only sweep, cost gate, rank-invariance audit, figures, daily diagnostic path, and manifest; no headline text. | Ignored/local |
| `outputs/23_fnspid_trend_volatility_sentiment_base/` | Aggregate base-comparison results, inference, exposure paths, costs, figures, and manifest; no headline text. | Ignored/local |
| `outputs/24_fnspid_har_sentiment_hysteresis/` | Aggregate direct-composition results, paired downside/return inference, matched-exposure attribution, costs, figures, and manifest; no headline text. | Ignored/local |
| `outputs/25_fnspid_har_sentiment_walkforward/` | Aggregate pre-2020 forecasts/fits, strategy results, paired inference, stability, costs, figures, and manifest; no headline text. | Ignored/local |
| `outputs/26_lseg_gemma_sparse_risk_brakes/` | Aggregate LSEG/Gemma coverage, signal activity, fixed-share basket paths, paired inference, cost/stability figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/27_fnspid_har_sentiment_composition/` | Aggregate cross-era composition results, exact upstream reproduction audit, paired inference, costs, exposure diagnostics, four-subperiod figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/28_lseg_gemma_sparse_extreme_rotation/` | Aggregate exact-event activity, capital-transfer audit, fixed-share paths, paired inference, costs, temporal halves, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/29_lseg_gemma_exact_extrema_spread/` | Aggregate dollar-neutral path, drift-aware ledger, inference, randomisation, concentration, attribution, leave-one-out and two-name diagnostics, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/30_lseg_gemma_capped_extrema_spread/` | Aggregate capped/uncapped paths, reproduction audit, paired inference, costs, temporal halves, concentration, attribution, leave-one-out diagnostics, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/31_lseg_gemma_sector_neutral_extrema_spread/` | Aggregate sector-neutral/capped paths, feasibility and construction audits, return-loss decomposition, inference, costs, temporal/factor diagnostics, exclusions, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/32_lseg_gemma_between_sector_extrema/` | Aggregate between-sector/capped paths, construction and reproduction audits, conditional randomisation, inference, costs, temporal/factor diagnostics, attribution, exclusions, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/33_lseg_gemma_original33_universe_robustness/` | Aggregate original-33 capped/projected paths, add-11 signal feasibility, frozen-universe comparison, conditional randomisation, inference, costs, temporal/factor diagnostics, attribution, exclusions, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/34_lseg_gemma_finbert_matched_count_substitution/` | Aggregate matched-count Gemma/FinBERT capped and projected paths, scorer-resolution/selection-overlap audit, inference, costs, temporal halves, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/35_lseg_gemma_event_timing_circular_shift/` | Aggregate observed hybrid paths, full 166-shift timing null, inference, cost curve, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/36_lseg_gemma_finbert_hybrid_alpha_audit/` | Aggregate capped-hybrid path, cash inference, market/sector factor intercepts, cost curve, temporal halves, company/sector exclusions, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/37_gemma_calibration_transfer/` | Aggregate hash/endpoint coverage, firm-open trigger transfer, permitted 31B hybrid path, cash/timing inference, costs, halves, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/38_gemma_calibrated_semantic_event_hybrid/` | Aggregate public calibration, semantic collision/schedule audit, capped hybrid path, cash/endpoint/timing/factor/dependence diagnostics, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/39_gemma_continuous_portability/` | Aggregate shared-headline and firm-open portability, daily rank/name overlap, adaptive schedule, complete-26B input-only deployment audit, figures, gates, and frozen future specification; no text or returns. | Ignored/local |
| `outputs/40_fnspid_lseg_gemma_continuous_rule_transfer/` | Aggregate translated schedules, drift-aware daily paths, costs, inference, mechanism nulls, factor intercepts, concentration, figures, gates, and manifest; no headline text. | Ignored/local |
| `outputs/41_input_only_turnover_persistence/` | Aggregate input-only episode, target-turnover, continuation-overlap, figure, gate, and manifest evidence; no targets, text, prices, or returns. | Ignored/local |
| `outputs/42_fnspid_2010_preperiod_transfer/` | Aggregate backward-transfer schedule, invalidation history, daily paths, costs, inference, mechanism nulls, factor intercepts, concentration, figures, and manifest; the licensed local checkpoint is ignored. | Ignored/local |
| `outputs/43_lseg_gemma_story_family_first_release/` | Aggregate story-family audit, filtered/baseline schedules and paths, costs, paired inference, halves, accounting audit, figures, and manifest; no headline text. | Ignored/local |
| `outputs/44_strategy_evidence_synthesis/` | Aggregate-only evidence registry, overlay scorecard, cross-regime transfer table, figures, source hashes, and manifest; no row-level prices, returns, positions, or text. | Ignored/local |
| `outputs/45_sentiment_downside_timing_placebo/` | Exact circular-shift nulls, reproduction audit, prospective-input stop, timing tables, figures, source hashes, and manifest; no headline text. | Ignored/local |
| `EXPLORATORY_EXPERIMENT_LEDGER.md` | Aggregate completion/result ledger, including blocked inputs and null arms. | Current |
| `lib/plots.py` | House figure style and the colour roles (categorical / ordinal / diverging / status). | Active |
| `outputs/07_strategy_analysis/` | Event-time CAR, quantile spread and monotonicity, sweep surface, monthly heatmap, book-health figures. | Ignored/local |
| `INVALIDATED_RUNS.md` | Source-controlled ledger for superseded/invalidated result chains and their generated snapshot locations. | Active |
| `plan.html` | Live phase plan and checklist. | Tracked |

All numbered stage notebooks now exist. Gate F1 and promotion remain decisions,
not missing notebook implementations.

## Running The Existing Notebooks

From the repository root, using Python 3.12 with the `tailrisk`, `finbert`, and figure dependencies installed:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
  final_experiments/00_data_inventory.ipynb
```

Repeat in numeric order for `01_panel.ipynb` through
`45_sentiment_downside_timing_placebo.ipynb`. Run from the repository root so relative paths
resolve consistently. The numbered `.ipynb` files are the sole notebook source;
reusable code remains under `lib/`.

### Refresh rule after a strategy update

Do not rerun only the notebook that was edited. After any change to a strategy
signal, portfolio construction, horizon, breadth, threshold, transaction cost,
turnover/accounting convention, deployment rule, or reported strategy result:

1. inspect all numbered notebooks in `final_experiments/` for generated-file
   dependencies and result/prose references;
2. rerun every notebook whose inputs, embedded outputs, figures, manifests, or
   conclusions could change, in dependency order;
3. treat uncertainty as affected and rerun the notebook rather than leaving a
   potentially stale saved output; and
4. reconcile `README.md`, `EXPLORATORY_EXPERIMENT_LEDGER.md`, `plan.html`, and
   `INVALIDATED_RUNS.md` wherever the recorded result or status changed.

For the current strategy chain, `06_strategy.ipynb` produces inputs consumed by
`07_strategy_analysis.ipynb`; threshold/deployment changes can also affect
`10_thresholds.ipynb`. This is a minimum known dependency set, not an exhaustive
allowlist.

These are not clone-only examples. They require local artifacts that are intentionally absent from Git:

- FNSPID archives under the external data root recorded in `00_data_inventory`;
- the Moment-2 FinBERT checkpoint and Gate-1 audit outputs under `reports/`;
- the local LSEG earnings-calendar files under `data/earnings/`;
- the local merged LSEG 44-company headline corpus, inherited score seed, and
  completed FinBERT/OpenRouter score artifacts under `Data/collections/`;
- the local 33-company and added-11 LSEG price exports used by Notebooks 13–15 and 17–18;
- the local licensed Reuters body corpus used to prepare Notebook 16's audit;
- FNSPID prices and SPY data inside the recorded archive/checkpoint chain.

If a path moves, update the notebook parameter cell or helper configuration and record the change. Do not silently substitute another dataset.

## Current Panel Contract

Grain:

```text
(symbol, session_date) where news exists and adjusted-open stock + SPY prices exist
```

Core fields include:

- `symbol`, `session_date`, and chronological `split`;
- FinBERT firm-day score moments and article/recap counts;
- split-adjusted stock/market open-to-open returns and `ar_open_h1`;
- earnings-session flags from the LSEG results calendar.

Known null/unavailable fields on the current FNSPID checkpoint grain:

- publisher/source identity suitable for publisher weighting;
- `story_family_id` revision lineage;
- point-in-time `availability_timestamp`;
- precise intraday publication time for most events;
- explicit sector mapping.

Those gaps constrain claims. They are not imputed.

## Data And Timing Rules

### News

FNSPID dates inherit the revised Gate-1 mapping already applied in the FinBERT checkpoint: date-only news maps to the first XNYS session strictly after the stated date. This supports next-open estimands; it does not prove intraday availability.

### Earnings

The local LSEG calendar uses:

- BMO: same XNYS session;
- AMC: next XNYS session;
- unknown time: conservative next session;
- during-market: currently same session, with the pre-news-open contamination caveat carried forward.

See [`data/earnings/README.md`](data/earnings/README.md) before using Workstream 6. The current gather has 22,883 quarterly rows and 490 of 574 cohort symbols with at least one kept quarterly event; recoverable mapping/title-filter gaps remain.

### Returns

The panel uses split-adjusted opens and exact next-exchange-session SPY-adjusted returns. A missing stock price on that immediate session remains missing rather than jumping to the next observed stock date (715,534 of 715,546 rows currently have `ret_open_h1`). Dividends are not back-adjusted. Rows are selected on both news and price availability, which creates a collider/selection limitation for inference.

## Output And Licence Boundary

`outputs/` and the earnings payloads are ignored because they may contain licensed or large data. Do not force-add them wholesale.

A result may be promoted only when:

1. the estimand, event grain, split, inference unit, seed, and multiplicity family are recorded;
2. the evaluation block was not used for selection;
3. attrition and known timing/data gaps are carried into the report;
4. the promoted artifact contains aggregate evidence only;
5. the run is registered in `experiments/manifest.toml` with code/data identities.

## Standing Analysis Rules

Learned from correcting the first W3/W4 pass. These bind every later arm.

- **Forward returns come from the exchange calendar, never from shifting inside
  an event panel.** `shift(-1)` on a news-bearing panel returns the firm's next
  *news* day. On this spine that is more than one session away for ~31% of rows
  (median gap 2 calendar days, 99th percentile 26, max 1,018), which silently
  paired a multi-week stock return with a one-session SPY return. Pass the dense
  price frame; `attach_open_returns` now requires it.
- **A cross-sectional book must be checked for neutrality, not assumed.** Record
  `net_exposure` on every arm and assert it. The historical `sign` mode, a
  breadth cut over a tie block, and row-level gating without separate long/short
  leg normalisation each produced a one-sided book that read as a signal result.
- **Every signal comparison needs a null nest.** An OOS R² of +0.003 is
  meaningless until the model with no sentiment in it is scored on the same rows.
  `MODEL_SPECS` carries `M_control_only` for this reason; do not drop it.
- **Rank within the session before taking positions.** `position_mode="sign"`
  shorts every name of a non-negative signal and never goes long, and it collapses
  any two rules that share a sign onto one book. Use `position_mode="cs_rank"`
  for signal comparisons; `"sign"` exists only to reproduce the first pass.
- **Report the effective number of tests beside the BH family.** Nine aggregation
  rules are about four independent statistics on this panel.
- **Condition on `n` before claiming a distribution-shape effect.** Half the
  panel is `n=1`, where most rules are identical and dispersion is zero, and
  dispersion is Spearman-0.87 correlated with the article count.
- **A ranking without an interval is not a result.** OOS R² differences get a
  date-block bootstrap on the paired loss differential.
- **A pooled rank correlation with clustered errors is not a cross-sectional
  IC.** Rank signal and return within each session, average the daily Spearman
  correlations through time, and use HAC or a date-block bootstrap.
- **Holding-period tails may not cross the frozen split.** A development
  formation is eligible only when every return through its declared horizon
  ends by 2019-12-31; final positions pay liquidation turnover.
- **Match the cost label to the turnover definition.** Turnover is half the L1
  weight change. A quoted per-side cost is therefore charged as
  `2 × turnover × cost_bps_per_side / 10_000`; break-even uses the same factor.
- **Let a strategy choose cash.** A ranking rule always returns a least-bad
  cell, even when every cell loses after costs. The deployment gate therefore
  requires sufficient history/breadth, positive net Sharpe, break-even at least
  equal to the charged cost, and a positive block-bootstrap lower bound. If no
  cell clears all four, the action is cash. When continued iteration on an open
  evaluation block is explicitly authorised, recompute and compare the policy
  directly while labelling it iterative/retrospective and preserving the prior
  traded result.
- **Compare threshold rules at comparable activity.** The historical fixed band
  was selected without the learned gates' activity floor and traded on only
  2.98% of validation sessions. Preserve that disclosed historical comparison,
  but require every fixed or learned gate on new data to average at least five
  active names and trade on at least half of validation sessions. Charge final
  liquidation turnover in every arm. The authorised iterative recomputation of
  the activity-matched band is reported beside the historical band and cash;
  it is not presented as a pristine holdout result.
- **An event-time shape is descriptive until its spread has dependence-aware
  inference.** `07_strategy_analysis` uses a 20-session date-block bootstrap and
  BH across all 4 displayed signals × 20 lags; zero of 80 cells currently survives.
- **Use event time diagnostically, not as an automatic rescue of a portfolio
  null.** `strategy.event_time_car` shows descriptive CAR by signal quantile;
  the Q5−Q1 path needs the block-bootstrap/BH procedure before interpretation.
- **Do not overclaim either a signal or a null.** The repaired lag-1 Q5−Q1
  spreads are about 1.2–1.8 bps versus a 20 bps round trip, but every displayed
  interval crosses zero. The supported statement is economic non-viability;
  neither a predictive effect nor a reversal is established by this family.
- **Figures: colour by the job it does** (`lib/plots.py`). Categorical for
  identity, one-hue ordinal for ranked groups, diverging with a **neutral grey**
  midpoint for signed quantities. A coloured midpoint makes zero look like a
  value; more than about five series becomes small multiples, not more hues.

## Working Style

- Prefer a clear notebook narrative and a decisive plot over another framework.
- Put reusable joins or metrics in `lib/`; keep exploratory glue in the notebook.
- Add tests only where a silent reusable-helper error would corrupt downstream results.
- Keep nulls, failed arms, and invalidated runs visible.
- Do not extend `src/sentiment_benchmark` unless a final accepted result genuinely needs stable library support.
