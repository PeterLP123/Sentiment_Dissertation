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
| Backward replication | Frozen same-33 LSEG 2024-01-01 to 2025-10-26 collection; no scoring or return access until its staged gates pass |
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
   cash-flow-distance audit; its labels remain outstanding. Notebook 62 now
   validates the frozen worksheet identities, reports aggregate coding
   progress, and conditionally executes the predeclared reliability gate; it
   currently stops at coder A 0/200 and coder B 0/60. Notebook 17 then
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
   Notebook 44 consolidates the aggregate-only strategy evidence without
   loading any price, return, position, or text rows and is now refreshed
   through Notebook 65. The sentiment-free HAR target remains the research
   benchmark from Notebook 21 (net Sharpe 0.673, +29.85% net return, −15.73%
   drawdown), but the later transfer failure means it is not a time-stable or
   deployment-qualified strategy. Across fifteen paired sentiment/construction comparisons,
   one favourable return interval excludes zero and six report lower downside
   loss. The sole favourable interval is Notebook 47's retrospective
   revision-delta comparison with a matched latest-score-level construction;
   it is not a cash-alpha result and does not pass promotion. The
   continuous-rule stress table is net-negative at 10
   bps/side in all five regime/split rows. This freezes the useful result as
   repeated de-risking evidence plus a directional-alpha null; it is an input
   to Gate F1, not the RQ decision.
   Notebook 55 then restores the omitted frozen 22-company LSEG mid-cap FinBERT
   event result without loading row-level outcomes or constructing a new return.
   Its evaluation net return/Sharpe are +4.00%/0.593 at 10 bps/side and its
   legacy local gate passes, but the 95% cash interval is
   [−8.86,+19.34] bps/session, only 37 evaluation sessions are active, and a
   name can carry 50% absolute weight. It is retained as a tier-2 historical
   LSEG/FinBERT strategy-shaped comparator, not a current base, Gemma
   head-to-head result, or alpha promotion.
   Notebook 56 then lineage-audits every frozen local LSEG/FinBERT strategy run
   using only aggregate manifests and metrics. Ten run directories collapse to
   five exact metric payloads and four declared strategy families; three
   payloads are repeated artifacts rather than replications. Only the same
   mid-cap event family clears the descriptive coverage/stability floor. The
   only other positive evaluation arm has net Sharpe 0.545 on five active
   sessions after development Sharpe −1.456. No distinct adequately covered
   positive local base was omitted, and no new return or signal is constructed.
   Notebook 57 then separates research-benchmark, prospective-primary,
   non-promoting risk-diagnostic, and historical-comparator roles using only
   aggregate artifacts. The unchanged Notebook 36 hybrid is the sole
   prospective strategy candidate: gross/net Sharpe 3.700/2.271 at 10 bps/side,
   25.24 bps/side break-even, and positive net Sharpe in both halves. At 5
   bps/side its scenario net Sharpe is 2.995. This is economically interesting,
   but not validated alpha: the sector-factor interval is [−0.44,+10.08]
   bps/session and the candidate was selected on the opened window. Under the
   frozen Y=1 impact scenario, net Sharpe falls from 1.679 at $1m to 0.413 at
   $10m and −0.632 at $25m. Notebook 58 then applies 9,999 paired five-session
   circular-block resamples at every predeclared AUM. No grid point clears the
   uncertainty-aware gate. Even at $1m, the mean-net interval is
   [−1.29,+11.64] bps/session, the probability of a positive terminal return is
   93.48%, and second-half Sharpe is only 0.016. At $10m the interval is
   [−5.20,+8.02], probability is 62.53%, and second-half Sharpe is −2.011.
   Therefore $10m is only an optimistic point-estimate scenario bound, not
   uncertainty-qualified capacity. Notebook 59 then exactly decomposes the
   unchanged open-to-open gross path: intraday contributes +6.47 bps/session
   [+1.42,+11.55] with BH q=0.0242, overnight contributes +4.80
   [+0.11,+10.14] with q=0.0611, and the long leg supplies 97.43% of total
   arithmetic gross contribution (Sharpe 4.344 versus 0.163 for the short leg).
   Notebook 60 preserves those long weights and finds +8.36 bps/session
   [+3.41,+13.78] versus same-date equal-weight market exposure (BH q=0.0038)
   and +3.66 [+0.95,+6.62] versus same-sector equal weight (q=0.0121), with
   both differences positive in both halves. This is useful opened-window
   gross mechanism evidence for intraday and within-sector name selection, not
   a new strategy or validation; the zero-crossing factor-alpha interval still
   governs. Notebook 61 then exactly decomposes the selected-long minus
   same-sector difference. Intraday is +1.45 bps/session [−0.32,+3.40] and
   overnight is +2.21 [−0.006,+4.85]; neither passes the two-test correction
   (both BH q=0.1264), and overnight reverses to −0.36 bps/session in the second
   half. The positive total therefore does not license an intraday or overnight
   variant. Notebook 63 then performs one frozen joint characteristic audit of
   the unchanged selected-long path. Strictly lagged 1/5-session returns,
   volatility, and log ADV explain −0.073 bps/session of the +3.662-bps raw
   same-sector difference, leaving +3.735 bps/session with a coefficient-refit
   block-bootstrap interval [+0.867,+6.746] (p=0.0086) and positive halves
   (+5.335/+2.116). This strengthens the opened-window name-selection mechanism
   claim, but it is not a residual strategy, independent confirmation, or
   validated alpha. Notebook 64 then performs the one user-authorized costed
   translation of Notebook 60's controls. Netting the selected-long weights
   against the equal-weight market produces net Sharpe 2.332, +8.70% net return,
   −1.85% maximum drawdown, 55.20 turnover, and 25.30-bps/side break-even at
   10 bps/side, versus 2.271, +11.82%, −2.90%, 74.56, and 25.24 for the unchanged
   incumbent. The market residual stays positive in both halves, but its
   +5.06-bps/session cash comparison has BH q=0.0966, its sector-factor alpha
   has q=0.0970, and its mean net return is 1.75 bps/session below the incumbent
   [−5.65,+1.99]. The same-sector residual has net Sharpe 0.818 and a negative
   second-half mean. Both frozen classifications are `nonviable`: the market
   hedge is a descriptive risk-efficiency result rather than improved return
   evidence, and the sector hedge is rejected. Historical session,
   characteristic-control, hedge, and scaling expansion is closed. Notebook 65
   then applies one frozen explicit-target, single-company, non-technical
   metadata filter only to Gemma's timing/count leg while preserving Notebook
   36's complete unfiltered FinBERT ranks. Defined Gemma strongest-event
   firm-opens increase from 3,841 to 4,826, but active both-sign sessions collapse
   from 70 to 22. Gross/net Sharpe falls to 1.508/0.881, net return to +3.09%,
   and the filtered rule trails the incumbent by −4.93 bps/session
   [−10.82,+0.59]. Its cash interval [−2.64,+7.01] and sector-spanning interval
   [−2.46,+4.46] both cross zero. The classification is `nonviable`: this is a
   useful specificity-versus-breadth mechanism null, and no further historical
   metadata-filter variants are permitted. Carry the equal-weight
   hybrid unchanged to the one prospective replay; do not retune it historically.
   Notebook 45 then asks whether the lower downside loss is unusually well
   timed or merely mechanical de-risking. It enumerates every circular placement
   of the exact frozen state in three disjoint regimes. Only FNSPID 2020–2023
   passes the three-test downside family (99.1st percentile; exact p=0.0100,
   BH q=0.0301). FNSPID 2013–2019 is suggestive but misses BH (p=0.0455,
   q=0.0682), while the LSEG/Gemma firm brake is ordinary (p=q=0.3653).
   Zero return-timing tests pass. The cross-regime mechanism gate therefore
   fails: retain one-period market-risk timing evidence, not a general semantic
   timing or Gemma-alpha claim. Notebook 53 now supplies the first local news
   batch after 2026-06-26, but its 27-session calendar upper bound is below the
   frozen 120-session minimum, so it stops before scoring, prices, or returns.
   Notebook 54 registers that batch in an append-only ledger and resolves
   130,692 strictly eligible unique normalized headlines from 162,049 eligible
   source rows: 19.35% fewer possible scoring requests and about $4.95 at the
   earlier realised Gemma average. This is a planning estimate, not a quote or
   authorisation; the 27/120 stop is unchanged.
   Notebook 46 then applies Notebook 21's unchanged 2011–2019 HAR coefficients
   to the 167-session LSEG S&P 500 index window and composes the exact Notebook
   26 Gemma brakes as `1 − 0.75 × exact-negative firms / 44`. HAR controls
   realised volatility to 9.74% and drawdown to −7.77%, but its net Sharpe is
   only 0.240 versus 0.940 for constant matched exposure; the paired timing loss
   is −2.731 bps/session [−4.394,−1.360], and both external-time base gates fail.
   The Gemma overlay adds only +0.013 bps/session and +0.007 Sharpe. Its downside
   reduction passes the two-test family (BH q=0.0244), but drawdown improves only
   1.4% and the downside advantage disappears against a constant multiplier
   (p=0.8218). This is mechanical de-risking evidence, not semantic timing or
   alpha; no deployment-qualified base remains.
   Notebook 47 then tests whether the *change* in Gemma sentiment within a
   Reuters story family carries information that its latest level misses. A
   hash-only, licence-safe reconstruction yields 8,654 first-to-current
   transitions and 1,606 nonzero original-33 firm-open updates, with 165 of
   167 sessions containing both positive and negative revisions. The frozen
   one-session revision-delta spread has gross/net Sharpe 1.523/−1.866,
   +11.43%/−13.26% return, and 4.49-bps/side break-even at 10 bps/side. Its
   paired net mean versus cash is −8.268 bps/session
   [−17.899,+1.200] (q=0.0914), but it beats a matched latest-score-level
   construction by +15.620 bps/session [+2.683,+28.725] (BH q=0.0368).
   Gross Sharpe is positive in both halves; net Sharpe is negative in both.
   Preserve this as incremental revision-information evidence, not validated
   alpha or a deployable strategy.
   Notebook 48 applies the only parameter-free implementation repair: carry
   each firm's last nonzero revision sign until it is replaced, with no expiry,
   decay, threshold, lookback, or horizon search. Target turnover falls 72.3%
   to 0.416 per session; median/p90 signal age is 2/18 sessions. Gross/net
   Sharpe improves to 1.459/−0.331, net return is −1.39%, and break-even rises
   to 8.15 bps/side. Performance is positive at 1–5 bps/side but fails the
   frozen 10-bps cost, cash, factor, stability, and dependence gates. The
   persistent-minus-event effect is +7.498 bps/session
   [−0.006,+15.165] (BH q=0.1064), and 10-bps half Sharpes are +0.434/−1.074.
   This is a useful cost frontier and implementation result; do not tune a
   persistence variant on the opened window.
   Notebook 49 then freezes the single scale-free semantic restriction suggested
   by Notebook 47: trade only Reuters families whose first and current Gemma
   scores have strictly opposite signs. The return-free gate passes with 467
   original-33 transition associations, 390 firm-opens across 32 companies, and
   74 both-sign sessions split 46/28 by half. It does not isolate the useful
   gross effect. Gross/net Sharpe is 0.256/−1.646, net return is −7.17%, and
   break-even is only 1.34 bps/side. Strict flips minus cash are −4.366
   bps/session [−9.428,+0.388] (BH q=0.161); versus the broad revision arm they
   add +3.902 bps/session [−5.289,+13.007] (BH q=0.409). The conditional
   same-day revision-update random-name p-value is 0.4227, gross Sharpe is
   0.452/−0.008 by half, and the sector-factor intercept is significantly
   negative. Every promotion gate fails. This closes dramatic zero-crossing
   revisions as the explanation and forbids another revision filter on the
   opened window.
   Notebook 50 then tests one conventional risk-allocation change without
   changing Gemma event timing, FinBERT-selected names, the 25% cap, horizon,
   or cost. Strictly lagged 20-session inverse-volatility sizing passes its
   return-blind gate, retains all 70 active sessions, lowers target turnover
   11.7%, and reduces annualised net volatility from 7.55% to 6.47%. It retains
   gross/net Sharpe 3.691/2.207 and +9.77% net return, but the equal-weight
   Notebook 36 hybrid remains slightly stronger at 3.700/2.271 and +11.82%.
   The paired difference is −1.136 bps/session [−2.679,+0.216]
   (BH q=0.123), the cash comparison also misses BH (q=0.111), and the
   sector-spanning alpha is +4.028 bps/session [−0.656,+8.712] (p=0.0919).
   Preserve the lower-risk/lower-turnover trade-off as a useful null; it does
   not replace the equal-weight hybrid or create validated alpha.
   Notebook 51 then freezes an implementation-capacity diagnostic before
   calculating it. The exact Notebook 36 targets and drift-aware orders are
   replayed over a predeclared $1m–$1bn AUM grid with strictly lagged 20-session
   dollar volume and open-return volatility, the existing 10-bps/side cost, and
   added square-root impact. Under the conservative primary coefficient Y=1,
   $1m retains net Sharpe 1.679 and +8.53% return; $10m retains 0.413 and
   +1.89%, with every order below 5% of lagged full-day ADV. At $25m, net
   Sharpe is −0.632, return is −3.29%, and ten orders breach the 5% ceiling,
   eight in DUK and two in PLD. The optimistic predeclared-grid capacity bound
   is therefore $10m. This is useful implementation evidence, not new alpha;
   full-day ADV is optimistic for open-auction execution.
   Notebook 52 then freezes the one analytical liquidity-sizing implication:
   within each unchanged leg, allocate in proportion to
   `ADV / max(sigma, 0.005)^2` with deterministic water filling at the 25%
   cap. All 70 Gemma dates, FinBERT names/signs, leg budgets, and exposure
   constraints survive, and the from-flat impact proxy falls 6.08%. However,
   target-path L1 turnover rises 0.54%, from 74.458 to 74.858, violating the
   predeclared no-higher-turnover gate. The notebook stops cleanly with
   `returns_constructed = false`; no modified next-open return is inspected.
   Preserve the input-only NO-GO and do not tune smoothing or shrinkage on the
   opened window.
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
| `16_lseg_cash_flow_distance_audit.ipynb` | Validates the Reuters body corpus against successful Gemma hashes and prepares a return-blind 200-event audit plus an independently ordered 60-event double-code subset. Immutable ID/headline/body hashes bind both worksheets to the frozen sample. | Executed; audit prepared, labels outstanding; do not rerun after coding starts |
| `lib/cash_flow_audit.py` | Hash-validates the licensed full-text source, creates the deterministic sample, validates worksheet identity/completion, and evaluates the frozen weighted-kappa/bootstrap gate without loading returns or sentiment. | Active; focused tests pass |
| `data/cash_flow_distance_codebook.md` | Frozen 0–3/NA definitions, coding fields, arbitration, jointly-numeric requirement, and reliability-before-returns gate. | Tracked |
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
| `44_strategy_evidence_synthesis.ipynb` | Aggregate-only evidence registry, paired sentiment/construction scorecard, cross-regime continuous-rule stress table, implementation-capacity evidence, prospective population status, historical mid-cap comparator reconciliation, local strategy-lineage coverage, candidate-base role separation, and post-result hybrid mechanism/translation controls under an ordinal policy that prevents retrospective Sharpe from overriding independence, stability, and inference. | Refreshed through N65 with 23 registry rows; HAR retained only as research benchmark, the Notebook 36 hybrid retained unchanged as the sole prospective strategy candidate, the positive mid-cap FinBERT result retained as tier-2 historical evidence, one of fifteen paired return intervals is favourable and excludes zero but is retrospective and non-promoting, no AUM clears the uncertainty-aware capacity gate, the market-residual hedge improves descriptive Sharpe/drawdown but fails corrected cash, factor, and incumbent-return gates, the Gemma specificity filter collapses active sessions and fails all return gates, the prospective ledger remains at 27/120 sessions, and no validated or deployment-qualified alpha exists |
| `45_sentiment_downside_timing_placebo.ipynb` | Exact all-shift timing placebo for the frozen HAR sentiment products in 2013–2019 and 2020–2023 plus the separate LSEG/Gemma exact-negative firm brake, preserving schedule structure and accounting. | Executed; 2020–2023 downside timing passes BH, older FNSPID and LSEG/Gemma do not; cross-regime and return gates fail |
| `46_lseg_external_time_har_base_transfer.ipynb` | Unchanged Notebook 21 HAR-model transfer to the later LSEG S&P 500 index window plus a parameter-free market aggregate of Notebook 26's exact-negative Gemma firm brakes, with matched-exposure controls, costs, paired/BH inference, halves, and plots. | Executed; HAR transfer and Gemma incremental gates fail; downside reduction is mechanical rather than unusually timed |
| `47_lseg_gemma_story_revision_delta.ipynb` | Original-33 one-session spread on the first-to-current Gemma score change within Reuters story families, compared with cash and a same-date/name/count/gross latest-score-level construction, with costs, BH inference, halves, factors, exclusions, and plots. | Executed; revision delta beats matched level, but cash, 10-bps quality, factor, and dependence gates fail |
| `48_lseg_gemma_persistent_revision_state.ipynb` | Parameter-free carry-forward of each firm's latest nonzero story-revision sign, with input-only turnover/age gate, unchanged spread accounting, 10-bps costs, comparison with Notebook 47, factors, exclusions, and plots. | Executed; turnover falls 72.3% and break-even rises to 8.15 bps/side, but net Sharpe is −0.331 and all promotion gates fail |
| `49_lseg_gemma_story_revision_sign_flip.ipynb` | Strict first-to-current Gemma zero-crossing subset on Reuters story families, with an input-first gate, exact Notebook 47 reproduction, same-day conditional random-name control, 10-bps costs, paired/BH inference, factors, halves, exclusions, and plots. | Executed; gross/net Sharpe 0.256/−1.646, 1.34-bps break-even, random-name p=0.4227, and every promotion gate fails |
| `50_lseg_gemma_hybrid_inverse_volatility.ipynb` | Return-blind inverse-volatility allocation of Notebook 36's unchanged Gemma-timed, FinBERT-ranked hybrid, using strictly lagged opens, the existing portfolio projector, the same cap/horizon/cost, paired/BH inference, factors, halves, exclusions, and plots. | Executed; risk and turnover fall, gross/net Sharpe remains 3.691/2.207, but paired improvement, cash, factor-alpha, and promotion gates fail |
| `51_lseg_gemma_hybrid_implementation_capacity.ipynb` | Frozen implementation-capacity replay of Notebook 36's exact targets with drift-aware per-symbol orders, lagged 20-session ADV/volatility, fixed 10-bps cost, square-root impact scenarios, a predeclared AUM grid, 5%-ADV feasibility gate, bottleneck attribution, and plots. | Executed; primary Y=1 remains positive through $10m but fails by $25m, giving an optimistic $10m grid bound; no alpha promotion |
| `frozen_specs/lseg_gemma_hybrid_capacity_v1.json` | Pre-outcome contract for Notebook 51's liquidity estimator, impact coefficients, AUM grid, participation ceiling, identity checks, and interpretation boundary. | Frozen at `71ac538`; complete |
| `52_lseg_gemma_hybrid_liquidity_aware_sizing.ipynb` | Input-first test of the analytical square-root-impact-minimising within-leg allocation, preserving Gemma dates/counts, FinBERT names/signs, leg budgets, cap, neutrality, and horizon. | Executed input-only NO-GO; proxy −6.08% but target turnover +0.54%, so modified returns were not constructed |
| `frozen_specs/lseg_gemma_hybrid_liquidity_sizing_v1.json` | Pre-outcome contract for Notebook 52's convex objective, capped water-filling solution, input gate, conditional AUM comparisons, and non-promoting stop rule. | Frozen at `e2112ed`; stopped at input gate |
| `frozen_specs/lseg_gemma_finbert_hybrid_prospective_v1.json` | Source-controlled prospective contract for the unchanged Notebook 36 equal-weight hybrid, with an explicitly non-promoting Notebook 50 risk diagnostic, exact scorer/privacy/timing/portfolio contracts, minimum sample, inference, and stop gates. | Frozen at `44adccd`; first accumulation batch complete but below the 120-session minimum |
| `frozen_specs/lseg_gemma_finbert_hybrid_prospective_acquisition_batch_20260806.json` | Pre-retrieval contract for the first immutable 33-company headline-only prospective batch, including source separation, closed dates, file/config hashes, population-first stop, and explicit paid-scoring authorisation boundary. | Frozen and hash-corrected before completion; collection complete |
| `frozen_specs/lseg_gemma_finbert_hybrid_prospective_batch_registry_v1.json` | Append-only ordered registry of immutable prospective acquisition specs and hashes; prior entries cannot be replaced when a new batch is added. | Active; one completed batch registered, scoring and return access remain false |
| `frozen_specs/lseg_gemma_finbert_hybrid_backward_v1.json` | Pre-retrieval contract for an earlier non-overlapping same-33 LSEG replication of the unchanged hybrid, including a Gemma endpoint-drift gate, return-blind population gate, one-shot gross/net replay, and timing/name-selection mechanism controls. | Frozen before retrieval; headline collection active, paid scoring and return access remain false |
| `frozen_specs/lseg_gemma_finbert_hybrid_backward_operational_amendment_20260806.json` | Post-start record of the researcher's authorization to use the full 10,000-request daily LSEG allowance through an identity-neutral CLI budget override. | Active; first day reserves 33 requests already used by the entitlement check and schedules a 467-request top-up after the frozen 9,500-request run |
| `frozen_specs/lseg_gemma_finbert_hybrid_backward_collection_order_amendment_20260806.json` | Post-day-one operational amendment switching future retrieval from company-major to date-major traversal while preserving every existing checkpoint and the final frozen population. | Active from the next quota reset; transition recorded at 1,469/21,912 windows and 4,841 pages, with scoring and returns still prohibited before completion |
| `53_lseg_prospective_population_gate.ipynb` | Licence-safe integrity audit and pre-score population gate for the first post-cutoff LSEG batch; scans only timestamps, IDs, and matched symbols, then stops before scorers or returns if the calendar upper bound is below 120 sessions. | Executed; 162,123 source story/headline rows, 162,049 strictly post-cutoff, 33/33 symbols, 27/120 possible XNYS sessions, earliest possible 120th session 2026-12-16; no performance inference |
| `54_lseg_prospective_batch_ledger.ipynb` | Cumulative licence-safe audit of every registered prospective batch, normalized-headline population, source hashes, planning spend, and the calendar-first stop. It cannot launch scorers or load prices/returns. | Executed; 130,692 eligible unique normalized headlines, 19.35% fewer requests than eligible source rows, about $4.95 at the prior realised Gemma average, and the same 27/120 stop |
| `55_lseg_midcap_finbert_baseline_reconciliation.ipynb` | Aggregate-only integrity, period-result, cash-interval, and stock-contribution audit of the frozen recent-news mid-cap FinBERT event rule; reads no corpus, scores, prices, daily P&L, positions, or text and constructs no new return. | Executed; legacy local gate passes, evaluation net return/Sharpe +4.00%/0.593 and break-even 16.52 bps/side, but the cash interval crosses zero, maximum name weight is 50%, and final promotion/deployment gates fail |
| `56_local_lseg_finbert_strategy_lineage.ipynb` | Aggregate-only inventory of every frozen local LSEG/FinBERT strategy run, exact metric-payload duplicate detection, declared-family reconciliation, and coverage/stability floor. | Executed; 10 run directories collapse to 5 exact metric payloads and 4 families, only `midcap_event_v1` clears the floor, N55 already captures it, and no distinct adequately covered positive base is omitted or promoted |
| `57_aggregate_candidate_base_frontier.ipynb` | Aggregate-only candidate-role, gate, fixed-cost, and Y=1 impact-capacity audit across already-frozen research and strategy artifacts; it constructs no signal or return and cannot reselect the prospective contract. | Executed; Notebook 36 remains the sole prospective primary with gross/net Sharpe 3.700/2.271 and 25.24-bps break-even, but its factor interval crosses zero, impact Sharpe falls to 0.413 at $10m and −0.632 at $25m, and no validation or deployment gate is promoted |
| `58_lseg_gemma_hybrid_capacity_uncertainty.ipynb` | Pre-outcome-frozen uncertainty audit of Notebook 51's unchanged primary Y=1 impact curve, using the full AUM grid, identical 9,999 five-session circular-block resamples, both-half stability, and the existing 5%-ADV ceiling. | Executed; no AUM passes. At $1m the mean-net interval is [−1.29,+11.64] bps/session, positive-terminal probability is 93.48%, and second-half Sharpe is 0.016; the $10m point bound is not uncertainty-qualified and no strategy or prospective contract changes |
| `frozen_specs/lseg_gemma_hybrid_capacity_uncertainty_v1.json` | Pre-bootstrap contract for Notebook 58's unchanged candidate, primary impact model, full Notebook 51 AUM grid, block bootstrap, uncertainty/stability/participation gate, and non-promoting claim boundary. | Frozen at `5dfee26`; complete |
| `59_lseg_gemma_hybrid_session_decomposition.ipynb` | Exact start-open-normalized decomposition of Notebook 36's unchanged gross path into entry-open-to-close and close-to-next-open components, with a frozen two-test block-bootstrap family and descriptive long/short attribution. | Executed; 57.38% of gross contribution is intraday and only intraday passes BH; 97.43% comes from the long leg, whose gross Sharpe is 4.344 versus 0.163 short; no exit rule or prospective contract changes |
| `frozen_specs/lseg_gemma_hybrid_session_decomposition_v1.json` | Pre-outcome contract for Notebook 59's exact additive identity, two component estimands, block-bootstrap family, and non-promoting accounting boundary. | Frozen at `fb72f2e`; complete |
| `60_lseg_gemma_hybrid_long_leg_exposure_control.ipynb` | Post-result gross mechanism audit of Notebook 36's unchanged selected long weights against same-date equal-weight market and sector-exposure-matched controls, with a frozen two-comparison BH family. | Executed; selected long beats market by +8.36 bps/session [+3.41,+13.78], q=0.0038, and same-sector equal weight by +3.66 [+0.95,+6.62], q=0.0121; both differences stay positive in both halves, but no control strategy or alpha promotion is constructed |
| `frozen_specs/lseg_gemma_hybrid_long_leg_exposure_control_v1.json` | Pre-outcome contract for Notebook 60's matched gross exposures, two paired comparisons, multiplicity family, and opened-window claim boundary. | Frozen at `daa8ebe`; complete |
| `61_lseg_gemma_hybrid_long_leg_sector_component.ipynb` | Exact intraday/overnight decomposition of Notebook 60's selected-long minus same-sector gross difference, with a frozen two-component BH family and unchanged weights/horizon. | Executed; intraday +1.45 bps/session [−0.32,+3.40] and overnight +2.21 [−0.006,+4.85] both have BH q=0.1264; overnight reverses in the second half, neither gate passes, and no exit or component strategy is promoted |
| `frozen_specs/lseg_gemma_hybrid_long_leg_sector_component_v1.json` | Pre-outcome contract for Notebook 61's exact additive identity, two component estimands, multiplicity family, and permanent no-exit/no-promotion boundary. | Frozen at `80f8912`; complete |
| `62_lseg_cash_flow_distance_reliability_gate.ipynb` | Aggregate-output reliability/readiness notebook for the frozen local human audit. It verifies immutable worksheet identity, reports coding progress, and evaluates weighted kappa, its 9,999-event-bootstrap interval, exact/adjacent agreement, and the full confusion matrix only after both sheets are complete. | Executed; waiting for human labels at A 0/200 and B 0/60; no reliability metric or return opened |
| `63_lseg_gemma_hybrid_long_leg_characteristic_attribution.ipynb` | Frozen post-result attribution of Notebook 60's unchanged within-sector selected-long difference to lagged 1/5-session returns, 20-session volatility, and log ADV. Coefficients are fitted on nonselected names and refitted inside each block-bootstrap replicate. | Executed; characteristics explain −0.073 bps/session, leaving +3.735 [+0.867,+6.746], p=0.0086, positive in both halves; useful retrospective mechanism evidence, not a residual strategy or alpha validation |
| `lib/characteristic_attribution.py` | Tested strictly lagged characteristic construction, exact sector-demeaned path decomposition, and coefficient-refitting circular-block bootstrap for a fixed selected path. | Active; no signal or weights constructed |
| `frozen_specs/lseg_gemma_hybrid_long_leg_characteristic_attribution_v1.json` | Pre-outcome contract for Notebook 63's four-control joint attribution, single primary residual estimand, inference, and permanent no-strategy/no-promotion boundary. | Frozen at `9102d3a`; complete |
| `64_lseg_gemma_hybrid_long_leg_residual_portfolios.ipynb` | User-authorized iterative retrospective translation of Notebook 60's unchanged selected-long minus market/sector gross paths into exact dollar-neutral portfolios, with drift-aware turnover, final liquidation, fixed costs, a four-test cash/incumbent family, a two-test factor family, halves, and plots. | Executed; market residual net Sharpe 2.332 and drawdown −1.85% versus incumbent 2.271/−2.90%, but cash and factor q-values are about 0.097 and mean return does not improve; sector residual Sharpe 0.818 with negative second-half mean; both classes `nonviable` |
| `lib/residual_portfolios.py` | Tested exact construction of market- and same-sector-equal-weight residual targets from a fixed long-selection path, including by-name netting and exposure identities. | Active; used by Notebook 64 |
| `frozen_specs/lseg_gemma_hybrid_long_leg_residual_portfolios_v1.json` | Pre-outcome contract for Notebook 64's two residual portfolios, accounting, four-test primary family, two-test factor family, classifications, and retrospective claim boundary. | Frozen at `7d2653f`; complete |
| `65_lseg_gemma_hybrid_specificity_filter.ipynb` | User-authorized iterative retrospective test of one deterministic explicit-target, single-company, non-technical Reuters-metadata filter on Gemma's timing/count leg, while keeping Notebook 36's unfiltered FinBERT ranks, cap, horizon, ledger, costs, inference, and prospective contract fixed. | Executed; defined Gemma firm-opens rise 3,841→4,826 but active sessions fall 70→22; gross/net Sharpe 1.508/0.881, +3.09% net return, cash q=0.4449, incumbent-improvement q=0.1754, sector-factor p=0.5723; `nonviable`, no further filter variants |
| `lib/hybrid_specificity.py` | Tested deterministic metadata filtering, missing-topic audit, and Gemma-count/unchanged-FinBERT-rank capped-target construction. | Active; used by Notebook 65 |
| `frozen_specs/lseg_gemma_hybrid_specificity_filter_v1.json` | Preserved first contract; execution stopped before return construction when filtered FinBERT ranks were missing for one firm-open. | Input-infeasible; no portfolio outcome opened |
| `frozen_specs/lseg_gemma_hybrid_specificity_filter_v2.json` | Pre-outcome replacement contract applying the one filter only to Gemma timing/counts and keeping the incumbent unfiltered FinBERT ranks exactly. | Frozen at `995d949`; complete |
| `66_lseg_gemma_finbert_backward_replication.ipynb` | Return-blind acquisition and methodology gate for the earlier same-33 LSEG arm, with complete-date/frontier coverage, per-date request density, storage/capacity projections, immutable-source checks, the frozen endpoint-drift design, signal-population stop, and predeclared one-shot gross/net and mechanism methodology. | Executed after the date-major amendment; 1,469/21,912 company-day windows and 4,841 pages, zero complete 33-company dates, frontier 2024-01-01 at 3/33 companies, an explicitly company-biased seven-full-quota-day planning estimate, and 8.7 GiB projected disk headroom; drift scoring remains unauthorized and no scores, prices, or returns were loaded |
| `lib/backward_validation.py` | Tested fail-closed collection audit, per-date checkpoint/frontier ledger, request and storage projections, deterministic census-plus-control Gemma drift sampling, and endpoint/continuous drift metrics. It imports no price or portfolio loader. | Active; used by Notebook 66 while collection accumulates |
| `lib/prospective.py` | Tested fail-closed multi-batch registry auditor: verifies parent/acquisition/config/source hashes, contiguous intervals, the frozen 33, headline-only retrieval, timestamps, cross-batch deduplication, and aggregate-only outputs. | Active; synthetic contiguous/dedup and gap-rejection tests pass |
| `lib/lseg_story_families.py` | Hash-only reconstruction of terminal-suffix LSEG story families, earliest releases, and first-to-current revision transitions with source-manifest verification; licensed headline text and raw story IDs are never returned. | Active; used by Notebooks 43 and 47 |
| `lib/sparse_spread.py` | Tested exact-extrema equal-leg targets, symmetric capacity capping, within-sector demeaning, equal-weight sector projection, fail-closed exposure audits, and compact conversion of the validated drift-aware ledger. | Active |
| `lib/capacity.py` | Tested point-in-time lagged ADV/volatility construction, convex capped impact-minimising leg allocation, and drift-aware square-root impact replay, including final liquidation, per-order participation, and capacity summaries. | Active; zero-impact identity reproduces the core ledger and N52's water-filling helper is unit tested |
| `lib/risk_overlay.py` | Leakage-safe monthly momentum, negative-risk flags, aggregate pressure state, fixed exposure rules, dense adjusted-open returns, explicit cost accounting, and paired block-bootstrap helpers for Notebooks 19–40. | Active |
| `lib/volatility_target.py` | Causal HAR variance features, development-only and expanding walk-forward fitting/calibration, unlevered volatility targeting, lagged price-trend construction, exposure composition, costs, and paired circular-block inference for Notebooks 21–25, 27, and 46. | Active |
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
| `outputs/46_lseg_external_time_har_base_transfer/` | Aggregate external-time forecast, exposure, strategy, matched-control, cost, timing, Gemma breadth, figure, source-hash, and manifest evidence; no headline text. | Ignored/local |
| `outputs/47_lseg_gemma_story_revision_delta/` | Aggregate revision-transition coverage, matched-construction paths, costs, inference, halves, factor/exclusion diagnostics, figures, source hashes, and manifest; no headline text or raw story IDs. | Ignored/local |
| `outputs/48_lseg_gemma_persistent_revision_state/` | Aggregate persistent-state age/turnover audit, paths, costs, inference, halves, factor/exclusion diagnostics, figures, source hashes, and manifest; no headline text. | Ignored/local |
| `outputs/49_lseg_gemma_story_revision_sign_flip/` | Aggregate strict-flip coverage, broad-revision reproduction, paths, costs, paired and random-name inference, factor/exclusion diagnostics, figures, source hashes, and manifest; no headline text or raw story IDs. | Ignored/local |
| `outputs/50_lseg_gemma_hybrid_inverse_volatility/` | Aggregate input gate, equal/inverse-volatility paths, costs, paired inference, factors, halves, exclusions, figures, and manifest; no headline text. | Ignored/local |
| `outputs/51_lseg_gemma_hybrid_implementation_capacity/` | Aggregate capacity curve, identity/liquidity audit, reference-AUM paths and order diagnostics, bottleneck attribution, figures, and manifest; no headline text. | Ignored/local |
| `outputs/52_lseg_gemma_hybrid_liquidity_aware_sizing/` | Aggregate input gate, target audit, input-only figure, empty conditional outcome tables, and manifest; no headline text or modified returns. | Ignored/local |
| `outputs/53_lseg_prospective_population_gate/` | Aggregate collection coverage, source hashes, population gate, figure, and stop manifest; no headline text, scorer output, prices, or returns. | Ignored/local |
| `outputs/54_lseg_prospective_batch_ledger/` | Aggregate batch/cumulative audits, normalized-headline population, source hashes, planning-cost summary, calendar gate, figure, and stop manifest; no headline text, scorer output, prices, or returns. | Ignored/local |
| `outputs/55_lseg_midcap_finbert_baseline_reconciliation/` | Aggregate period metrics, cash inference, stock-contribution concentration, figures, one N44 registry row, source hashes, and manifest; no corpus, score, price, daily-P&L, position, or headline row. | Ignored/local |
| `outputs/56_local_lseg_finbert_strategy_lineage/` | Aggregate run/arm/family inventories, exact-payload groups, coverage gates, lineage figure, source hashes, and manifest; no corpus, score, price, return, position, or headline row. | Ignored/local |
| `outputs/57_aggregate_candidate_base_frontier/` | Aggregate candidate scorecard, role-specific gate matrix, fixed-cost and Y=1 capacity frontiers, figures, source hashes, decision row, and manifest; no price, return-path, position, score, or headline row. | Ignored/local |
| `outputs/58_lseg_gemma_hybrid_capacity_uncertainty/` | Aggregate uncertainty frontier, Notebook 51/bootstrap identity audits, figure, and manifest; no row-level return, position, score, headline, or licensed-text output. | Ignored/local |
| `outputs/64_lseg_gemma_hybrid_long_leg_residual_portfolios/` | Aggregate target identities, costed paths, cost curve, cash/incumbent inference, factor controls, halves, classifications, figure, and manifest; no headline text. | Ignored/local |
| `outputs/65_lseg_gemma_hybrid_specificity_filter/` | Aggregate filter audits, topic-metadata audit, coverage, target identities, costed paths, inference, factor control, halves, gates, figure, and manifest; no headline text. | Ignored/local |
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
`65_lseg_gemma_hybrid_specificity_filter.ipynb`. Run from the repository root so relative paths
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
