# Evidence map

The numbered notebooks are intentionally non-contiguous: their names retain the identifiers used by frozen specifications and result manifests in the source archive.

| Claim or decision | Notebook(s) | Committed evidence | Status |
|---|---|---|---|
| Compare nine ways of aggregating same-day stories | `03_aggregation` | `results/03_aggregation/` and `fig_aggregation_ic.png` | Negative share is the only BH survivor |
| Test whether learned no-trade thresholds add value | `10_thresholds` | `results/10_thresholds/` | Null; cash beats learned policies; the label-shuffled negative control is explicitly insolvent under drift-aware accounting |
| Translate negative pressure into an aggregate risk state | `20`, `21`, `24`, `25` | respective result folders and `fig_har_sentiment_equity.png` | Bounded mapped-session FNSPID downside result; not an ex-ante timing test |
| Test mapped-session downside alignment against shifted schedules | `45_sentiment_downside_timing_placebo` | `timing_results.csv`, `fig_downside_placebo.png` | Unusual only in 2020-2023 after correction; conditional on the session mapping |
| Estimate negative share beyond the mean and transfer to LSEG | `71_conditional_negative_share_transfer` | `conditional_coefficients.csv` | FNSPID pass; LSEG FinBERT blocks imprecise |
| Apply the frozen FNSPID risk state to recent and backward LSEG | `72`, `73` | strategy, inference, timing and gate tables | Recent rule never activates; backward evidence is insufficient for a portability claim |
| Stress-test HAC, bootstrap, episode and story-count sensitivity | `74_conditional_and_pressure_robustness_diagnostics` | six diagnostic tables | Robustness/fragility boundary |
| Replicate the selected FNSPID estimands in 2020--2023 | `75_fnspid_evaluation_beyond_mean_replication` | exact source execution, frozen amendment, coefficient families, daily series, power context and baseline stability contrast | Completed once; both primary families fail and the baseline contrast is consistent with pipeline-level instability, not its cause |
| Show the headline effect without a regression | `76_fnspid_development_negative_share_double_sort` | spread summaries and `fig_conditional_double_sort.png` | -0.618 bps; interval crosses zero |
| Quantify power for important nulls | `77_null_minimum_detectable_effects` | MDE tables | LSEG is underpowered for an FNSPID-sized coefficient |
| Consolidate nulls, multiplicity and provenance | `78_null_results_and_provenance_tables` | result ledger and provenance table | 56 families, 264 decisions/tests; the family-level tables are kept here rather than in the manuscript |
| Rule out recent-return reversal and volatility as the explanation | `79_fnspid_development_reversal_confound` | coefficient tables and `fig_reversal_confound.png` | Primary price-control test passes |
| Search a bounded aggregate LSEG risk-rule family | `80_lseg33_economic_value_rule_search` | testing-period, inference and gate tables | No economic-value gate pass |
| Run the same economic search on FNSPID | `81_fnspid_economic_value_rule_search` | testing-period, inference and gate tables | No stable incremental economic gain |
| Search firm-level LSEG loss controls | `82_lseg33_firm_level_loss_control` | matched controls, halves and leave-one-out tables | Raw loss reduction, no incremental economic value |
| Test prompt-engineered LSEG scores | aggregate result snapshot from source Notebook 83 | selection and support tables, prompt figures | All prompt variants ineligible |
| Test a structured anticipated-price-reaction score | aggregate result snapshot from source Notebook 84 | selection and support tables, reaction figure | Selection gate fails |
| Test close-to-close, FF3 residual and soft negative mass | `85_fnspid_development_outcome_and_label_sensitivities` | `sensitivity_coefficients.csv` | 0/3 BH survivors; the family does not establish assigned-window specificity |
| Decompose the assigned FNSPID return window and correct reported scale/power/drawdown | `86_fnspid_outcome_leg_decomposition` | ten aggregate files under `results/86_fnspid_outcome_leg_decomposition/` | Training-period association is intraday, not post-close; prior-close evidence keeps assignment timing unresolved; observed-rank translation is -0.512 points; realised-precision power is 26.5%; mapped-session modifier drawdown is adverse |
| Complete the external-review robustness pack | `87_fnspid_external_review_robustness_pack` | fifteen aggregate files under `results/87_fnspid_external_review_robustness_pack/` | Previous-open probe is strongly negative; trailing-beta and HAC checks preserve the era contrast; the FNSPID downside result survives episode and crash exclusions; class and tie base rates are reported |

Historical Notebook 76--78 manifests retain the execution state recorded when
those analyses were frozen. Notebook 75 was subsequently authorised, executed
once and promoted without re-execution; `experiments/results/README.md` records
the current state and the preserved authority chain.

Return-convention metadata correction: the FNSPID adjusted-open proxy applies
the archived Yahoo adjusted-close ratio and therefore carries the split and
dividend adjustments encoded there; corporate actions are not independently
reconstructed. Historical risk manifests with the older
"not dividend-adjusted" wording remain hash-preserved; see
`experiments/results/RETURN_CONVENTION_CORRECTION_20260815.md`.

Portfolio-accounting correction: Notebooks 03, 10, 71 and the downstream
Notebook 77 precision audit now rebalance from return-drifted weights and charge
final liquidation. The frozen amendment and the observed negative-control
insolvency rule are in `experiments/specs/portfolio_turnover_*_20260815.json`;
the repaired aggregate snapshots preserve the statistical nulls and make the
economic results slightly more negative.

## Dissertation figures

Ten manuscript figures are regenerated by
`scripts/generate_dissertation_artifacts.py` from committed aggregate CSVs. The
script writes to `experiments/generated/dissertation` by default; `make
artifacts` promotes them into `manuscript/artifacts`. The eleventh analytical
figure, `fig_har_sentiment_equity.png`, is regenerated separately from Notebook
24's private archived daily paths. Those row-level paths cannot enter this clean
repository, so the committed PNG is auditable against the aggregate endpoint
and exposure summaries but is not covered by the aggregate-only byte-replay gate.

| Figure | Built from | Shows |
|---|---|---|
| `fig_aggregation_family` | `03_aggregation` | Nine-rule ICs with the single BH survivor emphasised |
| `fig_estimate_stability` | `74`, `79` | Price-path controls beside story-count strata on one shared scale |
| `fig_double_sort` | `76` | High-minus-low spreads by mean-sentiment quintile and pooled |
| `fig_break_even_cost_gap` | `03_aggregation` | Per-side break-even costs against the 10 bps charge |
| `fig_coefficient_drift` | `75` | Descriptive calendar-year baseline coefficients and period means |
| `fig_har_sentiment_equity` | `24` private daily paths | Mapped-session modifier and HAR exposure/equity paths |
| `fig_timing_placebo` | `45` | Mapped-session risk-off alignment against its circular-shift null |
| `fig_risk_regime_stability` | `24`, `25` | Mapped-session modifier-minus-base return by period |
| `fig_cross_source_coefficient_power` | `71`, `75`, `77`, `79` | Training-period, temporal-replication and cross-source estimates with power context |
| `fig_lseg_matched_control_value` | `82` | Risk metrics and the ending-wealth decomposition |
| `fig_prompt_gate` | `83`, `84` | Gross-to-net returns for all six prompt variants |

The generator also writes seven manuscript tables. Two are specific to the
external-review remediation: `tab_outcome_leg_decomposition.tex` reports the
Notebook 86/87 timing evidence and `tab_hypothesis_verdicts.tex` maps each final
claim to its bounded verdict.

Figures carry no in-image title or source line: the LaTeX caption names what the figure
shows and which notebooks it draws on, so the information is not duplicated.

`manuscript/figures/` retains the original notebook-rendered PNGs as an archive.
Three of them (`fig_aggregation_ic`, `fig_conditional_double_sort`,
`fig_downside_placebo`) were superseded by reproducible equivalents and are no
longer included in the manuscript. `fig_har_sentiment_equity` remains included
because its daily paths are private; it is regenerated from those paths when
they are available.

## Reporting units

Notebook 80 and 82 economic decompositions are stored in a column named
`gbp_per_million`, following the frozen specification's sterling notional. No FX
conversion exists anywhere in the pipeline; the value is a return difference times a
notional of one million. Because the underlying cash flows are USD-denominated US
equity returns, the manuscript reports these amounts in USD. The stored column name is
left untouched so the aggregate snapshot still matches the frozen spec.

## Why other experiments are absent

The source repository accumulated more than eighty notebooks. Excluded items fall into at least one of these categories:

- superseded data preparation or model bake-offs;
- exploratory LSEG strategies that do not bear directly on the final question;
- duplicated dashboards and supervisor-note renderings;
- large or licensed intermediate data;
- operational OpenRouter scoring code, raw prompts, attempts and responses;
- tests for infrastructure not required by the selected analyses.

Their absence is a scope decision, not deletion of the historical record. The original private repository remains the audit archive.

## Result directories

`experiments/results/` is a curated snapshot used for writing. `experiments/generated/` is the ignored target for reruns. This prevents an exploratory execution from silently changing a number already cited in the manuscript.
