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
   Notebook 13 preserves the resulting null return experiment. Human validation
   remains outstanding, and no expanded-corpus result is promoted.
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
| `lib/openrouter_validation.py` | Frozen, resumable public-benchmark gate for OpenRouter Gemma 4 26B pinned to DeepInfra with ZDR, no fallback, strict probability JSON, and cost/quality metrics. It refuses to use LSEG inputs by construction. | Gate passed: 1,000/1,000 coverage; accuracy 0.808; macro-F1 0.813 |
| `../scripts/run_openrouter_gemma4_validation.py` | Spend-gated command for preparing, executing, or explicitly retrying the 1,000-row OpenRouter validation. See [`../docs/openrouter_gemma4_validation.md`](../docs/openrouter_gemma4_validation.md). | Executed 2026-08-04; evidence ignored/local |
| `lib/openrouter_lseg.py` | Bounded, append-only and resumable licensed-corpus scorer. Freezes the 888,155-headline population, exact input hashes, provider/privacy/FP8 contract, prompt hash, and researcher permission; failed attempts remain auditable and are retried without duplicating successes. | Completed 2026-08-05: 888,155/888,155 unique successes; $33.6153 |
| `../scripts/run_openrouter_lseg_scoring.py` | Explicit-authorisation entrypoint for the full LSEG scorer. The private output and log paths are recorded in [`../docs/lseg_external_processing_authorisation.md`](../docs/lseg_external_processing_authorisation.md). | Executed; private append-only output remains off Git |
| `outputs/12_lseg_44_labelling/` | Licence-safe aggregate coverage, label-distribution, and LLM-design tables/figures. | Ignored/local |
| `outputs/13_lseg_44_gemma_robustness/` | Aggregate scorer/return tables, firm-open panels, daily portfolios, figures, and manifest; no headline text. | Ignored/local |
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
`13_lseg_44_gemma_robustness.ipynb`. Run from the repository root so relative paths
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
- the local 33-company and added-11 LSEG price exports used by Notebook 13;
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
