# Final Experiments

Notebook-first closing programme for the dissertation. This directory is intentionally lighter than `src/sentiment_benchmark`: rapid analysis, plots, and interpretable tables take priority over new CLI commands or production-style abstractions.

- Live decisions and checklist: [`plan.html`](plan.html)
- Scientific guardrails: [`../docs/research_protocol.md`](../docs/research_protocol.md)
- Stage plan: [`../docs/dissertation_execution_plan.md`](../docs/dissertation_execution_plan.md)

## Current Decisions

| Decision | Frozen/current value |
| --- | --- |
| Primary spine | FNSPID 2011–2023 |
| Robustness spine | LSEG sector-33 and midcap-22; never pooled with FNSPID |
| Firm-day panel | 715,546 news-bearing rows, 570 priced symbols, 3,262 sessions |
| Development block | `session_date <= 2019-12-31` |
| Evaluation block | `session_date >= 2020-01-01` |
| Primary scorer | Pinned ProsusAI/FinBERT checkpoint already used in the Moment-2 study |
| Final RQ | Not chosen; Gate F1 follows filtering/distribution EDA |
| Closed scope | New VaR/ES work, scorer bake-offs, and multi-scorer ensembles |

The split is a **chronological evaluation block**, not a pristine holdout. The samples have influenced earlier design work.

## Workstream Order

1. **Inventory and panel**: completed. Data paths, primary spine, earnings calendar, panel, attrition, and coverage plots exist.
2. **Filtering and distribution EDA**: next. Publisher/source structure, novelty/repetition/routine classification, audit labels, and within-firm-day score moments.
3. **Gate F1**: choose one primary RQ and at most one secondary after reading the EDA.
4. **Core experiment**: run the workstream serving the chosen RQ through one common evaluation harness.
5. **Secondary arms**: surprise, story type, earnings, and learned thresholds only where the primary design supports them.
6. **Promotion**: register the accepted run and export aggregate, licence-safe figures/tables to the dissertation.

## Files

| Path | Role | Status |
| --- | --- | --- |
| `00_data_inventory.py` / `.ipynb` | Inventories FNSPID/LSEG assets, profiles the earnings gather, and records the spine decision evidence. | Completed |
| `01_panel.py` / `.ipynb` | Builds the FNSPID firm-day panel and writes attrition/coverage evidence. | Completed |
| `lib/panel.py` | Thin reusable panel builder and frozen split constants. | Active |
| `data/earnings/README.md` | Earnings-calendar schema, session mapping, and quality caveats. | Tracked |
| `data/earnings/license_record.md` | LSEG source, access date, requested fields, redistribution boundary, and safe aggregate counts. | Tracked |
| `data/earnings/*.csv`, `*.json`, `checkpoints/` | Licensed LSEG results-calendar payloads. | Ignored/local |
| `outputs/00_data_inventory/` | Generated inventory tables and figures. | Ignored/local |
| `outputs/01_panel/` | Generated panel, manifest, attrition, schema, and coverage figures. | Ignored/local |
| `plan.html` | Live phase plan and checklist. | Tracked |

Planned notebook/helper names remain listed in `plan.html`; create them only when their workstream starts.

## Running The Existing Notebooks

From the repository root, using Python 3.12 with the `tailrisk`, `finbert`, and figure dependencies installed:

```bash
uv run python final_experiments/00_data_inventory.py
uv run python final_experiments/01_panel.py
```

The `.py` files are jupytext mirrors of the `.ipynb` notebooks. Run from the repository root so relative paths resolve consistently.

These are not clone-only examples. They require local artifacts that are intentionally absent from Git:

- FNSPID archives under the external data root recorded in `00_data_inventory`;
- the Moment-2 FinBERT checkpoint and Gate-1 audit outputs under `reports/`;
- the local LSEG earnings-calendar files under `data/earnings/`;
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

The panel uses split-adjusted opens and SPY-adjusted one-session returns. Dividends are not back-adjusted. Rows are selected on both news and price availability, which creates a collider/selection limitation for inference.

## Output And Licence Boundary

`outputs/` and the earnings payloads are ignored because they may contain licensed or large data. Do not force-add them wholesale.

A result may be promoted only when:

1. the estimand, event grain, split, inference unit, seed, and multiplicity family are recorded;
2. the evaluation block was not used for selection;
3. attrition and known timing/data gaps are carried into the report;
4. the promoted artifact contains aggregate evidence only;
5. the run is registered in `experiments/manifest.toml` with code/data identities.

## Working Style

- Prefer a clear notebook narrative and a decisive plot over another framework.
- Put reusable joins or metrics in `lib/`; keep exploratory glue in the notebook.
- Add tests only where a silent reusable-helper error would corrupt downstream results.
- Keep nulls, failed arms, and invalidated runs visible.
- Do not extend `src/sentiment_benchmark` unless a final accepted result genuinely needs stable library support.
