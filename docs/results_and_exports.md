# Results And Exports

This reference explains where benchmark evidence is stored, what gets exported, and how to interpret the main metrics.

## Storage Locations

| Path | Created by | Purpose | Git policy |
| --- | --- | --- | --- |
| `results/sentiment_benchmark.sqlite` | `run`, `run-baselines`, `run-self-consistency`, TUI workflows | Local SQLite database for runs, responses, metrics, and self-consistency samples. | Ignored. |
| `results/turso_replica.db` | Native `libsql` backend | Embedded local replica that syncs with Turso when `SENTIMENT_BENCH_DB_BACKEND=libsql`. | Ignored. |
| `results/exports/run_<id>/` | `export` or TUI Export Run | Reproducible run evidence for analysis and dissertation writing. | Ignored. |
| `Data/news/tavily_news_*/` | `fetch-news` or TUI News tab | Unlabeled Tavily article corpora. | Ignored except `.gitkeep`. |
| `experiments/manifest.toml` | Manual curation | Formal dissertation experiment registry. | Source-controlled. |

## Optional Turso/libSQL Cloud Database

By default, the project uses local SQLite. To share one run history across machines without syncing a live SQLite file, configure Turso through the native `libsql` Python package. The project targets Python 3.12 for this backend because native `libsql` wheels may not be available for newer Python versions.

```text
SENTIMENT_BENCH_DB_BACKEND=libsql
TURSO_DATABASE_URL=libsql://your-database-your-org.turso.io
TURSO_AUTH_TOKEN=your-database-token
TURSO_REPLICA_PATH=results/turso_replica.db
```

This creates an embedded local replica at `TURSO_REPLICA_PATH`, syncs from Turso when connections open, and syncs writes back to Turso after commits. Each machine should recreate its own `.venv/`, keep its own `.env`, and use its own local replica path while sharing the same Turso database URL and token.

Each benchmark run stores machine metadata so a shared Turso history can be traced back to the computer that produced it. Set a friendly label per machine in `.env`:

```text
SENTIMENT_BENCH_MACHINE_LABEL=desktop-3070
```

If `SENTIMENT_BENCH_MACHINE_ID` is not set, the project derives a short hashed identifier from a stable OS machine value where available. The raw OS machine value is not stored. New runs also store `environment_json` with package version, git commit/branch/dirty state, Python version, OS/platform, database backend, and local timezone.

Create credentials from Turso:

```bash
turso auth login
turso db create sentiment-dissertation-results
turso db show --url sentiment-dissertation-results
turso db tokens create sentiment-dissertation-results
```

If using the Turso dashboard instead of the CLI, copy the database URL and create a database auth token there. Keep `TURSO_AUTH_TOKEN` out of git.

Python 3.12 setup on a new Windows machine:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

After adding the Turso values to `.env`, initialize or migrate the schema through the normal storage layer:

```powershell
@'
from sentiment_benchmark.storage import BenchmarkStore
store = BenchmarkStore("results/sentiment_benchmark.sqlite")
store.initialize()
'@ | .\.venv\Scripts\python.exe -
```

This is safe to run more than once. It creates missing tables and adds the run metadata columns (`machine_id`, `machine_label`, `environment_json`) to older local or Turso databases.

Existing local SQLite runs are not automatically copied into Turso. Export them with `sentiment-bench export --run-id <id>` if you need the evidence, or migrate them deliberately after deciding which old runs belong in the shared history.

## Database Tables

The database is an execution store, not a source dataset. The same logical tables are used by local SQLite and the native libSQL/Turso backend.

| Table | Meaning |
| --- | --- |
| `dataset_items` | Loaded dataset rows, duplicate flags, conflict flags, and hidden labels. |
| `prompts` | Prompt text keyed by prompt hash. |
| `runs` | Run metadata, model list, selected rows, provider route, request settings, machine id, machine label, and environment snapshot. |
| `run_models` | Per-model run status. |
| `responses` | One model response per run/model/row/prompt hash. |
| `generation_metadata` | Provider cost, provider name, latency, and raw generation metadata when available. |
| `metrics` | Per-model metrics for `primary` and `all` scopes. |
| `sc_runs` | Self-consistency run metadata. |
| `sc_samples` | Repeated self-consistency samples. |
| `sc_results` | Aggregated self-consistency ambiguity metrics. |

## Export Files

Run:

```bash
sentiment-bench export --run-id 1
```

Default destination:

```text
results/exports/run_1/
```

| File | Contents |
| --- | --- |
| `responses.csv` | Spreadsheet-friendly response table with hidden label, normalized label, parse status, latency, tokens, provider cost, and errors. |
| `responses.json` | JSON version of response records. |
| `metrics.json` | Per-model metrics and per-class scores for each scope. |
| `run.json` | Run metadata, prompt metadata, package version, export timestamp, dataset SHA-256, prompt hash, provider, base URL, machine metadata, environment snapshot, and request settings. |
| `statistics.json` | Bootstrap confidence intervals, paired McNemar tests, and agreement statistics where applicable. |
| `summary.md` | Human-readable run summary for notes and dissertation drafting. |
| `figures/` | Optional PNG figures when installed with `python -m pip install -e ".[figures]"`. |

## Optional Figures

With the `figures` extra installed, exports include:

- Model leaderboard.
- Accuracy bootstrap-CI forest plot.
- Confusion-matrix heatmaps.
- Per-class score bars.

In the TUI, the Results tab can export a selected run and open the `figures/` folder.

## Metric Reference

| Metric | Use |
| --- | --- |
| Accuracy | Overall correct fraction. Useful but can be misleading with class imbalance. |
| Macro-F1 | Average F1 across labels with equal class weight. Good headline metric for imbalanced data. |
| Weighted-F1 | F1 weighted by label support. Useful for overall performance with the observed label distribution. |
| Balanced accuracy | Average recall across classes. Helps counter majority-class dominance. |
| Matthews correlation coefficient | Single-score classification quality that is robust to imbalance. |
| Per-class precision | Of predictions for a label, how many were correct. |
| Per-class recall | Of true rows for a label, how many were found. |
| Invalid-output count | Model produced output that could not be parsed into a valid label. |
| API-error count | Provider request failed after retry handling. |
| Bootstrap CI | Resampled uncertainty interval for accuracy or macro-F1. |
| McNemar test | Paired test for whether two models differ on shared rows. |
| Cohen's kappa | Pairwise inter-model agreement beyond chance. |
| Fleiss' kappa | Multi-model agreement beyond chance. |
| Krippendorff's alpha | Nominal agreement statistic tolerant of missing labels. |

## Scoring Scopes

| Scope | Definition | Use |
| --- | --- | --- |
| `primary` | Excludes rows in duplicate sentence groups with conflicting labels. | Preferred headline comparison scope. |
| `all` | Includes every selected row, including conflicting duplicates. | Audit and ambiguity-analysis scope. |

Both scopes are exported because duplicate-label conflict is central to the dissertation design.

## Prompt Sensitivity Outputs

`run-prompt-suite` creates one run per prompt variant. `prompt-sensitivity` then reports:

- Mean, standard deviation, minimum, maximum, and spread.
- Coefficient of variation.
- Variant-level metric table.

Use these outputs to describe prompt fragility rather than only reporting the best prompt.

## Self-Consistency Outputs

Self-consistency runs sample the same model multiple times per row at temperature > 0. Stored and reported values include:

- Row-level label distribution.
- Entropy.
- Majority fraction.
- Consistency rate.
- Majority-vote accuracy.
- Majority-vote balanced accuracy.
- Conflicting versus non-conflicting duplicate entropy comparison.

## Reproducibility Rules

For formal results:

1. Export the run.
2. Add the run or run family to `experiments/manifest.toml`.
3. Record the code commit SHA.
4. Preserve `run.json`, `metrics.json`, `statistics.json`, and `summary.md`.
5. Interpret headline metrics with `primary` scope and report `all` as an audit scope.
