# CLI Reference

The `sentiment-bench` command is installed by the package entry point `sentiment_benchmark.cli:app`.

```bash
sentiment-bench --help
```

On Windows, if PowerShell does not recognize `sentiment-bench`, activate the virtual environment or call the executable directly:

```powershell
.\.venv\Scripts\Activate.ps1
sentiment-bench --help

# Or:
.\.venv\Scripts\sentiment-bench.exe --help
```

## Global Behavior

| Behavior | Details |
| --- | --- |
| Dataset default | `Data/derived/labeled/financial_sentiment_v2.csv` |
| Database default | `results/sentiment_benchmark.sqlite` |
| Prompt config default | `configs/default_prompts.toml` |
| Provider default | `openrouter`, or `SENTIMENT_BENCH_PROVIDER` if set. |
| OpenRouter endpoint default | `https://openrouter.ai/api/v1` |
| Ollama endpoint default | `http://localhost:11434`, or `OLLAMA_HOST` if set. |
| Pilot sample default | 30 rows per class from the primary scope. |
| Seed default | `42` |
| Temperature default | `0.0` for standard benchmark runs. |
| Retries default | `3` |
| Concurrency default | `1` |
| Database backend | Local SQLite unless `SENTIMENT_BENCH_DB_BACKEND=libsql`. |
| Machine label | `SENTIMENT_BENCH_MACHINE_LABEL`, or the local hostname when unset. |

## Environment Variables

| Variable | Meaning |
| --- | --- |
| `SENTIMENT_BENCH_PROVIDER` | Default provider for CLI/TUI: `openrouter`, `cerebras`, or `ollama`. |
| `OPENROUTER_API_KEY` | OpenRouter authentication token. |
| `CEREBRAS_API_KEY` | Cerebras authentication token. |
| `TAVILY_API_KEY` | Tavily search and extraction token. |
| `NEWSAPI_API_KEY` | NewsAPI Everything endpoint token. |
| `OPENROUTER_BASE_URL` | OpenRouter-compatible endpoint. |
| `CEREBRAS_BASE_URL` | Cerebras endpoint, defaulting to `https://api.cerebras.ai/v1`. |
| `CEREBRAS_MAX_RPM` | Optional local request-rate ceiling; live Cerebras API-key headers take precedence. |
| `OLLAMA_HOST` | Ollama endpoint, such as `http://localhost:11434`. |
| `SENTIMENT_BENCH_DB_BACKEND` | `sqlite` by default; set `libsql` for Turso/native libSQL. |
| `TURSO_DATABASE_URL` | Turso/libSQL URL, such as `libsql://...turso.io`. |
| `TURSO_AUTH_TOKEN` | Turso database auth token. Keep out of git. |
| `TURSO_REPLICA_PATH` | Local embedded replica path, defaulting to `results/turso_replica.db`. |
| `SENTIMENT_BENCH_MACHINE_LABEL` | Friendly per-machine label stored with new runs. |
| `SENTIMENT_BENCH_MACHINE_ID` | Optional explicit stable machine id; otherwise a hashed OS machine value is used. |
| `SENTIMENT_BENCH_AUTO_FETCH_MODELS` | TUI local-Ollama auto-fetch toggle, default `1`. |

## Commands

Commands grouped by workflow. Every command is documented in a section below or in the linked guide.

### Benchmarking and results

| Command | Purpose |
| --- | --- |
| `validate-data` | Validate dataset columns, labels, duplicates, and primary scoring scope. |
| `list-models` | List models from OpenRouter, Cerebras, or Ollama. |
| `run` | Run LLM sentiment classification benchmarks. |
| `run-baselines` | Run non-LLM baseline classifiers. |
| `runs` | List stored benchmark runs. |
| `results` | Show stored metrics and optional confusion matrices. |
| `compare` | Compare two models with paired McNemar's test and bootstrap CIs. |
| `export` | Export a stored run to files under `results/exports/`. |
| `agreement` | Compute inter-model agreement for one run. |
| `tui` | Launch the terminal UI. |

### Prompt robustness and self-consistency

| Command | Purpose |
| --- | --- |
| `run-prompt-suite` | Run a model across prompt perturbation variants. |
| `prompt-sensitivity` | Summarize metric variation across prompt variants. |
| `run-self-consistency` | Sample one model repeatedly at temperature > 0. |
| `self-consistency` | Analyze an existing self-consistency run. |
| `self-consistency-list` | List self-consistency runs. |
| `sc-compare-by-conflict` | Test whether conflicting duplicate rows have higher entropy. |

### News sourcing (Tavily and NewsAPI)

| Command | Purpose |
| --- | --- |
| `news-check` | Run a small Tavily search to verify API connectivity. |
| `fetch-news` | Fetch Tavily-sourced articles into a timestamped derived corpus. |
| `newsapi-check` | Run a one-result NewsAPI Everything query. |
| `fetch-newsapi` | Page through NewsAPI titles and descriptions into a timestamped corpus. |

### LSEG Workspace corpus and formal analyses

| Command | Purpose |
| --- | --- |
| `lseg-init-config` | Write a preset collection TOML config. |
| `lseg-news-check` | Test the Workspace session and entitlements without writing data. |
| `fetch-lseg-news` | Atomically checkpoint an immutable, resumable raw collection. |
| `build-lseg-corpus` | Verify raw hashes and deterministically rebuild clean text offline. |
| `build-lseg-analysis-cohort` | Apply the configured relevance rules and write hashed development/holdout/L3 cohorts. |
| `lseg-catalog` | Refresh the metadata-only local corpus catalog. |
| `sample-lseg-validation` | Create seeded human-annotation sheets for relevance and sentiment. |
| `evaluate-lseg-annotations` | Score annotator agreement and adjudications. |
| `score-corpus-matrix` | Run the configured crossed model × prompt × sample scoring matrix, resumably. |
| `analyze-l2` | Consensus/ambiguity event study with clustered inference (H2). |
| `analyze-l3` | Variance components, reliability, and holdout trading rules (H3). |

### Trading pipeline and strategies

See the [trading pipeline guide](trading_pipeline.md) for the end-to-end workflow.

| Command | Purpose |
| --- | --- |
| `run-trading-strategy` | Run or preview a config-driven source-to-sentiment-to-return trading run. |
| `analyze-trading-run` | Robustness tables, plots, effectiveness battery, and a technical report for a completed run. |
| `sweep-trading-strategy` | Tune a strategy's parameters on a completed run, selecting on a training split only. |
| `list-strategies` | List registered trading strategies (ideas) and their sweepable parameters. |
| `analyze-headline-value` | Headline-only coverage, taxonomy, and trading-value screen for an LSEG collection. |

## `validate-data`

```bash
sentiment-bench validate-data
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset-path` | `Data/derived/labeled/financial_sentiment_v2.csv` | CSV containing `Sentence` and `Sentiment`. |

## `list-models`

```bash
sentiment-bench list-models --provider openrouter --limit 50
sentiment-bench list-models --provider cerebras --limit 50
sentiment-bench list-models --provider ollama --ollama-host http://desktop-pc:11434
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--provider` | `openrouter` | `openrouter`, `cerebras`, or `ollama`. |
| `--base-url` | `https://openrouter.ai/api/v1` | OpenRouter-compatible base URL. |
| `--ollama-host` | `http://localhost:11434` | Ollama host URL. |
| `--limit` | `50` | Maximum rows to show. |

## `run`

```bash
sentiment-bench run --models openai/gpt-4o-mini --mode pilot
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--models`, `-m` | required | Model ID. Repeat for multiple models. |
| `--mode` | `pilot` | `pilot` or `full`. |
| `--prompt-id` | `default_label_only` | Prompt ID from `configs/default_prompts.toml`. |
| `--dataset-path` | `Data/derived/labeled/financial_sentiment_v2.csv` | Dataset CSV. |
| `--db-path` | `results/sentiment_benchmark.sqlite` | Local SQLite storage path; when `SENTIMENT_BENCH_DB_BACKEND=libsql`, Turso env vars select the synced replica. |
| `--prompts-path` | `configs/default_prompts.toml` | Prompt config path. |
| `--provider` | `openrouter` | `openrouter`, `cerebras`, or `ollama`. |
| `--base-url` | `https://openrouter.ai/api/v1` | OpenRouter-compatible base URL. |
| `--ollama-host` | `http://localhost:11434` | Ollama host URL. |
| `--sample-per-class` | `30` | Pilot rows per class. |
| `--seed` | `42` | Row sampling and reproducibility seed. |
| `--temperature` | `0.0` | Model sampling temperature. |
| `--max-completion-tokens` | `64` | Default completion-token budget. |
| `--reasoning-max-tokens` | `2048` | Larger token budget for reasoning-model ID markers. |
| `--model-max-tokens` | none | Per-model override as `model_id=N`; repeatable. |
| `--concurrency` | provider default | Concurrent model requests. Defaults to `64` for Cerebras (quota-paced) and `1` otherwise, matching the TUI. |
| `--retries` | `3` | Retries per request. |
| `--few-shot-k` | `0` | Demonstrations per class. |
| `--few-shot-seed` | `--seed` | Demonstration sampling seed. |
| `--resume-run-id` | none | Resume an existing run without duplicating completed responses. |

Useful prompt IDs include:

| Prompt ID | Use |
| --- | --- |
| `default_label_only` | Strict zero-shot label-only benchmark prompt. |
| `finance_calibrated_label_only` | Finance-oriented label-only prompt for market/business/trading language; recommended for Gemma 4 pilots. |
| `default_with_explanation` | Label plus short explanation. |
| `default_soft_label` | Per-class probability JSON; enables Brier/ECE calibration metrics. Runs get a completion-token floor of 128 so the JSON is never truncated. |
| `default_chain_of_thought` | Reasoning-style output ending in `Answer: <label>`. |

## `run-baselines`

```bash
sentiment-bench run-baselines --mode pilot
sentiment-bench run-baselines --match-run-id 5 --baselines majority --baselines tfidf_logreg
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--baselines`, `-b` | `majority`, `tfidf_logreg` | Baseline names. Repeat for multiple. |
| `--mode` | `pilot` | `pilot` or `full`. |
| `--dataset-path` | `Data/derived/labeled/financial_sentiment_v2.csv` | Dataset CSV. |
| `--db-path` | `results/sentiment_benchmark.sqlite` | Local SQLite storage path; when `SENTIMENT_BENCH_DB_BACKEND=libsql`, Turso env vars select the synced replica. |
| `--sample-per-class` | `30` | Pilot rows per class. |
| `--seed` | `42` | Sampling seed. |
| `--folds` | `5` | Stratified CV folds for fitted baselines. |
| `--match-run-id` | none | Evaluate on the exact rows selected by an existing run. |

Supported baselines:

| Baseline | Dependency | Notes |
| --- | --- | --- |
| `majority` | Core install | Predicts most frequent training label. |
| `tfidf_logreg` | Core install | TF-IDF 1-2 grams plus multinomial logistic regression. |
| `vader` | `python -m pip install -e ".[baselines]"` | NLTK VADER lexicon with conventional compound thresholds. |
| `finbert` | `python -m pip install -e ".[finbert]"` | `ProsusAI/finbert`; downloads transformer weights. |

## `news-check`

```bash
sentiment-bench news-check --query "financial markets" --max-results 1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--query` | `financial markets` | Small Tavily query for connectivity check. |
| `--max-results` | `1` | Maximum results for the check. |

## `newsapi-check` and `fetch-newsapi`

```bash
sentiment-bench newsapi-check --query "Apple AAPL stock"
sentiment-bench fetch-newsapi --query "Apple AAPL stock news" \
  --from 2026-06-08T04:00:00Z --to 2026-06-11T04:00:00Z
```

`fetch-newsapi` calls the Everything endpoint with English results, publication-time ordering, `X-Api-Key` authentication, and 100 rows per page. It writes a `newsapi_news_*` corpus containing JSONL, CSV, and a manifest; full article bodies are not represented as available.

| Option | Default | Meaning |
| --- | --- | --- |
| `--query` | required | NewsAPI keyword/Boolean query. |
| `--from` | none | Oldest ISO 8601 publication date or timestamp. |
| `--to` | none | Newest ISO 8601 publication date or timestamp. |
| `--max-pages` | `10` | Maximum 100-result pages. |
| `--domain` | none | Included publisher domain; repeatable. |
| `--exclude-domain` | none | Excluded publisher domain; repeatable. |
| `--output-dir` | `Data/news` | Root for the timestamped corpus. |

## `fetch-news`

```bash
sentiment-bench fetch-news --query "bank earnings sentiment" \
  --max-results 10 --time-range week --output-dir Data/news
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--query` | required | Tavily news search query. |
| `--max-results` | `10` | Valid range: `1` to `20`. |
| `--topic` | `news` | `news`, `finance`, or `general`. |
| `--time-range` | `week` | `day`, `week`, `month`, or `year`. |
| `--search-depth` | `basic` | `basic`, `advanced`, `fast`, or `ultra-fast`. |
| `--start-date` | none | Only return results after `YYYY-MM-DD`. |
| `--end-date` | none | Only return results before `YYYY-MM-DD`. |
| `--include-domain` | none | Domain include filter; repeatable. |
| `--exclude-domain` | none | Domain exclude filter; repeatable. |
| `--extract / --no-extract` | `--extract` | Run Tavily Extract for full article text. |
| `--extract-depth` | `basic` | Tavily extract depth: `basic` or `advanced`. `advanced` returns cleaner article bodies. |
| `--min-text-chars` | `500` | Minimum extracted characters before text counts as usable; `0` disables the quality gate. |
| `--output-dir` | `Data/news` | Root directory for timestamped corpus outputs. |

Extracted text passes a quality gate before a record counts as usable: known error-page signatures (for example Yahoo Finance's "Oops, something went wrong") and bodies shorter than `--min-text-chars` are downgraded to `extraction_status = failed` with `text_quality` set to `error_page` or `too_short`. The raw extractor output is kept in `raw_extract_result` for auditing.

## `run-trading-strategy`

```bash
sentiment-bench run-trading-strategy --dry-run
sentiment-bench run-trading-strategy --config configs/trading_pilot_3co.toml
```

The fixed Week 3 config covers AAPL, AMZN, and TSLA on 8–10 June 2026. It reuses the Tavily ticker package, collects NewsAPI results, performs target-title screening and cross-provider deduplication, scores three OpenRouter models plus VADER, enters at the next observed adjusted open, and exports 1–7-session returns on $10,000 notional. Successful NewsAPI source pointers and LLM responses are resumable. A completed run cannot be overwritten with a changed config.

| Option | Default | Meaning |
| --- | --- | --- |
| `--config` | `configs/trading_pilot_3co.toml` | Fixed TOML run definition. |
| `--dry-run` | off | Print calls, dates, models, horizons, and entry rule without writes or API calls. |

### Optional index fallback for thin-coverage company-days

By default every company-day trades the company's own stock, and a company-day with no accepted
texts produces no signal (it is dropped, never imputed). For companies with sparse daily news you
can opt into the supervisor's "use a US index if you can't get enough texts" rule by adding an
`[index_fallback]` table to the run config:

```toml
[index_fallback]
enabled = true
symbol = "^GSPC"   # any yfinance index ticker, e.g. ^DJI (Dow) or ^IXIC (Nasdaq)
min_texts = 3      # company-days with fewer accepted texts trade the index instead
```

When enabled, a company-day with fewer than `min_texts` accepted texts keeps its (thin) sentiment
signal but executes against `symbol` rather than the individual stock; the index price series is
fetched alongside the companies. Each return row records `traded_symbol` and an `index_fallback`
flag, and the run manifest reports the index settings plus an `index_fallback_events` count, so
fallback trades stay fully auditable. The option is off unless a config sets `enabled = true`, and
the already-completed Week 3 runs do not use it.

## `analyze-trading-run`

```bash
sentiment-bench analyze-trading-run \
  --run-dir results/trading/week3_trading_broad_20260624_reviewed \
  --comparison-run-dir results/trading/week3_trading_pilot_20260624_reviewed \
  --output-dir results/trading/week3_trading_broad_20260624_reviewed_analysis
```

The analysis command refuses to overwrite an existing output directory. It writes seeded percentile-bootstrap intervals,
traded-only hit rates, company and date breakdowns, leave-one-company-out estimates, LLM/VADER agreement diagnostics,
source-yield tables, five static plots, a technical `summary.md`, a source map, and an SHA-256 manifest. Bootstrap
intervals are descriptive because company-day events overlap and are not independent. Install plotting support first with
`uv sync --extra figures` when the optional figure dependencies are not already present.

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-dir` | required | Completed trading run containing `run_manifest.json` and CSV outputs. |
| `--output-dir` | required | New directory for the non-overwriting analysis artifact. |
| `--comparison-run-dir` | none | Optional earlier run for descriptive panel-sensitivity comparison. |
| `--bootstrap-resamples` | `10000` | Number of percentile-bootstrap event resamples. |
| `--seed` | `42` | Random seed used by the bootstrap. |

The analysis includes the effectiveness battery — significance/direction tests with Benjamini–Hochberg correction, a buy-and-hold benchmark comparison, and risk-adjusted economics per scorer-horizon cell — written as `effectiveness_*.csv` and appended to `summary.md`. See [Trading pipeline](trading_pipeline.md#the-effectiveness-battery).

## `analyze-headline-value`

```bash
sentiment-bench analyze-headline-value \
  --collection-root Data/collections/lseg_us_sector_33_6m \
  --prices Data/derived/prices/lseg_us_sector_33.csv
```

Analyzes headline-only coverage, an event-type taxonomy, source mix, and lexicon-scored trading value for an LSEG collection, without requiring story bodies. When the prices CSV is omitted or missing, the trading arm is reported as blocked instead of failing. Outputs (panel, event table, category and source summaries, daily signals, a raw-headline calibration template, trading returns/decisions/summary, `summary.md`, and a manifest) contain licensed headline text and stay local.

| Option | Default | Meaning |
| --- | --- | --- |
| `--collection-root` | `Data/collections/lseg_us_sector_33_6m` | Collection folder or raw LSEG directory containing `headlines.jsonl`. |
| `--prices` | `Data/derived/prices/lseg_us_sector_33.csv` | Optional daily prices CSV; if missing, trading value is reported as blocked. |
| `--output-dir` | `<collection>/derived/headline_value_analysis` | Output directory for local artifacts. |
| `--sample-size` | `2000` | Rows in the local raw-headline calibration template. |
| `--seed` | `42` | Deterministic sample seed. |
| `--timezone` | `America/New_York` | Exchange timezone used by the backtest core. |
| `--horizons` | `1,5,10` | Comma-separated holding horizons in trading sessions. |
| `--transaction-cost-bps-per-side` | `10.0` | Per-side trading cost applied to headline signals. |
| `--notional-usd` | `10000` | Per-signal notional for P&L summaries. |
| `--overwrite` | off | Replace files in an existing non-empty output directory. |

## `sweep-trading-strategy`

```bash
sentiment-bench sweep-trading-strategy \
  --run-dir results/trading/week3_trading_pilot_20260624_reviewed \
  --scorer consensus/majority --strategy sentiment_magnitude_v1 \
  --thresholds 0.0,0.1,0.2,0.3 --horizons 1,3,5 \
  --metric sharpe --train-fraction 0.6
```

Tunes a registered strategy on a completed run without re-scoring. It reloads `daily_signals.csv` and `prices.csv`, filters to one `--scorer`, and evaluates the strategy's declared parameter space × horizon on a chronological train/test split. `--strategy` selects the idea (default `sentiment_threshold_v1`; `--thresholds` overrides the decision-threshold axis for any strategy, while idea-specific params such as the magnitude idea's `scale` use the strategy's declared grid). The single point with the best **training** metric is selected; **held-out** test metrics are reported alongside, so tuning cannot leak. Writes `sweep.csv` (all points, with the selected row flagged and a `params` column for idea-specific values) and a heatmap PNG named after the CSV stem — `sweep_heatmap.png` by default, or `<stem>_heatmap.png` when `--output` is set — showing the threshold × horizon test metric, and prints the grid with the selected point starred. Selection metrics are notional-independent ratios, so no portfolio assumptions enter the tuning.

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-dir` | required | Completed run with `daily_signals.csv` and `prices.csv`. |
| `--scorer` | `consensus/majority` | `scorer_id` to tune (e.g. `openai/gpt-4o-mini`, or a `#masked` arm). |
| `--strategy` | `sentiment_threshold_v1` | Registered strategy id to tune (see `list-strategies`). |
| `--thresholds` | `0.0,0.1,0.2,0.3` | Comma-separated decision thresholds (overrides the threshold axis). |
| `--horizons` | `1,3,5` | Comma-separated holding horizons (trading sessions). |
| `--metric` | `sharpe` | Selection metric: `mean_return`, `hit_rate`, or `sharpe`. |
| `--train-fraction` | `0.6` | Fraction of distinct news dates used to tune (split is by date, not row). |
| `--output` | `<run-dir>/sweep.csv` | Where to write the sweep CSV. |

## `list-strategies`

```bash
sentiment-bench list-strategies
```

Lists the registered trading strategies (ideas): id, the pipeline seams each customizes, its evaluation frame, its default sweepable parameter space, and a description. Use the id with `sweep-trading-strategy --strategy` or in a config `[strategy]` table. Built-ins: `sentiment_threshold_v1` (equal-weight ±1 default), `sentiment_magnitude_v1` (conviction-weighted sizing), and `headline_sentiment_threshold_v1` (headline information-value pipeline).

## `runs`

```bash
sentiment-bench runs
```

Lists stored benchmark runs, most recent first.

The table includes the machine label/id for new runs, which is useful when reading a shared Turso history.

## `results`

```bash
sentiment-bench results --run-id 1 --confusion
```

Shows stored metrics. Add `--confusion` to print per-model confusion matrices.

## `compare`

```bash
sentiment-bench compare --run-a 1 --model-a openai/gpt-4o-mini \
  --model-b anthropic/claude-3.5-sonnet --metric macro_f1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-a` | required | Run ID for model A. |
| `--model-a` | required | Model ID for side A. |
| `--model-b` | required | Model ID for side B. |
| `--run-b` | `--run-a` | Run ID for model B. |
| `--scope` | `primary` | `primary` or `all`. |
| `--metric` | `accuracy` | `accuracy` or `macro_f1`. |
| `--db-path` | `results/sentiment_benchmark.sqlite` | Local SQLite storage path; when `SENTIMENT_BENCH_DB_BACKEND=libsql`, Turso env vars select the synced replica. |
| `--n-resamples` | `1000` | Bootstrap resamples for CIs. |
| `--confidence` | `0.95` | CI confidence level. |
| `--seed` | `42` | Bootstrap seed. |
| `--alpha` | `0.05` | McNemar significance threshold. |

## `export`

```bash
sentiment-bench export --run-id 1
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--run-id` | required | Run ID to export. |
| `--db-path` | `results/sentiment_benchmark.sqlite` | Local SQLite storage path; when `SENTIMENT_BENCH_DB_BACKEND=libsql`, Turso env vars select the synced replica. |
| `--output-dir` | `results/exports/run_<id>` | Optional export destination override. |

Exports include `tables/` with dissertation-ready booktabs LaTeX tables and, when matplotlib is installed, `figures/` with publication figures (see [Results and exports](results_and_exports.md)).

## Prompt Robustness Commands

Run perturbation variants:

```bash
sentiment-bench run-prompt-suite --models openai/gpt-4o-mini \
  --base-prompt-id default_label_only --include label_order --include paraphrase
```

Summarize metric variation:

```bash
sentiment-bench prompt-sensitivity --run-id 3 --run-id 4 --run-id 5 \
  --model openai/gpt-4o-mini --metric accuracy
```

Add `--latex-output results/exports/prompt_sensitivity.tex` to also write the variant table as a dissertation-ready booktabs LaTeX table.

Common options include `--provider`, `--base-url`, `--ollama-host`, `--mode`, `--dataset-path`, `--db-path`, `--prompts-path`, `--sample-per-class`, `--seed`, `--temperature`, `--max-completion-tokens`, `--reasoning-max-tokens`, `--concurrency`, and `--retries`.

## Agreement Commands

```bash
sentiment-bench agreement --run-id 1 --scope primary
```

Computes Cohen's kappa for model pairs, Fleiss' kappa for multi-model agreement, and Krippendorff's alpha for nominal agreement over valid labels.

## Self-Consistency Commands

Collect repeated samples:

```bash
sentiment-bench run-self-consistency --model openai/gpt-4o-mini \
  --mode pilot --temperature 0.7 --num-samples 5
```

Analyze a run:

```bash
sentiment-bench self-consistency --sc-run-id 1 --top-rows 10
```

List runs:

```bash
sentiment-bench self-consistency-list
```

Compare ambiguity for conflicting duplicate rows:

```bash
sentiment-bench sc-compare-by-conflict --sc-run-id 1
```

Key `run-self-consistency` options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--model`, `-m` | required | Model ID to sample. |
| `--mode` | `pilot` | `pilot` or `full`. |
| `--prompt-id` | `default_label_only` | Prompt ID. |
| `--provider` | `openrouter` | `openrouter`, `cerebras`, or `ollama`. |
| `--temperature`, `-t` | `0.7` | Sampling temperature; use > 0 for diversity. |
| `--num-samples`, `-n` | `5` | Repeated samples per row. |
| `--max-completion-tokens` | `64` | Completion-token budget. |
| `--concurrency` | provider default | Concurrent requests: `64` for Cerebras, `1` otherwise. |
| `--retries` | `3` | Retries per request. |

## `tui`

```bash
sentiment-bench tui
```

Launches the Textual terminal UI. See [TUI guide](tui_guide.md).
## LSEG Workspace Commands

These commands require `python -m pip install -e ".[lseg]"` only on the collection machine. Full details are in [LSEG to Ollama pipeline](lseg_ollama_pipeline.md).

```bash
sentiment-bench lseg-init-config --collection-id my_lseg_collection --output configs/my_lseg_collection.toml
sentiment-bench lseg-news-check --config configs/lseg_workspace_example.toml
sentiment-bench fetch-lseg-news --config configs/lseg_workspace_example.toml
sentiment-bench build-lseg-corpus --source Data/news/lseg_lseg_workspace_example
sentiment-bench build-lseg-analysis-cohort --corpus-manifest Data/derived/lseg/<id>/manifest.json
sentiment-bench score-corpus-matrix \
  --input Data/collections/lseg_us_sector_33_6m/derived/us_sector_33_6m_analysis/cohort.jsonl \
  --subset Data/collections/lseg_us_sector_33_6m/derived/us_sector_33_6m_analysis/l3_subset.jsonl \
  --kind lseg --output-dir results/scoring/lseg_us_sector_33_formal --dry-run
sentiment-bench lseg-catalog
```

`lseg-init-config` writes a preset TOML config and refuses to replace an existing file unless `--overwrite` is passed. The default preset is the 8-company Week 4 universe over `2025-06-26T00:00:00Z` to `2026-06-26T00:00:00Z` with daily windows; `configs/lseg_us_mega_cap_1y.toml` is the checked-in template. `lseg-news-check` tests the desktop session plus headline and story entitlements without writing data. `fetch-lseg-news` atomically checkpoints an immutable collection. `build-lseg-corpus` verifies raw hashes and deterministically rebuilds clean text offline. `lseg-catalog` refreshes metadata-only `Data/derived/lseg/catalog.json` and `catalog.csv`.

`build-lseg-analysis-cohort` leaves the verified corpus unchanged, applies the configured relevance and earliest-revision rules, and writes hashed development, holdout, and L3 cohort artifacts. It refuses incomplete corpora, undersized splits, and existing output directories.

`score-corpus-matrix` runs the configured crossed design (default: five models, three prompts, five samples — swappable in `configs/crossed_scoring.toml`). Scores are append-only and resumable by item content, model digest, prompt hash, and sample index. `--dry-run` validates inputs and prints the call counts for the current configuration without contacting a provider.

```bash
sentiment-bench analyze-l2 \
  --scores results/scoring/lseg_us_sector_33_formal/scores.jsonl \
  --prices Data/derived/prices/l2_market_model.csv \
  --output-dir results/l2/us_sector_33
```

`analyze-l2` derives five-model majority readings, pairwise agreement, self-consistency entropy, company-day weights, 120-session market-model CARs at 1/5/10 sessions, two-way clustered tests, and BH-adjusted H2 results. The price CSV must contain `symbol`, `session_date`, and `close` rows for every company and `^GSPC`.

```bash
sentiment-bench analyze-l3 \
  --scores results/scoring/lseg_us_sector_33_formal/scores.jsonl \
  --benchmark-scores results/scoring/financial_sentiment_formal/scores.jsonl \
  --event-returns results/l2/us_sector_33/event_returns.csv \
  --l2-hypotheses results/l2/us_sector_33/hypotheses.csv \
  --output-dir results/l3/us_sector_33
```

`analyze-l3` fits the crossed variance-component model, reports the G and dependability coefficients, derives item/day reliability, tunes the fixed threshold on development data, compares all three rules on holdout with 10 bps per side, validates PhraseBank tiers, and recomputes one BH family across the supplied H2 and H3 tests.

```bash
sentiment-bench sample-lseg-validation \
  --cohort-manifest Data/collections/lseg_us_sector_33_6m/derived/us_sector_33_6m_analysis/manifest.json \
  --output-dir Data/derived/lseg/<id>/validation_seed42

sentiment-bench evaluate-lseg-annotations \
  --primary <annotation_primary.csv> --secondary <annotation_secondary.csv> \
  --output-dir Data/derived/lseg/<id>/validation_evaluated
```

Sampling defaults to 150 ticker/date-stratified stories with seed 42 and 30 double-coded stories with seed 43. The sheets collect company relevance and sentiment separately; annotation evaluation requires allowed values and adjudication of every disagreement. `--corpus-manifest` remains available for validation work that intentionally precedes cohort construction.

For `run-trading-strategy`, `[scoring].provider` defaults to `openrouter` for existing configs. A pure-LSEG Ollama config supplies `sources.lseg_corpus_manifest`, exact tagged `models`, `primary_model`, optional `baselines`, Ollama host/keep-alive/thinking/structured-output settings, and `[signal_policy]`. See `configs/lseg_ollama_trading_example.toml`.

Three optional tables add the reproducibility and contamination controls (all default to existing behaviour, so current configs are unaffected):

```toml
[prices]
provider = "lseg"                 # "lseg" (preferred; needs a Workspace session) or "yfinance"
cache_dir = "Data/derived/prices" # opt-in per-(symbol,window) cache; omit to disable
[prices.rics]                     # LSEG only: symbol → RIC (unmapped symbols pass through)
AAPL = "AAPL.O"
adjusted = true                   # adjusted (auto_adjust) prices

[cutoff]
policy = "stratify"               # ignore | stratify (default) | post_only
[cutoff.overrides]                # optional authoritative cutoffs, by model id or suffix
"openai/gpt-4o-mini" = "2023-10-01"

[scoring]
masking_mode = "both"             # off (default) | on | both
```

- `[prices]` makes runs deterministic and offline-replayable when `cache_dir` is set. `provider = "lseg"` (preferred for formal runs) fetches licensed daily bars through the same Workspace session as the news collection; note it applies exchange/manual corrections and split adjustments but does **not** back-adjust dividends, so returns follow the price-return convention (yfinance's `auto_adjust` folds dividends in). Existing configs that omit `[prices]` keep the yfinance default.
- `[cutoff].policy = stratify` only annotates scores and emits `sensitivity_cutoff.csv`, leaving the declared primary cell untouched; `post_only` additionally restricts the traded signal to contamination-free (post-cutoff) scores; `ignore` disables annotation. Cutoffs are best-effort — verify against provider model cards or pin them via `[cutoff.overrides]`.
- `[scoring].masking_mode = both` runs an extra anonymised arm under `#masked` scorer ids and writes `sensitivity_masking.csv`; `on` scores only masked text; `off` is the default.

An optional `[strategy]` table selects which registered idea a run uses (default is the equal-weight threshold strategy, so existing configs are unaffected):

```toml
[strategy]
id = "sentiment_magnitude_v1"     # see `sentiment-bench list-strategies`
scale = 1.0                        # idea-specific params (here: conviction sizing scale)
max_position = 1.0
eval_frame = "event_study"        # event_study (implemented) | cross_sectional (planned)
```

- Shared decision params (`threshold`, `min_valid_stories`, costs) still come from `[signal_policy]`; `[strategy]` only adds the idea id, idea-specific params, and the evaluation frame.
- The default `sentiment_threshold_v1` reproduces the legacy behaviour byte-for-byte; only opt-in ideas (e.g. the conviction-weighted `sentiment_magnitude_v1`) use fractional position sizing.
- The resolved `strategy_id` and params are recorded in `experiments/manifest.toml` for each run. `eval_frame = "cross_sectional"` is a guarded planned fast-follow and currently errors.
