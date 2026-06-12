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
| Dataset default | `Data/data.csv` |
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
| `SENTIMENT_BENCH_PROVIDER` | Default provider for CLI/TUI: `openrouter` or `ollama`. |
| `OPENROUTER_API_KEY` | OpenRouter authentication token. |
| `OPENROUTER_BASE_URL` | OpenRouter-compatible endpoint. |
| `OLLAMA_HOST` | Ollama endpoint, such as `http://localhost:11434`. |
| `SENTIMENT_BENCH_DB_BACKEND` | `sqlite` by default; set `libsql` for Turso/native libSQL. |
| `TURSO_DATABASE_URL` | Turso/libSQL URL, such as `libsql://...turso.io`. |
| `TURSO_AUTH_TOKEN` | Turso database auth token. Keep out of git. |
| `TURSO_REPLICA_PATH` | Local embedded replica path, defaulting to `results/turso_replica.db`. |
| `SENTIMENT_BENCH_MACHINE_LABEL` | Friendly per-machine label stored with new runs. |
| `SENTIMENT_BENCH_MACHINE_ID` | Optional explicit stable machine id; otherwise a hashed OS machine value is used. |
| `SENTIMENT_BENCH_AUTO_FETCH_MODELS` | TUI local-Ollama auto-fetch toggle, default `1`. |

## Commands

| Command | Purpose |
| --- | --- |
| `validate-data` | Validate dataset columns, labels, duplicates, and primary scoring scope. |
| `news-check` | Run a small Tavily search to verify API connectivity. |
| `fetch-news` | Fetch Tavily-sourced articles into a timestamped derived corpus. |
| `fetch-news-batch` | Fetch a TOML query matrix across repeated date windows and optionally package the results. |
| `package-news` | Package existing Tavily corpora into a colleague-shareable source dataset. |
| `news-quality` | Summarize text quality across fetched Tavily corpora without calling Tavily. |
| `list-models` | List models from OpenRouter or Ollama. |
| `run` | Run LLM sentiment classification benchmarks. |
| `run-baselines` | Run non-LLM baseline classifiers. |
| `export` | Export a stored run to files under `results/exports/`. |
| `runs` | List stored benchmark runs. |
| `results` | Show stored metrics and optional confusion matrices. |
| `compare` | Compare two models with paired McNemar's test and bootstrap CIs. |
| `run-prompt-suite` | Run a model across prompt perturbation variants. |
| `prompt-sensitivity` | Summarize metric variation across prompt variants. |
| `agreement` | Compute inter-model agreement for one run. |
| `tui` | Launch the terminal UI. |
| `run-self-consistency` | Sample one model repeatedly at temperature > 0. |
| `self-consistency` | Analyze an existing self-consistency run. |
| `self-consistency-list` | List self-consistency runs. |
| `sc-compare-by-conflict` | Test whether conflicting duplicate rows have higher entropy. |

## `validate-data`

```bash
sentiment-bench validate-data --dataset-path Data/data.csv
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset-path` | `Data/data.csv` | CSV containing `Sentence` and `Sentiment`. |

## `list-models`

```bash
sentiment-bench list-models --provider openrouter --limit 50
sentiment-bench list-models --provider ollama --ollama-host http://desktop-pc:11434
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--provider` | `openrouter` | `openrouter` or `ollama`. |
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
| `--dataset-path` | `Data/data.csv` | Dataset CSV. |
| `--db-path` | `results/sentiment_benchmark.sqlite` | Local SQLite storage path; when `SENTIMENT_BENCH_DB_BACKEND=libsql`, Turso env vars select the synced replica. |
| `--prompts-path` | `configs/default_prompts.toml` | Prompt config path. |
| `--provider` | `openrouter` | `openrouter` or `ollama`. |
| `--base-url` | `https://openrouter.ai/api/v1` | OpenRouter-compatible base URL. |
| `--ollama-host` | `http://localhost:11434` | Ollama host URL. |
| `--sample-per-class` | `30` | Pilot rows per class. |
| `--seed` | `42` | Row sampling and reproducibility seed. |
| `--temperature` | `0.0` | Model sampling temperature. |
| `--max-completion-tokens` | `64` | Default completion-token budget. |
| `--reasoning-max-tokens` | `2048` | Larger token budget for reasoning-model ID markers. |
| `--model-max-tokens` | none | Per-model override as `model_id=N`; repeatable. |
| `--concurrency` | `1` | Concurrent model requests. |
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
| `--dataset-path` | `Data/data.csv` | Dataset CSV. |
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

## `fetch-news-batch`

```bash
sentiment-bench fetch-news-batch --weeks 5 --package-id tavily_shared_v3
```

`--weeks N` generates N contiguous 7-day windows ending today (or at `--end-date`). Explicit windows are also supported and can be combined with `--weeks`:

```bash
sentiment-bench fetch-news-batch \
  --date-window 2026-05-07:2026-05-14 \
  --date-window 2026-06-05:2026-06-11 \
  --package-id tavily_shared_v3
```

Reads `configs/tavily_query_matrix.toml`, runs every query across every date window, writes timestamped corpora under `Data/news`, then packages those newly fetched corpora into `Data/derived/<package-id>` by default.

During execution, the command prints a Rich progress display with the current fetch number, query id, date window, per-fetch record count, usable article count, failed extraction count, and output directory. Packaging has its own progress step after all fetches finish.

Use dry run first to confirm the number of Tavily calls:

```bash
sentiment-bench fetch-news-batch --weeks 5 --dry-run
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--date-window` | none | Date window as `YYYY-MM-DD:YYYY-MM-DD`; repeat for multiple batches. At least one window or `--weeks` is required. |
| `--weeks` | none | Generate this many contiguous 7-day windows ending at `--end-date`. |
| `--end-date` | today | Last day covered by `--weeks`, as `YYYY-MM-DD`. |
| `--query-matrix` | `configs/tavily_query_matrix.toml` | TOML query matrix defining reusable Tavily queries. |
| `--query-id` | all queries | Restrict to one matrix query id; repeat for multiple ids. |
| `--extract / --no-extract` | `--extract` | Run Tavily Extract for each fetched result. |
| `--extract-depth` | matrix value | Override the matrix `extract_depth` for every query: `basic` or `advanced`. |
| `--output-dir` | `Data/news` | Root directory for fetched timestamped corpora. |
| `--package / --no-package` | `--package` | Package newly fetched corpora after the batch completes. |
| `--package-id` | `tavily_shared_v1` | Share package id used when packaging is enabled. |
| `--package-output-dir` | `Data/derived` | Root directory for generated share package. |
| `--text-policy` | `metadata` | Package text policy: `metadata` or `internal-extracts`. |
| `--skip-existing / --refetch` | `--skip-existing` | Skip fetches whose exact parameters already produced a corpus under `--output-dir`; skipped corpora still join the package, so interrupted batches resume without paying for completed fetches again. Any parameter change (window, depth, domains, quality floor) triggers a refetch. |
| `--dry-run` | off | Print planned fetches without calling Tavily or writing files, including how many would be skipped. |

Each fetch is attempted up to 3 times with a short backoff before being recorded as failed, and the batch continues past persistent failures instead of aborting. When any fetch fails, packaging is skipped and the command exits non-zero — re-run the same command to retry only the failed fetches and build the package. Search and extract calls use 90s/120s timeouts — the SDK defaults are too tight for advanced extraction over 20 URLs, and 120s is the maximum the Tavily SDK accepts. Corpora whose search returned zero records are refetched on the next run rather than treated as complete.

## `package-news`

```bash
sentiment-bench package-news \
  --source Data/news/tavily_news_20260607T120000Z_bank-earnings-sentiment \
  --package-id tavily_shared_v1 \
  --output-dir Data/derived \
  --query-matrix configs/tavily_query_matrix.toml \
  --text-policy metadata
```

Packages existing Tavily corpus directories into a versioned source dataset for colleague review. This command does not call Tavily, label examples, run models, or modify `Data/data.csv`.

Records from corpora fetched before the quality gate existed are re-assessed during packaging, so error pages and stub bodies in older corpora are reported with the correct `text_quality` and excluded from `extract_text_available` and `extracts.jsonl`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--source` | required | Existing Tavily corpus directory; repeat for multiple corpora. |
| `--package-id` | `tavily_shared_v1` | Output package directory name under `--output-dir`; spaces are converted to hyphens. |
| `--output-dir` | `Data/derived` | Root directory for generated share packages. |
| `--query-matrix` | `configs/tavily_query_matrix.toml` | Optional TOML query matrix used to add query IDs and families. |
| `--text-policy` | `metadata` | `metadata` writes no full article bodies; `internal-extracts` also writes `extracts.jsonl`. |

Output files:

| File | Purpose |
| --- | --- |
| `sources.csv` | Metadata-first source table with URLs, titles, snippets, provenance, hashes, text quality, and text availability. |
| `screening_index.csv` | Editable review worksheet with screening fields, a `text_quality` column, and blank optional future label columns. |
| `package_manifest.json` | Package metadata, source corpora, counts, duplicate URL count, file hashes, and sharing notes. |
| `README.md` | Handoff notes for colleagues. |
| `extracts.jsonl` | Optional internal full-text bundle, written only with `--text-policy internal-extracts`. |

## `news-quality`

```bash
sentiment-bench news-quality
```

Summarizes text quality across fetched Tavily corpora without making any API calls. Prints an overview table (records, unique URLs, usable unique URLs, `text_quality` breakdown, publication-date provenance), per-query-family counts, and the top source domains with usable shares. Query families whose corpora returned zero records remain visible as coverage gaps. Corpora that predate the quality gate are re-assessed on the fly.

The domain table includes a `Shared prefix` column: the longest identical opening shared by at least two distinct usable articles from that domain. A large value (highlighted at 200+) means the extraction is carrying site boilerplate into article bodies or the domain published near-duplicate articles — both worth reviewing before labeling.

| Option | Default | Meaning |
| --- | --- | --- |
| `--source` | all under `--news-dir` | Specific Tavily corpus directory; repeat for multiple. |
| `--news-dir` | `Data/news` | Root directory scanned for corpora when `--source` is not given. |
| `--query-matrix` | `configs/tavily_query_matrix.toml` | Optional TOML matrix used to group corpora by query family. |
| `--top-domains` | `10` | Number of source domains to list. |

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
| `--provider` | `openrouter` | `openrouter` or `ollama`. |
| `--temperature`, `-t` | `0.7` | Sampling temperature; use > 0 for diversity. |
| `--num-samples`, `-n` | `5` | Repeated samples per row. |
| `--max-completion-tokens` | `64` | Completion-token budget. |
| `--concurrency` | `1` | Concurrent requests. |
| `--retries` | `3` | Retries per request. |

## `tui`

```bash
sentiment-bench tui
```

Launches the Textual terminal UI. See [TUI guide](tui_guide.md).
