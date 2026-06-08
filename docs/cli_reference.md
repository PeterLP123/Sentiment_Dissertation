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
| `--output-dir` | `Data/news` | Root directory for timestamped corpus outputs. |

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
