# Getting Started With The Benchmark

You will install the project, validate the dissertation dataset, run a small model benchmark, inspect the stored metrics, export the evidence, and optionally fetch a small Tavily news corpus.

![CLI news fetch screenshot](assets/cli-news-fetch.svg)

## What You'll Need

- Python 3.12.
- The repository checked out locally.
- `Data/data.csv` present in the repository.
- An OpenRouter API key for OpenRouter model runs, or an Ollama server reachable from this machine.
- A Tavily API key if you want to source news articles.

## Step 1: Install The Package

From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows, you can run the setup helper instead:

```powershell
.\scripts\setup_windows_py312.ps1
```

This installs the `sentiment-bench` command, runtime dependencies, and development tools used by the test suite.

The `.venv/` folder is local to each machine and is intentionally ignored by git. Do not copy it between machines; recreate it on each machine with the setup command above. The repository carries the reproducible setup instructions through `.python-version`, `pyproject.toml`, and `scripts/setup_windows_py312.ps1`.

## Step 2: Add Local Credentials

Create `.env` in the repository root:

```bash
OPENROUTER_API_KEY=your-openrouter-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

TAVILY_API_KEY=your-tavily-key
TAVILY_PROJECT=sentiment-dissertation
```

Do not commit `.env`. It is ignored by git.

For Ollama instead of OpenRouter, add:

```bash
SENTIMENT_BENCH_PROVIDER=ollama
OLLAMA_HOST=http://desktop-pc:11434
```

See [Model providers](model_providers.md) for the desktop PC setup.

## Step 3: Validate The Dataset

Run:

```bash
sentiment-bench validate-data --dataset-path Data/data.csv
```

You should see a Rich table summarizing row counts, label counts, duplicates, conflicting duplicate groups, and the primary scoring scope. This step is the fastest proof that the environment and dataset path are correct.

## Step 4: Run A Pilot Benchmark

Run one small OpenRouter pilot:

```bash
sentiment-bench run --models openai/gpt-4o-mini --mode pilot
```

Pilot mode samples 30 primary rows per class with seed `42` by default. Results are stored in `results/sentiment_benchmark.sqlite`.

To use a local Gemma model through Ollama:

```bash
sentiment-bench run --provider ollama --ollama-host http://desktop-pc:11434 \
  --models gemma3 --mode pilot
```

## Step 5: Inspect Results

List recent runs:

```bash
sentiment-bench runs
```

Show metrics for one run:

```bash
sentiment-bench results --run-id 1 --confusion
```

The metrics include accuracy, macro-F1, weighted-F1, balanced accuracy, MCC, invalid-output counts, API-error counts, and confusion matrices when requested.

## Step 6: Export A Run

Export the run evidence:

```bash
sentiment-bench export --run-id 1
```

The default export path is `results/exports/run_1/`. It contains response tables, metrics, prompt/run metadata, statistical summaries, and optional figures if the plotting extra is installed.

## Step 7: Fetch A Small News Corpus

Check Tavily connectivity:

```bash
sentiment-bench news-check --query "financial markets" --max-results 1
```

Fetch articles into a timestamped derived corpus:

```bash
sentiment-bench fetch-news --query "bank earnings sentiment" \
  --max-results 10 --time-range week --output-dir Data/news
```

Each fetch writes:

- `articles.jsonl` for provenance-rich records.
- `articles.csv` for spreadsheet review.
- `manifest.json` for reproducibility metadata.

These articles are unlabeled source material. The command does not modify `Data/data.csv` and does not create benchmark rows automatically.

## Step 8: Open The TUI

Run:

```bash
sentiment-bench tui
```

Use tabs `1` through `6` for Dashboard, Models, Prompt, Run, Results, and News. The TUI persists its local session state in `results/tui_session.json`.

## What You Built

You now have a working dissertation benchmark environment with:

- A validated immutable source dataset.
- At least one stored benchmark run.
- Inspectable metrics and exports.
- Optional Tavily article sourcing.
- Optional remote Ollama routing for desktop-hosted local models.

For exact command options, use [CLI reference](cli_reference.md). For output interpretation, use [Results and exports](results_and_exports.md).
