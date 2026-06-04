# Sentiment Dissertation Benchmark

Terminal UI and CLI tooling for benchmarking OpenRouter language models on the dissertation sentiment dataset.

## Setup

Install the package in editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
```

Create a local `.env` file for OpenRouter credentials:

```bash
OPENROUTER_API_KEY=your-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

`OPENROUTER_BASE_URL` is optional. The default points to OpenRouter.

## Usage

Validate the dataset:

```bash
sentiment-bench validate-data --dataset-path Data/data.csv
```

Run a pilot benchmark from the CLI:

```bash
sentiment-bench run --models openai/gpt-4o-mini --mode pilot
```

Launch the terminal UI:

```bash
sentiment-bench tui
```

Run non-LLM baselines for context:

```bash
# Default baselines (majority, tfidf_logreg) on the balanced pilot sample.
sentiment-bench run-baselines --mode pilot

# Evaluate baselines on the exact rows used by an existing LLM run (fair comparison).
sentiment-bench run-baselines --match-run-id 5 --baselines majority --baselines tfidf_logreg

# Optional baselines (require extra dependencies):
#   pip install ".[baselines]"   # enables vader
#   pip install ".[finbert]"     # enables finbert (large model download)
sentiment-bench run-baselines --mode full --baselines vader --baselines finbert
```

Inspect stored results without opening the TUI:

```bash
# List recent runs (id, mode, status, models).
sentiment-bench runs

# Metrics table for one run; add --confusion for per-model confusion matrices.
sentiment-bench results --run-id 1 --confusion
```

Compare two models with a paired significance test:

```bash
# Same run, two models (defaults to --scope primary, --metric accuracy).
sentiment-bench compare --run-a 1 --model-a openai/gpt-4o-mini --model-b anthropic/claude-3.5-sonnet

# Across runs (e.g. prompt sensitivity), on macro-F1.
sentiment-bench compare --run-a 1 --model-a openai/gpt-4o-mini \
    --run-b 2 --model-b openai/gpt-4o-mini --metric macro_f1
```

Predictions are paired by row so McNemar's test is valid, and each side reports a
bootstrap confidence interval computed on the shared rows. The Results tab in the
TUI shows the same metrics plus a colour-coded confusion matrix per model/scope.

Export a completed run:

```bash
sentiment-bench export --run-id 1
```

Exports are written under `results/exports/run_<id>/` and include responses, metrics, run configuration, prompt data, and reproducibility metadata.

If the optional plotting extra is installed (`pip install ".[figures]"`), each export also writes publication-ready PNGs to `results/exports/run_<id>/figures/`: a model leaderboard, an accuracy bootstrap-CI forest plot, and per-model confusion-matrix heatmaps and per-class score bars. In the TUI, the Results tab's **View Figures** button exports the selected run and opens its `figures/` folder in your OS image viewer.

## Reproducibility Notes

- The source dataset in `Data/` is treated as source material. Do not edit it in place for experiments.
- Pilot row selection records the random seed and selected rows in SQLite.
- Exports include the dataset path, dataset SHA-256 hash, prompt hash, model list, request settings, package version, and export timestamp.
- Primary metrics exclude rows whose duplicate sentence has conflicting labels. Audit metrics include all selected rows.

## Baselines

Baselines are stored, scored, and exported exactly like LLM runs (same metrics, primary/audit scopes, and `export` command), so they appear alongside models in the runs list and can be compared directly.

- `majority`: predicts the most frequent training label. Reported via stratified k-fold cross-validation (out-of-fold predictions), so it never sees a row it is scored on.
- `tfidf_logreg`: TF-IDF (1–2 grams) features with multinomial logistic regression, also evaluated out-of-fold.
- `vader`: NLTK VADER lexicon; the compound score is mapped to a label using the conventional ±0.05 threshold. Pretrained, predicts on every row.
- `finbert`: `ProsusAI/finbert`, a transformer fine-tuned on financial sentiment. Pretrained, predicts on every row.

Fitted baselines (`majority`, `tfidf_logreg`) use the recorded `--seed` and `--folds` for reproducibility. Use `--match-run-id` to evaluate baselines on the identical row selection as a prior LLM run.
