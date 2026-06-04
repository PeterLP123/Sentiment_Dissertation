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

Export a completed run:

```bash
sentiment-bench export --run-id 1
```

Exports are written under `results/exports/run_<id>/` and include responses, metrics, run configuration, prompt data, and reproducibility metadata.

## Reproducibility Notes

- The source dataset in `Data/` is treated as source material. Do not edit it in place for experiments.
- Pilot row selection records the random seed and selected rows in SQLite.
- Exports include the dataset path, dataset SHA-256 hash, prompt hash, model list, request settings, package version, and export timestamp.
- Primary metrics exclude rows whose duplicate sentence has conflicting labels. Audit metrics include all selected rows.
