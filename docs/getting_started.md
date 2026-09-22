# Getting started with the benchmark

The `sentiment-bench` application validates labelled sentiment data, runs model
comparisons and stores their results. It is the earlier tooling developed during the
dissertation. To reproduce the final study, use the [submission guide](reproducing_the_submission.md).

For a complete walkthrough without provider credentials, start with the
[offline portfolio demo](portfolio_demo.md). It validates the benchmark and runs
the separate historical strategy pipeline on synthetic fixtures.

## Install and check the data

With Python 3.12 and `uv` installed, run from the repository root:

```bash
uv sync --locked
uv run --locked sentiment-bench validate-data
```

This check runs locally. No credentials are needed.

The committed dataset has **5,947 rows**: 2,884 neutral, 2,082 positive and 981 negative.
The validation table should report zero duplicate sentence groups and zero conflicting
labels. The [dataset card](dataset_card.md) records its Financial PhraseBank/FiQA sources
and the corruption corrected in the older `Data/data.csv` merge.

If the derived CSV is missing, rebuild it with `python scripts/build_labeled_dataset.py`.
To install development dependencies and run all benchmark checks, use
`make benchmark-setup` and `make benchmark-check` instead.

<details>
<summary>Install without uv, including Windows PowerShell</summary>

On macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
sentiment-bench validate-data
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
sentiment-bench validate-data
```

This pip fallback resolves dependencies from `pyproject.toml` rather than `uv.lock`.
With an activated virtual environment, omit `uv run --locked` from the commands below.

</details>

## Run a pilot

Model runs require a configured provider. For OpenRouter, add your key to a local `.env`
file, which Git ignores:

```dotenv
OPENROUTER_API_KEY=your-openrouter-key
```

List the available models:

```bash
uv run --locked sentiment-bench list-models --provider openrouter --limit 5
```

Replace `MODEL_ID` below with an available model ID. This command sends benchmark text
to OpenRouter and incurs the selected model's usage charges:

```bash
uv run --locked sentiment-bench run --provider openrouter --models MODEL_ID --mode pilot
```

Pilot mode samples 30 rows per class with seed `42`. Results are saved locally in
`results/sentiment_benchmark.sqlite` by default. For Ollama, Cerebras or a shared
Turso/libSQL store, follow the [provider guide](model_providers.md).

## Inspect and export results

Find the run ID, then use it in place of `1`:

```bash
uv run --locked sentiment-bench runs
uv run --locked sentiment-bench results --run-id 1 --confusion
uv run --locked sentiment-bench export --run-id 1
```

Results include macro-F1, balanced accuracy, MCC, per-class scores, invalid-output and
API-error counts. The export contains response tables, metrics and prompt/run metadata
under `results/exports/run_1/`. See [results and exports](results_and_exports.md) for the
file schemas and the optional plotting extra.

## Use the terminal interface

```bash
uv run --locked sentiment-bench tui
```

The TUI uses the same services as the CLI and saves preferences in
`results/tui_session.json`. The [TUI guide](tui_guide.md) covers its tabs, keyboard
shortcuts and logs. News collection is a separate workflow with its own provider
credentials, described in [news sourcing](news_sourcing.md).
