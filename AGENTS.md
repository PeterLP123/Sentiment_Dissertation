# Agent Instructions

This repository contains code, data, experiments, and supporting artifacts for a dissertation on sentiment analysis. Treat it as a research workspace: changes should be reproducible, documented, and conservative around data.

## Repository Shape

- `Data/` contains datasets and related data files.
- Future code may include scripts, notebooks, model experiments, evaluation utilities, and reports.
- Keep new files organized by purpose. Prefer clear top-level folders such as `src/`, `notebooks/`, `experiments/`, `reports/`, `models/`, and `tests/` if the project grows.

## Working Principles

- Preserve research reproducibility. Record important assumptions, parameters, random seeds, model names, package versions, and dataset versions.
- Do not silently overwrite datasets, experiment outputs, trained models, or dissertation artifacts.
- Keep experimental code and production-style utilities separate when possible.
- Prefer small, focused changes over broad refactors unless the user explicitly asks for restructuring.
- Use descriptive names for files, functions, notebooks, and experiment outputs.

## Data Handling

- Treat files in `Data/` as important source material.
- Before modifying or cleaning a dataset, create a derived file rather than editing the original in place.
- Avoid committing large generated outputs, model checkpoints, cache directories, or temporary files unless the user asks.
- If data contains personal, sensitive, or proprietary text, avoid printing large raw samples in logs or final responses.
- The default labeled benchmark dataset is `Data/derived/labeled/financial_sentiment_v2.csv`.
  It is rebuilt by `scripts/build_labeled_dataset.py` from Financial PhraseBank and FiQA
  sources. Do not manually edit it; rebuild it from the script if provenance logic changes.
- Treat `Data/data.csv` as legacy Kaggle source material, not the default benchmark.
  It contains documented merge corruption: 514 negative PhraseBank sentences were duplicated
  under an incorrect `neutral` label. Do not use those duplicate conflicts as an ambiguity
  signal or silently switch defaults back to this file.
- When changing dataset defaults or examples, update the code default, CLI/TUI docs,
  dataset card, research protocol, tests, and `.gitignore` tracking rules together.
  Verify with `sentiment-bench validate-data` and focused dataset tests.
- When creating processed datasets, document:
  - Source file
  - Cleaning steps
  - Label mapping
  - Train/validation/test split logic
  - Random seed

## Python and Experiments

- Prefer Python for sentiment analysis tooling unless the existing codebase establishes another stack.
- Use common libraries where appropriate, such as `pandas`, `numpy`, `scikit-learn`, `nltk`, `spacy`, `transformers`, `torch`, or `tensorflow`.
- Keep reusable logic in scripts or modules rather than burying everything in notebooks.
- Notebooks are acceptable for exploration, but important results should be reproducible from scripts when practical.
- Set random seeds for experiments and record them near the code that uses them.
- Make evaluation metrics explicit. For classification tasks, include accuracy only when paired with more informative metrics such as precision, recall, F1, confusion matrices, or class-level scores.

## Code Quality

- Follow the style already present in the repo. If no style exists yet, use:
  - Python 3
  - Clear function boundaries
  - Type hints for reusable functions
  - `pytest` for tests where practical
  - `requirements.txt`, `pyproject.toml`, or `environment.yml` for dependencies
- Keep comments useful and specific. Avoid comments that merely restate the code.
- Prefer structured parsing and data APIs over ad hoc string manipulation.
- Add focused tests for reusable data processing, model evaluation, or metric code.
- Preserve LF line endings for repository files. The `.gitattributes` file sets
  this convention; do not introduce CRLF-only changes.

## Dissertation Context

- Prioritize traceability: future readers should be able to understand how a result was produced.
- When adding experiment results, include enough metadata to compare runs fairly.
- Avoid overstating conclusions in generated reports or summaries. Distinguish observed results from interpretation.
- If using external models, datasets, or papers, cite or record the source clearly.

## Command Guidance

- Use `rg` or `rg --files` for searching.
- Inspect the worktree before editing with `git status --short`.
- Do not run destructive commands such as `git reset --hard`, force pushes, or dataset deletion unless explicitly requested.
- If running long experiments, make outputs resumable or clearly timestamped.

## Agent Handoff Notes

When leaving work in progress, summarize:

- What changed
- What commands or experiments were run
- Key results or failures
- Files that are generated versus source-controlled
- Recommended next step
