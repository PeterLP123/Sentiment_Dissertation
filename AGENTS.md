# Agent Instructions

This repository contains code, data, experiments, and supporting artifacts for a dissertation on sentiment analysis. Treat it as a research workspace: changes should be reproducible, documented, and conservative around data.

## Repository Shape

- `Data/` contains datasets and related data files.
- `submission/news-sentiment-beyond-mean/` contains the completed dissertation and its self-contained reproduction package.
- `src/`, `scripts/`, `configs/`, and `tests/` support the benchmark application; `final_experiments/` and the older `dissertation/` retain the research history.

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

## Completed Dissertation And Research Archive

The canonical completed project is `submission/news-sentiment-beyond-mean/`. Its submitted
manuscript, selected experiment sources, frozen specifications, aggregate evidence,
`uv.lock`, and validation scripts are the authority for final claims and reproduction.
Preserve that package as a self-contained project. Do not silently replace its sources,
locked environment, frozen results, or submitted PDF with files from the wider workspace.

The final research question is settled: whether hard negative-story share is associated
with the assigned-window return beyond rank-linear mean sentiment and news volume in the
training period, where that association is measured, and whether the unchanged baseline
persists in the frozen 2020--2023 testing period. The submitted result is a bounded
temporal non-replication and timing diagnosis, not validated alpha or a market mechanism.
Notebook 75 is the sealed one-shot evaluation record. Do not rerun, retune, or
extend the testing-period study during repository maintenance.

`final_experiments/`, `docs/research_protocol.md`,
`docs/dissertation_execution_plan.md`, and `final_experiments/plan.html` are historical
research records. Retain them for auditability. The archive also contains the distinct
Notebook 85 novelty-conditioning and Notebook 86 economic-value studies, published
from existing local work on 8 September 2026. Keep them separate from the submitted
package. Notebook 86's recorded evaluation is a retrospective replay of an already-opened
period, not a new confirmatory test. Earlier open-RQ,
workstream, and live-plan language describes the historical decision process; it does not
reopen the submitted design or supersede the submission package.

### Reproducibility guardrails

- Preserve frozen specifications, invalidation records, chronological boundaries, and
  all null or adverse findings.
- Do not overwrite source data, experiment outputs, submitted artifacts, or evidence
  snapshots. Write exploratory reruns to new, clearly identified paths.
- Keep FNSPID and licensed LSEG text local and gitignored. Do not pool source regimes.
- State the estimand, sample, split, timing rule, clustering or bootstrap unit,
  multiplicity family, model identity, and costs where applicable.
- Treat missing row-level availability times as a binding limit on predictive claims.
- Changes to a signal, portfolio rule, holding period, threshold, cost, turnover rule, or
  reported strategy result require tracing and refreshing the full affected notebook
  chain. Update the archive README, experiment ledger, and invalidation history where the
  recorded status changes.
- New exploratory helpers belong in `final_experiments/lib/`; reusable benchmark code
  belongs in `src/sentiment_benchmark`. Add tests where a silent helper error would corrupt
  downstream results.

## Command Guidance

- Use `rg` or `rg --files` for searching.
- Use `make validate` for the final package and `make benchmark-check` for the
  public benchmark. These commands select separate locked Python environments.
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
