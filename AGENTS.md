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

## Final Experiments Phase (opened 2026-07-30)

`final_experiments/` holds the consolidated closing work for the dissertation.
Read `final_experiments/README.md` for the notebook/data map and
`final_experiments/plan.html` for live decisions and checklist status. For scientific
constraints and stage order, use `docs/research_protocol.md` and
`docs/dissertation_execution_plan.md`. Dated protocols/results are historical evidence,
not competing current plans.

### The research question is not decided

- **Do not hard-code a single RQ** into new code, docs, or dissertation text. Gate F1
  follows the filtering/distribution EDA and precedes any evaluation-return comparison.
  Candidate framings are enumerated in `final_experiments/plan.html`; record which
  candidate each result speaks to.
- The "Beyond the Mean" cross-model-agreement RQ remains the standing fallback/default.
  That is not a decision.

### Working style: researcher, not production engineer

The rest of this repository is a hardened pipeline. `final_experiments/` is deliberately
looser so iteration is fast:

- Prefer Jupyter notebooks (jupytext `# %%` `.py` pairs, as in `notebooks/`) with plots
  and tables over new CLI subcommands and frozen-manifest machinery.
- Write **fewer tests**. Add a `pytest` test only for a reusable data-processing or metric
  helper where a silent error would corrupt every downstream result. Do not test
  exploratory analysis, plotting, or notebook glue.
- Skip full-suite runs, `mypy`, and hash/manifest ceremony unless a result is being
  promoted into the dissertation.
- Optimise for academic rigour and explainability: state the estimand, the split, the
  clustering, the multiplicity family, and the figure that shows the effect. A clear plot
  beats another abstraction layer.
- Treat `src/sentiment_benchmark` as a library to import from, not a place to extend. New
  helpers belong in `final_experiments/lib/`.

### Current state and workstreams

Workstream 1 is complete for the current checkpoint chain: FNSPID 2011-2023 is the
primary spine; the panel has 715,546 news-bearing firm-days, 570 priced symbols and 3,262
sessions. LSEG sector-33 + midcap-22 is a non-pooled robustness arm. The chronological
split is frozen at development through 2019-12-31 and evaluation from 2020-01-01.
Workstream 2 is next.

1. **Data consolidation (complete).** FNSPID provides breadth/history; LSEG provides
   recent Reuters metadata/body text for separate robustness. Never pool the source
   regimes.
2. **Story filtering and weighting.** Publisher/source weighting, plus rules or a
   classifier separating genuinely new breaking news from repetition/syndication and from
   routine scheduled corporate reporting. Includes exploratory analysis of the daily
   distribution of scores within a company-day. FNSPID publisher/story-family fields are
   unavailable on the current checkpoint grain; do not invent proxies.
3. **Aggregation.** Compare ways of collapsing many stories into one signal: mean of hard
   labels (the current rule), mean continuous score, median, trimmed mean, negative
   share, dispersion, strongest-event selection, attention weighting, decayed state.
4. **Sentiment surprise.** A larger, better-controlled retry of the 2026-07-17 NO-GO
   pilot, stripping out both the market return and the general sentiment level
   (cross-sectional daily mean and the firm's own trailing baseline).
5. **Story-type conditioning.** Whether the signal should be scaled by event type, using
   the existing 12-type taxonomy, under an explicitly declared multiplicity family.
6. **Earnings-date effect.** An LSEG scheduled-results calendar now exists locally under
   `final_experiments/data/earnings/`; its mapping, title-filter and coverage caveats are
   binding until validated.

### Scope notes

- **De-prioritise VaR / ES tail risk.** The FNSPID tail-risk factorial is finished and
  registered (news arrival matters, semantics do not). Cite it; do not extend it.
- Neural nets are in scope for one narrow purpose: learning the trade/no-trade threshold
  for a sentiment signal, potentially per stock or per sector, in place of a fixed
  no-trade band. Fit development-only, compare against the fixed band, and prefer the
  large FNSPID sample over the ~85-session LSEG development window.

### Still binding in this phase

Speed does not relax honesty. Freeze downstream fitting/model selection before opening
evaluation outcomes; use date-clustered or block-bootstrap inference; report costs and
break-even where portfolio outcomes appear; keep licensed LSEG/FNSPID text local and
gitignored; report nulls as results. `final_experiments/outputs/` is gitignored - commit
figures and tables only when they are aggregate and licence-safe.

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
