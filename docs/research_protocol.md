# Research Protocol

Last updated: 2026-06-07

Operational documentation:

- [Getting started](getting_started.md)
- [CLI reference](cli_reference.md)
- [Model providers](model_providers.md)
- [Tavily news sourcing](news_sourcing.md)
- [Results and exports](results_and_exports.md)
- [Architecture](architecture.md)

## Purpose

This repository supports a dissertation study of financial sentiment analysis. It has two co-primary aims:

1. Benchmark large language models and non-LLM baselines on a shared financial sentiment classification task.
2. Investigate sentiment ambiguity through duplicate-label conflict, model disagreement, prompt sensitivity, and within-model self-consistency.

The protocol is intended to make formal experiments reproducible, comparable, and appropriately limited in interpretation.

## Research Questions

RQ1. How accurately do selected LLMs classify financial text into `positive`, `negative`, and `neutral` labels compared with established non-LLM baselines?

RQ2. How robust are model rankings and per-class outcomes to prompt design, label ordering, and instruction paraphrase?

RQ3. Do rows with conflicting human labels or high model disagreement show higher signs of sentiment ambiguity than rows without label conflict?

RQ4. How do cost, latency, invalid-output rate, API error rate, and completion-token budget affect the practical use of LLMs for financial sentiment classification?

## Hypotheses

H1. Stronger LLMs and domain-specialized models will outperform trivial and lexical baselines on macro-F1, balanced accuracy, and MCC, not only raw accuracy.

H2. Prompt perturbations will change measured performance for at least some models, so prompt sensitivity should be reported alongside headline benchmark scores.

H3. Conflicting duplicate rows and rows with higher inter-model disagreement will show higher ambiguity indicators, such as self-consistency entropy or lower agreement statistics.

H4. Cost, latency, parse failures, and API errors will create meaningful tradeoffs between model quality and operational practicality.

These hypotheses describe expected empirical patterns. They do not imply causal claims about market behavior, investor decision-making, or the true psychological sentiment of text authors.

## Dataset Policy

The source dataset is `Data/data.csv`. Treat it as immutable source material. Do not clean, relabel, deduplicate, or overwrite it in place.

The dataset has two required columns:

- `Sentence`: financial text to classify.
- `Sentiment`: gold label, expected to be one of `positive`, `negative`, or `neutral`.

Experiments use two scoring scopes:

- `primary`: excludes duplicate sentence groups where the same sentence appears with conflicting labels. This is the preferred scope for headline model comparisons.
- `all`: includes all selected rows, including conflicting duplicates. This is an audit scope for sensitivity and ambiguity analysis.

Pilot experiments use a balanced sample of 30 primary rows per class by default. Full experiments may evaluate every row, subject to cost and runtime constraints. Any derived dataset or cleaned view must document its source file, cleaning steps, label mapping, split logic, and random seed.

News articles sourced through Tavily are unlabeled derived source material. They may support future corpus construction, annotation, or qualitative context, but they must not be treated as benchmark labels until a separate labeling protocol is defined and documented.

## Model Selection

Formal comparisons may include:

- OpenRouter-accessed LLMs selected for relevance, availability, cost, and diversity of provider/model family.
- Ollama-hosted local models, including Gemma-family models running on a separate desktop PC, when provider route, model tag, host configuration, and sampling settings are recorded.
- `baseline/majority`, as a lower-bound sanity check.
- `baseline/tfidf_logreg`, as a lightweight supervised lexical baseline evaluated out-of-fold.
- `baseline/vader`, as a lexicon-based sentiment baseline.
- `baseline/finbert`, as a financial-domain transformer baseline.

Model lists must be recorded in the experiment registry and exports. LLM runs should record provider route, model ID, prompt ID/hash, temperature, retry policy, concurrency, token limits, run date, machine label/id, package version, Python version, git commit, and database backend. External model behavior may drift over time, and local model behavior may vary by machine, so results should be interpreted as observations from the recorded run context.

## Default Experimental Settings

Unless a formal experiment states otherwise, use these defaults from the benchmark package:

- Dataset path: `Data/data.csv`
- Labels: `positive`, `negative`, `neutral`
- Random seed: `42`
- Pilot sample: `30` rows per class from the primary scope
- LLM temperature: `0.0` for deterministic benchmark runs
- Self-consistency temperature: greater than `0.0`, with the value and sample count recorded
- Default maximum completion tokens: `64`
- Reasoning-model maximum completion-token budget: `2048`
- Concurrency: `1`
- Retries: `3` for LLM API runs

Few-shot runs must record `few_shot_k`, `few_shot_seed`, selected demonstration logic, and prompt hash. Demonstration rows must not overlap with evaluation rows.

## Metrics and Statistical Tests

Classification quality should be reported with more than one metric because the dataset is imbalanced.

Primary classification metrics:

- Accuracy
- Macro-F1
- Weighted-F1
- Balanced accuracy
- Matthews correlation coefficient
- Per-class precision, recall, F1, and support
- Confusion matrices
- Invalid-output and API-error counts

Uncertainty and comparison statistics:

- Bootstrap confidence intervals for accuracy and macro-F1
- Paired McNemar tests for model comparisons on shared rows
- Cohen's kappa for pairwise model agreement
- Fleiss' kappa for multi-model agreement
- Krippendorff's alpha for nominal agreement
- Prompt sensitivity summaries across label-order and paraphrase variants
- Self-consistency entropy, majority fraction, and majority-vote accuracy for repeated stochastic samples

Where many pairwise comparisons are made, results should be interpreted cautiously and correction or grouping strategy should be documented in the dissertation text.

## Interpretation Limits

The dataset is financial-domain text and should not be generalized to non-financial sentiment without further evidence.

The label distribution is neutral-heavy, so raw accuracy can overstate practical performance. Macro-F1, balanced accuracy, MCC, and per-class scores should be emphasized.

Conflicting duplicate labels are treated as evidence of annotation uncertainty or dataset inconsistency, not as proof that any individual label is wrong.

LLM outputs are sensitive to prompts, model versions, provider routing, token budgets, and API behavior. Exports and registry entries must be used to contextualize every result.

Self-consistency entropy and model disagreement are proxies for ambiguity. They support interpretation when aligned with error analysis, but they do not directly measure human uncertainty.

No experiment in this repository should be framed as causal evidence about financial markets, future asset performance, or investor behavior.

## Required Experiment Record

Every formal dissertation run or run family should have an entry in `experiments/manifest.toml` and an export under `results/exports/` when available. At minimum, record:

- Run ID or run family ID
- Commit SHA used for the run
- Dataset path and SHA-256 hash
- Prompt ID and prompt hash
- Model IDs
- Provider route, machine label/id, package version, Python version, and git commit/dirty state
- Mode, seed, sample logic, temperature, and token settings
- Metrics emphasized in the dissertation
- Cost and latency notes when available
- Interpretation notes and dissertation section relevance
