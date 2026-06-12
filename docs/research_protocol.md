# Research Protocol

Last updated: 2026-06-12

Operational documentation:

- [Getting started](getting_started.md)
- [CLI reference](cli_reference.md)
- [Model providers](model_providers.md)
- [Tavily news sourcing](news_sourcing.md)
- [Results and exports](results_and_exports.md)
- [Architecture](architecture.md)

## Purpose

This repository is the measurement engine for the dissertation **"Beyond the Score"**
(direction confirmed with the supervisor on 2026-06-12; full research design lives in
the Obsidian vault, `wiki/thesis/Beyond the Score Plan.md`).

The dissertation's central claim: a sentiment pipeline that compresses each day's news
into one scalar S_t and trades on it is using the wrong object. When multiple models,
prompts, and stochastic samples score the same text, they produce a *distribution* of
readings, and properties of that distribution carry economically meaningful information
that the scalar destroys.

The research is organized into layers, and this repository feeds all of them:

| Layer | Distributional property | Repository's role |
| --- | --- | --- |
| **L1** (baseline) | Mean (S_t) | Per-article sentiment scoring feeding daily aggregation and the supervisor's threshold trading strategy |
| **L2** (consensus → crowding) | Cross-model agreement | Full-roster scoring of timestamped news items; agreement statistics per item |
| **L3** (reliability → sizing) | Reliability (G-coefficient) | Crossed-design runs (item × model × prompt × sample) supplying the G-study variance decomposition |
| **L4** (stretch, gated) | Score shift on machine text | Writer×scorer scoring matrix, if the gate opens at M2 |

The repository's original two aims — benchmarking LLMs against non-LLM baselines, and
investigating sentiment ambiguity via duplicate-label conflict, model disagreement,
prompt sensitivity, and self-consistency — survive intact as *instruments*: the
benchmark comparison is the extraction-comparison milestone (M3), and the ambiguity
toolkit provides L2's ambiguity proxy and L3's measurement facets.

The protocol is intended to make formal experiments reproducible, comparable, and
appropriately limited in interpretation.

## Research Questions

Dissertation-level questions (from the Beyond the Score plan):

- **RQ1 (L1, baseline):** Can a daily aggregated sentiment signal, traded via a tuned
  threshold rule, generate profit on index futures vs price-only and buy-and-hold
  baselines?
- **RQ2 (L2, consensus → crowding):** Does the level of agreement among independent
  models about a news item predict how its initial price reaction resolves — reversal
  after high-consensus readings vs drift after low-consensus readings?
- **RQ3 (L3, reliability → sizing):** Does a trading rule that knows the measurement
  error of its sentiment signal — shrinking unreliable scores and sizing positions by
  signal-to-noise — outperform the fixed-threshold rule on the same signal?
- **RQ4 (L4, stretch):** Do sentiment scorers shift systematically on machine-written
  versions of the same news?

Operational measurement questions answered inside this repository (these support the
RQs; they are not the dissertation's headline questions):

- **OQ1.** How accurately do the selected LLMs classify financial text into `positive`,
  `negative`, and `neutral` labels compared with established non-LLM baselines?
- **OQ2.** How robust are model rankings and per-class outcomes to prompt design, label
  ordering, and instruction paraphrase? (Feeds the L3 prompt facet.)
- **OQ3.** Do rows with conflicting human labels or high model disagreement show higher
  signs of sentiment ambiguity than rows without label conflict? (Validates the L2
  ambiguity proxy against human-disagreement ground truth.)
- **OQ4.** How do cost, latency, invalid-output rate, API error rate, and
  completion-token budget affect the practical use of LLMs for this pipeline?

## Hypotheses

The dissertation's pre-registered hypothesis families are stated in the Beyond the
Score plan and are fixed in advance:

- **H2a–H2c** (L2): reversal after high-agreement readings; drift after low-agreement
  readings; effects concentrated where machine consensus occurs despite textual
  ambiguity.
- **H3a–H3c** (L3): measurement variance is structured (model and prompt facets
  material); item-level variance correlates with human-agreement tiers; shrinkage and
  signal-to-noise sizing weakly dominate the fixed threshold on risk-adjusted P/L.

Benjamini–Hochberg correction is applied across the H2/H3 family, and event-study
horizons are fixed in advance at 1, 5, and 10 days.

Measurement-level expectations for the repository's own benchmark and ambiguity runs
(numbered E to avoid collision with the pre-registered families):

**E1.** Stronger LLMs and domain-specialized models will outperform trivial and lexical
baselines on macro-F1, balanced accuracy, and MCC, not only raw accuracy.

**E2.** Prompt perturbations will change measured performance for at least some models,
so prompt sensitivity should be reported alongside headline benchmark scores.

**E3.** Conflicting duplicate rows and rows with higher inter-model disagreement will
show higher ambiguity indicators, such as self-consistency entropy or lower agreement
statistics.

**E4.** Cost, latency, parse failures, and API errors will create meaningful tradeoffs
between model quality and operational practicality.

These expectations describe empirical patterns in the measurement layer. They do not
imply causal claims about market behavior, investor decision-making, or the true
psychological sentiment of text authors; market-facing inference belongs to the L1–L3
designs (chronological splits, leakage controls, pre-registered tests) and is bounded
by the interpretation limits below.

## Dataset Policy

### Labeled benchmark set

The source dataset is `Data/data.csv`. Treat it as immutable source material. Do not
clean, relabel, deduplicate, or overwrite it in place.

The dataset has two required columns:

- `Sentence`: financial text to classify.
- `Sentiment`: gold label, expected to be one of `positive`, `negative`, or `neutral`.

Provenance was verified on 2026-06-12 (M1-5; full findings in the dataset card):
`data.csv` is PhraseBank `Sentences_66Agree.txt` + FiQA 2018, plus 514 corrupt rows —
every PhraseBank negative duplicated under a wrong `neutral` label. The 514 conflicting
duplicate groups are therefore **merge corruption, not human disagreement**, and must
not be used as an ambiguity signal. The genuine graded human-disagreement ground truth
for the L2 ambiguity proxy (H2c) and the L3 tier check (H3b) is PhraseBank's
annotator-agreement tier, carried as `pb_agreement_tier` in the provenance-clean
rebuild `Data/derived/labeled/financial_sentiment_v2.csv`
(`scripts/build_labeled_dataset.py`). Formal runs should prefer the rebuild — on
`data.csv` the primary scope contains zero PhraseBank negatives — with the final choice
frozen at M3-1. License (CC BY-NC-SA) and LLM training-data contamination caveats are
recorded at ingest and acknowledged in the write-up.

Experiments use two scoring scopes:

- `primary`: excludes duplicate sentence groups where the same sentence appears with
  conflicting labels. This is the preferred scope for headline model comparisons.
- `all`: includes all selected rows, including conflicting duplicates. This is an audit
  scope for sensitivity and ambiguity analysis.

Pilot experiments use a balanced sample of 30 primary rows per class by default. Full
experiments may evaluate every row, subject to cost and runtime constraints. Any
derived dataset or cleaned view must document its source file, cleaning steps, label
mapping, split logic, and random seed.

### News corpus

News articles sourced through Tavily are unlabeled, timestamped derived source
material. Their dissertation role is to supply event-time text for L1 daily aggregation
and the L2 event study; ingestion must remain source-agnostic so FNSPID or GDELT can
substitute if Tavily coverage proves inadequate (tripwire 28 Jun: no working
timestamped source → drop the L2 futures arm, run the L2 single-stock arm on FNSPID
historical, promote L4). News articles must not be treated as benchmark labels until a
separate labeling protocol is defined and documented.

### Price data

Daily prices come from Yahoo Finance: index futures (FTSE, DOW, Hang Seng) for L1 and
the L2 futures overlay, plus single-name constituents for the L2 event-study arm.
Timestamp alignment follows the news-before-decision rule (see Leakage below).

## Model Selection

Formal comparisons may include:

- OpenRouter-accessed LLMs selected for relevance, availability, cost, and diversity of
  provider/model family (4–5 hosted models, including Gemini given its role in the
  supervisor's prior work).
- Ollama-hosted local models, including Gemma-family models running on a separate
  desktop PC, when provider route, model tag, host configuration, and sampling settings
  are recorded.
- `baseline/majority`, as a lower-bound sanity check.
- `baseline/tfidf_logreg`, as a lightweight supervised lexical baseline evaluated
  out-of-fold.
- `baseline/vader`, as a lexicon-based sentiment baseline and non-LLM contrast in L2.
- `baseline/finbert`, as a financial-domain transformer baseline and non-LLM contrast
  in L2.

**Roster freeze.** The model roster is frozen before formal runs begin (M1-3/W3 at the
latest). Because L2 agreement statistics and the L3 crossed design depend on every
model scoring every item, adding a model after the freeze requires re-running
everything; additions are therefore prohibited without an explicit registry entry
recording the re-run.

Model lists must be recorded in the experiment registry and exports. LLM runs should
record provider route, model ID, prompt ID/hash, temperature, retry policy,
concurrency, token limits, run date, machine label/id, package version, Python version,
git commit, and database backend. External model behavior may drift over time, and
local model behavior may vary by machine, so results should be interpreted as
observations from the recorded run context.

## Default Experimental Settings

Unless a formal experiment states otherwise, use these defaults from the benchmark
package:

- Dataset path: `Data/data.csv`
- Labels: `positive`, `negative`, `neutral`
- Random seed: `42`
- Pilot sample: `30` rows per class from the primary scope
- LLM temperature: `0.0` for deterministic benchmark runs
- Self-consistency temperature: greater than `0.0`, with the value and sample count
  recorded
- Default maximum completion tokens: `64`
- Reasoning-model maximum completion-token budget: `2048`
- Concurrency: `1`
- Retries: `3` for LLM API runs

Few-shot runs must record `few_shot_k`, `few_shot_seed`, selected demonstration logic,
and prompt hash. Demonstration rows must not overlap with evaluation rows.

**Crossed-design requirement (L3).** Formal scoring runs are planned as a crossed
design — item × model × prompt-variant × stochastic sample — so that the same runs that
produce the extraction comparison (M3) also supply the G-study variance components.
Facet levels (which prompt variants, how many stochastic samples) must be fixed and
registered before the run.

**Calibration prerequisite (blocking).** A soft-label prompt variant (per-class
probability output) plus Brier score and expected calibration error (ECE) metrics must
be in the repository **before** any formal scoring run, so that runs are not paid for
twice. This is M1-7 and is an L3 prerequisite.

## Metrics and Statistical Tests

Classification quality should be reported with more than one metric because the dataset
is imbalanced.

Primary classification metrics:

- Accuracy
- Macro-F1
- Weighted-F1
- Balanced accuracy
- Matthews correlation coefficient
- Per-class precision, recall, F1, and support
- Confusion matrices
- Invalid-output and API-error counts

Calibration metrics (required once the soft-label variant lands):

- Brier score
- Expected calibration error (ECE)

Uncertainty, agreement, and comparison statistics:

- Bootstrap confidence intervals for accuracy and macro-F1
- Paired McNemar tests for model comparisons on shared rows
- Cohen's kappa for pairwise model agreement
- Fleiss' kappa for multi-model agreement
- Krippendorff's alpha for nominal agreement
- Prompt sensitivity summaries across label-order and paraphrase variants
- Self-consistency entropy, majority fraction, and majority-vote accuracy for repeated
  stochastic samples (within-model noise, kept separate from between-model disagreement
  by design)

Layer-level analysis (computed from repository outputs, reported in the dissertation):

- L2: per-item inter-model agreement (pairwise correlation, Fleiss' kappa on
  discretized labels, score variance); CAR(0,h) event-study sorts at h = 1, 5, 10 days
- L3: G-study variance components (mixed-effects), G-coefficient, per-item and per-day
  reliability; three-rule backtest comparison (fixed threshold vs shrinkage vs
  signal-to-noise sizing, plus an abstention variant)
- Benjamini–Hochberg correction across the pre-registered H2/H3 family; transaction-cost
  sensitivity on every P/L claim

Where additional pairwise comparisons are made outside the pre-registered family,
results should be interpreted cautiously and the correction or grouping strategy
documented in the dissertation text.

## Leakage and Split Policy

- Chronological train/test splits everywhere; no tuning on the test window. The
  threshold *L* and any sizing parameters are tuned on the training split only.
- News-before-decision timestamp rule: a trading decision may only use news with
  timestamps strictly before the decision point.
- Memorization caveat: models may have memorized outcomes for pre-training-cutoff text;
  this look-ahead risk is acknowledged and cited rather than instrumented.

## Interpretation Limits

The dataset is financial-domain text and should not be generalized to non-financial
sentiment without further evidence.

The label distribution is neutral-heavy, so raw accuracy can overstate practical
performance. Macro-F1, balanced accuracy, MCC, and per-class scores should be
emphasized.

Conflicting duplicate labels are treated as evidence of annotation uncertainty or
dataset inconsistency, not as proof that any individual label is wrong.

LLM outputs are sensitive to prompts, model versions, provider routing, token budgets,
and API behavior. Exports and registry entries must be used to contextualize every
result.

Self-consistency entropy and model disagreement are proxies for ambiguity. They support
interpretation when aligned with error analysis (and, where provenance allows,
validation against PhraseBank human-agreement tiers), but they do not directly measure
human uncertainty.

Backtest results are not deployable alpha. L1–L3 P/L claims are statements about the
recorded backtest context, reported with and without transaction costs; profitability
is not required for the layers to be informative, and publishable nulls ("machine
consensus carries no information about reaction resolution"; "reliability information
adds nothing tradable") are acceptable outcomes. No experiment in this repository
should be framed as causal evidence about financial markets, future asset performance,
or investor behavior.

## Required Experiment Record

Every formal dissertation run or run family should have an entry in
`experiments/manifest.toml` and an export under `results/exports/` when available. At
minimum, record:

- Run ID or run family ID
- Commit SHA used for the run
- Dataset path and SHA-256 hash
- Prompt ID and prompt hash
- Model IDs
- Provider route, machine label/id, package version, Python version, and git
  commit/dirty state
- Mode, seed, sample logic, temperature, and token settings
- Metrics emphasized in the dissertation
- Cost and latency notes when available
- Interpretation notes and dissertation section relevance
- For crossed-design runs: the registered facet levels (prompt variants, sample count)
  and which layer(s) the run feeds (L1 scoring, L2 agreement, L3 G-study, L4 matrix)
