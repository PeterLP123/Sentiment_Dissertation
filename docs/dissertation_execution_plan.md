# Dissertation Execution Plan

Last updated: 2026-07-12

This is the code-repository execution plan for the refocused dissertation. The Obsidian vault's `03 Dissertation Plan.md` is the cross-project master; this document translates it into repository changes, artifacts, tests, gates, and writing dependencies.

## Outcome contract

Working title: **Beyond the Mean: Cross-Model Agreement and Post-News Return Resolution**

Primary research question:

> Does cross-model agreement—assessed against graded human annotation agreement—add out-of-sample information about firm-level post-news abnormal returns beyond mean sentiment and the initial price reaction?

The critical path has three empirical stages:

1. benchmark competence and PhraseBank agreement validation;
2. a timestamp-valid LSEG event panel;
3. a held-out comparison of strongest-scorer, mean-only, and mean-plus-agreement models.

Profitability, G-theory, reliability-aware sizing, prompt factorials, futures, and writer–scorer analysis are not completion criteria.

## Worktree protection

The repository currently contains unrelated modified and untracked funded-portfolio work in the CLI/trading stack. Before dissertation-core implementation:

1. inspect and deliberately commit/park that work, or create a separate `codex/dissertation-core` worktree;
2. do not mix it into the core research commits;
3. keep each dissertation deliverable independently testable and reviewable.

## Critical path

| Deliverable | Due | Repository outcome | Scientific/writing dependency |
| --- | --- | --- | --- |
| **C0 — design lock** | 14 Jul | Protocol, config identities, primary outcome, split and fallback recorded | Introduction/methods promises become stable |
| **C1 — corpus + price feasibility** | 19 Jul | Canonical corpus, full price panel, 100-event CAR dry run, gate report | Final market or measurement-only chapter route |
| **C2 — PhraseBank validity** | 18 Jul | Registered agreement-validation bundle | First complete results subsection |
| **C3 — four-scorer LSEG panel** | 23 Jul | Frozen cohort and complete score matrix | Final roster/cohort prose |
| **C4 — held-out event study** | 27 Jul | Nested predictive comparison and explanatory outputs | Main results chapter |
| **C5 — result freeze** | 31 Jul | Reproduced manifests, generated tables/figures, no new estimands | Results/discussion finalisation |
| **C6 — dissertation QA** | 8 Aug | Assets/build/count/reference checks pass | Complete supervisor PDF 9–10 Aug |

## C0 — Freeze the core design

Record before formal LSEG calls:

- scorer roster and exact model IDs/versions;
- one prompt ID/hash, temperature 0, one sample;
- event unit and timestamp-to-session rule;
- market benchmark and estimation window;
- primary `CAR(+1,+5)` outcome excluding the initial reaction;
- `+1` and `+10` sensitivity outcomes;
- development/evaluation boundary;
- strongest-scorer, mean-only, and mean-plus-agreement model formulas;
- primary held-out loss comparison and bootstrap plan;
- 19 July fallback rule.

Create a dated manifest/config before any hosted request. Holdout outcomes must not change the roster, prompt, agreement definition, or cut points.

## C1 — Canonical LSEG corpus and price panel

### Code targets

- Update `src/sentiment_benchmark/lseg_corpus.py` and `tests/test_lseg_corpus.py` so a verified auxiliary `csv/` export can coexist with canonical outputs while unrecognised non-empty derived content still causes a hard refusal.
- Add `src/sentiment_benchmark/price_panel.py`.
- Add `tests/test_price_panel.py`.
- Add `configs/lseg_us_sector_33_prices.toml`.
- Add CLI command `build-price-panel`.

### Data targets

- Canonical corpus rebuilt from `Data/collections/lseg_us_sector_33_6m/raw/lseg_us_sector_33_6m` with the current cleaner version and a complete manifest.
- Analysis cohort keeps the earliest eligible `story_family × symbol` revision and records all attrition.
- Price output: `Data/derived/prices/lseg_us_sector_33_event_study.csv` plus manifest.
- Price universe: retained stocks plus the verified broad-market series (`^GSPC`, using the confirmed LSEG alias) and enough history to support 120 estimation sessions, a 21-session gap, the initial event session, and ten subsequent sessions.

### Required dry run

Run 100 representative events end-to-end and produce:

- timestamp classification and assigned event session;
- initial abnormal reaction;
- post-event `CAR(+1,+1)`, `CAR(+1,+5)`, and `CAR(+1,+10)`;
- failure/attrition reasons;
- coverage by symbol and independent calendar date.

### Gate

Proceed with the market RQ only if:

- at least 80% of holdout events align to valid return windows;
- at least 25 symbols remain;
- audited relevance precision is at least 0.85;
- support across calendar dates is adequate for the declared date-block inference;
- the 100-event run verifies that the initial reaction is not included in subsequent CAR.

Otherwise activate the measurement-validity fallback and retain LSEG only as an exploratory case study.

## C2 — PhraseBank agreement validation

### Core roster

- ProsusAI/FinBERT;
- VADER;
- Cerebras `gemma-4-31b`;
- Cerebras `gpt-oss-120b`.

TF–IDF is optional because its completed benchmark predictions are out-of-fold, whereas LSEG use would require a separately fitted full-data model.

### Code targets

- Add `src/sentiment_benchmark/agreement_validation.py`.
- Add `tests/test_agreement_validation.py`.
- Add CLI command `analyze-agreement-validity`.

### Inputs and validation

- `Data/derived/labeled/financial_sentiment_v2.csv`;
- registered FinBERT/VADER benchmark predictions;
- registered Gemma/GPT-OSS benchmark predictions;
- join using stable row identity and verify sentence/content hashes;
- restrict the human-tier test to the 4,836 PhraseBank rows.

Primary item agreement is the fraction of exactly agreeing model pairs. It remains a descriptive model-output property; do not rename it reliability or trust.

### Artifact bundle

Write to `results/agreement_validation/phrasebank_core_v1/`:

- `item_metrics.csv`;
- `tier_summary.csv`;
- `trend_tests.csv`;
- `model_error_by_tier.csv`;
- `leave_one_model_out.csv`;
- `agreement_by_tier.png`;
- `summary.md`;
- `manifest.json`.

Report tier means/intervals, Spearman trend, gold-class/text-length-adjusted sensitivity, consensus and individual-model error, risk–coverage/selective accuracy, and leave-one-model-out results. Exact complete coverage, hashes, model IDs, and deterministic replay are engineering acceptance criteria; the scientific result may be null.

## C3 — Lean LSEG scoring

The old crossed design makes approximately 147,500 calls and is no longer on the critical path. The core requires roughly 6,000 hosted calls for two LLMs over the development/evaluation cohort, with deterministic FinBERT and VADER scores assembled into the same item panel.

### Code/config targets

- Add `configs/dissertation_core_scoring.toml` with the two LLMs, one target-company label prompt, temperature 0, and one sample.
- Update `src/sentiment_benchmark/corpus_scoring.py` and its tests to support a one-prompt, one-sample run with no facet subset; preserve `configs/crossed_scoring.toml` unchanged for optional work.
- Add `src/sentiment_benchmark/corpus_baseline_scoring.py` and tests for FinBERT/VADER scoring on the identical cohort.
- Add validated `assemble-score-panel` logic/CLI that rejects duplicate identities, model drift, content mismatch, and partial coverage beyond the declared tolerance.

### Artifact bundle

Write to `results/scoring/lseg_us_sector_33_core_v1/`:

- `llm_scores.jsonl`;
- `baseline_scores.jsonl`;
- `scores.jsonl`;
- `coverage.csv`;
- `manifest.json`.

Acceptance requires a frozen dry-run identity list, model/version/prompt/content hashes, at least 98% success per scorer and at least 95% complete four-scorer item coverage. Do not change the prompt or roster after viewing evaluation outcomes.

## C4 — Refocused event study

### Code targets

- Refactor `src/sentiment_benchmark/l2_event_study.py`.
- Expand `tests/test_l2_event_study.py` with synthetic timestamp, initial/post-CAR, aggregation, split-leak and bootstrap cases.
- Leave `src/sentiment_benchmark/l3_reliability.py` untouched as optional work.

### Event and return rules

- Convert `version_created` from UTC to `America/New_York`.
- Assign same-session treatment only when publication precedes the declared close rule; otherwise use the next session.
- Treat the event-session abnormal return as `initial_reaction`.
- Exclude that session from subsequent `CAR(+1,+5)`.
- Aggregate multiple eligible stories to one `symbol × event_session` observation before inference.
- Retain daily-data intraday contamination as an explicit limitation.

### Nested comparisons

Development-fit models:

1. strongest individual scorer + initial reaction;
2. ensemble mean sentiment + initial reaction;
3. ensemble mean + continuous agreement + mean×agreement + initial reaction.

Standardisation and any fixed choices use development data only. The primary held-out estimand is weighted `MSE(model 3) - MSE(model 2)`; a negative value favours the agreement-augmented model. Report MAE, out-of-sample R-squared, and a date-block bootstrap interval. A secondary coefficient table may use company/date-clustered uncertainty. Date fixed effects must not create unseen evaluation categories in the predictive model.

### Artifact bundle

Write to `results/l2/lseg_us_sector_33_core_v1/`:

- `item_metrics.csv`;
- `event_panel.csv`;
- `development_coefficients.csv`;
- `holdout_predictions.csv`;
- `model_comparison.csv`;
- `bootstrap.csv`;
- `robustness.csv`;
- `figures/`;
- `summary.md`;
- `manifest.json`.

Include a full attrition table and all null results. The mean×agreement sign can describe strengthening, attenuation, continuation, or reversal; it is not evidence of investor crowding.

## C5 — Registration, exports, and freeze

Register these run families in `experiments/manifest.toml`:

- `phrasebank-agreement-core-v1`;
- `lseg-core-scoring-v1`;
- `lseg-agreement-event-study-v1`.

Each record must contain Git commit, input/config/model/prompt hashes, command, split identity, output hashes, and deviations. Export final LaTeX tables/figures into `dissertation/generated/`; final result values must not be typed manually.

Verification sequence:

```bash
pytest -q <focused core tests>
pytest -q
ruff check .
mypy src/sentiment_benchmark
sentiment-bench validate-data
cd dissertation
make assets
make count
make pdf
```

Before the 31 July freeze, replay the core results into a fresh output directory and compare manifests/artifacts. After freeze, permit only bug fixes, declared robustness, exact reproduction, and regenerated assets.

## Writing deliverables in this repository

The current scaffold reports 3,399 chapter words, but most files contain prompts/placeholders. The target is seven chapters and 9,800–10,100 main-text words:

| Chapter | Target |
| --- | ---: |
| Introduction | 800–900 |
| Literature Review | 1,700–1,900 |
| Data and Provenance | 1,200–1,400 |
| Methodology | 1,700–1,900 |
| Results | 2,200–2,500 |
| Discussion and Limitations | 1,200–1,400 |
| Conclusion | 400–600 |

Writing is Peter-authored under the supervisor's no-AI-prose rule. Code-generated tables/figures and planning comments do not count as prose.

Repository writing checkpoints:

- **14 Jul:** target structure compiles; title/RQ aligned; L3/L4 excluded from core.
- **18 Jul:** Data and fixed Methods first drafts; 3,000–3,500 substantive words.
- **23 Jul:** Literature and Data complete; benchmark/agreement assets inserted; 4,500–5,000 words.
- **27 Jul:** Methodology complete and Results shell populated; 6,000–6,500 words.
- **31 Jul:** Results synchronized to frozen evidence; 7,500–8,000 words.
- **7 Aug:** 9,800–10,100 words, abstract, project summary, appendices, bibliography, AI-use log.
- **8 Aug:** no placeholders/todos/invented examples; full build and visual PDF QA.
- **9–10 Aug:** complete supervisor PDF.

## Nice-to-haves

An extension can start only after core reproduction on 31 July, at least 7,500 substantive words, no core evidence placeholders, and no risk to the supervisor deadline. At most one enters the main text.

Priority order:

1. one fixed after-cost agreement-conditioned portfolio comparison;
2. TF–IDF fifth scorer;
3. prompt order/paraphrase variants and stochastic self-consistency;
4. one powered sector or event-type interaction;
5. training-cutoff placebo;
6. legacy threshold/index-futures replication;
7. G-theory reliability and position sizing;
8. writer–scorer Echo analysis;
9. wider universes, countries, intraday data, or persona dispersion.

The old crossed/L3 workflow remains available for future work but is explicitly not part of the core delivery contract.
