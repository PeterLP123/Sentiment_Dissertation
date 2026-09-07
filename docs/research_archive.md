# Research Archive

This page maps the broader research workspace retained around the completed dissertation. The canonical submitted study is [`submission/news-sentiment-beyond-mean/`](../submission/news-sentiment-beyond-mean/README.md); archive documents describe earlier designs or additional exploratory work and should not override the final package.

## Exploratory notebook programme

[`final_experiments/`](../final_experiments/README.md) preserves the full notebook-first closing programme, its frozen specifications, decision records, invalidations, and licence-safe aggregate outputs. Its long chain includes the selected negative-story-share analysis and later bounded investigations, including:

- Notebook 75: the one-shot 2020–2023 temporal evaluation;
- Notebooks 79–84: robustness, economic-value, prompt, and structured-score diagnostics.

Separate local novelty-conditioning and already-known-story-share studies also
use numbers 85 and 86. Those unpublished additions are outside this release and
must not be confused with the final package's notebooks of the same numbers.

The [experiment ledger](../final_experiments/EXPLORATORY_EXPERIMENT_LEDGER.md) records the status and evidential boundary of exploratory runs. The [historical plan](../final_experiments/plan.html), [research protocol](research_protocol.md), and [execution plan](dissertation_execution_plan.md) preserve the decisions that governed that work.

## Frozen results and protocols

| Area | Records |
| --- | --- |
| FNSPID tail risk | [Variant comparison](fnspid_tail_risk_variant_comparison.md) |
| Baseline trading design | [Sentiment trading baseline](sentiment_trading_baseline.md) · [Trading preregistration](trading_preregistration.md) |
| Mid-cap Reuters study | [Protocol](recent_news_midcap_finbert_protocol.md) · [Result](recent_news_midcap_finbert_strategy.md) |
| Sector-33 robustness | [Protocol](recent_news_sector33_finbert_protocol.md) · [Result](recent_news_sector33_finbert_result.md) |
| Signal persistence | [Protocol and result](recent_news_sentiment_persistence_protocol.md) |
| Portfolio combination | [Sentiment signal portfolio](sentiment_signal_portfolio.md) |
| Model comparisons | [Local Gemma study](../reports/local_gemma4_headline_return_study.md) · [Full-population FinBERT/VADER study](../reports/week6_finbert_vader_full_population_test.md) |

These records remain valid for their declared samples and dates. They are not additional confirmatory evidence for the submitted primary claim.

## Collection and operations

| Guide | Scope |
| --- | --- |
| [LSEG to Ollama pipeline](lseg_ollama_pipeline.md) | Licensed-news collection, cleaning, scoring, and validation. |
| [UCL GPU labeling](ucl_gpu_labeling.md) | Shared-GPU model labeling. |
| [UCL GPU SSH handoff](ucl_gpu_ssh_handoff.md) | Gateway and workstation workflow. |
| [Mid-cap deadline runbook](midcap_deadline_runbook.md) | Completed July 2026 collection run. |
| [Historical strategy pipeline](strategy_research_pipeline.md) | Point-in-time portfolio and evaluation infrastructure. |

## Data and source-control boundary

Licensed article corpora, model checkpoints, caches, databases, and large generated outputs remain local or ignored. The public sentiment benchmark and its documented source fixtures are tracked, alongside code, specifications, manifests, documentation, notebooks and aggregate evidence. Do not manually edit the default benchmark dataset at `Data/derived/labeled/financial_sentiment_v2.csv`; rebuild it with `scripts/build_labeled_dataset.py` if its provenance logic changes.
