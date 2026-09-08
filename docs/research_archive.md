# Research archive

The completed study is in [`submission/news-sentiment-beyond-mean/`](../submission/README.md).
The directories below record the earlier research, including superseded designs,
invalidated runs and exploratory findings. Use their stated samples and dates when
interpreting a result.

## Notebooks and decisions

[`final_experiments/`](../final_experiments/README.md) contains the notebook sequence,
frozen specifications and aggregate outputs. Notebook 75 records the one-shot
2020–2023 evaluation. Notebooks 79–84 cover robustness, economic value, prompts and
structured-score diagnostics.

Separate unpublished novelty-conditioning and already-known-story-share studies in
the local workspace also use numbers 85 and 86. They are outside this release and must
not be merged by number with the final package's notebooks.

The [experiment ledger](../final_experiments/EXPLORATORY_EXPERIMENT_LEDGER.md) records
run status and limitations. The [historical plan](../final_experiments/plan.html),
[research protocol](research_protocol.md) and [execution plan](dissertation_execution_plan.md)
retain the decisions made during the work. They do not reopen the submitted design.

## Earlier results and protocols

| Study | Records |
| --- | --- |
| FNSPID tail risk | [Variant comparison](fnspid_tail_risk_variant_comparison.md) |
| Baseline trading design | [Sentiment trading baseline](sentiment_trading_baseline.md) · [Trading preregistration](trading_preregistration.md) |
| Mid-cap Reuters study | [Protocol](recent_news_midcap_finbert_protocol.md) · [Result](recent_news_midcap_finbert_strategy.md) |
| Sector-33 robustness | [Protocol](recent_news_sector33_finbert_protocol.md) · [Result](recent_news_sector33_finbert_result.md) |
| Signal persistence | [Protocol and result](recent_news_sentiment_persistence_protocol.md) |
| Portfolio combination | [Sentiment signal portfolio](sentiment_signal_portfolio.md) |
| Model comparisons | [Local Gemma study](../reports/local_gemma4_headline_return_study.md) · [FinBERT/VADER study](../reports/week6_finbert_vader_full_population_test.md) |

These studies are part of the research history, not additional confirmatory tests of
the dissertation's primary claim. Consult the ledger for corrections and invalidations.

## Collection and operations

| Guide | Scope |
| --- | --- |
| [LSEG to Ollama pipeline](lseg_ollama_pipeline.md) | Licensed-news collection, cleaning and scoring |
| [UCL GPU labelling](ucl_gpu_labeling.md) | Model labelling on shared GPUs |
| [UCL SSH access](ucl_gpu_ssh_handoff.md) | Gateway and workstation workflow |
| [Mid-cap deadline runbook](midcap_deadline_runbook.md) | Completed July 2026 collection |
| [Historical strategy pipeline](strategy_research_pipeline.md) | Portfolio construction and evaluation |

## Directory guides

| Path | Contents |
| --- | --- |
| [`scripts/`](../scripts/README.md) | Collection, scoring, validation and reporting commands |
| [`configs/`](../configs/README.md) | Dated experiment parameters and provider settings |
| [`reports/`](../reports/README.md) | Result narratives, manifests and compact evidence |
| [`dissertation/`](../dissertation/README.md) | Superseded manuscript workspace |

Historical paths remain stable for provenance. Some tools require licensed inputs,
external services or UCL access. Raw news corpora, checkpoints and large generated
outputs stay local. The tracked public benchmark has separate
[provenance and rebuild instructions](dataset_card.md).
