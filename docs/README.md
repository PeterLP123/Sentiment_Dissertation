# Documentation Home

![Project workflow overview](assets/research-workflow.svg)

This documentation is organized by reader task ([Diátaxis](https://diataxis.fr/)): the tutorial when you are new, how-to guides when you need to complete a job, references for exact options and schemas, and explanations for dissertation-ready rationale.

## The Full Map

| Document | Type | Best for |
| --- | --- | --- |
| [Getting started](getting_started.md) | Tutorial | First install, validation, pilot run, export, and news fetch. |
| [Model providers](model_providers.md) | How-to | OpenRouter setup and local/remote Ollama setup for Gemma models. |
| [News sourcing](news_sourcing.md) | How-to | Search, extract, package, and troubleshoot Tavily and NewsAPI corpora. |
| [LSEG to Ollama pipeline](lseg_ollama_pipeline.md) | How-to | Collect, clean, score, validate, and evaluate local-only Workspace news. |
| [Frozen LSEG analysis workflow](lseg_analysis_workflow.md) | How-to | The gated corpus → cohort → validation → crossed scoring → L2/L3 sequence. |
| [Trading pipeline](trading_pipeline.md) | How-to + explanation | The news-to-price pipeline: runs, effectiveness battery, sweeps, and pluggable strategies. |
| [TUI guide](tui_guide.md) | How-to | Interactive model, prompt, run, results, and news workflows. |
| [CLI reference](cli_reference.md) | Reference | Every command, option, default, and example. |
| [Results and exports](results_and_exports.md) | Reference | Storage backends, export files, metrics, LaTeX tables, and trading artifacts. |
| [Dataset card](dataset_card.md) | Reference | Dataset identity, provenance, label policy, limitations, and ethics. |
| [Architecture](architecture.md) | Explanation | Why source data, providers, runs, corpora, and exports are separated. |
| [Research protocol](research_protocol.md) | Explanation | "Beyond the Score" research questions, hypotheses, layers, and experiment rules. |
| [Week 4 pre-registration](week4_preregistration.md) | Explanation | The frozen confirmatory trading test: hypothesis, design, and decision rule. |

## Choose A Starting Point

| I need to... | Read this |
| --- | --- |
| Install the project and run the first benchmark. | [Getting started](getting_started.md) |
| Use Gemma or another local model running on a desktop PC. | [Model providers](model_providers.md) |
| Share run history across machines with Turso/libSQL. | [Results and exports](results_and_exports.md) |
| Fetch articles from Tavily or NewsAPI into `Data/news`. | [News sourcing](news_sourcing.md) |
| Batch-fetch and package a shareable Tavily source dataset. | [News sourcing](news_sourcing.md) |
| Build the licensed Workspace-to-local-model pipeline. | [LSEG to Ollama pipeline](lseg_ollama_pipeline.md) |
| Execute the frozen 33-company LSEG cohort, scoring, L2, and L3 workflow. | [Frozen LSEG analysis workflow](lseg_analysis_workflow.md) |
| Run, analyze, tune, or extend the trading strategy. | [Trading pipeline](trading_pipeline.md) |
| Understand the confirmatory Week 4 trading test. | [Week 4 pre-registration](week4_preregistration.md) |
| Work through the terminal UI. | [TUI guide](tui_guide.md) |
| Look up a command, option, or default. | [CLI reference](cli_reference.md) |
| Understand exported files and metrics. | [Results and exports](results_and_exports.md) |
| Explain the system design in a methods chapter. | [Architecture](architecture.md) and [Research protocol](research_protocol.md) |
| Cite or describe the benchmark dataset. | [Dataset card](dataset_card.md) |

## Project At A Glance

```mermaid
flowchart TB
    subgraph Source["Source material"]
        D["financial_sentiment_v2.csv<br/>labeled benchmark"]
        N["Tavily + NewsAPI corpora<br/>Data/news/*"]
        LS["LSEG Workspace corpus<br/>local-only, licensed"]
    end
    subgraph Execution["Execution"]
        CLI["sentiment-bench CLI"]
        TUI["sentiment-bench tui"]
        P["Provider adapter<br/>OpenRouter or Ollama"]
    end
    subgraph Evidence["Evidence"]
        DB["SQLite or Turso/libSQL<br/>run history"]
        E["results/exports/run_&lt;id&gt;/"]
        TR["results/trading/&lt;run-id&gt;/<br/>+ analysis + sweep"]
        L23["results/l2 + results/l3<br/>event study + reliability"]
        M["experiments/manifest.toml"]
    end
    D --> CLI
    N --> CLI
    LS --> CLI
    D --> TUI
    CLI --> P
    TUI --> P
    CLI --> DB
    TUI --> DB
    DB --> E
    CLI --> TR
    CLI --> L23
    E --> M
    TR --> M
    L23 --> M
```

## What Is Source-Controlled

Source-controlled files should explain how an experiment was produced. Local generated files hold the heavy or changing evidence.

| Source-controlled | Local/generated |
| --- | --- |
| Code under `src/` and tests under `tests/`. | SQLite run database and Turso local replica under `results/`. |
| `Data/derived/labeled/financial_sentiment_v2.csv` default benchmark dataset. | Export folders under `results/exports/` and trading evidence under `results/trading/`. |
| `Data/data.csv` legacy Kaggle source dataset. | Tavily corpora and licensed LSEG checkpoints under `Data/news/` or `Data/collections/`. |
| Prompt and run configs under `configs/`. | Local credentials in `.env`. |
| Experiment registry `experiments/manifest.toml`. | Temporary or ad hoc derived datasets outside the tracked labeled default. |

## Recommended Reading Order

1. [Getting started](getting_started.md)
2. [Model providers](model_providers.md)
3. [CLI reference](cli_reference.md)
4. [Results and exports](results_and_exports.md)
5. [Trading pipeline](trading_pipeline.md)
6. [Research protocol](research_protocol.md)
