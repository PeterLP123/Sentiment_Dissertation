# Architecture

## LSEG Research Path

```mermaid
flowchart LR
    W["Workspace desktop session"] --> R["Data/news/lseg_<id><br/>immutable raw checkpoints"]
    R --> C["Data/derived/lseg/<id><br/>verified clean revisions"]
    C --> O["Ollama primary + secondary<br/>VADER + FinBERT"]
    O --> D["TradingDecision<br/>buy / sell / hold"]
    D --> P["next observed session<br/>gross + net returns"]
```

Collection and cleaning are separate commands so a corpus can be rebuilt and audited without another licensed API call. Native story and revision identifiers cross the trading boundary; URL-based web records retain their existing representation. Sentiment records and trading decisions are separate artifacts so model behavior can be inspected independently from policy thresholds. Python owns the orchestration because network and model inference dominate latency; a C++ component would not improve those boundaries.

This project keeps source data, external article sourcing, model execution, scoring, and export evidence separated so dissertation results can be reproduced and audited.

## The Problem

Sentiment benchmark projects can become difficult to defend when they mix raw data, cleaned data, live API behavior, run results, and analysis exports in one place. The main failure modes are:

- The source dataset gets silently edited.
- API behavior drifts without recorded run context.
- Prompt changes are compared as if they were the same experiment.
- Generated article corpora are mistaken for labeled benchmark data.
- Results are reported without enough metadata to reproduce the row selection, model route, prompt, machine, or runtime environment.

## The Approach

```mermaid
flowchart LR
    D["Default clean dataset<br/>financial_sentiment_v2.csv"] --> V["Validation<br/>dataset summary"]
    V --> R["Run config<br/>prompt hash, seed, provider"]
    R --> P["Provider adapter"]
    P --> O["OpenRouter"]
    P --> L["Ollama"]
    R --> S["SQLite or Turso/libSQL store<br/>responses + metrics"]
    M["Machine/runtime metadata"] --> S
    S --> A["Analysis commands<br/>results, compare, agreement"]
    S --> E["Export bundle<br/>run.json, metrics, statistics, figures"]
    T["Tavily"] --> N["Data/news<br/>unlabeled derived corpora"]
    NA["NewsAPI"] --> N
    N --> TR["Provider-neutral trading runner<br/>screen, score, align prices"]
    TR --> TE["Data/derived/trading + results/trading"]
```

Core boundaries:

| Boundary | Responsibility |
| --- | --- |
| `Data/derived/labeled/financial_sentiment_v2.csv` | Default provenance-clean labeled benchmark dataset. |
| `Data/data.csv` | Immutable legacy Kaggle source with documented merge corruption. |
| `src/sentiment_benchmark/dataset.py` | Dataset loading, validation, duplicate/conflict metadata. |
| `src/sentiment_benchmark/providers.py` | Chooses OpenRouter or Ollama client from provider settings. |
| `src/sentiment_benchmark/runner.py` | Executes model runs and stores responses. |
| `src/sentiment_benchmark/metrics.py` | Computes classification metrics and comparison statistics. |
| `src/sentiment_benchmark/storage.py` | Owns database schema and persistence for SQLite or native libSQL/Turso. |
| `src/sentiment_benchmark/libsql_backend.py` | Adapts the native `libsql` client to the SQLite-like store interface. |
| `src/sentiment_benchmark/runtime_metadata.py` | Captures machine id/label, git state, Python version, OS/platform, database backend, and timezone for new runs. |
| `src/sentiment_benchmark/exporter.py` | Writes reproducible run export bundles. |
| `src/sentiment_benchmark/news_source.py` | Sources Tavily article corpora without touching benchmark labels. |
| `src/sentiment_benchmark/newsapi_source.py` | Pages through NewsAPI discovery results and writes provenance-rich snippet corpora. |
| `src/sentiment_benchmark/trading_strategy.py` | Merges provider records, screens target relevance, scores sentiment, aligns sessions, and exports event returns. |
| `src/sentiment_benchmark/tui.py` + `tui_*.py` | Interactive Textual interface over the same services: an app shell (`tui.py`) that composes per-feature mixins (`tui_run`, `tui_results`, `tui_queue`, `tui_models`, `tui_monitor`, `tui_news`, `tui_baselines`) on a shared `tui_base` foundation, with `tui_format`/`tui_screens` helpers. |

## Why Two Scoring Scopes Exist

The default v2 dataset has zero conflicting duplicate groups, so `primary` and `all` currently contain the same rows. The two-scope design remains because legacy `Data/data.csv` and future derived datasets can contain conflicting duplicates.

The project therefore stores both:

- `primary`: excludes rows from conflicting duplicate groups and is used for headline comparisons.
- `all`: includes every selected row and is used for audit and ambiguity analysis.

This gives the dissertation a stable comparison scope without mixing legacy merge corruption into headline accuracy.

## Why Prompt Hashes Are Stored

Prompt wording, label order, few-shot demonstrations, and output format all affect model behavior. The benchmark stores prompt content by hash so two runs can be compared only when their prompt context is known.

Few-shot demonstrations are included in the prompt hash. A k-shot run is therefore a distinct reproducible prompt, even if it uses the same model and dataset rows as a zero-shot run.

## Why Providers Are Abstracted

OpenRouter and Ollama expose different Python clients and metadata. The benchmark normalizes them behind a small provider boundary so the runner can ask for the same operation:

```text
classify(model_id, prompt, sentence, request_settings) -> response record
```

The provider boundary keeps the dissertation storage format consistent while still recording provider route, base URL, latency, token usage, cost metadata when available, and raw response fragments.

## Why Runtime Metadata Is Stored

Runs can now be produced on more than one machine and synced through Turso/libSQL. A run therefore records both a queryable `machine_id`/`machine_label` and an `environment_json` snapshot. The machine id is either explicitly configured with `SENTIMENT_BENCH_MACHINE_ID` or derived as a short hash from a stable OS value; raw OS machine identifiers are not stored.

The environment snapshot records package version, git commit/branch/dirty state, Python version, OS/platform, database backend, and local timezone. This makes a shared run history easier to audit when local Gemma behavior differs across machines.

## Why News And Trading Outputs Are Separate

Tavily and NewsAPI supply article source material, not gold benchmark labels. Writing raw provider corpora under `Data/news`, merged screening data under `Data/derived/trading`, and strategy evidence under `results/trading` prevents accidental mutation of the labeled benchmark and keeps external-data provenance explicit. Trading sentiment is stored outside the benchmark database because it has no hidden human label.

The Tavily corpus writer records:

- Query configuration.
- Fetch timestamp.
- Request IDs and usage metadata when returned.
- URL-normalized deduplication.
- Extraction status and errors.
- Raw Tavily payload fragments.
- Schema version.

NewsAPI records the equivalent query, pagination, publication timestamp, publisher, and raw result fragment. The trading runner then canonicalizes URLs, removes exact headline syndications, assigns exchange-local news dates, and records every automatic screening decision. Its run manifest hashes the config and generated evidence, and it refuses to overwrite a completed run created from a different config.

## Trade-Offs

| Choice | Benefit | Cost |
| --- | --- | --- |
| SQLite local store by default | Simple, inspectable, no service dependency. | Not ideal for sharing one run history across machines. |
| Optional Turso/libSQL backend | Shared run history with per-machine local replicas. | Requires Python 3.12 setup, Turso credentials, and careful secret handling. |
| Ignored generated outputs | Keeps git clean and avoids large artifacts. | Formal outputs must be exported and registered deliberately. |
| Provider abstraction | Same benchmark flow works for OpenRouter and Ollama. | Lowest common denominator interface hides provider-specific advanced controls. |
| Primary/all scopes | Separates headline comparison from ambiguity audit. | Readers must understand which scope is being discussed. |
| Tavily corpora unlabeled by default | Prevents accidental weak labeling. | A later labeling workflow is required before articles can become benchmark rows. |
| Next-session-open trading entry | Permits all news on day D without look-ahead. | It differs from a literal close-to-close teaching example. |

## Alternatives Considered

The current code favors conservative research traceability over automation. It does not automatically transform Tavily articles into a `Sentence,Sentiment` dataset because there is no labeling protocol yet. It does not edit the source CSV because derived data should carry provenance. It does not hard-code a single provider because the dissertation may compare API-hosted models with local Gemma-family models. It does not copy old local SQLite runs into Turso automatically because a shared dissertation history should include only deliberate, reviewable runs.
