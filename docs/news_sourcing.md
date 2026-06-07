# How To Source News Articles With Tavily

Use Tavily when you need current article source material from the internet. The feature searches for articles, optionally extracts full text, deduplicates URLs, and writes a timestamped derived corpus under `Data/news`.

![CLI Tavily fetch screenshot](assets/cli-news-fetch.svg)

## Prerequisites

- A Tavily account and API key.
- Project dependencies installed, including `tavily-python>=0.7`.
- `.env` available in the repository root.

Add:

```bash
TAVILY_API_KEY=your-tavily-key
TAVILY_PROJECT=sentiment-dissertation
```

`TAVILY_PROJECT` is optional and is passed through for project tracking when configured. The API key is never printed by the CLI or TUI.

## Check Connectivity

Run a minimal Tavily news search:

```bash
sentiment-bench news-check --query "financial markets" --max-results 1
```

The command prints a compact status table with record count, request ID when returned, response time, and usage metadata when Tavily provides it.

## Fetch Articles

Run:

```bash
sentiment-bench fetch-news --query "bank earnings sentiment" \
  --max-results 10 --time-range week --output-dir Data/news
```

Defaults:

| Option | Default | Notes |
| --- | --- | --- |
| `--topic` | `news` | Allowed: `news`, `finance`, `general`. |
| `--time-range` | `week` | Allowed: `day`, `week`, `month`, `year`. |
| `--search-depth` | `basic` | Allowed: `basic`, `advanced`, `fast`, `ultra-fast`. |
| `--extract` | enabled | Runs Tavily Extract on returned URLs. |
| `--max-results` | `10` | Valid range: `1` to `20`. |
| `--output-dir` | `Data/news` | Timestamped subdirectory is created per fetch. |

Use snippet-only mode when you want a quick provenance list without full extraction:

```bash
sentiment-bench fetch-news --query "central bank inflation outlook" \
  --max-results 5 --no-extract
```

Constrain domains:

```bash
sentiment-bench fetch-news --query "bank earnings sentiment" \
  --include-domain reuters.com --include-domain ft.com \
  --exclude-domain example.com
```

Constrain dates:

```bash
sentiment-bench fetch-news --query "market volatility" \
  --start-date 2026-06-01 --end-date 2026-06-07
```

## Output Files

Each fetch writes to a directory like:

```text
Data/news/tavily_news_20260607T120000Z_bank-earnings-sentiment/
```

| File | Purpose |
| --- | --- |
| `articles.jsonl` | One provenance-rich JSON record per article. Use this for reproducible processing. |
| `articles.csv` | Compact review table for spreadsheets and manual article screening. |
| `manifest.json` | Query parameters, fetched timestamp, counts, package/schema version, request IDs, usage, and failure counts. |

## Record Fields

`articles.jsonl` preserves fields only when Tavily returns them.

| Field | Meaning |
| --- | --- |
| `record_id` | `sha256(normalized_url)[:16]`; stable across repeated fetches of the same normalized URL. |
| `url` | Original URL returned by Tavily. |
| `normalized_url` | Lower-cased host, stripped fragment, normalized trailing path. |
| `title` | Article title when available. |
| `snippet` | Search result snippet. |
| `article_text` | Extracted full text when extraction succeeds. |
| `published_date` | Tavily-provided publication date when available. |
| `score` | Tavily relevance score when available. |
| `favicon` | Tavily-provided favicon URL when available. |
| `extraction_status` | `success`, `failed`, or `skipped`. |
| `extract_error` | Failure message when extraction fails. |
| `search_request_id` | Tavily search request ID when returned. |
| `extract_request_id` | Tavily extract request ID when returned. |
| `raw_search_result` | JSON-safe fragment of the original Tavily search result. |
| `raw_extract_result` | JSON-safe fragment of the Tavily extract result. |

## Data Handling Rules

Tavily corpora are unlabeled source material. They are not converted into `Sentence,Sentiment` benchmark rows automatically.

The source benchmark dataset is not modified:

```text
Data/data.csv   source dataset, unchanged
Data/news/      generated article corpora, ignored by git except .gitkeep
```

If you later label article snippets or full text for benchmarking, create a separate derived dataset and document the labeling protocol, inclusion rules, label mapping, and random seed.

## TUI Workflow

Open:

```bash
sentiment-bench tui
```

Go to tab `6` or select News. The News tab has:

- API check button.
- Query input.
- Topic selector.
- Time range selector.
- Max results input.
- Extract toggle.
- Output directory input.
- Status/log output.

The TUI uses the same Tavily client and output writer as the CLI.

## Verification

After a fetch:

```bash
find Data/news -maxdepth 2 -type f | sort
```

You should see `articles.jsonl`, `articles.csv`, and `manifest.json` under one timestamped directory.

Check the manifest without printing article bodies:

```bash
python -m json.tool Data/news/tavily_news_*/manifest.json | head -80
```

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `TAVILY_API_KEY is required` | `.env` missing or key not exported. | Add `TAVILY_API_KEY` to `.env`. |
| `max_results must be between 1 and 20` | Out-of-range input. | Choose a value from `1` to `20`. |
| `topic must be one of...` | Unsupported topic. | Use `news`, `finance`, or `general`. |
| Extracted text is empty | Source blocks extraction or Tavily returned no content. | Review `extraction_status`, `extract_error`, and `raw_extract_result`. |
| Too much generated data | Repeated broad fetches. | Use narrower queries, domain filters, date filters, and keep `Data/news` ignored unless a specific corpus is curated. |
