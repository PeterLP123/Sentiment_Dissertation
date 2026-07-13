# Mid-Cap Reuters Headline Strategy Runbook

Last updated: 2026-07-13

This is the deadline-bounded exploratory extension requested for completion by
Wednesday 15 July 2026. It keeps the existing 33-company collection and Week 6
evidence untouched.

## Frozen scope

- Universe: 22 US mid-cap companies, two per sector, defined in
  `configs/lseg_us_midcap_22_1y.toml`.
- News interval: `2025-06-26T00:00:00Z` to `2026-06-26T00:00:00Z`.
- Source: Reuters headlines (`Source:RTRS`) only.
- Story bodies: disabled. This run tests headline sentiment, not full-text
  sentiment.
- Retrieval: seven-day windows, 3 requests/second, hard stop at 8,000 requests.
- Scoring: one pinned local Ollama model, exact direct-company headline filter,
  at most 15,000 unique headlines.
- Baseline comparison: the existing headline-value VADER/rule-based scorers.
- Evaluation: chronological development/evaluation split; no return-driven
  changes to the company list, text filter, prompt or model tag.

Generated LSEG text, local scores and results remain ignored local artifacts.
Do not commit or redistribute them.

## 0. Environment

Workspace Desktop must be running and signed in. Set a reachable Ollama endpoint
and exact installed model tag:

```bash
export OLLAMA_HOST="http://<reachable-host>:11434"
export OLLAMA_MODEL="<exact-installed-tag>"

curl --fail --silent --show-error "$OLLAMA_HOST/api/tags"
```

The model tag must be present in that response. Do not silently substitute a
different tag after scoring begins.

## 1. Validate all RICs and entitlements

```bash
.venv/bin/sentiment-bench lseg-news-check \
  --config configs/lseg_us_midcap_22_1y.toml \
  --all-companies
```

Investigate any zero-headline company before collection. Replace a bad RIC only
before the collection starts; a changed universe requires a new collection ID.

## 2. Collect Reuters headlines

```bash
caffeinate -is .venv/bin/sentiment-bench fetch-lseg-news \
  --config configs/lseg_us_midcap_22_1y.toml
```

The command checkpoints every page. The 8,000-request ceiling fails safely with
instructions to resume; it never spills into story retrieval because
`fetch_story_bodies = false`.

Expected output:

```text
Data/collections/lseg_us_midcap_22_1y/raw/lseg_us_midcap_22_1y/
```

## 3. Freeze the local scoring population

Count the exact Reuters/direct-company population without opening Ollama:

```bash
.venv/bin/sentiment-bench score-headlines \
  --collection-root Data/collections/lseg_us_midcap_22_1y \
  --provider ollama \
  --model "$OLLAMA_MODEL" \
  --source-code NS:RTRS \
  --direct-company-only \
  --max-population 15000 \
  --dry-run
```

If this exceeds 15,000, freeze a reduced sector-balanced universe under a new
collection ID. Do not use `--limit` to create the final backtest population.

## 4. Smoke test and complete local scoring

Use one immutable output file so the full run resumes the 100-item smoke test:

```bash
caffeinate -is .venv/bin/sentiment-bench score-headlines \
  --collection-root Data/collections/lseg_us_midcap_22_1y \
  --provider ollama \
  --ollama-host "$OLLAMA_HOST" \
  --model "$OLLAMA_MODEL" \
  --prompt-id finance_calibrated_label_only \
  --source-code NS:RTRS \
  --direct-company-only \
  --max-population 15000 \
  --concurrency 1 \
  --limit 100 \
  --output Data/collections/lseg_us_midcap_22_1y/derived/headline_scores_ollama_primary.csv

caffeinate -is .venv/bin/sentiment-bench score-headlines \
  --collection-root Data/collections/lseg_us_midcap_22_1y \
  --provider ollama \
  --ollama-host "$OLLAMA_HOST" \
  --model "$OLLAMA_MODEL" \
  --prompt-id finance_calibrated_label_only \
  --source-code NS:RTRS \
  --direct-company-only \
  --max-population 15000 \
  --concurrency 1 \
  --output Data/collections/lseg_us_midcap_22_1y/derived/headline_scores_ollama_primary.csv
```

Stop if the smoke test contains malformed responses or if projected completion
misses Wednesday morning. A smaller exact model tag is a design change and must
be recorded before restarting in a new score file.

## 5. Fetch the matching LSEG price panel

```bash
.venv/bin/sentiment-bench fetch-lseg-prices \
  --config configs/lseg_us_midcap_22_1y.toml \
  --start 2025-06-20 \
  --end 2026-07-10 \
  --output Data/derived/prices/lseg_us_midcap_22_1y.csv
```

The adjacent manifest records the RIC mapping, coverage, config hash, output
hash and price-return convention.

## 6. Build signals and event-level diagnostics

```bash
.venv/bin/sentiment-bench analyze-headline-value \
  --collection-root Data/collections/lseg_us_midcap_22_1y \
  --prices Data/derived/prices/lseg_us_midcap_22_1y.csv \
  --output-dir Data/collections/lseg_us_midcap_22_1y/derived/headline_value_ollama_v1 \
  --source-code NS:RTRS \
  --direct-company-only \
  --horizons 1,3,5,7 \
  --transaction-cost-bps-per-side 10 \
  --llm-scores Data/collections/lseg_us_midcap_22_1y/derived/headline_scores_ollama_primary.csv
```

## 7. Run the funded chronological evaluation

The scorer ID in `daily_signals.csv` is `llm/<exact model tag>`:

```bash
.venv/bin/sentiment-bench analyze-week6-pnl \
  --signals Data/collections/lseg_us_midcap_22_1y/derived/headline_value_ollama_v1/daily_signals.csv \
  --prices Data/derived/prices/lseg_us_midcap_22_1y.csv \
  --scorer "llm/$OLLAMA_MODEL" \
  --run-id us_midcap22_ollama_20260715 \
  --development-fraction 0.75 \
  --thresholds 0,0.25,0.50 \
  --holding-periods 1,3,5,7 \
  --transaction-cost-bps-per-side 10
```

Treat the all-company portfolio as primary. The low-correlation subset is a
predeclared sensitivity, not a replacement when the primary result is weak.

## Completion checks

Before reporting:

- LSEG manifest status is `completed` and request count is at most 8,000.
- All price symbols have non-empty coverage.
- The final scoring file contains no model-tag drift.
- Scoring success is at least 98%; failures and exclusions are reported.
- At least 14 companies have usable signals; otherwise report event-level
  evidence and do not force a funded-portfolio conclusion.
- Existing 33-company and Week 6 artifacts remain unchanged.
- Result interpretation remains exploratory and includes nulls, costs,
  attrition and the headline-only limitation.
