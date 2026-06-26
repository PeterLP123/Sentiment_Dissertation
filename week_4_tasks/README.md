# Week 4 LSEG/Eikon Task

Use this folder for the Week 4 handoff task:

> Get the LSEG/Eikon scripts working so news data can be downloaded and distributed. Download 50-100 stories per company at a time. Well-known companies should produce roughly 5 stories per day. If Workspace access is unavailable, work on cleaning scripts that remove boilerplate text.

## Self-Contained Gist Files

The two files to share are:

- `LSEG.py`
- `clean_lseg_articles.py`

They are self-contained scripts. `LSEG.py` needs `pandas` and `lseg-data`; `clean_lseg_articles.py` needs `pandas` and `lxml`.

## What To Do

1. Open LSEG Workspace Desktop and sign in.
2. Use the project Python environment, not the system Python.
3. Run a one-row check for one company.
4. Download 50-100 rows per company/date window.
5. Review the CSV before distributing: check `story_error` and spot-check the `story_html` column.
6. Share only with people who are allowed to receive LSEG licensed story text.

For a balanced sample, use `--per-day-count` instead of a single whole-window `--count`. The script will make one LSEG headline request per calendar day, then combine the results into one CSV. The output filename still reflects the requested date range; use the `request_date`, `request_start`, and `request_end` columns to audit the actual daily requests.

## One Company

```bash
cd /Users/peterprendergast/Documents/Sentiment_Dissertation
.venv/bin/python week_4_tasks/LSEG.py \
  --company MSFT \
  --count 1 \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

## Provided-Script Style

With no company flag, the script behaves like the provided query-based script and defaults to `R:MSFT.O and Language:LEN`:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --count 1 \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

## One Company Download

After the one-row check works, run about five stories for one day:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --company MSFT \
  --per-day-count 5 \
  --dedupe-story-ids \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

Outputs are written to `week_4_tasks/outputs/` by default.

For a 14-day balanced sample, use a 14-day start/end range. With `--per-day-count 5`, this requests up to 70 headlines per company:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --company MSFT \
  --per-day-count 5 \
  --dedupe-story-ids \
  --start "2026-06-11T00:00:00" \
  --end "2026-06-24T23:59:59"
```

## Other Companies

The script has common RICs built in for `MSFT`, `AAPL`, `NVDA`, `AMZN`, `GOOGL`, `META`, `TSLA`, and `JPM`.

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --company AAPL \
  --per-day-count 5 \
  --dedupe-story-ids \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

You can also pass a direct LSEG query:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --query "R:MSFT.O and Language:LEN" \
  --per-day-count 5
```

To work through the built-in Week 4 company list, use all-company batch mode:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --all-companies \
  --per-day-count 5 \
  --dedupe-story-ids \
  --start "2026-06-11T00:00:00" \
  --end "2026-06-24T23:59:59"
```

## Clean Downloaded Articles

After reviewing the raw download CSVs, build derived cleaned CSVs:

```bash
.venv/bin/python week_4_tasks/clean_lseg_articles.py
```

By default, this reads `week_4_tasks/outputs/*.csv` and writes `week_4_tasks/outputs/cleaned/*_cleaned.csv` plus `cleaning_summary.csv`. Relative `--output-dir` paths are resolved under `week_4_tasks/`, not the shell cwd. It accepts both old raw files with a `story` column and newer raw files with a `story_html` column. The cleaned CSVs keep metadata, derive `story_type` when missing, add `clean_text`, `cleaning_quality`, character counts, `clean_text_sha256`, `article_story_type`, and `usable_article`, and drop raw HTML unless `--keep-raw-html` is passed. The script refuses to replace existing cleaned files unless `--overwrite` is passed.

The default keeps all rows so missing `webnews` rows and non-article `social` rows remain auditable. To write only usable cleaned article rows to `week_4_tasks/outputs/cleaned_articles/`, use:

```bash
.venv/bin/python week_4_tasks/clean_lseg_articles.py --articles-only --overwrite
```

Useful options:

```bash
.venv/bin/python week_4_tasks/clean_lseg_articles.py --ok-only
.venv/bin/python week_4_tasks/clean_lseg_articles.py --overwrite
.venv/bin/python week_4_tasks/clean_lseg_articles.py --input week_4_tasks/outputs/MSFT_20260611_20260624.csv
```

## Notes

- Generated outputs are ignored by Git. Do not commit licensed story CSVs by accident.
- `--all-companies` uses the company list embedded in `LSEG.py`, so the downloader is self-contained for sharing.
- `story_type` is derived from `storyId` so later cleaning can separate `newsroom`, `reuters`, `webnews`, and `social` rows.
- `webnews` rows often have no downloaded story text and `social` rows should usually be cleaned as a separate dataset from article stories.
- If you do not have LSEG/Eikon login access, use received CSV outputs for cleaning in a separate script later.
