# LSEG Week 4 News Download + Cleaning

Two small scripts:

- `LSEG.py` downloads LSEG Workspace headlines and full story HTML to CSV.
- `clean_lseg_articles.py` converts the downloaded story HTML into plain text and writes derived cleaned CSVs.

## Install

```bash
python -m pip install pandas lxml lseg-data
```

Open LSEG Workspace Desktop and sign in before downloading.

## Balanced Download

This asks for up to five headlines per calendar day:

```bash
python LSEG.py \
  --company MSFT \
  --per-day-count 5 \
  --dedupe-story-ids \
  --start "2026-06-11T00:00:00" \
  --end "2026-06-24T23:59:59"
```

For the built-in company list:

```bash
python LSEG.py \
  --all-companies \
  --per-day-count 5 \
  --dedupe-story-ids \
  --start "2026-06-11T00:00:00" \
  --end "2026-06-24T23:59:59"
```

Raw CSVs are written to `outputs/`.

## Clean Articles

Keep all rows for audit:

```bash
python clean_lseg_articles.py --overwrite
```

Write only usable article rows (`newsroom` or `reuters`, with valid clean text):

```bash
python clean_lseg_articles.py \
  --articles-only \
  --output-dir outputs/cleaned_articles \
  --overwrite
```

Cleaned CSVs include `clean_text`, `cleaning_quality`, `story_type`, `article_story_type`, and `usable_article`.
