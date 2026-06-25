# Week 4 LSEG/Eikon Task

Use this folder for the Week 4 handoff task:

> Get the LSEG/Eikon scripts working so news data can be downloaded and distributed. Download 50-100 stories per company at a time. Well-known companies should produce roughly 5 stories per day. If Workspace access is unavailable, work on cleaning scripts that remove boilerplate text.

## What To Do

1. Open LSEG Workspace Desktop and sign in.
2. Use the project Python environment, not the system Python.
3. Run a one-row check for one company.
4. Download 50-100 rows per company/date window.
5. Review the CSV before distributing: check `story_error` and spot-check the `story` column.
6. Share only with people who are allowed to receive LSEG licensed story text.

`LSEG.py` is intentionally close to the provided script. It opens a Workspace session, gets headlines, loops over each `storyId`, downloads each full story, and writes a CSV. Cleaning is not done here; do that in a separate step later.

## One Company

```bash
cd /Users/peterprendergast/Documents/Sentiment_Dissertation
.venv/bin/python week_4_tasks/LSEG.py \
  --company MSFT \
  --count 1 \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

## One Company Download

After the one-row check works, run 50-100 rows:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --company MSFT \
  --count 50 \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

Outputs are written to `week_4_tasks/outputs/` by default.

## Other Companies

The script has common RICs built in for `MSFT`, `AAPL`, `NVDA`, `AMZN`, `GOOGL`, `META`, `TSLA`, and `JPM`.

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --company AAPL \
  --count 50 \
  --start "2026-06-24T00:00:00" \
  --end "2026-06-24T23:59:59"
```

You can also pass a direct LSEG query:

```bash
.venv/bin/python week_4_tasks/LSEG.py \
  --query "R:MSFT.O and Language:LEN" \
  --count 50
```

## Notes

- Generated outputs are ignored by Git. Do not commit licensed story CSVs by accident.
- `week4_companies.csv` is just a reference list of companies/RICs to work through manually.
- If you do not have LSEG/Eikon login access, use received CSV outputs for cleaning in a separate script later.
