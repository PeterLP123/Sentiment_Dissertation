# FNSPID earnings calendar (LSEG)

Scheduled earnings / results-release dates for the FNSPID coherent cohort
(574 symbols, 2011-01-01 through 2023-12-31), pulled from a live LSEG
Workspace desktop session.

## Canonical files

| File | Use |
| --- | --- |
| `fnspid_earnings_calendar_2011_2023_quarterly.csv` | **Recommended for Workstream 6** — Q/FY/H releases only |
| `fnspid_earnings_calendar_2011_2023.csv` | All issuer-matched RES events (includes Progressive-style monthly) |
| `symbol_ric_map.csv` | ticker → LSEG RIC |
| `earnings_events_raw.csv` | Raw LSEG dump before filters |
| `earnings_events_title_rejected.csv` | Dropped predecessor / same-ticker contamination |
| `manifest.json` | Counts and conventions |
| `checkpoints/` | Per-batch resume shards |

CSV / JSON artifacts are **gitignored** (licensed LSEG data). Re-fetch with:

```bash
.venv/bin/python scripts/fetch_fnspid_earnings_lseg.py
```

Requires Workspace running and signed in. Existing checkpoints are reused.

## Session mapping (frozen for this gather)

| `timing_flag` | Rule |
| --- | --- |
| `BMO` | trade that XNYS session open |
| `AMC` | first XNYS session strictly after `report_date` |
| `during_market` | same session |
| `unknown` (missing time) | conservative `next_session` |

`Event Start Time` is treated as UTC and converted to America/New_York before
the BMO/AMC cutoffs (09:30 / 16:00 local).

## Quality notes

- LSEG often returns predecessor-bank or same-ticker earnings under a surviving
  RIC. Rows are kept only when the Event Title matches the issuer phrase from
  the RIC `DocumentTitle`.
- Zero-event breakdown (84 of 574 cohort symbols, classified from the RIC map
  `DocumentTitle` and raw/rejected row counts — see
  `final_experiments/00_data_inventory` §8): 35 ETF/fund (genuinely no
  earnings), 22 delisted/acquired (expected attrition for a 2011–2023
  cohort), 10 tickers mis-resolved by symbol conversion onto futures/options
  roots (`BK`, `FL`, `GOL`, `HA`, `HBI`, `MDC`, `NS`, `SOL`, `TRI`, `TUP`),
  and 17 equity/ADR names dropped by the issuer-title filter — 15 with a full
  raw RES history (`AEM`, `B`, `BG`, `BTI`, `BXP`, `CVE`, `DEO`, `GOLD`,
  `HIG`, `MDRX`, `NAT`, `SLB`, `SU`, `TTEK`, `VOD`), mostly name-change or
  parenthetical-phrase failures. Treat the last two groups as recoverable
  coverage holes, not as "no earnings occurred."
- The fetch script's issuer-phrase rule was hardened after this build
  (parenthetical qualifiers stripped from the `DocumentTitle` phrase) and
  symbol conversion now flags non-equity resolutions. **Re-fetch when
  Workspace is available** to close the recoverable holes.
- Panel-builder flags: 17 duplicated `(symbol, report_date)` rows and
  multi-entity issuers (e.g. `ENB`, 72 events) — dedupe on
  `(symbol, report_date)` at panel build; 8 weekend `report_date`s; 202
  `during_market` rows map to `same_session`, which under open-to-open
  returns pulls pre-news trading into the event window — declare a rule.
