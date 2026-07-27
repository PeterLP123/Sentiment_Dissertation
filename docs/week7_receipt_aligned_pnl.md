# Week 7 receipt-aligned P/L

## Scope

This Week 7 export applies the supervisor's receipt-date convention to the
current exploratory strategy, `recent-news-midcap-finbert-event-v1-dfdd88113a5f`.
The frozen strategy result is not overwritten.

For an interval entered at `session` and exited at `next_session`, gross P/L is
reported on `next_session`. This is the portfolio equivalent of
`Price(exit) - Price(entry)`. Every trading date in the strategy's session
spine is present; dates without a realised interval P/L contain zero.

The `portfolio_pnl_matrix.csv` output follows that gross receipt-date definition
exactly. Transaction costs and net-cash accounting are outside this Week 7
deliverable.

## Result

| Model | Trading dates | P/L receipt dates | Zero slots | Gross P/L | Receipt P/L Sharpe |
| --- | ---: | ---: | ---: | ---: | ---: |
| FinBERT | 252 | 62 | 190 | $188,851.03 | 1.472 |

The Sharpe uses all 252 aligned trading dates, including zero slots, and is
annualised as `sqrt(252) * mean(daily P/L) / sample standard deviation`.

These are arithmetic USD P/L summaries. They should not be substituted for the
frozen source report's compounded return or its return-based Sharpe. The source
ledger resets its USD NAV at the development/evaluation boundary; the Week 7
export preserves that ledger and only changes the reporting date of gross P/L.

## Outputs

Generated files are local and gitignored under:

`results/week7_pnl/recent-news-midcap-finbert-event-v1-dfdd88113a5f-receipt-aligned/`

- `portfolio_pnl_matrix.csv`: requested date-by-model P/L array.
- `daily_pnl_receipt_aligned.csv`: long-form accounting audit.
- `performance.csv`: gross P/L total and Sharpe calculation.
- `cumulative_pnl_by_receipt_date.png`: cumulative receipt-date P/L figure.
- `summary.md` and `manifest.json`: human- and machine-readable provenance.

The generated manifest verifies the immutable source ledger SHA-256 as
`1bd11490c579bba9cde12ee22c17861916c125bc9c18cf24c16d84670708d4a6`.

## Reproduce

From the repository root:

```bash
.venv/bin/python scripts/build_week7_receipt_pnl.py
```

The focused accounting tests are:

```bash
.venv/bin/python -m pytest -q tests/strategy_research/test_receipt_pnl.py
```
