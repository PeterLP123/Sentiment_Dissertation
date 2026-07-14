# Week 6 trade decision explorer

The local web app links every position-changing execution to its price path,
signal date, contributing headline associations, transaction cost, and realised
next-session profit or loss. Generated files remain under `results/` because
they embed licensed LSEG headline text and must not be redistributed.

## Frozen trading rule

1. For each company and news date, classify every matched headline association
   with the deterministic headline lexicon and take the association-weighted
   arithmetic mean.
2. A mean strictly above `+0.10` targets `+1` (long); a mean strictly below
   `-0.10` targets `-1` (short); values on the boundaries, inside the band, or
   missing target `0` (flat).
3. The signal is actioned at the next observed trading-session close. The new
   position earns only the following close-to-close return, so no same-session
   return is credited.
4. Hold for five trading sessions unless a newer signal supersedes the current
   position and restarts the holding clock. Incomplete cross-split and trailing
   holds are omitted.
5. Allocate `£100,000 / 33 = £3,030.30` to each stock. Charge 10 basis points
   per side on the absolute position change; a direct `-1` to `+1` reversal
   therefore costs two sides.

The `0.10` threshold and five-session hold were selected using development data
only from 12 threshold/holding-period candidates. Evaluation results were not
used to select the rule.

## Rebuild and run locally

Set `RUN_DIR` to the frozen Week 6 P&L run, then execute:

```bash
RUN_DIR=/absolute/path/to/lseg_headline_sentiment_all_20260713_final

.venv/bin/python scripts/build_week6_trade_dashboard.py \
  --run-dir "$RUN_DIR" \
  --signals Data/collections/lseg_us_sector_33_6m/derived/headline_value_analysis_lseg_priced/sweep_ws/daily_signals.csv \
  --prices Data/collections/lseg_us_sector_33_6m/derived/headline_value_analysis_lseg_priced/sweep_ws/prices.csv \
  --raw-dir Data/collections/lseg_us_sector_33_6m/raw/lseg_us_sector_33_6m \
  --output results/week6_trade_explorer/week6-trade-explorer.html \
  --standalone-output results/week6_trade_explorer/index.html

cd results/week6_trade_explorer
python3 -m http.server 8765 --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/`. The build fails rather than publishing a
dashboard when the reconstructed headline counts or mean scores disagree with
the frozen signal artifacts.
