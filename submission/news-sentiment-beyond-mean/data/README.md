# Local data mount

No research data are committed here. FNSPID and LSEG/Reuters inputs are licensed or too large for Git.

Place authorised local inputs under this directory, or create local symlinks to the corresponding files in the protected source archive. The notebooks also consume staged intermediates under `experiments/generated/`; their manifests record hashes of the historical inputs used for the committed results.

Minimum source families:

- FNSPID scored firm-day news and adjusted-open prices;
- SPY or S&P 500 market price series used by the abnormal-return and HAR analyses;
- LSEG sector-33 price panels;
- pinned FinBERT LSEG firm-day aggregates for the backward and recent blocks;
- local earnings calendar for the earnings-exclusion robustness check.

The public Ken French daily three-factor file under `data/factors/` is the exception: it is small, unlicensed news, and hashed in the Notebook 85 specification.

Do not add FNSPID or LSEG input files to Git. See `docs/data_and_licensing.md` for the exact policy.
