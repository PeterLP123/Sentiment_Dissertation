# FNSPID adjusted-open return-convention correction

Recorded: 15 August 2026

The historical manifests for Notebooks 20, 21, 24 and 25 describe FNSPID or
SPY adjusted-open returns as split-adjusted but not dividend-adjusted. That
description is incorrect. The underlying FNSPID price archive was collected
from Yahoo Finance, and the code constructs
`adjusted_open = open * (adj_close / close)`. Yahoo defines adjusted close to
reflect applicable split and dividend multipliers, so the formula transfers
whichever adjustments are encoded in the archived ratio. It does not
independently reconstruct or verify corporate actions.

This is a metadata correction, not a numerical recomputation. The saved paths
already use the same `adjusted_open` values; the historical aggregate manifests
are retained unchanged so their hashes remain auditable. The source notebooks,
panel helper and manuscript now use the corrected wording. The estimand should
be read as a Yahoo adjusted-open proxy, not as an audited corporate-action
ledger, a directly observed cash-dividend return series or an exact executable
total return.

Primary verification:

- FNSPID reference scraper: <https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/main/data_scraper/stock_price_scraper/get_price_from_yahoo.py>
- Yahoo adjusted-close definition: <https://help.yahoo.com/kb/adjusted-close-sln28256.html>
