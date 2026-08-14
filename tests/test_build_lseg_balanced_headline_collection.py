from __future__ import annotations

import pandas as pd
import pytest

from final_experiments.lib.backward_validation import LsegCollectionCoverageAudit
from scripts.build_lseg_balanced_headline_collection import (
    BalancedCoverageError,
    validate_balanced_coverage,
)


def _audit(*, final_complete: bool = True) -> LsegCollectionCoverageAudit:
    summary = pd.DataFrame(
        [
            {
                "collection_id": "first",
                "manifest_status": "completed",
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-02T00:00:00Z",
                "unsafe_pagination_reasons": "",
                "all_windows_completed": True,
            },
            {
                "collection_id": "second",
                "manifest_status": "completed",
                "start": "2024-01-02T00:00:00Z",
                "end": "2024-01-03T00:00:00Z",
                "unsafe_pagination_reasons": "",
                "all_windows_completed": final_complete,
            },
        ]
    )
    ledger = pd.DataFrame(
        [
            {
                "collection_label": collection,
                "collection_id": collection,
                "symbol": symbol,
                "window_start": date,
                "complete": complete,
            }
            for collection, date, complete in (
                ("first", "2024-01-01T00:00:00Z", True),
                ("second", "2024-01-02T00:00:00Z", final_complete),
            )
            for symbol in ("AAA", "BBB")
        ]
    )
    return LsegCollectionCoverageAudit(summary=summary, company_date_completion=ledger)


def test_validate_balanced_coverage_accepts_rectangular_complete_panel() -> None:
    result = validate_balanced_coverage(_audit())

    assert result.companies == 2
    assert result.dates == 2
    assert result.company_date_cells == 4
    assert result.complete_company_date_cells == 4
    assert result.symbols == ("AAA", "BBB")
    assert result.source_collection_ids == ("first", "second")


def test_validate_balanced_coverage_rejects_incomplete_company_date() -> None:
    with pytest.raises(BalancedCoverageError, match="incomplete company-date windows"):
        validate_balanced_coverage(_audit(final_complete=False))
