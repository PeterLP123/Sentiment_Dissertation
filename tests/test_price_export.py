import csv
import json
from pathlib import Path

import pytest

from sentiment_benchmark.lseg_source import LsegCollectionConfig, LsegCompanyConfig
from sentiment_benchmark.price_export import PriceExportError, export_lseg_prices
from sentiment_benchmark.prices import PriceRow


def _config(tmp_path: Path) -> LsegCollectionConfig:
    return LsegCollectionConfig(
        collection_id="prices",
        start="2025-06-26T00:00:00Z",
        end="2026-06-26T00:00:00Z",
        raw_output_root=tmp_path / "raw",
        derived_output_root=tmp_path / "derived",
        companies=(
            LsegCompanyConfig("DLB", "Dolby", "DLB.N", "R:DLB.N", ("Dolby",)),
            LsegCompanyConfig("LFUS", "Littelfuse", "LFUS.O", "R:LFUS.O", ("Littelfuse",)),
        ),
    )


class FakeProvider:
    def fetch(self, symbols, start, end):
        assert tuple(symbols) == ("DLB", "LFUS")
        assert (start, end) == ("2025-06-20", "2026-07-10")
        return [
            PriceRow(symbol, "2025-06-20", 10.0, 11.0, 9.0, 10.5, 100.0, False)
            for symbol in symbols
        ]


def test_export_lseg_prices_writes_panel_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "prices.csv"

    result = export_lseg_prices(
        _config(tmp_path),
        start="2025-06-20",
        end="2026-07-10",
        output=output,
        provider=FakeProvider(),
    )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert {row["symbol"] for row in rows} == {"DLB", "LFUS"}
    assert manifest["counts"]["rows_by_symbol"] == {"DLB": 1, "LFUS": 1}
    assert manifest["rics"] == {"DLB": "DLB.N", "LFUS": "LFUS.O"}
    assert manifest["file"]["sha256"]


def test_export_lseg_prices_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "prices.csv"
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(PriceExportError, match="refusing to overwrite"):
        export_lseg_prices(
            _config(tmp_path),
            start="2025-06-20",
            end="2026-07-10",
            output=output,
            provider=FakeProvider(),
        )
