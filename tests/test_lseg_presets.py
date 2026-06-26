from pathlib import Path

import pytest

from sentiment_benchmark.lseg_presets import (
    LSEG_PRESET_WEEK4_EIGHT,
    lseg_config_from_preset,
    write_lseg_config,
)
from sentiment_benchmark.lseg_source import LsegConfigurationError, load_lseg_collection_config


def test_lseg_week4_preset_writes_exact_company_rics(tmp_path: Path) -> None:
    config = lseg_config_from_preset(
        preset=LSEG_PRESET_WEEK4_EIGHT,
        collection_id="test_collection",
        start="2025-06-26T00:00:00Z",
        end="2026-06-26T00:00:00Z",
    )
    output = write_lseg_config(config, tmp_path / "lseg.toml")

    loaded = load_lseg_collection_config(output)

    assert loaded.collection_id == "test_collection"
    assert loaded.window_days == 1
    assert {company.symbol: company.ric for company in loaded.companies} == {
        "AAPL": "AAPL.O",
        "AMZN": "AMZN.O",
        "GOOGL": "GOOGL.O",
        "JPM": "JPM.N",
        "META": "META.O",
        "MSFT": "MSFT.O",
        "NVDA": "NVDA.O",
        "TSLA": "TSLA.O",
    }
    assert all(company.news_query == f"R:{company.ric} and Language:LEN" for company in loaded.companies)


def test_lseg_config_writer_refuses_overwrite(tmp_path: Path) -> None:
    config = lseg_config_from_preset(
        preset=LSEG_PRESET_WEEK4_EIGHT,
        collection_id="test_collection",
        start="2025-06-26T00:00:00Z",
        end="2026-06-26T00:00:00Z",
    )
    output = write_lseg_config(config, tmp_path / "lseg.toml")

    with pytest.raises(LsegConfigurationError, match="refusing to overwrite"):
        write_lseg_config(config, output)

    write_lseg_config(config, output, overwrite=True)
