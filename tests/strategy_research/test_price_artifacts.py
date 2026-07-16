from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.strategy_research.price_artifacts import (
    StrategyPriceArtifactError,
    build_strategy_price_artifact,
)


def _source(tmp_path: Path, *, omit_last: bool = False) -> tuple[Path, Path]:
    panel = tmp_path / "lseg.csv"
    rows = [
        (symbol, session, value)
        for symbol, value in (("AAA", 100), ("BBB", 200))
        for session in ("2026-01-05", "2026-01-06", "2026-01-07")
    ]
    if omit_last:
        rows.pop()
    with panel.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("symbol", "session_date", "open", "high", "low", "close", "volume", "repaired"))
        for symbol, session, value in rows:
            writer.writerow((symbol, session, value, value + 1, value - 1, value, 1000, False))
    manifest = tmp_path / "lseg.manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "provider": "lseg",
                "return_convention": "split-adjusted price returns; dividends not back-adjusted",
                "symbols": ["AAA", "BBB"],
                "rics": {"AAA": "AAA.N", "BBB": "BBB.N"},
                "requested_start": "2026-01-05",
                "requested_end": "2026-01-07",
                "file": {"path": panel.name, "sha256": sha256_file(panel)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return panel, manifest


def test_build_strategy_price_artifact_is_hash_verified_and_complete(tmp_path: Path) -> None:
    source, source_manifest = _source(tmp_path)
    panel = tmp_path / "strategy.csv"
    manifest = tmp_path / "strategy.manifest.json"

    result = build_strategy_price_artifact(
        source,
        source_manifest,
        panel,
        manifest,
        expected_symbol_count=2,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert result.row_count == 6
    assert result.session_count == 3
    assert payload["split_adjusted"] is True
    assert payload["dividend_adjusted"] is False
    assert payload["files"]["price_panel"]["sha256"] == sha256_file(panel)
    assert panel.read_text(encoding="utf-8").splitlines()[0] == "symbol,session_date,adjusted_open"


def test_build_strategy_price_artifact_rejects_incomplete_grid(tmp_path: Path) -> None:
    source, source_manifest = _source(tmp_path, omit_last=True)

    with pytest.raises(StrategyPriceArtifactError, match="complete symbol-session grid"):
        build_strategy_price_artifact(
            source,
            source_manifest,
            tmp_path / "strategy.csv",
            tmp_path / "strategy.manifest.json",
            expected_symbol_count=2,
        )


def test_build_strategy_price_artifact_refuses_overwrite(tmp_path: Path) -> None:
    source, source_manifest = _source(tmp_path)
    output = tmp_path / "strategy.csv"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(StrategyPriceArtifactError, match="refusing to overwrite"):
        build_strategy_price_artifact(
            source,
            source_manifest,
            output,
            tmp_path / "strategy.manifest.json",
            expected_symbol_count=2,
        )
