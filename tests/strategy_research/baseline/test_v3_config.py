from __future__ import annotations

from pathlib import Path

import pytest

from sentiment_benchmark.strategy_research.baseline.v3.config import (
    FinbertV3ConfigurationError,
    load_finbert_v3_config,
)

CONFIG = Path("configs/strategy_research/baselines/lseg_sector33_finbert_rank_reversal_v3.toml")


def test_repository_v3_config_is_frozen_and_discloses_post_selection() -> None:
    config = load_finbert_v3_config(CONFIG)
    assert config.run.specified_after_level_reversal_results is True
    assert config.run.confirmation_required is True
    assert config.signal.direction == "contrarian"
    assert config.signal.formation_lag_sessions == 1
    assert config.signal.lookback_sessions == 5
    assert config.signal.tail_fraction == 0.20


def test_v3_config_rejects_false_confirmatory_provenance(tmp_path: Path) -> None:
    text = CONFIG.read_text(encoding="utf-8").replace(
        "specified_after_level_reversal_results = true",
        "specified_after_level_reversal_results = false",
    )
    path = tmp_path / "invalid.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(FinbertV3ConfigurationError, match="post-result"):
        load_finbert_v3_config(path)
