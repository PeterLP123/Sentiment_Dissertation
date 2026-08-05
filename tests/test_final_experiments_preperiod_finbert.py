from __future__ import annotations

from pathlib import Path

import pytest

from final_experiments.lib.preperiod_finbert import (
    PreperiodCheckpointError,
    PreperiodConfig,
    _initialise_store,
    _metadata_set,
    load_aggregate_signal,
)


def test_aggregate_loader_exposes_no_licensed_text(tmp_path: Path) -> None:
    db_path = tmp_path / "preperiod.sqlite3"
    config = PreperiodConfig()
    connection = _initialise_store(db_path, config)
    try:
        connection.executemany(
            """
            INSERT INTO events(
                event_key, symbol, session_date, headline,
                p_positive, p_negative, p_neutral, finbert_score
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("a", "AAA", "2010-01-04", "licensed one", 0.7, 0.1, 0.2, 0.6),
                ("b", "AAA", "2010-01-04", "licensed two", 0.2, 0.5, 0.3, -0.3),
            ],
        )
        _metadata_set(connection, "finbert_scoring_complete", True)
        connection.commit()
    finally:
        connection.close()

    aggregate = load_aggregate_signal(db_path)

    assert aggregate.to_dict(orient="records") == [
        {
            "symbol": "AAA",
            "session_date": aggregate.loc[0, "session_date"],
            "mean_continuous": pytest.approx(0.15),
            "article_count": 2,
        }
    ]
    assert "headline" not in aggregate.columns


def test_checkpoint_refuses_a_configuration_change(tmp_path: Path) -> None:
    db_path = tmp_path / "preperiod.sqlite3"
    connection = _initialise_store(db_path, PreperiodConfig())
    connection.close()

    with pytest.raises(PreperiodCheckpointError, match="different frozen configuration"):
        _initialise_store(
            db_path,
            PreperiodConfig(start_session="2009-09-01"),
        )
