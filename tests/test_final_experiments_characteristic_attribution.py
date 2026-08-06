from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from final_experiments.lib.characteristic_attribution import (
    FEATURE_COLUMNS,
    bootstrap_residual_mean,
    build_lagged_characteristics,
    fit_characteristic_attribution,
)


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = []
    for symbol, offset in (("AAA", 0.0), ("BBB", 10.0)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "session_date": date,
                    "symbol": symbol,
                    "open": 100.0 + offset + index,
                    "close": 100.5 + offset + index,
                    "volume": 1_000.0 + 10 * index,
                }
            )
    return pd.DataFrame(rows)


def test_lagged_characteristics_exclude_current_session_values() -> None:
    prices = _prices()
    baseline = build_lagged_characteristics(prices)
    changed = prices.copy()
    target_date = pd.Timestamp("2026-01-09")
    mask = changed["session_date"].eq(target_date) & changed["symbol"].eq("AAA")
    changed.loc[mask, ["open", "close", "volume"]] = [9_999.0, 8_888.0, 7_777.0]

    altered = build_lagged_characteristics(changed)
    key = baseline["session_date"].eq(target_date) & baseline["symbol"].eq("AAA")
    pd.testing.assert_series_equal(
        baseline.loc[key, FEATURE_COLUMNS].iloc[0],
        altered.loc[key, FEATURE_COLUMNS].iloc[0],
    )


def _attribution_panel() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    beta = np.array([0.004, -0.003, 0.002, 0.001])
    for day_index, date in enumerate(pd.date_range("2026-01-01", periods=12)):
        for sector_index, sector in enumerate(("s1", "s2")):
            features = rng.normal(size=(3, len(FEATURE_COLUMNS)))
            features -= features.mean(axis=0)
            noise = np.array([0.001, -0.0005, -0.0005])
            returns = features @ beta + noise + 0.0002 * day_index
            for name_index in range(3):
                row = {
                    "session_date": date,
                    "symbol": f"{sector}-{name_index}",
                    "sector": sector,
                    "forward_return": returns[name_index],
                    "selected_weight": 0.25 if name_index == sector_index else 0.0,
                }
                row.update(dict(zip(FEATURE_COLUMNS, features[name_index], strict=True)))
                rows.append(row)
    return pd.DataFrame(rows)


def test_characteristic_attribution_identities_and_is_deterministic() -> None:
    attribution = fit_characteristic_attribution(_attribution_panel())
    daily = attribution.daily

    np.testing.assert_allclose(
        daily["raw_sector_difference"],
        daily["explained_characteristic_component"] + daily["residual_characteristic_adjusted"],
        atol=1e-12,
    )
    feature_columns = [f"explained_{feature}" for feature in FEATURE_COLUMNS]
    np.testing.assert_allclose(
        daily[feature_columns].sum(axis=1),
        daily["explained_characteristic_component"],
        atol=1e-12,
    )
    first_metrics, first_distribution = bootstrap_residual_mean(
        attribution,
        block_length=3,
        replications=100,
        seed=9,
    )
    second_metrics, second_distribution = bootstrap_residual_mean(
        attribution,
        block_length=3,
        replications=100,
        seed=9,
    )
    assert first_metrics == second_metrics
    np.testing.assert_array_equal(first_distribution, second_distribution)


def test_attribution_fails_when_selected_characteristics_are_missing() -> None:
    panel = _attribution_panel()
    selected = panel["selected_weight"].gt(0)
    panel.loc[selected.idxmax(), "sigma20"] = np.nan
    with pytest.raises(ValueError, match="every selected firm-session"):
        fit_characteristic_attribution(panel)
