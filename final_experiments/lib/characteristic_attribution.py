"""Point-in-time characteristic attribution for a fixed selected-long path.

The functions in this module explain an already selected portfolio. They do
not select names, alter weights, or construct a tradable control strategy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "lagged_return_1",
    "lagged_return_5",
    "sigma20",
    "log_adv20",
)


@dataclass(frozen=True)
class CharacteristicAttribution:
    """Fixed-path attribution and sufficient statistics for session bootstrap."""

    daily: pd.DataFrame
    coefficients: pd.DataFrame
    scaling: pd.DataFrame
    xtx_by_session: np.ndarray
    xty_by_session: np.ndarray
    exposure_by_session: np.ndarray


def build_lagged_characteristics(
    prices: pd.DataFrame,
    *,
    window_sessions: int = 20,
    minimum_observations: int = 5,
) -> pd.DataFrame:
    """Return conventional price/liquidity features known before the signal open."""

    if window_sessions < 5:
        raise ValueError("window_sessions must be at least five")
    if not 5 <= minimum_observations <= window_sessions:
        raise ValueError("minimum_observations must be in [5, window_sessions]")
    required = {"session_date", "symbol", "open", "close", "volume"}
    if missing := required - set(prices.columns):
        raise ValueError(f"prices missing columns: {sorted(missing)}")

    frame = prices.loc[:, sorted(required)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    if frame[["session_date", "symbol"]].isna().any().any():
        raise ValueError("session_date and symbol must be non-missing")
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError("prices must be unique by session_date and symbol")
    for column in ("open", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{column} must be finite and positive")

    frame = frame.sort_values(["symbol", "session_date"], kind="mergesort")
    grouped = frame.groupby("symbol", sort=False)
    frame["open_return"] = grouped["open"].pct_change(fill_method=None)
    frame["lagged_return_1"] = grouped["open_return"].shift(1)
    frame["lagged_return_5"] = grouped["open"].transform(lambda values: values.shift(1).div(values.shift(6)).sub(1.0))
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["sigma20"] = grouped["open_return"].transform(
        lambda values: (
            values.rolling(
                window_sessions,
                min_periods=minimum_observations,
            )
            .std(ddof=1)
            .shift(1)
        )
    )
    adv = grouped["dollar_volume"].transform(
        lambda values: (
            values.rolling(
                window_sessions,
                min_periods=minimum_observations,
            )
            .mean()
            .shift(1)
        )
    )
    frame["log_adv20"] = np.log(adv)
    return frame.loc[:, ["session_date", "symbol", *FEATURE_COLUMNS]].reset_index(drop=True)


def fit_characteristic_attribution(
    panel: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
    identity_tolerance: float = 1e-12,
) -> CharacteristicAttribution:
    """Decompose a fixed within-sector selected path into explained and residual."""

    required = {
        "session_date",
        "symbol",
        "sector",
        "forward_return",
        "selected_weight",
        *feature_columns,
    }
    if missing := required - set(panel.columns):
        raise ValueError(f"attribution panel missing columns: {sorted(missing)}")
    if not feature_columns:
        raise ValueError("at least one feature is required")
    if identity_tolerance < 0 or not np.isfinite(identity_tolerance):
        raise ValueError("identity_tolerance must be finite and non-negative")

    frame = panel.loc[:, sorted(required)].copy()
    frame["session_date"] = pd.to_datetime(frame["session_date"]).dt.normalize()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["sector"] = frame["sector"].astype(str)
    if frame.duplicated(["session_date", "symbol"]).any():
        raise ValueError("attribution panel must be unique by session and symbol")
    for column in ("forward_return", "selected_weight"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be finite")
    if frame["selected_weight"].lt(0).any():
        raise ValueError("selected_weight must be non-negative")

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    complete_features = frame.loc[:, feature_columns].notna().all(axis=1)
    if (~complete_features & frame["selected_weight"].gt(0)).any():
        raise ValueError("every selected firm-session must have complete characteristics")
    training = complete_features & frame["selected_weight"].eq(0)
    if not training.any():
        raise ValueError("no complete nonselected observations are available")
    scaling_rows: list[dict[str, Any]] = []
    for column in feature_columns:
        mean = float(frame.loc[training, column].mean())
        std = float(frame.loc[training, column].std(ddof=1))
        if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
            raise ValueError(f"feature {column} has invalid training scale")
        frame[f"z_{column}"] = (frame[column] - mean) / std
        scaling_rows.append({"feature": column, "mean": mean, "std": std})

    group_columns = ["session_date", "sector"]
    frame["demeaned_return"] = frame["forward_return"] - frame.groupby(
        group_columns,
        sort=False,
    )["forward_return"].transform("mean")
    demeaned_features: list[str] = []
    for column in feature_columns:
        z_column = f"z_{column}"
        demeaned = f"demeaned_{column}"
        frame[demeaned] = frame[z_column] - frame.groupby(
            group_columns,
            sort=False,
        )[z_column].transform("mean")
        demeaned_features.append(demeaned)

    sessions = pd.DatetimeIndex(sorted(frame["session_date"].unique()))
    n_features = len(feature_columns)
    xtx = np.zeros((len(sessions), n_features, n_features), dtype=float)
    xty = np.zeros((len(sessions), n_features), dtype=float)
    exposure = np.zeros((len(sessions), n_features), dtype=float)
    raw = np.zeros(len(sessions), dtype=float)
    selected_names = np.zeros(len(sessions), dtype=int)
    training_rows = np.zeros(len(sessions), dtype=int)
    session_index = {session: index for index, session in enumerate(sessions)}
    for session, day in frame.groupby("session_date", sort=True):
        index = session_index[pd.Timestamp(session)]
        day_complete = day.loc[day.loc[:, demeaned_features].notna().all(axis=1)]
        day_training = day_complete.loc[day_complete["selected_weight"].eq(0)]
        x_train = day_training.loc[:, demeaned_features].to_numpy(dtype=float)
        y_train = day_training["demeaned_return"].to_numpy(dtype=float)
        xtx[index] = x_train.T @ x_train
        xty[index] = x_train.T @ y_train
        weights = day_complete["selected_weight"].to_numpy(dtype=float)
        x_all = day_complete.loc[:, demeaned_features].to_numpy(dtype=float)
        exposure[index] = weights @ x_all
        raw[index] = float(weights @ day_complete["demeaned_return"].to_numpy(dtype=float))
        selected_names[index] = int(day["selected_weight"].gt(0).sum())
        training_rows[index] = int(len(day_training))

    total_xtx = xtx.sum(axis=0)
    total_xty = xty.sum(axis=0)
    if np.linalg.matrix_rank(total_xtx) != n_features:
        raise ValueError("characteristic design is rank deficient")
    beta = np.linalg.solve(total_xtx, total_xty)
    explained = exposure @ beta
    residual = raw - explained
    if np.max(np.abs(raw - explained - residual)) > identity_tolerance:
        raise RuntimeError("characteristic attribution identity failed")

    daily = pd.DataFrame(
        {
            "session_date": sessions,
            "selected_names": selected_names,
            "training_rows": training_rows,
            "raw_sector_difference": raw,
            "explained_characteristic_component": explained,
            "residual_characteristic_adjusted": residual,
        }
    )
    coefficient_rows = []
    for index, feature in enumerate(feature_columns):
        contribution = exposure[:, index] * beta[index]
        daily[f"explained_{feature}"] = contribution
        coefficient_rows.append(
            {
                "feature": feature,
                "coefficient_return_per_sd": float(beta[index]),
                "mean_selected_exposure_sd_weight": float(exposure[:, index].mean()),
                "mean_explained_bps_session": float(contribution.mean() * 10_000),
            }
        )
    if (
        np.max(
            np.abs(daily[[f"explained_{feature}" for feature in feature_columns]].sum(axis=1) - daily["explained_characteristic_component"])
        )
        > identity_tolerance
    ):
        raise RuntimeError("feature contribution identity failed")
    return CharacteristicAttribution(
        daily=daily,
        coefficients=pd.DataFrame(coefficient_rows),
        scaling=pd.DataFrame(scaling_rows),
        xtx_by_session=xtx,
        xty_by_session=xty,
        exposure_by_session=exposure,
    )


def bootstrap_residual_mean(
    attribution: CharacteristicAttribution,
    *,
    block_length: int = 5,
    replications: int = 9_999,
    seed: int = 20260820,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Block-bootstrap the residual mean while refitting coefficients each time."""

    daily = attribution.daily
    n_sessions = len(daily)
    if not 1 <= block_length <= n_sessions:
        raise ValueError("block_length must be in [1, n_sessions]")
    if replications <= 0:
        raise ValueError("replications must be positive")
    raw = daily["raw_sector_difference"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n_sessions / block_length))
    distribution = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = rng.integers(0, n_sessions, size=blocks_needed)
        positions = np.concatenate([(start + np.arange(block_length, dtype=int)) % n_sessions for start in starts])[:n_sessions]
        xtx = attribution.xtx_by_session[positions].sum(axis=0)
        xty = attribution.xty_by_session[positions].sum(axis=0)
        if np.linalg.matrix_rank(xtx) != xtx.shape[0]:
            raise RuntimeError("bootstrap characteristic design is rank deficient")
        beta = np.linalg.solve(xtx, xty)
        residual = raw[positions] - attribution.exposure_by_session[positions] @ beta
        distribution[replication] = float(residual.mean())
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    probability_nonpositive = (np.count_nonzero(distribution <= 0.0) + 1) / (replications + 1)
    probability_nonnegative = (np.count_nonzero(distribution >= 0.0) + 1) / (replications + 1)
    metrics: dict[str, float | int] = {
        "n_sessions": n_sessions,
        "block_length": block_length,
        "replications": replications,
        "seed": seed,
        "mean_residual": float(daily["residual_characteristic_adjusted"].mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "p_two_sided": float(min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative))),
    }
    return metrics, distribution


__all__ = [
    "FEATURE_COLUMNS",
    "CharacteristicAttribution",
    "bootstrap_residual_mean",
    "build_lagged_characteristics",
    "fit_characteristic_attribution",
]
