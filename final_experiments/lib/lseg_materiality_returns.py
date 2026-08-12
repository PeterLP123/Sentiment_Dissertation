"""Pre-2026 model-only return analysis for the LSEG materiality pilot.

This module is deliberately narrow.  It joins the frozen body-available model
annotations to the return-blind sample, constructs the predeclared ordinal
signal, and evaluates only returns whose end session is before 2026-01-01.
The 2026 confirmation labels may exist in the immutable measurement artifact,
but no 2026 numeric price is parsed and no 2026 return is materialised here.

The model annotations are not human validated.  All outputs from this module
are exploratory development evidence, never validated alpha.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from sentiment_benchmark.artifact_io import sha256_file

CONFIRMATION_START = pd.Timestamp("2026-01-01")
PRIMARY_HORIZONS = (1, 5)
SECONDARY_HORIZONS = (20,)
RANDOM_SEED = 20_260_812

DIRECTION_ORDINAL = {
    "very_negative": -2,
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "very_positive": 2,
}
LEVEL_ORDINAL = {
    "none": 0,
    "low": 1,
    "moderate": 2,
    "high": 3,
    "very_high": 4,
}

OOS_FOLDS: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = (
    (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-01-01")),
    (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")),
    (pd.Timestamp("2025-07-01"), CONFIRMATION_START),
)


class MaterialityReturnError(RuntimeError):
    """Raised when an input or analysis step could breach the frozen contract."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MaterialityReturnError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _require_hash(path: Path, expected_sha256: str) -> None:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise MaterialityReturnError(
            f"immutable input hash mismatch for {path}: {actual} != {expected_sha256}"
        )


def load_model_only_development_events(
    sample_path: str | Path,
    score_path: str | Path,
    *,
    sample_sha256: str,
    score_sha256: str,
    expected_development_rows: int = 1_200,
    confirmation_start: pd.Timestamp = CONFIRMATION_START,
) -> pd.DataFrame:
    """Load the frozen annotations and return only pre-confirmation events."""

    sample_file = Path(sample_path)
    score_file = Path(score_path)
    _require_hash(sample_file, sample_sha256)
    _require_hash(score_file, score_sha256)

    sample = pd.DataFrame(_read_jsonl(sample_file))
    scores = pd.DataFrame(_read_jsonl(score_file))
    sample_need = {
        "audit_id",
        "headline_sha256",
        "symbol",
        "entry_session",
        "score_gemma",
        "score_finbert",
    }
    score_need = {
        "audit_id",
        "headline_sha256",
        "direction_severity",
        "materiality",
        "novelty",
        "valuation_horizon",
        "target_specific",
        "evidence_sufficient",
    }
    if sample_need - set(sample):
        raise MaterialityReturnError(f"sample is missing fields: {sorted(sample_need - set(sample))}")
    if score_need - set(scores):
        raise MaterialityReturnError(f"scores are missing fields: {sorted(score_need - set(scores))}")
    if sample["audit_id"].duplicated().any() or scores["audit_id"].duplicated().any():
        raise MaterialityReturnError("audit_id must be unique in both frozen inputs")

    joined = sample.merge(
        scores,
        on=["audit_id", "headline_sha256"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_model"),
    )
    joined["entry_session"] = pd.to_datetime(joined["entry_session"], errors="raise").dt.normalize()
    development = joined.loc[joined["entry_session"].lt(confirmation_start)].copy()
    if len(development) != expected_development_rows:
        raise MaterialityReturnError(
            f"pre-2026 body-available population changed: {len(development)} != {expected_development_rows}"
        )
    if development["entry_session"].ge(confirmation_start).any():
        raise MaterialityReturnError("confirmation event reached the development frame")
    if development["symbol"].nunique() != 33:
        raise MaterialityReturnError("development frame does not retain all 33 companies")

    development["direction_ordinal"] = development["direction_severity"].map(DIRECTION_ORDINAL)
    development["materiality_ordinal"] = development["materiality"].map(LEVEL_ORDINAL)
    development["novelty_ordinal"] = development["novelty"].map(LEVEL_ORDINAL)
    ordinal_cols = ["direction_ordinal", "materiality_ordinal", "novelty_ordinal"]
    if development[ordinal_cols].isna().any().any():
        raise MaterialityReturnError("model output contains an unknown ordinal label")

    development["direction_only"] = development["direction_ordinal"] / 2.0
    development["signed_materiality"] = (
        development["direction_only"] * development["materiality_ordinal"] / 4.0
    )
    development["signed_materiality_novelty"] = (
        development["signed_materiality"] * development["novelty_ordinal"] / 4.0
    )
    for column in ("score_gemma", "score_finbert"):
        development[column] = pd.to_numeric(development[column], errors="raise")
        std = float(development[column].std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            raise MaterialityReturnError(f"cannot standardise comparator: {column}")
        development[f"z_{column}"] = (development[column] - development[column].mean()) / std
    development["combined_polarity"] = (
        development["z_score_gemma"] + development["z_score_finbert"]
    ) / 2.0
    development["measurement_status"] = "model_only_unvalidated"
    return development.sort_values(["entry_session", "symbol", "audit_id"], kind="mergesort").reset_index(drop=True)


def load_preconfirmation_lseg_prices(
    path: str | Path,
    *,
    expected_sha256: str,
    confirmation_start: pd.Timestamp = CONFIRMATION_START,
    expected_symbols: int = 33,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Hash-check a full export but parse numeric values only before 2026.

    The row's date is inspected first.  For rows on or after the confirmation
    boundary the numeric price fields are never converted, retained or returned.
    """

    price_path = Path(path)
    manifest_path = price_path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise MaterialityReturnError(f"price manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or manifest.get("provider") != "lseg":
        raise MaterialityReturnError("price source is not a completed LSEG export")
    convention = str(manifest.get("return_convention") or "").lower()
    if "split-adjusted" not in convention or "not back-adjusted" not in convention:
        raise MaterialityReturnError("price return convention changed")
    declared_hash = str((manifest.get("file") or {}).get("sha256") or "")
    if declared_hash != expected_sha256:
        raise MaterialityReturnError("price manifest does not declare the frozen file hash")
    _require_hash(price_path, expected_sha256)

    rows: list[dict[str, Any]] = []
    with price_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        needed = {"symbol", "session_date", "open"}
        if needed - set(reader.fieldnames or ()):
            raise MaterialityReturnError(f"price file lacks columns: {sorted(needed)}")
        for row in reader:
            session = pd.Timestamp(row["session_date"]).normalize()
            if session >= confirmation_start:
                # Do not parse, persist or expose the confirmation-period price.
                continue
            rows.append(
                {
                    "symbol": str(row["symbol"]),
                    "session_date": session,
                    "adjusted_open": float(row["open"]),
                }
            )
    prices = pd.DataFrame(rows)
    if prices.empty:
        raise MaterialityReturnError("pre-confirmation price panel is empty")
    if prices["session_date"].ge(confirmation_start).any():
        raise MaterialityReturnError("confirmation price reached the numeric frame")
    if prices.duplicated(["symbol", "session_date"]).any():
        raise MaterialityReturnError("pre-confirmation price panel has duplicate symbol-sessions")
    if prices["symbol"].nunique() != expected_symbols:
        raise MaterialityReturnError("pre-confirmation price panel does not contain 33 symbols")
    valid = np.isfinite(prices["adjusted_open"]) & prices["adjusted_open"].gt(0)
    if not bool(valid.all()):
        raise MaterialityReturnError("pre-confirmation price panel contains invalid opens")

    prices = prices.sort_values(["symbol", "session_date"], kind="mergesort").reset_index(drop=True)
    audit = {
        "source_sha256": expected_sha256,
        "numeric_confirmation_prices_parsed": 0,
        "confirmation_returns_materialised": 0,
        "first_numeric_session": prices["session_date"].min().date().isoformat(),
        "last_numeric_session": prices["session_date"].max().date().isoformat(),
        "preconfirmation_rows": int(len(prices)),
        "symbols": int(prices["symbol"].nunique()),
    }
    return prices, audit


def dense_open_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Return the complete 33-company pre-confirmation adjusted-open rectangle."""

    wide = (
        prices.pivot(index="session_date", columns="symbol", values="adjusted_open")
        .sort_index()
        .dropna(axis=0, how="any")
    )
    if wide.empty or wide.shape[1] != 33:
        raise MaterialityReturnError("cannot construct a dense 33-company development price panel")
    if wide.index.max() >= CONFIRMATION_START:
        raise MaterialityReturnError("dense price panel crossed the confirmation boundary")
    return wide


def attach_forward_open_returns(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = (*PRIMARY_HORIZONS, *SECONDARY_HORIZONS),
    confirmation_start: pd.Timestamp = CONFIRMATION_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach exact forward-session open returns without crossing into 2026."""

    out = events.copy()
    if out["entry_session"].ge(confirmation_start).any():
        raise MaterialityReturnError("confirmation event supplied to return attachment")
    wide = dense_open_panel(prices)
    for horizon in horizons:
        if horizon < 1:
            raise ValueError("return horizons must be positive")
        end_session = pd.Series(wide.index, index=wide.index).shift(-horizon)
        returns = wide.shift(-horizon) / wide - 1.0
        long = returns.rename_axis(index="entry_session", columns="symbol").stack(future_stack=True)
        long.name = f"ret_open_h{horizon}"
        lookup = long.reset_index()
        lookup[f"return_end_h{horizon}"] = lookup["entry_session"].map(end_session)
        lookup.loc[
            lookup[f"return_end_h{horizon}"].ge(confirmation_start),
            [f"ret_open_h{horizon}", f"return_end_h{horizon}"],
        ] = [np.nan, pd.NaT]
        out = out.merge(lookup, on=["entry_session", "symbol"], how="left", validate="many_to_one")
        end_col = f"return_end_h{horizon}"
        if out[end_col].dropna().ge(confirmation_start).any():
            raise MaterialityReturnError(f"h{horizon} return crossed the confirmation boundary")
    return out, wide


def _zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if not math.isfinite(std) or std <= 0:
        raise MaterialityReturnError("predictor has zero or invalid variance")
    return (values - float(values.mean())) / std


def _regression_design(
    frame: pd.DataFrame,
    *,
    signal_col: str,
    include_signal: bool,
) -> pd.DataFrame:
    continuous = pd.DataFrame(
        {
            "gemma_polarity_z": _zscore(frame["score_gemma"].astype(float)),
            "finbert_polarity_z": _zscore(frame["score_finbert"].astype(float)),
        },
        index=frame.index,
    )
    if include_signal:
        continuous["materiality_signal_z"] = _zscore(frame[signal_col].astype(float))
    company = pd.get_dummies(frame["symbol"].astype(str), prefix="company", drop_first=True, dtype=float)
    design = pd.concat([continuous, company], axis=1)
    return sm.add_constant(design, has_constant="add").astype(float)


def fit_incremental_regression(
    frame: pd.DataFrame,
    *,
    horizon: int,
    signal_col: str = "signed_materiality_novelty",
) -> dict[str, Any]:
    """Fit the frozen nested OLS with entry-session clustered covariance."""

    outcome = f"ret_open_h{horizon}"
    use = frame.dropna(subset=[outcome, signal_col, "score_gemma", "score_finbert"]).copy()
    if use["entry_session"].ge(CONFIRMATION_START).any():
        raise MaterialityReturnError("confirmation event reached regression")
    if len(use) < 100 or use["entry_session"].nunique() < 30:
        raise MaterialityReturnError(f"insufficient support for h{horizon} regression")
    y = use[outcome].astype(float)
    base_x = _regression_design(use, signal_col=signal_col, include_signal=False)
    full_x = _regression_design(use, signal_col=signal_col, include_signal=True)
    base = sm.OLS(y, base_x).fit()
    full_plain = sm.OLS(y, full_x).fit()
    full = full_plain.get_robustcov_results(
        cov_type="cluster",
        groups=pd.Categorical(use["entry_session"]).codes,
        use_correction=True,
    )
    names = list(full.model.exog_names)
    position = names.index("materiality_signal_z")
    beta = float(full.params[position])
    se = float(full.bse[position])
    p = float(full.pvalues[position])
    ci = np.asarray(full.conf_int(alpha=0.05), dtype=float)[position]
    return {
        "horizon": horizon,
        "signal": signal_col,
        "n_events": int(len(use)),
        "n_sessions": int(use["entry_session"].nunique()),
        "n_companies": int(use["symbol"].nunique()),
        "beta_per_signal_sd": beta,
        "beta_bps_per_signal_sd": beta * 10_000.0,
        "cluster_se": se,
        "cluster_se_bps": se * 10_000.0,
        "ci_low_bps": float(ci[0] * 10_000.0),
        "ci_high_bps": float(ci[1] * 10_000.0),
        "p_two_sided": p,
        "base_r2": float(base.rsquared),
        "expanded_r2": float(full_plain.rsquared),
        "delta_r2": float(full_plain.rsquared - base.rsquared),
        "expected_direction_positive": beta > 0,
    }


def _training_encoded_design(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    signal_col: str,
    include_signal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictors = ["score_gemma", "score_finbert"] + ([signal_col] if include_signal else [])
    train_parts = [np.ones((len(train), 1), dtype=float)]
    test_parts = [np.ones((len(test), 1), dtype=float)]
    for column in predictors:
        mean = float(train[column].mean())
        std = float(train[column].std(ddof=0))
        if not math.isfinite(std) or std <= 0:
            raise MaterialityReturnError(f"training predictor has zero variance: {column}")
        train_parts.append((train[[column]].to_numpy(dtype=float) - mean) / std)
        test_parts.append((test[[column]].to_numpy(dtype=float) - mean) / std)
    levels = sorted(train["symbol"].astype(str).unique())
    if len(levels) < 2:
        raise MaterialityReturnError("training fold has fewer than two companies")
    for symbol in levels[1:]:
        train_parts.append(train["symbol"].astype(str).eq(symbol).to_numpy(dtype=float)[:, None])
        test_parts.append(test["symbol"].astype(str).eq(symbol).to_numpy(dtype=float)[:, None])
    return np.hstack(train_parts), np.hstack(test_parts)


def expanding_oos_predictions(
    frame: pd.DataFrame,
    *,
    horizon: int,
    signal_col: str = "signed_materiality_novelty",
    folds: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = OOS_FOLDS,
) -> pd.DataFrame:
    """Generate frozen expanding-window base and expanded OLS predictions."""

    outcome = f"ret_open_h{horizon}"
    end_col = f"return_end_h{horizon}"
    need = [outcome, end_col, signal_col, "score_gemma", "score_finbert"]
    use = frame.dropna(subset=need).copy()
    rows: list[pd.DataFrame] = []
    for fold_number, (test_start, test_end) in enumerate(folds, start=1):
        train = use.loc[use[end_col].lt(test_start)].copy()
        test = use.loc[
            use["entry_session"].ge(test_start)
            & use["entry_session"].lt(test_end)
            & use[end_col].lt(test_end)
        ].copy()
        if len(train) < 100 or test.empty:
            raise MaterialityReturnError(f"insufficient rows in OOS fold {fold_number} for h{horizon}")
        if train[end_col].max() >= test_start:
            raise MaterialityReturnError("training outcome overlaps OOS test start")
        base_train, base_test = _training_encoded_design(
            train, test, signal_col=signal_col, include_signal=False
        )
        full_train, full_test = _training_encoded_design(
            train, test, signal_col=signal_col, include_signal=True
        )
        y_train = train[outcome].to_numpy(dtype=float)
        base_beta = np.linalg.lstsq(base_train, y_train, rcond=None)[0]
        full_beta = np.linalg.lstsq(full_train, y_train, rcond=None)[0]
        pred = test[["audit_id", "entry_session", "symbol", outcome]].copy()
        pred["fold"] = fold_number
        pred["prediction_base"] = base_test @ base_beta
        pred["prediction_expanded"] = full_test @ full_beta
        rows.append(pred)
    out = pd.concat(rows, ignore_index=True)
    if out["entry_session"].ge(CONFIRMATION_START).any():
        raise MaterialityReturnError("confirmation row reached OOS predictions")
    return out.sort_values(["entry_session", "symbol", "audit_id"], kind="mergesort").reset_index(drop=True)


def oos_daily_differences(predictions: pd.DataFrame, *, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return date-level paired squared-loss and cross-sectional IC differences."""

    outcome = f"ret_open_h{horizon}"
    pred = predictions.copy()
    pred["loss_base"] = (pred[outcome] - pred["prediction_base"]) ** 2
    pred["loss_expanded"] = (pred[outcome] - pred["prediction_expanded"]) ** 2
    pred["loss_improvement"] = pred["loss_base"] - pred["loss_expanded"]
    loss = pred.groupby("entry_session", as_index=False).agg(
        loss_improvement=("loss_improvement", "mean"),
        loss_base=("loss_base", "mean"),
        loss_expanded=("loss_expanded", "mean"),
        n_events=("audit_id", "size"),
    )

    ic_rows: list[dict[str, Any]] = []
    for session, day in pred.groupby("entry_session", sort=True):
        if len(day) < 3 or day[outcome].nunique() < 2:
            continue
        base_ic = float(spearmanr(day["prediction_base"], day[outcome]).statistic)
        expanded_ic = float(spearmanr(day["prediction_expanded"], day[outcome]).statistic)
        if not (math.isfinite(base_ic) and math.isfinite(expanded_ic)):
            continue
        ic_rows.append(
            {
                "entry_session": session,
                "ic_base": base_ic,
                "ic_expanded": expanded_ic,
                "ic_improvement": expanded_ic - base_ic,
                "n_events": int(len(day)),
            }
        )
    return loss, pd.DataFrame(ic_rows)


def circular_block_mean_test(
    values: Iterable[float],
    *,
    block_length: int = 5,
    replications: int = 9_999,
    seed: int = RANDOM_SEED,
    alternative: str = "greater",
) -> dict[str, float]:
    """Circular block bootstrap for a sample mean and directional p-value."""

    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        raise MaterialityReturnError("block bootstrap requires at least two observations")
    if block_length < 1 or replications < 1:
        raise ValueError("block length and replications must be positive")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less or two-sided")
    rng = np.random.default_rng(seed)
    n = len(x)
    blocks = int(math.ceil(n / block_length))
    means = np.empty(replications, dtype=float)
    offsets = np.arange(block_length)
    for replication in range(replications):
        starts = rng.integers(0, n, size=blocks)
        indices = (starts[:, None] + offsets[None, :]) % n
        means[replication] = float(x[indices.ravel()[:n]].mean())
    point = float(x.mean())
    # Centre the series to generate the null distribution.  The uncentred
    # bootstrap distribution is retained for the percentile confidence
    # interval, but using it directly as a p-value would not impose H0: mean=0.
    centred = x - point
    null_means = np.empty(replications, dtype=float)
    for replication in range(replications):
        starts = rng.integers(0, n, size=blocks)
        indices = (starts[:, None] + offsets[None, :]) % n
        null_means[replication] = float(centred[indices.ravel()[:n]].mean())
    if alternative == "greater":
        p = (1.0 + float(np.sum(null_means >= point))) / (replications + 1.0)
    elif alternative == "less":
        p = (1.0 + float(np.sum(null_means <= point))) / (replications + 1.0)
    else:
        p = (1.0 + float(np.sum(np.abs(null_means) >= abs(point)))) / (replications + 1.0)
    return {
        "n": int(n),
        "mean": point,
        "se": float(means.std(ddof=1)),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "p": float(p),
        "block_length": int(block_length),
        "replications": int(replications),
    }


def benjamini_hochberg_qvalues(p_values: Sequence[float]) -> np.ndarray:
    """Return monotone Benjamini-Hochberg adjusted p-values in input order."""

    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be a finite one-dimensional vector in [0, 1]")
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def sparse_formation_books(
    events: pd.DataFrame,
    wide_prices: pd.DataFrame,
    *,
    signal_col: str,
    horizon: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build the frozen single-positive/single-negative event book."""

    sessions = pd.DatetimeIndex(wide_prices.index)
    symbols = list(wide_prices.columns.astype(str))
    session_index = {session: i for i, session in enumerate(sessions)}
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    formation = np.zeros((len(sessions), len(symbols)), dtype=float)
    audit_rows: list[dict[str, Any]] = []
    for session, day in events.groupby("entry_session", sort=True):
        row = session_index.get(pd.Timestamp(session))
        if row is None or row + horizon >= len(sessions):
            continue
        positive = day.loc[day[signal_col].gt(0)].sort_values(
            [signal_col, "audit_id"], ascending=[False, True], kind="mergesort"
        )
        negative = day.loc[day[signal_col].lt(0)].sort_values(
            [signal_col, "audit_id"], ascending=[True, True], kind="mergesort"
        )
        if positive.empty or negative.empty:
            continue
        long_symbol = str(positive.iloc[0]["symbol"])
        short_symbol = str(negative.iloc[0]["symbol"])
        if long_symbol == short_symbol:
            raise MaterialityReturnError("sparse event book selected the same symbol on both legs")
        formation[row, symbol_index[long_symbol]] = 0.5
        formation[row, symbol_index[short_symbol]] = -0.5
        audit_rows.append(
            {
                "entry_session": session,
                "long_symbol": long_symbol,
                "short_symbol": short_symbol,
                "long_signal": float(positive.iloc[0][signal_col]),
                "short_signal": float(negative.iloc[0][signal_col]),
            }
        )
    return formation, pd.DataFrame(audit_rows)


def run_sparse_strategy(
    events: pd.DataFrame,
    wide_prices: pd.DataFrame,
    *,
    signal_col: str = "signed_materiality_novelty",
    horizon: int,
    cost_bps_per_side: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run fixed overlapping tranches on the full cash-inclusive calendar."""

    if horizon < 1 or cost_bps_per_side < 0:
        raise ValueError("horizon must be positive and cost non-negative")
    if wide_prices.index.max() >= CONFIRMATION_START:
        raise MaterialityReturnError("strategy price panel crossed into confirmation")
    formation, formation_audit = sparse_formation_books(
        events, wide_prices, signal_col=signal_col, horizon=horizon
    )
    targets = np.zeros_like(formation)
    for formed_at in np.flatnonzero(np.abs(formation).sum(axis=1) > 0):
        targets[formed_at : formed_at + horizon] += formation[formed_at] / float(horizon)

    asset_returns = wide_prices.shift(-1) / wide_prices - 1.0
    asset_matrix = asset_returns.iloc[:-1].to_numpy(dtype=float)
    targets = targets[:-1]
    sessions = pd.DatetimeIndex(wide_prices.index[:-1])
    end_sessions = pd.DatetimeIndex(wide_prices.index[1:])
    if not np.isfinite(asset_matrix).all():
        raise MaterialityReturnError("dense development return matrix contains missing values")

    rows: list[dict[str, Any]] = []
    previous_target = np.zeros(targets.shape[1], dtype=float)
    previous_asset_return = np.zeros(targets.shape[1], dtype=float)
    previous_gross = 0.0
    cost_rate = cost_bps_per_side / 10_000.0
    for index, (session, end_session) in enumerate(zip(sessions, end_sessions, strict=True)):
        target = targets[index]
        if index == 0:
            pretrade = np.zeros_like(target)
        else:
            denominator = 1.0 + previous_gross
            if denominator <= 0:
                raise MaterialityReturnError("portfolio lost all capital during drift accounting")
            pretrade = previous_target * (1.0 + previous_asset_return) / denominator
        turnover = 0.5 * float(np.abs(target - pretrade).sum())
        gross = float(np.dot(target, asset_matrix[index]))
        cost = 2.0 * turnover * cost_rate
        rows.append(
            {
                "session_date": session,
                "return_end_date": end_session,
                "gross_return": gross,
                "turnover": turnover,
                "cost": cost,
                "net_return": gross - cost,
                "gross_exposure": float(np.abs(target).sum()),
                "net_exposure": float(target.sum()),
                "n_long": int((target > 0).sum()),
                "n_short": int((target < 0).sum()),
            }
        )
        previous_target = target
        previous_asset_return = asset_matrix[index]
        previous_gross = gross

    daily = pd.DataFrame(rows)
    if not daily.empty:
        # Every frozen tranche is complete by the last development open.  Close
        # residual drifted weights at that open and charge the final trade.
        denominator = 1.0 + previous_gross
        final_pretrade = previous_target * (1.0 + previous_asset_return) / denominator
        liquidation_turnover = 0.5 * float(np.abs(final_pretrade).sum())
        liquidation_cost = 2.0 * liquidation_turnover * cost_rate
        last = daily.index[-1]
        daily.loc[last, "turnover"] += liquidation_turnover
        daily.loc[last, "cost"] += liquidation_cost
        daily.loc[last, "net_return"] -= liquidation_cost
    if daily["return_end_date"].ge(CONFIRMATION_START).any():
        raise MaterialityReturnError("strategy materialised a confirmation return")
    return daily, formation_audit


def annualized_sharpe(values: Iterable[float], *, periods_per_year: int = 252) -> float:
    series = pd.Series(list(values), dtype=float).dropna()
    if len(series) < 2 or float(series.std(ddof=1)) <= 0:
        return float("nan")
    return float(math.sqrt(periods_per_year) * series.mean() / series.std(ddof=1))


def summarize_sparse_strategy(daily: pd.DataFrame, *, cost_bps_per_side: float) -> dict[str, Any]:
    """Summarise a cash-inclusive sparse strategy path."""

    if daily.empty:
        raise MaterialityReturnError("cannot summarise an empty strategy")
    gross_mean = float(daily["gross_return"].mean())
    turnover_mean = float(daily["turnover"].mean())
    breakeven = (
        gross_mean * 10_000.0 / (2.0 * turnover_mean) if turnover_mean > 0 else float("nan")
    )
    net_equity = (1.0 + daily["net_return"]).cumprod()
    drawdown = net_equity / net_equity.cummax() - 1.0
    return {
        "n_sessions": int(len(daily)),
        "active_holding_sessions": int(daily["gross_exposure"].gt(0).sum()),
        "mean_gross_bps_session": gross_mean * 10_000.0,
        "mean_net_bps_session": float(daily["net_return"].mean() * 10_000.0),
        "gross_sharpe": annualized_sharpe(daily["gross_return"]),
        "net_sharpe": annualized_sharpe(daily["net_return"]),
        "total_return_gross": float((1.0 + daily["gross_return"]).prod() - 1.0),
        "total_return_net": float(net_equity.iloc[-1] - 1.0),
        "max_drawdown_net": float(drawdown.min()),
        "mean_turnover": turnover_mean,
        "annualized_turnover": turnover_mean * 252.0,
        "breakeven_bps_per_side": float(breakeven),
        "cost_bps_per_side": float(cost_bps_per_side),
    }


def fit_horizon_matched_materiality(
    frame: pd.DataFrame,
    *,
    horizon: int,
    horizon_label: str,
) -> dict[str, Any]:
    """Test whether materiality scales direction-aligned return at prompt horizon."""

    outcome = f"ret_open_h{horizon}"
    use = frame.loc[
        frame["valuation_horizon"].eq(horizon_label)
        & frame["direction_ordinal"].ne(0)
    ].dropna(subset=[outcome]).copy()
    if len(use) < 20 or use["entry_session"].nunique() < 10:
        return {
            "horizon": horizon,
            "valuation_horizon": horizon_label,
            "n_events": int(len(use)),
            "status": "insufficient_support",
            "p_one_sided": 1.0,
        }
    use["aligned_return"] = np.sign(use["direction_ordinal"]) * use[outcome]
    continuous = pd.DataFrame(
        {
            "materiality_z": _zscore(use["materiality_ordinal"].astype(float)),
            "abs_gemma_z": _zscore(use["score_gemma"].abs().astype(float)),
            "abs_finbert_z": _zscore(use["score_finbert"].abs().astype(float)),
        },
        index=use.index,
    )
    company = pd.get_dummies(use["symbol"].astype(str), prefix="company", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([continuous, company], axis=1), has_constant="add").astype(float)
    plain = sm.OLS(use["aligned_return"].astype(float), design).fit()
    result = plain.get_robustcov_results(
        cov_type="cluster",
        groups=pd.Categorical(use["entry_session"]).codes,
        use_correction=True,
    )
    position = list(result.model.exog_names).index("materiality_z")
    beta = float(result.params[position])
    two_sided = float(result.pvalues[position])
    return {
        "horizon": horizon,
        "valuation_horizon": horizon_label,
        "n_events": int(len(use)),
        "n_sessions": int(use["entry_session"].nunique()),
        "status": "estimated",
        "beta_bps_per_materiality_sd": beta * 10_000.0,
        "cluster_se_bps": float(result.bse[position] * 10_000.0),
        "p_one_sided": two_sided / 2.0 if beta > 0 else 1.0 - two_sided / 2.0,
        "expected_direction_positive": beta > 0,
    }


def fit_reaction_magnitude_regression(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> dict[str, Any]:
    """Estimate incremental absolute-return information in model materiality."""

    outcome = f"ret_open_h{horizon}"
    use = frame.dropna(
        subset=[outcome, "materiality_ordinal", "novelty_ordinal", "score_gemma", "score_finbert"]
    ).copy()
    if use["entry_session"].ge(CONFIRMATION_START).any():
        raise MaterialityReturnError("confirmation event reached reaction-magnitude regression")
    use["absolute_return"] = use[outcome].abs()
    continuous = pd.DataFrame(
        {
            "materiality_z": _zscore((use["materiality_ordinal"] / 4.0).astype(float)),
            "novelty_z": _zscore((use["novelty_ordinal"] / 4.0).astype(float)),
            "abs_gemma_z": _zscore(use["score_gemma"].abs().astype(float)),
            "abs_finbert_z": _zscore(use["score_finbert"].abs().astype(float)),
        },
        index=use.index,
    )
    company = pd.get_dummies(use["symbol"].astype(str), prefix="company", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([continuous, company], axis=1), has_constant="add").astype(float)
    plain = sm.OLS(use["absolute_return"].astype(float), design).fit()
    result = plain.get_robustcov_results(
        cov_type="cluster",
        groups=pd.Categorical(use["entry_session"]).codes,
        use_correction=True,
    )
    position = list(result.model.exog_names).index("materiality_z")
    beta = float(result.params[position])
    two_sided = float(result.pvalues[position])
    ci = np.asarray(result.conf_int(alpha=0.05), dtype=float)[position]
    return {
        "horizon": int(horizon),
        "n_events": int(len(use)),
        "n_sessions": int(use["entry_session"].nunique()),
        "n_companies": int(use["symbol"].nunique()),
        "beta_bps_per_materiality_sd": beta * 10_000.0,
        "cluster_se_bps": float(result.bse[position] * 10_000.0),
        "ci_low_bps": float(ci[0] * 10_000.0),
        "ci_high_bps": float(ci[1] * 10_000.0),
        "p_one_sided": two_sided / 2.0 if beta > 0 else 1.0 - two_sided / 2.0,
        "p_two_sided": two_sided,
        "r2": float(plain.rsquared),
        "expected_direction_positive": beta > 0,
    }


def expanding_oos_reaction_magnitude(
    frame: pd.DataFrame,
    *,
    horizon: int,
    folds: Sequence[tuple[pd.Timestamp, pd.Timestamp]] = OOS_FOLDS,
) -> pd.DataFrame:
    """Chronologically compare magnitude models before and after materiality."""

    outcome = f"ret_open_h{horizon}"
    end_col = f"return_end_h{horizon}"
    need = [outcome, end_col, "materiality_ordinal", "novelty_ordinal", "score_gemma", "score_finbert"]
    use = frame.dropna(subset=need).copy()
    use["target_absolute_return"] = use[outcome].abs()
    use["abs_score_gemma"] = use["score_gemma"].abs()
    use["abs_score_finbert"] = use["score_finbert"].abs()
    use["materiality_scale"] = use["materiality_ordinal"] / 4.0
    use["novelty_scale"] = use["novelty_ordinal"] / 4.0
    rows: list[pd.DataFrame] = []
    base_predictors = ["abs_score_gemma", "abs_score_finbert", "novelty_scale"]
    expanded_predictors = [*base_predictors, "materiality_scale"]

    def encoded(
        train: pd.DataFrame,
        test: pd.DataFrame,
        predictors: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        train_parts = [np.ones((len(train), 1), dtype=float)]
        test_parts = [np.ones((len(test), 1), dtype=float)]
        for column in predictors:
            mean = float(train[column].mean())
            std = float(train[column].std(ddof=0))
            if not math.isfinite(std) or std <= 0:
                raise MaterialityReturnError(f"magnitude training predictor has zero variance: {column}")
            train_parts.append((train[[column]].to_numpy(dtype=float) - mean) / std)
            test_parts.append((test[[column]].to_numpy(dtype=float) - mean) / std)
        levels = sorted(train["symbol"].astype(str).unique())
        for symbol in levels[1:]:
            train_parts.append(train["symbol"].astype(str).eq(symbol).to_numpy(dtype=float)[:, None])
            test_parts.append(test["symbol"].astype(str).eq(symbol).to_numpy(dtype=float)[:, None])
        return np.hstack(train_parts), np.hstack(test_parts)

    for fold_number, (test_start, test_end) in enumerate(folds, start=1):
        train = use.loc[use[end_col].lt(test_start)].copy()
        test = use.loc[
            use["entry_session"].ge(test_start)
            & use["entry_session"].lt(test_end)
            & use[end_col].lt(test_end)
        ].copy()
        if len(train) < 100 or test.empty:
            raise MaterialityReturnError(
                f"insufficient rows in reaction-magnitude OOS fold {fold_number} for h{horizon}"
            )
        base_train, base_test = encoded(train, test, base_predictors)
        full_train, full_test = encoded(train, test, expanded_predictors)
        y_train = train["target_absolute_return"].to_numpy(dtype=float)
        base_beta = np.linalg.lstsq(base_train, y_train, rcond=None)[0]
        full_beta = np.linalg.lstsq(full_train, y_train, rcond=None)[0]
        predicted = test[["audit_id", "entry_session", "symbol", "target_absolute_return"]].copy()
        predicted["fold"] = fold_number
        predicted["prediction_base"] = base_test @ base_beta
        predicted["prediction_expanded"] = full_test @ full_beta
        predicted["loss_base"] = (
            predicted["target_absolute_return"] - predicted["prediction_base"]
        ) ** 2
        predicted["loss_expanded"] = (
            predicted["target_absolute_return"] - predicted["prediction_expanded"]
        ) ** 2
        predicted["loss_improvement"] = predicted["loss_base"] - predicted["loss_expanded"]
        rows.append(predicted)
    out = pd.concat(rows, ignore_index=True)
    if out["entry_session"].ge(CONFIRMATION_START).any():
        raise MaterialityReturnError("confirmation row reached magnitude OOS predictions")
    return out.sort_values(["entry_session", "symbol", "audit_id"], kind="mergesort").reset_index(drop=True)


__all__ = [
    "CONFIRMATION_START",
    "DIRECTION_ORDINAL",
    "LEVEL_ORDINAL",
    "MaterialityReturnError",
    "OOS_FOLDS",
    "PRIMARY_HORIZONS",
    "SECONDARY_HORIZONS",
    "annualized_sharpe",
    "attach_forward_open_returns",
    "benjamini_hochberg_qvalues",
    "circular_block_mean_test",
    "dense_open_panel",
    "expanding_oos_predictions",
    "expanding_oos_reaction_magnitude",
    "fit_horizon_matched_materiality",
    "fit_incremental_regression",
    "fit_reaction_magnitude_regression",
    "load_model_only_development_events",
    "load_preconfirmation_lseg_prices",
    "oos_daily_differences",
    "run_sparse_strategy",
    "sparse_formation_books",
    "summarize_sparse_strategy",
]
