"""Expanded LSEG 44-company scorer-comparison helpers.

The analysis unit is the frozen unique normalized headline used by both
scorers.  Each headline is associated with every matched company, mapped to the
first exchange open at least 15 minutes after its first timestamp, and then
aggregated to ``(symbol, entry_session)``.  Licensed headline text never leaves
the ignored source files and is not returned by these helpers.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.aggregators import (
    AGGREGATOR_NAMES,
    benjamini_hochberg,
    build_aggregation_panel,
    cross_sectional_daily_ic,
)
from final_experiments.lib.evaluate import (
    DEFAULT_SIGNAL_ORIENT,
    TradeConfig,
    build_daily_portfolio,
    summarize_daily_portfolio,
)
from sentiment_benchmark.artifact_io import sha256_file

EXPECTED_POPULATION = 888_155
INFORMATION_BUFFER_MINUTES = 15
EXPLORATORY_SPLIT = "exploratory_full_period"
PRIMARY_FAMILY = "two scorers x nine pre-existing aggregators x h1; BH-FDR q=0.05"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
GEMMA_MODEL = "google/gemma-4-26b-a4b-it"
GEMMA_PROMPT_ID = "investor_headline_soft_label_v1"
GEMMA_PROMPT_HASH = "81596538d99b29b8"

SCORE_MANIFEST_CONTRACTS: dict[str, tuple[tuple[tuple[str, ...], Any], ...]] = {
    "finbert": (
        (("contract", "scorer"), "finbert"),
        (("contract", "model_id"), "ProsusAI/finbert"),
        (("contract", "revision"), FINBERT_REVISION),
        (("contract", "revision_enforced"), True),
    ),
    "gemma4_26b": (
        (("model",), GEMMA_MODEL),
        (("provider",), "DeepInfra"),
        (("quantization",), "fp8"),
        (("prompt_id",), GEMMA_PROMPT_ID),
        (("prompt_hash",), GEMMA_PROMPT_HASH),
        (("request", "provider", "only"), ["deepinfra"]),
        (("request", "provider", "quantizations"), ["fp8"]),
        (("request", "provider", "allow_fallbacks"), False),
        (("request", "provider", "data_collection"), "deny"),
        (("request", "provider", "require_parameters"), True),
        (("request", "provider", "zdr"), True),
    ),
}

SCORE_COLUMNS = (
    "headline_sha256",
    "headline",
    "matched_symbols",
    "first_timestamp",
    "explicit_target",
    "contextual",
    "market_price_technical",
    "label",
    "p_positive",
    "p_negative",
    "p_neutral",
    "score",
    "status",
    "reported_cost_usd",
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("boolean score metadata contains values other than true/false")
    return mapped.astype(bool)


def _validate_score_manifest_contract(manifest: dict[str, Any], *, scorer: str) -> None:
    contract = SCORE_MANIFEST_CONTRACTS.get(scorer)
    if contract is None:
        raise ValueError(f"no frozen score-manifest contract registered for scorer: {scorer}")

    missing = object()
    mismatches: list[str] = []
    for path, expected in contract:
        value: Any = manifest
        for part in path:
            value = value.get(part, missing) if isinstance(value, dict) else missing
            if value is missing:
                break
        if value != expected:
            dotted = ".".join(path)
            mismatches.append(f"{dotted}={value!r}, expected {expected!r}")
    if mismatches:
        raise ValueError("score manifest contract mismatch: " + "; ".join(mismatches))


def load_success_scores(
    path: str | Path,
    *,
    scorer: str,
    expected_population: int = EXPECTED_POPULATION,
    include_headline: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load successful rows and fail closed on the frozen score contract.

    Licensed headline text is excluded by default.  The explicit opt-in exists
    only for local feature construction; callers must not write it to tracked
    or aggregate result artifacts.
    """

    score_path = Path(path)
    manifest_path = score_path.with_suffix(score_path.suffix + ".manifest.json")
    if not score_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"score artifact or manifest is missing: {score_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"score manifest is not completed: {manifest_path}")
    _validate_score_manifest_contract(manifest, scorer=scorer)

    output_entry = manifest.get("output")
    output_hash = (
        output_entry.get("sha256") if isinstance(output_entry, dict) else None
    ) or manifest.get("output_sha256")
    actual_hash = sha256_file(score_path)
    if output_hash != actual_hash:
        raise ValueError(f"score hash mismatch for {score_path}")

    chunks: list[pd.DataFrame] = []
    status_counts: dict[str, int] = {}
    total_cost = 0.0
    total_rows = 0
    with score_path.open(newline="", encoding="utf-8") as handle:
        fieldnames = tuple(csv.reader(handle).__next__())
    required = set(SCORE_COLUMNS) - {"headline", "reported_cost_usd"}
    if include_headline:
        required.add("headline")
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"score artifact is missing required columns: {sorted(missing)}")
    usecols = [
        column
        for column in SCORE_COLUMNS
        if column in fieldnames and (include_headline or column != "headline")
    ]
    for chunk in pd.read_csv(score_path, usecols=usecols, chunksize=200_000):
        total_rows += len(chunk)
        counts = chunk["status"].astype(str).value_counts()
        for status, count in counts.items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
        if "reported_cost_usd" in chunk:
            total_cost += float(
                pd.to_numeric(chunk["reported_cost_usd"], errors="coerce").fillna(0).sum()
            )
        chunks.append(chunk.loc[chunk["status"].eq("success")].copy())
    success = pd.concat(chunks, ignore_index=True)

    if len(success) != expected_population or success["headline_sha256"].nunique() != expected_population:
        raise ValueError("successful score rows do not give one-to-one frozen-population coverage")
    probabilities = success[["p_positive", "p_negative", "p_neutral"]].apply(
        pd.to_numeric, errors="raise"
    )
    score = pd.to_numeric(success["score"], errors="raise")
    valid = (
        probabilities.ge(0).all(axis=1)
        & probabilities.le(1).all(axis=1)
        & np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        & np.isclose(score, probabilities["p_positive"] - probabilities["p_negative"], atol=1e-6)
    )
    if not bool(valid.all()):
        raise ValueError("successful rows contain invalid probabilities or score margins")

    success[["p_positive", "p_negative", "p_neutral"]] = probabilities
    success["score"] = score
    for column in ("explicit_target", "contextual", "market_price_technical"):
        success[column] = _as_bool(success[column])
    success["first_timestamp"] = pd.to_datetime(
        success["first_timestamp"], utc=True, errors="raise", format="mixed"
    )
    success["scorer"] = scorer
    success = success.drop(columns=["status", "reported_cost_usd"], errors="ignore")
    audit = {
        "scorer": scorer,
        "rows_including_historical_attempts": total_rows,
        "status_counts": status_counts,
        "unique_successes": int(success["headline_sha256"].nunique()),
        "duplicate_successes": int(success["headline_sha256"].duplicated().sum()),
        "invalid_success_probabilities": int((~valid).sum()),
        "reported_cost_usd_sum": total_cost,
        "output_sha256": actual_hash,
    }
    return success, audit


def assert_paired_population(finbert: pd.DataFrame, gemma: pd.DataFrame) -> pd.DataFrame:
    """Return a one-to-one pair using current-population metadata from Gemma.

    The inherited FinBERT export preserves some metadata from the older seed
    row.  Exact-hash labels remain reusable, but company associations and first
    timestamps must come from the newly enumerated 44-company population.  The
    OpenRouter artifact was built directly from that population, so its
    scorer-blind fields are canonical here.  Mismatch counts are retained in
    ``DataFrame.attrs`` for the notebook audit.
    """

    metadata = [
        "headline_sha256",
        "matched_symbols",
        "first_timestamp",
        "explicit_target",
        "contextual",
        "market_price_technical",
    ]
    mismatch_counts: dict[str, int] = {}
    metadata_check = finbert[metadata].merge(
        gemma[metadata],
        on="headline_sha256",
        how="inner",
        suffixes=("_finbert", "_gemma"),
        validate="one_to_one",
    )
    if len(metadata_check) != EXPECTED_POPULATION:
        raise ValueError("scorer populations do not match exactly")
    for column in metadata[1:]:
        mismatch_counts[column] = int(
            (~metadata_check[f"{column}_finbert"].eq(metadata_check[f"{column}_gemma"])).sum()
        )

    paired = gemma[metadata + ["label", "score"]].merge(
        finbert[["headline_sha256", "label", "score"]],
        on="headline_sha256",
        how="inner",
        suffixes=("_gemma", "_finbert"),
        validate="one_to_one",
    )
    paired.attrs["finbert_seed_metadata_mismatch_counts"] = mismatch_counts
    return paired


def canonical_score_frame(paired: pd.DataFrame, *, scorer: str) -> pd.DataFrame:
    """Materialise one scorer with the pair's canonical 44-company metadata."""

    if scorer not in {"finbert", "gemma"}:
        raise ValueError("scorer must be 'finbert' or 'gemma'")
    metadata = [
        "headline_sha256",
        "matched_symbols",
        "first_timestamp",
        "explicit_target",
        "contextual",
        "market_price_technical",
    ]
    out = paired[metadata + [f"label_{scorer}", f"score_{scorer}"]].rename(
        columns={f"label_{scorer}": "label", f"score_{scorer}": "score"}
    )
    out["scorer"] = "gemma4_26b" if scorer == "gemma" else "finbert"
    return out.copy()


def load_price_exports(*paths: str | Path, expected_symbols: int = 44) -> pd.DataFrame:
    """Load and hash-check separate LSEG exports into one adjusted-open panel."""

    pieces: list[pd.DataFrame] = []
    declared_symbols: set[str] = set()
    for value in paths:
        path = Path(value)
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest.get("provider") != "lseg":
            raise ValueError(f"price manifest is not a completed LSEG export: {manifest_path}")
        convention = str(manifest.get("return_convention") or "").lower()
        if "split-adjusted" not in convention or "not back-adjusted" not in convention:
            raise ValueError("price return convention is not the frozen LSEG price-return convention")
        if (manifest.get("file") or {}).get("sha256") != sha256_file(path):
            raise ValueError(f"price hash mismatch for {path}")
        piece = pd.read_csv(path, usecols=["symbol", "session_date", "open"])
        piece = piece.rename(columns={"open": "adjusted_open"})
        pieces.append(piece)
        declared_symbols.update(str(symbol) for symbol in manifest.get("symbols") or ())

    prices = pd.concat(pieces, ignore_index=True)
    prices["symbol"] = prices["symbol"].astype(str)
    prices["session_date"] = pd.to_datetime(prices["session_date"]).dt.normalize()
    prices["adjusted_open"] = pd.to_numeric(prices["adjusted_open"], errors="raise")
    if prices.duplicated(["symbol", "session_date"]).any():
        raise ValueError("combined price panel has duplicate symbol-session rows")
    observed = set(prices["symbol"])
    if len(observed) != expected_symbols or observed != declared_symbols:
        raise ValueError("combined price symbols do not reconcile to the two manifests")
    if not np.isfinite(prices["adjusted_open"]).all() or not prices["adjusted_open"].gt(0).all():
        raise ValueError("price panel contains invalid adjusted opens")
    return prices.sort_values(["symbol", "session_date"], kind="mergesort").reset_index(drop=True)


def map_headlines_to_entry_sessions(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    information_buffer_minutes: int = INFORMATION_BUFFER_MINUTES,
) -> pd.DataFrame:
    """Explode company associations and map each headline to its first eligible open."""

    if information_buffer_minutes < 0:
        raise ValueError("information buffer must be non-negative")
    events = scores[
        [
            "headline_sha256",
            "matched_symbols",
            "first_timestamp",
            "explicit_target",
            "contextual",
            "market_price_technical",
            "label",
            "score",
            "scorer",
        ]
    ].copy()
    events["matched_symbol_count"] = events["matched_symbols"].str.count(r"\|") + 1
    events["symbol"] = events["matched_symbols"].str.split("|")
    events = events.explode("symbol", ignore_index=True)
    events["symbol"] = events["symbol"].astype(str)
    unknown = sorted(set(events["symbol"]) - set(prices["symbol"]))
    if unknown:
        raise ValueError(f"score associations lack prices for: {unknown}")

    sessions = pd.DatetimeIndex(sorted(prices["session_date"].unique())).as_unit("ns")
    local_opens = sessions.tz_localize("America/New_York") + np.timedelta64(570, "m")
    open_utc = local_opens.tz_convert("UTC")
    cutoff = events["first_timestamp"].dt.as_unit("ns") + np.timedelta64(
        information_buffer_minutes, "m"
    )
    positions = open_utc.searchsorted(pd.DatetimeIndex(cutoff), side="left")
    valid = positions < len(sessions)
    events = events.loc[valid].copy()
    events["entry_session"] = sessions.take(positions[valid]).to_numpy()
    events["session_date"] = events["entry_session"]
    events["finbert_score"] = events["score"].astype(float)
    events["polarity"] = events["label"].astype(str)
    events["is_recap"] = False
    return events.drop(columns=["matched_symbols"])


def build_expanded_aggregation_panel(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Apply the pre-existing nine aggregation rules and attach h1 returns."""

    price = prices.copy()
    price["next_adjusted_open"] = price.groupby("symbol", sort=False)["adjusted_open"].shift(-1)
    price["raw_open_h1"] = price["next_adjusted_open"] / price["adjusted_open"] - 1.0
    panel = (
        events[["symbol", "entry_session"]]
        .drop_duplicates()
        .rename(columns={"entry_session": "session_date"})
        .merge(
            price[["symbol", "session_date", "raw_open_h1"]],
            on=["symbol", "session_date"],
            how="left",
            validate="one_to_one",
        )
    )
    # ``build_aggregation_panel`` preserves the FNSPID-compatible outcome name.
    panel["ar_open_h1"] = panel["raw_open_h1"]
    panel["split"] = EXPLORATORY_SPLIT
    aggregated = build_aggregation_panel(panel, stories=events)
    return aggregated.rename(
        columns={"n": "unique_headline_count", "ar_open_h1": "raw_open_h1"}
    )


def _charge_final_liquidation(daily: pd.DataFrame, *, cost_bps_per_side: float) -> pd.DataFrame:
    """Charge the final close of an active gross-one book."""

    out = daily.copy()
    if out.empty or int(out.iloc[-1]["n_long"] + out.iloc[-1]["n_short"]) == 0:
        return out
    liquidation_turnover = 0.5
    liquidation_cost = cost_bps_per_side / 10_000.0
    last = out.index[-1]
    out.loc[last, "turnover"] += liquidation_turnover
    out.loc[last, "cost"] += liquidation_cost
    out.loc[last, "net_return"] -= liquidation_cost
    return out


def evaluate_expanded_arms(
    panels: dict[str, pd.DataFrame],
    *,
    cost_bps_per_side: float = 10.0,
    bootstrap_replications: int = 1_999,
    seed: int = 20260805,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Evaluate the fixed 18-arm scorer/aggregator family without model selection."""

    cfg = TradeConfig(
        outcome="raw_open_h1",
        position_mode="cs_rank",
        threshold=0.0,
        min_names=10,
        cost_bps_per_side=cost_bps_per_side,
        bootstrap_block_length=5,
        bootstrap_replications=bootstrap_replications,
        random_seed=seed,
    )
    rows: list[dict[str, Any]] = []
    daily_outputs: dict[tuple[str, str], pd.DataFrame] = {}
    for scorer, panel in panels.items():
        use = panel.loc[panel["raw_open_h1"].notna()].copy()
        for aggregator in AGGREGATOR_NAMES:
            ic = cross_sectional_daily_ic(
                use[aggregator], use["raw_open_h1"], use["session_date"], min_names=3, hac_lags=5
            )
            daily = build_daily_portfolio(
                use,
                aggregator,
                config=cfg,
                orient=DEFAULT_SIGNAL_ORIENT[aggregator],
                split=EXPLORATORY_SPLIT,
            )
            daily = _charge_final_liquidation(daily, cost_bps_per_side=cost_bps_per_side)
            daily_outputs[(scorer, aggregator)] = daily
            summary = summarize_daily_portfolio(daily, config=cfg)
            rows.append(
                {
                    "scorer": scorer,
                    "aggregator": aggregator,
                    "orient": DEFAULT_SIGNAL_ORIENT[aggregator],
                    "ic": ic["ic"],
                    "ic_se": ic["se"],
                    "ic_t": ic["t"],
                    "ic_p": ic["p"],
                    "ic_n": ic["n"],
                    "ic_n_clusters": ic["n_clusters"],
                    **summary,
                }
            )
    table = pd.DataFrame(rows)
    table["bh_reject_ic_q05"] = benjamini_hochberg(table["ic_p"].fillna(1.0).tolist())
    table["multiplicity_family"] = PRIMARY_FAMILY
    table["status"] = "retrospective_exploratory_non_pooled_robustness"
    return table, daily_outputs


def filter_sensitivity_events(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return pre-declared metadata filters; these are not human-validated classes."""

    return {
        "all_unique_headlines": events,
        "single_company": events.loc[events["matched_symbol_count"].eq(1)].copy(),
        "explicit_single_company": events.loc[
            events["matched_symbol_count"].eq(1) & events["explicit_target"]
        ].copy(),
        "exclude_market_price_technical": events.loc[~events["market_price_technical"]].copy(),
    }


__all__ = [
    "EXPECTED_POPULATION",
    "EXPLORATORY_SPLIT",
    "INFORMATION_BUFFER_MINUTES",
    "PRIMARY_FAMILY",
    "assert_paired_population",
    "build_expanded_aggregation_panel",
    "canonical_score_frame",
    "evaluate_expanded_arms",
    "filter_sensitivity_events",
    "load_price_exports",
    "load_success_scores",
    "map_headlines_to_entry_sessions",
]
