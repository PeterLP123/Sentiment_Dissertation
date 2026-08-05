"""Publisher-conditioning helpers for the expanded 44-company LSEG corpus.

Only Reuters (``NS:RTRS``) is assigned a named ex-ante publisher class.  The
other 3,000+ LSEG source codes remain an unresolved non-Reuters pool; this
module never guesses prestige tiers from code names or realised returns.

Licensed headline text is read locally only to reconstruct the frozen
normalised headline hash.  Returned frames contain hashes and aggregate source
features, never headline text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.aggregators import (
    benjamini_hochberg,
    cross_sectional_daily_ic,
)
from final_experiments.lib.evaluate import (
    TradeConfig,
    build_daily_portfolio,
    summarize_daily_portfolio,
)
from final_experiments.lib.lseg_expanded import (
    EXPLORATORY_SPLIT,
    _charge_final_liquidation,
    map_headlines_to_entry_sessions,
)
from final_experiments.lib.novelty import headline_norm_sha256
from sentiment_benchmark.artifact_io import sha256_file

REUTERS_SOURCE_CODE = "NS:RTRS"
PUBLISHER_ARMS: tuple[str, ...] = (
    "all_source_mean",
    "reuters_only_mean",
    "non_reuters_only_mean",
    "reuters_2x_mean",
)
PUBLISHER_PRIMARY_FAMILY = (
    "four predeclared expanded-LSEG Gemma publisher signals x h1; "
    "daily cross-sectional IC, BH-FDR q=0.05"
)
PUBLISHER_COST_BPS_PER_SIDE = 10.0
PUBLISHER_BOOTSTRAP_REPLICATIONS = 1_999
PUBLISHER_SEED = 20260805


def load_headline_source_features(
    corpus_path: str | Path,
    *,
    expected_hashes: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build source flags and source-specific first timestamps by headline hash."""

    path = Path(corpus_path)
    manifest_path = path.with_name("manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"merged corpus or manifest is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("merged corpus manifest is not completed")
    file_contract = (manifest.get("files") or {}).get("headlines_jsonl") or {}
    if file_contract.get("sha256") != sha256_file(path):
        raise ValueError("merged headline corpus hash mismatch")

    # hash -> [has Reuters, has non-Reuters, first Reuters, first non-Reuters,
    #          corpus rows].  ISO timestamps are comparable as strings on this
    # frozen UTC corpus and are parsed strictly only after aggregation.
    records: dict[str, list[Any]] = {}
    source_rows: dict[str, int] = {}
    rows_read = 0
    unexpected_hash_rows = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows_read += 1
            digest = headline_norm_sha256(str(row.get("headline") or ""))
            if digest not in expected_hashes:
                unexpected_hash_rows += 1
                continue
            source = str(row.get("source_code") or "")
            timestamp = str(
                row.get("first_created") or row.get("version_created") or ""
            )
            if not timestamp:
                raise ValueError("merged headline row lacks a usable timestamp")
            source_rows[source] = source_rows.get(source, 0) + 1
            record = records.setdefault(digest, [False, False, None, None, 0])
            record[4] += 1
            if source == REUTERS_SOURCE_CODE:
                record[0] = True
                if record[2] is None or timestamp < record[2]:
                    record[2] = timestamp
            else:
                record[1] = True
                if record[3] is None or timestamp < record[3]:
                    record[3] = timestamp

    missing = expected_hashes - set(records)
    if missing:
        raise ValueError(
            f"merged source map misses {len(missing)} frozen successful hashes"
        )
    features = pd.DataFrame.from_records(
        (
            {
                "headline_sha256": digest,
                "has_reuters": bool(record[0]),
                "has_non_reuters": bool(record[1]),
                "first_reuters_timestamp": record[2],
                "first_non_reuters_timestamp": record[3],
                "source_corpus_rows": int(record[4]),
            }
            for digest, record in records.items()
        )
    )
    for column in ("first_reuters_timestamp", "first_non_reuters_timestamp"):
        features[column] = pd.to_datetime(
            features[column], utc=True, errors="coerce", format="mixed"
        )
    if features.loc[features["has_reuters"], "first_reuters_timestamp"].isna().any():
        raise ValueError("Reuters-tagged hash lacks a Reuters timestamp")
    if features.loc[
        features["has_non_reuters"], "first_non_reuters_timestamp"
    ].isna().any():
        raise ValueError("non-Reuters-tagged hash lacks a non-Reuters timestamp")

    audit = {
        "rows_read": rows_read,
        "unexpected_hash_rows": unexpected_hash_rows,
        "successful_hashes_mapped": int(len(features)),
        "hashes_with_reuters": int(features["has_reuters"].sum()),
        "hashes_with_non_reuters": int(features["has_non_reuters"].sum()),
        "hashes_with_both": int(
            (features["has_reuters"] & features["has_non_reuters"]).sum()
        ),
        "distinct_source_codes": int(len(source_rows)),
        "source_row_counts": source_rows,
        "corpus_sha256": file_contract.get("sha256"),
    }
    return features.sort_values("headline_sha256", kind="mergesort").reset_index(
        drop=True
    ), audit


def aggregate_publisher_events(
    events: pd.DataFrame,
    *,
    weight_reuters: bool = False,
) -> pd.DataFrame:
    """Aggregate mapped unique headlines to one company-entry-open signal."""

    required = {"symbol", "entry_session", "headline_sha256", "score", "has_reuters"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    frame = events.copy()
    frame["_weight"] = np.where(
        weight_reuters & frame["has_reuters"].astype(bool), 2.0, 1.0
    )
    frame["_weighted_score"] = frame["score"].astype(float) * frame["_weight"]
    grouped = (
        frame.groupby(["symbol", "entry_session"], observed=True)
        .agg(
            _weighted_score=("_weighted_score", "sum"),
            _weight=("_weight", "sum"),
            unique_headlines=("headline_sha256", "nunique"),
            reuters_headlines=("has_reuters", "sum"),
        )
        .reset_index()
    )
    grouped["publisher_signal"] = grouped["_weighted_score"] / grouped["_weight"]
    return grouped.drop(columns=["_weighted_score", "_weight"])


def _attach_next_open_return(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    price = prices.sort_values(["symbol", "session_date"], kind="mergesort").copy()
    price["raw_open_h1"] = (
        price.groupby("symbol", sort=False)["adjusted_open"].shift(-1)
        / price["adjusted_open"]
        - 1.0
    )
    panel = signals.rename(columns={"entry_session": "session_date"}).merge(
        price[["symbol", "session_date", "raw_open_h1"]],
        on=["symbol", "session_date"],
        how="left",
        validate="one_to_one",
    )
    panel["split"] = EXPLORATORY_SPLIT
    return panel


def build_publisher_panels(
    scores: pd.DataFrame,
    source_features: pd.DataFrame,
    prices: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Construct the frozen four publisher-conditioned Gemma h1 panels."""

    joined = scores.merge(
        source_features,
        on="headline_sha256",
        how="inner",
        validate="one_to_one",
    )
    if len(joined) != len(scores):
        raise ValueError("source features do not cover the score population one-to-one")

    def mapped_for(frame: pd.DataFrame, timestamp_col: str | None = None) -> pd.DataFrame:
        use = frame.copy()
        if timestamp_col is not None:
            use["first_timestamp"] = use[timestamp_col]
        mapped = map_headlines_to_entry_sessions(use, prices)
        return mapped.merge(
            source_features[["headline_sha256", "has_reuters", "has_non_reuters"]],
            on="headline_sha256",
            how="left",
            validate="many_to_one",
        )

    mapped_all = mapped_for(joined)
    mapped_reuters = mapped_for(
        joined.loc[joined["has_reuters"]], "first_reuters_timestamp"
    )
    mapped_non_reuters = mapped_for(
        joined.loc[joined["has_non_reuters"]], "first_non_reuters_timestamp"
    )
    events_by_arm = {
        "all_source_mean": (mapped_all, False),
        "reuters_only_mean": (mapped_reuters, False),
        "non_reuters_only_mean": (mapped_non_reuters, False),
        "reuters_2x_mean": (mapped_all, True),
    }
    panels: dict[str, pd.DataFrame] = {}
    coverage_rows: list[dict[str, Any]] = []
    for arm, (events, weighted) in events_by_arm.items():
        aggregated = aggregate_publisher_events(events, weight_reuters=weighted)
        panel = _attach_next_open_return(aggregated, prices)
        panels[arm] = panel
        coverage_rows.append(
            {
                "arm": arm,
                "unique_headline_company_associations": int(
                    events[["headline_sha256", "symbol"]].drop_duplicates().shape[0]
                ),
                "firm_open_rows": int(len(panel)),
                "priced_firm_open_rows": int(panel["raw_open_h1"].notna().sum()),
                "symbols": int(panel["symbol"].nunique()),
                "entry_sessions": int(panel["session_date"].nunique()),
                "mean_headlines_per_firm_open": float(panel["unique_headlines"].mean()),
            }
        )
    return panels, pd.DataFrame(coverage_rows)


def evaluate_publisher_arms(
    panels: dict[str, pd.DataFrame],
    *,
    cost_bps_per_side: float = PUBLISHER_COST_BPS_PER_SIDE,
    bootstrap_replications: int = PUBLISHER_BOOTSTRAP_REPLICATIONS,
    seed: int = PUBLISHER_SEED,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Evaluate the frozen four-arm publisher family without arm selection."""

    if tuple(panels) != PUBLISHER_ARMS:
        raise ValueError(f"publisher panels must follow frozen order: {PUBLISHER_ARMS}")
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
    daily_outputs: dict[str, pd.DataFrame] = {}
    for arm in PUBLISHER_ARMS:
        panel = panels[arm].dropna(subset=["publisher_signal", "raw_open_h1"])
        ic = cross_sectional_daily_ic(
            panel["publisher_signal"],
            panel["raw_open_h1"],
            panel["session_date"],
            min_names=3,
            hac_lags=5,
        )
        daily = build_daily_portfolio(
            panel,
            "publisher_signal",
            config=cfg,
            orient=1.0,
            split=EXPLORATORY_SPLIT,
        )
        daily = _charge_final_liquidation(
            daily, cost_bps_per_side=cost_bps_per_side
        )
        daily_outputs[arm] = daily
        rows.append(
            {
                "arm": arm,
                "ic": ic["ic"],
                "ic_se": ic["se"],
                "ic_t": ic["t"],
                "ic_p": ic["p"],
                "ic_n": ic["n"],
                "ic_n_clusters": ic["n_clusters"],
                **summarize_daily_portfolio(daily, config=cfg),
            }
        )
    results = pd.DataFrame(rows)
    results["bh_reject_ic_q05"] = benjamini_hochberg(
        results["ic_p"].fillna(1.0).tolist(), q=0.05
    )
    results["economically_viable"] = (
        results["bh_reject_ic_q05"]
        & results["sharpe_net"].gt(0)
        & results["breakeven_bps_per_side"].ge(cost_bps_per_side)
        & results["bootstrap_net_ci_low"].gt(0)
    )
    results["multiplicity_family"] = PUBLISHER_PRIMARY_FAMILY
    results["status"] = "retrospective_non_pooled_expanded_lseg_robustness"
    return results, daily_outputs


__all__ = [
    "PUBLISHER_ARMS",
    "PUBLISHER_BOOTSTRAP_REPLICATIONS",
    "PUBLISHER_COST_BPS_PER_SIDE",
    "PUBLISHER_PRIMARY_FAMILY",
    "PUBLISHER_SEED",
    "REUTERS_SOURCE_CODE",
    "aggregate_publisher_events",
    "build_publisher_panels",
    "evaluate_publisher_arms",
    "load_headline_source_features",
]
