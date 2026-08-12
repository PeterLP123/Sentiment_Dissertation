"""Return-blind joining of the frozen same-33 LSEG scorer populations.

The earlier and later LSEG collections remain distinct source blocks.  This
module validates their score artifacts, restricts the later 44-company block
to the immutable original 33, and creates a normalized-headline join without
loading prices or returns.  Licensed headline text is never returned.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from final_experiments.lib.lseg_expanded import (
    assert_paired_population,
    load_success_scores,
)
from sentiment_benchmark.artifact_io import sha256_file, sha256_text

BACKWARD_POPULATION = 1_219_608
RECENT_POPULATION = 888_155
BACKWARD_START = pd.Timestamp("2024-01-01", tz="UTC")
JOIN_BOUNDARY = pd.Timestamp("2025-10-26", tz="UTC")
RECENT_END = pd.Timestamp("2026-06-26", tz="UTC")

METADATA_COLUMNS = (
    "headline_sha256",
    "matched_symbols",
    "first_timestamp",
    "explicit_target",
    "contextual",
    "market_price_technical",
)
GEMMA_MODEL = "google/gemma-4-26b-a4b-it"
GEMMA_PROVIDER = "DeepInfra"
GEMMA_PROMPT_HASH = "81596538d99b29b8"


@dataclass(frozen=True)
class JoinedLsegInputs:
    backward_finbert: Path
    backward_gemma_seed: Path
    backward_gemma_delta: Path
    recent_finbert: Path
    recent_gemma: Path


def _manifest_output_hash(manifest: dict[str, Any]) -> str:
    output = manifest.get("output")
    if isinstance(output, dict):
        return str(output.get("sha256") or "")
    return str(manifest.get("output_sha256") or "")


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("score metadata contains a non-boolean flag")
    return mapped.astype(bool)


def _read_gemma_part(path: Path, *, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.is_file() or not manifest_path.is_file():
        raise ValueError(f"missing Gemma artifact or manifest: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"Gemma manifest is not completed: {manifest_path}")
    expected_hash = _manifest_output_hash(manifest)
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise ValueError(f"Gemma output hash mismatch: {path}")

    usecols = [
        *METADATA_COLUMNS,
        "model_id",
        "provider",
        "quantization",
        "prompt_hash",
        "label",
        "p_positive",
        "p_negative",
        "p_neutral",
        "score",
        "status",
    ]
    chunks: list[pd.DataFrame] = []
    status_counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        fields = tuple(next(csv.reader(handle)))
    missing = set(usecols) - set(fields)
    if missing:
        raise ValueError(f"Gemma artifact lacks columns: {sorted(missing)}")
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000):
        counts = chunk["status"].astype(str).value_counts()
        for status, count in counts.items():
            status_counts[status] = status_counts.get(status, 0) + int(count)
        success = chunk.loc[chunk["status"].eq("success")].copy()
        contract_ok = (
            success["model_id"].eq(GEMMA_MODEL)
            & success["provider"].astype(str).str.casefold().eq(GEMMA_PROVIDER.casefold())
            & success["quantization"].eq("fp8")
            & success["prompt_hash"].eq(GEMMA_PROMPT_HASH)
        )
        if not bool(contract_ok.all()):
            raise ValueError(f"Gemma row-level scorer contract mismatch: {path}")
        chunks.append(success)
    frame = pd.concat(chunks, ignore_index=True)
    probabilities = frame[["p_positive", "p_negative", "p_neutral"]].apply(
        pd.to_numeric, errors="raise"
    )
    scores = pd.to_numeric(frame["score"], errors="raise")
    valid = (
        probabilities.ge(0).all(axis=1)
        & probabilities.le(1).all(axis=1)
        & np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        & np.isclose(scores, probabilities["p_positive"] - probabilities["p_negative"], atol=1e-6)
    )
    if not bool(valid.all()):
        raise ValueError(f"Gemma probabilities or signed scores are invalid: {path}")
    frame[["p_positive", "p_negative", "p_neutral"]] = probabilities
    frame["score"] = scores
    frame["first_timestamp"] = pd.to_datetime(frame["first_timestamp"], utc=True, format="mixed")
    for column in ("explicit_target", "contextual", "market_price_technical"):
        frame[column] = _as_bool(frame[column])
    frame = frame.drop(
        columns=["model_id", "provider", "quantization", "prompt_hash", "status"]
    )
    return frame, {
        "path": str(path),
        "manifest": str(manifest_path),
        "output_sha256": actual_hash,
        "rows": int(sum(status_counts.values())),
        "status_counts": status_counts,
        "successful_rows": len(frame),
    }


def _pair_scores(
    finbert: pd.DataFrame,
    gemma: pd.DataFrame,
    *,
    expected_population: int,
    block: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if finbert["headline_sha256"].duplicated().any():
        raise ValueError(f"duplicate FinBERT success in {block}")
    if gemma["headline_sha256"].duplicated().any():
        raise ValueError(f"duplicate Gemma success in {block}")
    finbert_hashes = set(finbert["headline_sha256"])
    gemma_hashes = set(gemma["headline_sha256"])
    if len(finbert_hashes) != expected_population or finbert_hashes != gemma_hashes:
        raise ValueError(f"paired scorer population mismatch in {block}")

    check = gemma[list(METADATA_COLUMNS)].merge(
        finbert[list(METADATA_COLUMNS)],
        on="headline_sha256",
        how="inner",
        suffixes=("_gemma", "_finbert"),
        validate="one_to_one",
    )
    mismatch_counts: dict[str, int] = {}
    for column in METADATA_COLUMNS[1:]:
        left = check[f"{column}_gemma"]
        right = check[f"{column}_finbert"]
        mismatch_counts[column] = int((~left.eq(right)).sum())

    paired = gemma[list(METADATA_COLUMNS) + ["label", "score"]].merge(
        finbert[["headline_sha256", "label", "score"]],
        on="headline_sha256",
        how="inner",
        suffixes=("_gemma", "_finbert"),
        validate="one_to_one",
    )
    paired["source_block"] = block
    return paired, {"metadata_mismatch_counts": mismatch_counts}


def _restrict_associations(frame: pd.DataFrame, symbols: set[str]) -> tuple[pd.DataFrame, int]:
    out = frame.copy()
    restricted = out["matched_symbols"].astype(str).map(
        lambda value: tuple(sorted(set(value.split("|")) & symbols))
    )
    keep = restricted.map(bool)
    removed = int((~keep).sum())
    out = out.loc[keep].copy()
    out["matched_symbols"] = restricted.loc[keep].map("|".join)
    return out.reset_index(drop=True), removed


def _validate_block_dates(frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp, block: str) -> None:
    valid = frame["first_timestamp"].ge(start) & frame["first_timestamp"].lt(end)
    if not bool(valid.all()):
        invalid = int((~valid).sum())
        raise ValueError(f"{block} contains {invalid} timestamps outside its frozen interval")


def _deduplicate_join(blocks: Iterable[pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, Any]]:
    combined = pd.concat(blocks, ignore_index=True)
    combined = combined.sort_values(
        ["first_timestamp", "source_block", "headline_sha256"], kind="mergesort"
    ).reset_index(drop=True)
    overlap = combined.loc[combined["headline_sha256"].duplicated(keep=False)].copy()
    overlap_hashes = int(overlap["headline_sha256"].nunique())
    score_collisions = {"gemma": 0, "finbert": 0}
    if overlap_hashes:
        grouped = overlap.groupby("headline_sha256", sort=False)
        for scorer in ("gemma", "finbert"):
            score_collisions[scorer] = int(
                grouped[f"score_{scorer}"].nunique(dropna=False).gt(1).sum()
            )

    first = combined.drop_duplicates("headline_sha256", keep="first").set_index("headline_sha256")
    associations = combined.groupby("headline_sha256", sort=False)["matched_symbols"].agg(
        lambda values: "|".join(
            sorted({symbol for value in values for symbol in str(value).split("|") if symbol})
        )
    )
    first["matched_symbols"] = associations
    for column in ("explicit_target", "market_price_technical"):
        first[column] = combined.groupby("headline_sha256", sort=False)[column].any()
    first["contextual"] = ~first["explicit_target"]
    joined = first.reset_index().sort_values(
        ["first_timestamp", "headline_sha256"], kind="mergesort", ignore_index=True
    )
    population_sha256 = sha256_text("\n".join(sorted(joined["headline_sha256"])))
    return joined, {
        "input_rows": len(combined),
        "joined_unique_headlines": len(joined),
        "cross_block_overlap_hashes": overlap_hashes,
        "cross_block_score_collision_hashes": score_collisions,
        "joined_population_sha256": population_sha256,
    }


def load_joined_same33_scores(
    inputs: JoinedLsegInputs,
    *,
    original_symbols: Iterable[str],
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Load the frozen earlier/later blocks and return paired score populations."""

    symbols = {str(symbol) for symbol in original_symbols}
    if len(symbols) != 33:
        raise ValueError("the joined LSEG universe must contain exactly 33 symbols")

    backward_finbert, backward_finbert_audit = load_success_scores(
        inputs.backward_finbert,
        scorer="finbert",
        expected_population=BACKWARD_POPULATION,
    )
    seed_manifest = inputs.backward_gemma_seed.with_suffix(
        inputs.backward_gemma_seed.suffix + ".seed.json"
    )
    delta_manifest = inputs.backward_gemma_delta.with_suffix(
        inputs.backward_gemma_delta.suffix + ".manifest.json"
    )
    backward_seed, seed_audit = _read_gemma_part(
        inputs.backward_gemma_seed, manifest_path=seed_manifest
    )
    backward_delta, delta_audit = _read_gemma_part(
        inputs.backward_gemma_delta, manifest_path=delta_manifest
    )
    backward_gemma = pd.concat([backward_seed, backward_delta], ignore_index=True)
    backward, backward_pair_audit = _pair_scores(
        backward_finbert,
        backward_gemma,
        expected_population=BACKWARD_POPULATION,
        block="backward_2024_2025",
    )
    _validate_block_dates(
        backward, start=BACKWARD_START, end=JOIN_BOUNDARY, block="backward_2024_2025"
    )

    recent_finbert, recent_finbert_audit = load_success_scores(
        inputs.recent_finbert,
        scorer="finbert",
        expected_population=RECENT_POPULATION,
    )
    recent_gemma, recent_gemma_audit = load_success_scores(
        inputs.recent_gemma,
        scorer="gemma4_26b",
        expected_population=RECENT_POPULATION,
    )
    recent_all = assert_paired_population(recent_finbert, recent_gemma)
    recent_all["source_block"] = "opened_2025_2026"
    recent, recent_removed = _restrict_associations(recent_all, symbols)
    _validate_block_dates(
        recent, start=JOIN_BOUNDARY, end=RECENT_END, block="opened_2025_2026"
    )

    joined, join_audit = _deduplicate_join((backward, recent))
    audit = {
        "backward": {
            "paired_unique_headlines": len(backward),
            "population_sha256": sha256_text(
                "\n".join(sorted(backward["headline_sha256"]))
            ),
            "finbert": backward_finbert_audit,
            "gemma_parts": [seed_audit, delta_audit],
            "pair": backward_pair_audit,
        },
        "recent_original33": {
            "paired_unique_headlines": len(recent),
            "removed_add11_only_headlines": recent_removed,
            "finbert": recent_finbert_audit,
            "gemma": recent_gemma_audit,
            "finbert_metadata_mismatch_counts_before_restriction": recent_all.attrs.get(
                "finbert_seed_metadata_mismatch_counts", {}
            ),
        },
        "join": join_audit,
        "prices_or_returns_loaded": False,
    }
    return {"backward": backward, "recent": recent, "joined": joined}, audit


__all__ = [
    "BACKWARD_POPULATION",
    "JOIN_BOUNDARY",
    "JoinedLsegInputs",
    "load_joined_same33_scores",
]
