"""Local baseline scoring and no-look-ahead headline return analysis."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .artifact_io import atomic_write_text, sha256_file, sha256_text
from .baselines import SoftSentiment, iter_finbert_text_batches, score_finbert_texts, score_vader_text
from .headline_value import collect_scorable_headline_records
from .runtime_metadata import collect_run_environment

BASELINE_COLUMNS = (
    "headline_sha256",
    "headline",
    "matched_symbols",
    "first_timestamp",
    "explicit_target",
    "contextual",
    "market_price_technical",
    "baseline",
    "label",
    "p_positive",
    "p_negative",
    "p_neutral",
    "score",
    "score_100",
    "status",
    "error",
)

POPULATIONS = {
    "primary_explicit_nontechnical": lambda frame: frame["explicit_target"] & ~frame["market_price_technical"],
    "all_direct_candidates": lambda frame: pd.Series(True, index=frame.index),
    "explicit_including_technical": lambda frame: frame["explicit_target"],
    "all_nontechnical": lambda frame: ~frame["market_price_technical"],
    "contextual_nontechnical": lambda frame: frame["contextual"] & ~frame["market_price_technical"],
}


@dataclass(frozen=True)
class BaselineScoringSummary:
    output_path: Path
    manifest_path: Path
    input_rows: int
    succeeded: int
    failed: int
    runtime_seconds: float


@dataclass(frozen=True)
class ReturnStudySummary:
    output_dir: Path
    manifest_path: Path
    report_path: Path
    aligned_rows: int
    split_date: str


@dataclass(frozen=True)
class ScoreAuditSummary:
    output_path: Path
    rows: int
    unique_headlines: int
    succeeded: int
    failed: int
    valid: bool


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def audit_headline_scores(
    scores_path: str | Path,
    output_path: str | Path,
    *,
    expected_rows: int,
) -> ScoreAuditSummary:
    """Validate score schema and reconciliation without reproducing headline text."""
    source = Path(scores_path)
    output = Path(output_path)
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "headline_sha256",
        "model_id",
        "model_digest",
        "prompt_hash",
        "label",
        "p_positive",
        "p_negative",
        "p_neutral",
        "score",
        "score_100",
        "status",
        "error",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    missing = sorted(required - set(rows[0] if rows else ()))
    errors: list[str] = []
    if missing:
        errors.append(f"missing columns: {missing}")
    hashes = [row.get("headline_sha256", "") for row in rows]
    if len(rows) != expected_rows:
        errors.append(f"expected {expected_rows} rows, found {len(rows)}")
    if len(set(hashes)) != len(hashes) or "" in hashes:
        errors.append("headline hashes are blank or duplicated")
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    for index, row in enumerate(successes, start=1):
        try:
            probabilities = [float(row[f"p_{label}"]) for label in ("positive", "negative", "neutral")]
            score = float(row["score"])
            score_100 = float(row["score_100"])
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
                raise ValueError("probability outside [0, 1]")
            if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
                raise ValueError("probabilities do not sum to one")
            if not math.isclose(score, probabilities[0] - probabilities[1], abs_tol=1e-12):
                raise ValueError("score does not equal p_positive - p_negative")
            if not math.isclose(score_100, 100 * score, abs_tol=1e-9):
                raise ValueError("score_100 does not equal 100 * score")
            if not -1 <= score <= 1 or row["label"] not in {"positive", "negative", "neutral"}:
                raise ValueError("score or normalized label is invalid")
            if not row["model_digest"] or not row["prompt_hash"]:
                raise ValueError("model digest or prompt hash is blank")
            if any(float(row[column]) < 0 for column in ("latency_ms", "prompt_tokens", "completion_tokens", "total_tokens")):
                raise ValueError("negative timing or token count")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"successful row {index}: {exc}")
    if any(not row.get("error") for row in failures):
        errors.append("one or more failed rows lacks a documented error")
    digests = sorted({row.get("model_digest", "") for row in rows})
    prompt_hashes = sorted({row.get("prompt_hash", "") for row in rows})
    report = {
        "schema_version": 1,
        "audited_at": datetime.now(UTC).isoformat(),
        "input": {"path": str(source), "sha256": sha256_file(source)},
        "counts": {
            "expected": expected_rows,
            "rows": len(rows),
            "unique_headlines": len(set(hashes)),
            "succeeded": len(successes),
            "failed": len(failures),
        },
        "model_digests": digests,
        "prompt_hashes": prompt_hashes,
        "valid": not errors,
        "errors": errors[:100],
        "sharing": {"contains_licensed_headline_text": False},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ScoreAuditSummary(output, len(rows), len(set(hashes)), len(successes), len(failures), not errors)


def _append_csv(path: Path, rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> None:
    batch = list(rows)
    if not batch:
        return
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if write_header:
            writer.writeheader()
        writer.writerows(batch)


def _baseline_row(source: dict[str, str], name: str, result: SoftSentiment) -> dict[str, Any]:
    return {
        **{key: source.get(key, "") for key in BASELINE_COLUMNS[:7]},
        "baseline": name,
        "label": result.label,
        "p_positive": result.p_positive,
        "p_negative": result.p_negative,
        "p_neutral": result.p_neutral,
        "score": result.score,
        "score_100": 100 * result.score,
        "status": "success",
        "error": "",
    }


def _finbert_revision() -> str | None:
    try:
        from huggingface_hub import scan_cache_dir

        for repository in scan_cache_dir().repos:
            if repository.repo_id == "ProsusAI/finbert" and repository.revisions:
                return sorted(revision.commit_hash for revision in repository.revisions)[-1]
    except Exception:
        return None
    return None


def score_headline_baselines(
    gemma_scores_path: str | Path,
    output_path: str | Path,
    *,
    finbert_batch_size: int = 32,
) -> BaselineScoringSummary:
    """Score the frozen Gemma population locally with VADER and FinBERT."""
    started = time.monotonic()
    started_at = datetime.now(UTC)
    source_path = Path(gemma_scores_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite baseline scores: {output}")
    with source_path.open(encoding="utf-8", newline="") as handle:
        source_rows = [row for row in csv.DictReader(handle) if row.get("status") == "success"]
    hashes = [row["headline_sha256"] for row in source_rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Gemma scores must contain exactly one successful row per headline hash")

    failures = 0
    vader_rows: list[dict[str, Any]] = []
    for row in source_rows:
        try:
            vader_rows.append(_baseline_row(row, "vader", score_vader_text(row["headline"])))
        except Exception as exc:
            failures += 1
            vader_rows.append(
                {
                    **{key: row.get(key, "") for key in BASELINE_COLUMNS[:7]},
                    "baseline": "vader",
                    "label": "",
                    "p_positive": "",
                    "p_negative": "",
                    "p_neutral": "",
                    "score": "",
                    "score_100": "",
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}"[:500],
                }
            )
    _append_csv(output, vader_rows, BASELINE_COLUMNS)

    finbert_rows: list[dict[str, Any]] = []
    try:
        results = score_finbert_texts([row["headline"] for row in source_rows], batch_size=finbert_batch_size)
        if len(results) != len(source_rows):
            raise ValueError(f"FinBERT returned {len(results)} scores for {len(source_rows)} inputs")
        finbert_rows = [
            _baseline_row(row, "finbert", result)
            for row, result in zip(source_rows, results, strict=True)
        ]
    except Exception as exc:
        failures += len(source_rows)
        finbert_rows = [
            {
                **{key: row.get(key, "") for key in BASELINE_COLUMNS[:7]},
                "baseline": "finbert",
                "label": "",
                "p_positive": "",
                "p_negative": "",
                "p_neutral": "",
                "score": "",
                "score_100": "",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}"[:500],
            }
            for row in source_rows
        ]
    _append_csv(output, finbert_rows, BASELINE_COLUMNS)

    runtime = time.monotonic() - started
    succeeded = len(vader_rows) + len(finbert_rows) - failures
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime,
        "input": {"path": str(source_path), "sha256": sha256_file(source_path), "rows": len(source_rows)},
        "models": {
            "vader": {"implementation": "nltk.sentiment.vader"},
            "finbert": {
                "model_id": "ProsusAI/finbert",
                "local_model_path": os.getenv("SENTIMENT_FINBERT_MODEL"),
                "revision": os.getenv("SENTIMENT_FINBERT_REVISION") or _finbert_revision(),
            },
        },
        "inference": {
            "local_only": True,
            "finbert_batch_size": finbert_batch_size,
            "torch_device": _torch_device(),
        },
        "environment": {
            "run": collect_run_environment(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("nltk", "torch", "transformers")
            },
        },
        "counts": {"attempted": 2 * len(source_rows), "succeeded": succeeded, "failed": failures},
        "output": {"path": str(output), "sha256": sha256_file(output)},
        "sharing": {"contains_licensed_headline_text": True, "source_control": False, "redistribute": False},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return BaselineScoringSummary(output, manifest_path, len(source_rows), succeeded, failures, runtime)


def _collection_baseline_source_rows(collection_root: str | Path) -> list[dict[str, str]]:
    records = collect_scorable_headline_records(collection_root)
    return [
        {
            "headline_sha256": record.headline_sha256,
            "headline": record.headline,
            "matched_symbols": "|".join(record.matched_symbols),
            "first_timestamp": record.first_timestamp,
            "explicit_target": str(record.explicit_target),
            "contextual": str(record.contextual),
            "market_price_technical": str(record.market_price_technical),
        }
        for record in records
    ]


def _existing_baseline_successes(path: Path, population: set[str]) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BASELINE_COLUMNS:
            raise ValueError(f"cannot resume baseline scores with an incompatible schema: {path}")
        successes: set[tuple[str, str]] = set()
        for row in reader:
            if row.get("status") != "success":
                continue
            key = (str(row.get("headline_sha256") or ""), str(row.get("baseline") or ""))
            if key[0] not in population or key[1] not in {"finbert", "vader"}:
                raise ValueError(f"cannot resume baseline scores with an unexpected successful row: {key}")
            if key in successes:
                raise ValueError(f"cannot resume baseline scores with a duplicated successful row: {key}")
            successes.add(key)
    return successes


def score_collection_baselines(
    collection_root: str | Path,
    output_path: str | Path,
    *,
    finbert_batch_size: int = 32,
    finbert_checkpoint_size: int = 256,
    vader_flush_size: int = 5_000,
) -> BaselineScoringSummary:
    """Resumably score the complete frozen collection with local VADER and FinBERT."""
    started = time.monotonic()
    started_at = datetime.now(UTC)
    root = Path(collection_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = _collection_baseline_source_rows(root)
    population = {row["headline_sha256"] for row in source_rows}
    if len(population) != len(source_rows):
        raise ValueError("collection population contains duplicated headline hashes")
    population_sha256 = sha256_text("\n".join(sorted(population)))
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_population = str(previous.get("input", {}).get("population_sha256") or "")
        if previous_population and previous_population != population_sha256:
            raise ValueError("cannot resume baseline scores after the frozen population changed")
    successes = _existing_baseline_successes(output, population)
    finbert_revision = os.getenv("SENTIMENT_FINBERT_REVISION") or _finbert_revision()
    run_environment = collect_run_environment()
    package_versions = {
        name: importlib.metadata.version(name)
        for name in ("nltk", "torch", "transformers")
    }

    def write_manifest(*, status: str) -> None:
        completed = len(successes)
        manifest = {
            "schema_version": 1,
            "status": status,
            "started_at": started_at.isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "runtime_seconds_this_invocation": time.monotonic() - started,
            "input": {
                "collection_root": str(root),
                "unique_headlines": len(source_rows),
                "population_sha256": population_sha256,
            },
            "models": {
                "vader": {"implementation": "nltk.sentiment.vader"},
                "finbert": {
                    "model_id": "ProsusAI/finbert",
                    "local_model_path": os.getenv("SENTIMENT_FINBERT_MODEL"),
                    "revision": finbert_revision,
                },
            },
            "inference": {
                "local_only": True,
                "finbert_batch_size": finbert_batch_size,
                "finbert_checkpoint_size": finbert_checkpoint_size,
                "torch_device": _torch_device(),
            },
            "environment": {
                "run": run_environment,
                "packages": package_versions,
            },
            "counts": {
                "expected": 2 * len(source_rows),
                "successful_unique_model_headlines": completed,
                "remaining": 2 * len(source_rows) - completed,
                "vader_successes": sum(name == "vader" for _, name in successes),
                "finbert_successes": sum(name == "finbert" for _, name in successes),
            },
            "output": {
                "path": str(output),
                "sha256": sha256_file(output) if status == "completed" else None,
            },
            "sharing": {"contains_licensed_headline_text": True, "source_control": False, "redistribute": False},
        }
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    vader_pending = [row for row in source_rows if (row["headline_sha256"], "vader") not in successes]
    for start in range(0, len(vader_pending), vader_flush_size):
        batch_rows: list[dict[str, Any]] = []
        for row in vader_pending[start : start + vader_flush_size]:
            result = score_vader_text(row["headline"])
            batch_rows.append(_baseline_row(row, "vader", result))
            successes.add((row["headline_sha256"], "vader"))
        _append_csv(output, batch_rows, BASELINE_COLUMNS)
        write_manifest(status="running")

    finbert_pending = [row for row in source_rows if (row["headline_sha256"], "finbert") not in successes]
    texts = [row["headline"] for row in finbert_pending]
    offset = 0
    for result_batch in iter_finbert_text_batches(
        texts,
        batch_size=finbert_checkpoint_size,
        inference_batch_size=finbert_batch_size,
    ):
        batch_sources = finbert_pending[offset : offset + len(result_batch)]
        if len(batch_sources) != len(result_batch):
            raise ValueError("FinBERT returned more scores than requested")
        _append_csv(
            output,
            [_baseline_row(row, "finbert", result) for row, result in zip(batch_sources, result_batch, strict=True)],
            BASELINE_COLUMNS,
        )
        successes.update((row["headline_sha256"], "finbert") for row in batch_sources)
        offset += len(result_batch)
        write_manifest(status="running")
    if offset != len(finbert_pending):
        raise ValueError(f"FinBERT returned {offset} scores for {len(finbert_pending)} requested headlines")
    if len(successes) != 2 * len(source_rows):
        raise ValueError("baseline scoring finished without complete unique model/headline coverage")
    write_manifest(status="completed")
    runtime = time.monotonic() - started
    return BaselineScoringSummary(output, manifest_path, len(source_rows), len(successes), 0, runtime)


def _torch_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_signal_frames(gemma_path: Path, baseline_path: Path) -> pd.DataFrame:
    gemma = pd.read_csv(gemma_path)
    gemma = gemma.loc[gemma["status"].eq("success")].copy()
    gemma["scorer"] = "gemma4:e4b-it-qat"
    baseline = pd.read_csv(baseline_path)
    baseline = baseline.loc[baseline["status"].eq("success")].copy()
    baseline["scorer"] = baseline["baseline"]
    columns = [
        "headline_sha256",
        "matched_symbols",
        "first_timestamp",
        "explicit_target",
        "contextual",
        "market_price_technical",
        "label",
        "score",
        "score_100",
        "scorer",
    ]
    signals = pd.concat([gemma[columns], baseline[columns]], ignore_index=True)
    for column in ("explicit_target", "contextual", "market_price_technical"):
        signals[column] = signals[column].map(_as_bool)
    signals["first_timestamp"] = pd.to_datetime(signals["first_timestamp"], format="mixed", utc=True)
    signals["score"] = pd.to_numeric(signals["score"], errors="raise")
    signals["association_count"] = signals["matched_symbols"].str.count(r"\|") + 1
    signals["symbol"] = signals["matched_symbols"].str.split("|")
    return signals.explode("symbol", ignore_index=True).drop(columns="matched_symbols")


def align_next_open_returns(signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Assign each signal to the first available open strictly after publication."""
    required = {"symbol", "session_date", "open"}
    if not required.issubset(prices.columns):
        raise ValueError(f"price panel is missing columns: {sorted(required - set(prices.columns))}")
    prices = prices.copy()
    prices["session_date"] = pd.to_datetime(prices["session_date"]).dt.date
    prices = prices.sort_values(["symbol", "session_date"])
    aligned: list[dict[str, Any]] = []
    for symbol, symbol_signals in signals.groupby("symbol", sort=True):
        panel = prices.loc[prices["symbol"].eq(symbol)].copy()
        if panel.empty:
            continue
        local_opens = pd.to_datetime(panel["session_date"].astype(str) + " 09:30").dt.tz_localize(
            "America/New_York",
            ambiguous="raise",
            nonexistent="raise",
        )
        open_utc = local_opens.dt.tz_convert("UTC").array
        opens = panel["open"].astype(float).to_numpy()
        dates = panel["session_date"].to_numpy()
        for record in symbol_signals.to_dict("records"):
            index = int(open_utc.searchsorted(record["first_timestamp"], side="right"))
            if index >= len(panel) - 1:
                continue
            entry_open = opens[index]
            exit_open = opens[index + 1]
            if not math.isfinite(entry_open) or entry_open <= 0 or not math.isfinite(exit_open):
                continue
            aligned.append(
                {
                    **record,
                    "entry_date": dates[index],
                    "exit_date": dates[index + 1],
                    "entry_open": entry_open,
                    "exit_open": exit_open,
                    "forward_return": exit_open / entry_open - 1,
                }
            )
    return pd.DataFrame(aligned)


def _rank_correlation(frame: pd.DataFrame) -> float:
    if len(frame) < 3 or frame["score"].nunique() < 2 or frame["forward_return"].nunique() < 2:
        return float("nan")
    return float(frame["score"].rank(method="average").corr(frame["forward_return"].rank(method="average")))


def _bootstrap_intervals(frame: pd.DataFrame, samples: int, seed: int) -> dict[str, float]:
    if frame.empty or samples <= 0:
        return {
            key: float("nan")
            for key in ("mean_low", "mean_high", "hit_low", "hit_high", "rho_low", "rho_high")
        }
    dates = np.array(sorted(frame["entry_date"].unique()))
    rng = np.random.default_rng(seed)
    means: list[float] = []
    hits: list[float] = []
    correlations: list[float] = []
    groups = {date: frame.loc[frame["entry_date"].eq(date)] for date in dates}
    for _ in range(samples):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        draw = pd.concat([groups[date] for date in sampled], ignore_index=True)
        means.append(float(draw["strategy_return"].mean()))
        active = draw.loc[draw["position"].ne(0)]
        hits.append(float((active["strategy_return"] > 0).mean()) if not active.empty else float("nan"))
        correlations.append(_rank_correlation(draw))
    valid_hits = np.array([value for value in hits if math.isfinite(value)])
    valid_correlations = np.array([value for value in correlations if math.isfinite(value)])
    return {
        "mean_low": float(np.quantile(means, 0.025)),
        "mean_high": float(np.quantile(means, 0.975)),
        "hit_low": float(np.quantile(valid_hits, 0.025)) if len(valid_hits) else float("nan"),
        "hit_high": float(np.quantile(valid_hits, 0.975)) if len(valid_hits) else float("nan"),
        "rho_low": float(np.quantile(valid_correlations, 0.025)) if len(valid_correlations) else float("nan"),
        "rho_high": float(np.quantile(valid_correlations, 0.975)) if len(valid_correlations) else float("nan"),
    }


def calculate_return_metrics(frame: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> dict[str, Any]:
    company_days = (
        frame.groupby(["entry_date", "symbol"], as_index=False)
        .agg(score=("score", "mean"), forward_return=("forward_return", "first"), headlines=("headline_sha256", "nunique"))
        .sort_values(["entry_date", "symbol"])
    )
    company_days["position"] = np.sign(company_days["score"])
    company_days["strategy_return"] = company_days["position"] * company_days["forward_return"]
    daily = company_days.groupby("entry_date")["strategy_return"].mean().sort_index()
    daily_std = float(daily.std(ddof=1)) if len(daily) > 1 else float("nan")
    sharpe = float(math.sqrt(252) * daily.mean() / daily_std) if daily_std > 0 else float("nan")
    equity = (1 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1 if not equity.empty else pd.Series(dtype=float)
    positions = company_days.pivot(index="entry_date", columns="symbol", values="position").fillna(0).sort_index()
    previous = positions.shift(1, fill_value=0)
    turnover = float((positions - previous).abs().sum(axis=1).div(2 * max(1, positions.shape[1])).mean())
    active = company_days.loc[company_days["position"].ne(0)]
    intervals = _bootstrap_intervals(company_days, bootstrap_samples, seed)
    return {
        "company_day_observations": len(company_days),
        "headline_observations": int(company_days["headlines"].sum()),
        "trading_dates": int(company_days["entry_date"].nunique()),
        "mean_forward_return": float(company_days["forward_return"].mean()),
        "mean_strategy_return": float(company_days["strategy_return"].mean()),
        "mean_strategy_return_ci_low": intervals["mean_low"],
        "mean_strategy_return_ci_high": intervals["mean_high"],
        "hit_rate": float((active["strategy_return"] > 0).mean()) if not active.empty else float("nan"),
        "hit_rate_ci_low": intervals["hit_low"],
        "hit_rate_ci_high": intervals["hit_high"],
        "rank_correlation": _rank_correlation(company_days),
        "rank_correlation_ci_low": intervals["rho_low"],
        "rank_correlation_ci_high": intervals["rho_high"],
        "mean_one_way_turnover": turnover,
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if not drawdown.empty else float("nan"),
    }


def run_headline_return_study(
    gemma_scores_path: str | Path,
    baseline_scores_path: str | Path,
    prices_path: str | Path,
    output_dir: str | Path,
    *,
    development_fraction: float = 0.7,
    bootstrap_samples: int = 1_000,
    seed: int = 42,
    event_records_path: str | Path | None = None,
) -> ReturnStudySummary:
    """Run the frozen open-to-open study and emit aggregate-only artifacts."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one")
    started = time.monotonic()
    started_at = datetime.now(UTC)
    gemma_path = Path(gemma_scores_path)
    baseline_path = Path(baseline_scores_path)
    price_path = Path(prices_path)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty study directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    signals = _load_signal_frames(gemma_path, baseline_path)
    prices = pd.read_csv(price_path)
    # A normalized headline observed separately for more than one queried company
    # has only one global first timestamp in the frozen score row. Exclude those
    # rare ambiguous associations rather than assign another company too early.
    analysis_signals = signals.loc[signals["association_count"].eq(1)].copy()
    aligned = align_next_open_returns(analysis_signals, prices)
    dates = sorted(aligned["entry_date"].unique())
    split_index = max(1, min(len(dates) - 1, int(len(dates) * development_fraction)))
    split_date = dates[split_index - 1]
    aligned["sample"] = np.where(aligned["entry_date"] <= split_date, "development", "holdout")

    metric_rows: list[dict[str, Any]] = []
    for scorer in sorted(aligned["scorer"].unique()):
        scorer_frame = aligned.loc[aligned["scorer"].eq(scorer)]
        for population, selector in POPULATIONS.items():
            population_frame = scorer_frame.loc[selector(scorer_frame)]
            for sample in ("all", "development", "holdout"):
                sample_frame = population_frame if sample == "all" else population_frame.loc[population_frame["sample"].eq(sample)]
                metric_rows.append(
                    {
                        "scorer": scorer,
                        "population": population,
                        "sample": sample,
                        **calculate_return_metrics(
                            sample_frame,
                            bootstrap_samples=bootstrap_samples,
                            seed=seed,
                        ),
                    }
                )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(destination / "return_metrics.csv", index=False, lineterminator="\n")

    unique_signals = signals.drop_duplicates(["scorer", "headline_sha256"])
    quantiles = unique_signals.groupby("scorer")["score_100"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]).unstack()
    quantiles.columns = [f"q{int(float(value) * 100):02d}" for value in quantiles.columns]
    summary = unique_signals.groupby("scorer")["score_100"].agg(["count", "mean", "std", "min", "max"]).join(quantiles)
    summary.to_csv(destination / "score_distribution.csv", lineterminator="\n")
    unique_signals.groupby(["scorer", "label"]).size().rename("count").reset_index().to_csv(
        destination / "label_counts.csv", index=False, lineterminator="\n"
    )
    coverage = unique_signals.groupby("scorer").agg(
        headlines=("headline_sha256", "nunique"),
        explicit_target=("explicit_target", "sum"),
        contextual=("contextual", "sum"),
        market_price_technical=("market_price_technical", "sum"),
    )
    coverage["companies"] = signals.groupby("scorer")["symbol"].nunique()
    coverage.to_csv(destination / "coverage.csv", lineterminator="\n")
    if event_records_path is not None:
        events = pd.read_csv(event_records_path, usecols=["normalized_headline_sha256", "event_type"])
        events = events.rename(columns={"normalized_headline_sha256": "headline_sha256"}).drop_duplicates()
        event_coverage = (
            unique_signals[["scorer", "headline_sha256"]]
            .merge(events, on="headline_sha256", how="left", validate="many_to_many")
            .fillna({"event_type": "unclassified"})
            .groupby(["scorer", "event_type"])["headline_sha256"]
            .nunique()
            .rename("headlines")
            .reset_index()
        )
        event_coverage.to_csv(destination / "event_type_coverage.csv", index=False, lineterminator="\n")

    primary_holdout = metrics.loc[
        metrics["population"].eq("primary_explicit_nontechnical") & metrics["sample"].eq("holdout")
    ]
    report_path = destination / "summary.md"
    report_path.write_text(_render_report(primary_holdout, split_date, len(aligned)), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": time.monotonic() - started,
        "inputs": {
            "gemma_scores": {"path": str(gemma_path), "sha256": sha256_file(gemma_path)},
            "baseline_scores": {"path": str(baseline_path), "sha256": sha256_file(baseline_path)},
            "prices": {"path": str(price_path), "sha256": sha256_file(price_path)},
            "event_records": (
                {"path": str(event_records_path), "sha256": sha256_file(event_records_path)}
                if event_records_path is not None
                else None
            ),
        },
        "design": {
            "entry": "first US/Eastern 09:30 session open strictly after publication timestamp",
            "return": "next-open to following-open simple return",
            "development_fraction": development_fraction,
            "split_date": str(split_date),
            "bootstrap": {"unit": "entry_date", "samples": bootstrap_samples, "seed": seed},
            "position": "sign(mean company-day headline score)",
            "turnover": "mean one-way absolute position change divided by 2N symbols",
            "association_exclusion": "normalized score rows associated with more than one company",
        },
        "counts": {
            "aligned_model_headline_rows": len(aligned),
            "excluded_multi_company_score_rows": int(
                signals.loc[signals["association_count"].gt(1), ["scorer", "headline_sha256"]]
                .drop_duplicates()
                .shape[0]
            ),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "pid": os.getpid()},
        "outputs": {
            path.name: sha256_file(path)
            for path in destination.iterdir()
            if path.is_file() and path.name != "manifest.json"
        },
        "sharing": {"contains_licensed_headline_text": False, "safe_aggregate_only": True},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ReturnStudySummary(destination, manifest_path, report_path, len(aligned), str(split_date))


def _render_report(primary_holdout: pd.DataFrame, split_date: object, aligned_rows: int) -> str:
    lines = [
        "# Local headline sentiment return study",
        "",
        "## Observed results",
        "",
        f"The fixed chronological development period ends on {split_date}. "
        f"The alignment produced {aligned_rows:,} model-headline-company rows "
        "before company-day aggregation.",
        "",
        "Primary holdout results exclude contextual and market-price/technical headlines:",
        "",
        "| Scorer | Company-days | Mean strategy return | 95% CI | Hit rate | Rank correlation | Sharpe | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary_holdout.to_dict("records"):
        lines.append(
            f"| {row['scorer']} | {int(row['company_day_observations']):,} | {row['mean_strategy_return']:.4%} | "
            f"[{row['mean_strategy_return_ci_low']:.4%}, {row['mean_strategy_return_ci_high']:.4%}] | "
            f"{row['hit_rate']:.2%} | {row['rank_correlation']:.3f} | {row['annualized_sharpe']:.2f} | {row['max_drawdown']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These are exploratory associations, not evidence of a causal or deployable trading edge. "
            "The holdout was not used to tune prompts, filters, thresholds, or model selection. "
            "Confidence intervals are clustered by entry date and do not remove all cross-company "
            "or news-event dependence.",
            "",
        ]
    )
    return "\n".join(lines)
