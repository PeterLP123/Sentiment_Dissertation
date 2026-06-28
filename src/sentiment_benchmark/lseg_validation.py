from __future__ import annotations

import csv
import io
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .agreement import cohen_kappa
from .artifact_io import atomic_write_json, atomic_write_text, sha256_file, sha256_text
from .constants import VALID_LABELS
from .lseg_cohort import load_verified_lseg_cohort
from .lseg_corpus import load_verified_lseg_corpus
from .lseg_source import LsegNewsError, utc_now


@dataclass(frozen=True)
class LsegValidationSample:
    output_dir: Path
    primary_path: Path
    secondary_path: Path
    manifest_path: Path
    sample_count: int
    double_code_count: int


@dataclass(frozen=True)
class LsegAnnotationEvaluation:
    output_dir: Path
    labeled_dataset_path: Path
    metrics_path: Path
    percent_agreement: float
    cohen_kappa: float
    double_code_count: int
    relevance_percent_agreement: float = 1.0
    relevance_cohen_kappa: float = 1.0


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    return atomic_write_text(path, stream.getvalue())


def _assigned_symbol(article: dict[str, Any]) -> str:
    symbols = sorted(str(value) for value in article.get("matched_symbols") or [] if str(value))
    if not symbols:
        raise LsegNewsError(f"eligible LSEG revision has no matched symbol: {article.get('revision_id')}")
    digest = int(sha256_text(str(article.get("revision_id")))[:8], 16)
    return symbols[digest % len(symbols)]


def _stratified_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size < 1:
        raise LsegNewsError("sample size must be positive")
    if len(rows) < size:
        raise LsegNewsError(f"requested {size} LSEG stories, but only {len(rows)} eligible revisions are available")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["sample_symbol"]), str(row["news_date"]))].append(row)
    generator = random.Random(seed)
    keys = sorted(groups)
    generator.shuffle(keys)
    for key in keys:
        generator.shuffle(groups[key])
    selected: list[dict[str, Any]] = []
    while len(selected) < size:
        made_progress = False
        for key in keys:
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop())
                made_progress = True
        if not made_progress:
            break
    return sorted(selected, key=lambda row: (str(row["sample_symbol"]), str(row["news_date"]), str(row["revision_id"])))


def create_lseg_validation_sample(
    corpus_manifest: str | Path,
    output_dir: str | Path,
    *,
    sample_size: int = 150,
    double_code_size: int = 30,
    seed: int = 42,
    double_code_seed: int = 43,
    cohort_manifest: str | Path | None = None,
) -> LsegValidationSample:
    """Create local-only, deterministic annotation sheets from a verified LSEG corpus."""
    if double_code_size < 1 or double_code_size > sample_size:
        raise LsegNewsError("double-code size must be between one and the full sample size")
    destination = Path(output_dir)
    if destination.exists():
        raise LsegNewsError(f"refusing to overwrite validation sample directory: {destination}")
    source_manifest = Path(cohort_manifest or corpus_manifest)
    if cohort_manifest is not None:
        _, articles = load_verified_lseg_cohort(cohort_manifest)
    else:
        _, articles = load_verified_lseg_corpus(corpus_manifest)
    eligible: list[dict[str, Any]] = []
    for article in articles:
        if not article.get("scoring_eligible") or not article.get("version_created"):
            continue
        eligible.append(
            {
                "article_id": article["article_id"],
                "revision_id": article["revision_id"],
                "story_id": article["story_id"],
                "sample_symbol": str(article.get("symbol") or _assigned_symbol(article)),
                "news_date": str(article.get("news_date") or article["version_created"])[:10],
                "version_created": article["version_created"],
                "headline": article.get("headline") or "",
                "clean_text": article.get("clean_text") or "",
                "primary_relevance": "",
                "relevance_adjudication": "",
                "primary_sentiment": "",
                "sentiment_adjudication": "",
            }
        )
    selected = _stratified_sample(eligible, sample_size, seed)
    double_generator = random.Random(double_code_seed)
    double_ids = set(double_generator.sample([str(row["revision_id"]) for row in selected], double_code_size))
    secondary = [
        {
            "article_id": row["article_id"],
            "revision_id": row["revision_id"],
            "story_id": row["story_id"],
            "sample_symbol": row["sample_symbol"],
            "news_date": row["news_date"],
            "version_created": row["version_created"],
            "headline": row["headline"],
            "clean_text": row["clean_text"],
            "secondary_relevance": "",
            "secondary_sentiment": "",
        }
        for row in selected
        if str(row["revision_id"]) in double_ids
    ]
    destination.mkdir(parents=True, exist_ok=False)
    common = [
        "article_id",
        "revision_id",
        "story_id",
        "sample_symbol",
        "news_date",
        "version_created",
        "headline",
        "clean_text",
    ]
    primary_path = _write_csv(
        destination / "annotation_primary.csv",
        selected,
        [
            *common,
            "primary_relevance",
            "relevance_adjudication",
            "primary_sentiment",
            "sentiment_adjudication",
        ],
    )
    secondary_path = _write_csv(
        destination / "annotation_secondary.csv",
        secondary,
        [*common, "secondary_relevance", "secondary_sentiment"],
    )
    manifest_path = destination / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "annotation_pending",
            "created_at": utc_now(),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "source_kind": "cohort" if cohort_manifest is not None else "corpus",
            "sampling": {
                "method": "round_robin_ticker_date_strata",
                "sample_size": sample_size,
                "seed": seed,
                "double_code_size": double_code_size,
                "double_code_seed": double_code_seed,
            },
            "files": {
                "primary": {"path": primary_path.name, "sha256": sha256_file(primary_path)},
                "secondary": {"path": secondary_path.name, "sha256": sha256_file(secondary_path)},
            },
            "sharing": {"redistribute": False, "licensed_full_text": True},
        },
    )
    return LsegValidationSample(
        destination,
        primary_path,
        secondary_path,
        manifest_path,
        sample_size,
        double_code_size,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _validated_label(value: str, *, field: str, revision_id: str) -> str:
    label = value.strip().lower()
    if label not in VALID_LABELS:
        raise LsegNewsError(f"{field} for revision {revision_id} must be positive, negative, or neutral")
    return label


def _validated_relevance(value: str, *, field: str, revision_id: str) -> str:
    relevance = value.strip().lower()
    if relevance not in {"relevant", "irrelevant"}:
        raise LsegNewsError(f"{field} for revision {revision_id} must be relevant or irrelevant")
    return relevance


def evaluate_lseg_annotations(
    primary_path: str | Path,
    secondary_path: str | Path,
    output_dir: str | Path,
) -> LsegAnnotationEvaluation:
    """Validate joint double coding and export adjudicated relevant stories."""
    primary_file = Path(primary_path)
    secondary_file = Path(secondary_path)
    destination = Path(output_dir)
    if destination.exists():
        raise LsegNewsError(f"refusing to overwrite annotation evaluation directory: {destination}")
    primary = _read_csv(primary_file)
    secondary = {row["revision_id"]: row for row in _read_csv(secondary_file)}
    sentiment_pairs: list[tuple[str, str]] = []
    relevance_pairs: list[tuple[str, str]] = []
    benchmark_rows: list[dict[str, str]] = []
    joint_schema = bool(primary and "primary_relevance" in primary[0])
    for row in primary:
        revision_id = row["revision_id"]
        if joint_schema:
            first_relevance = _validated_relevance(
                row.get("primary_relevance", ""),
                field="primary_relevance",
                revision_id=revision_id,
            )
            first_sentiment = (
                _validated_label(
                    row.get("primary_sentiment", ""),
                    field="primary_sentiment",
                    revision_id=revision_id,
                )
                if first_relevance == "relevant"
                else ""
            )
        else:
            first_relevance = "relevant"
            first_sentiment = _validated_label(
                row.get("primary_label", ""),
                field="primary_label",
                revision_id=revision_id,
            )
        final_relevance = first_relevance
        final_sentiment = first_sentiment
        if revision_id in secondary:
            if joint_schema:
                second_relevance = _validated_relevance(
                    secondary[revision_id].get("secondary_relevance", ""),
                    field="secondary_relevance",
                    revision_id=revision_id,
                )
                second_sentiment = (
                    _validated_label(
                        secondary[revision_id].get("secondary_sentiment", ""),
                        field="secondary_sentiment",
                        revision_id=revision_id,
                    )
                    if second_relevance == "relevant"
                    else ""
                )
                if first_relevance != second_relevance:
                    final_relevance = _validated_relevance(
                        row.get("relevance_adjudication", ""),
                        field="relevance_adjudication",
                        revision_id=revision_id,
                    )
            else:
                second_relevance = "relevant"
                second_sentiment = _validated_label(
                    secondary[revision_id].get("secondary_label", ""),
                    field="secondary_label",
                    revision_id=revision_id,
                )
            relevance_pairs.append((first_relevance, second_relevance))
            if first_sentiment and second_sentiment:
                sentiment_pairs.append((first_sentiment, second_sentiment))
            if final_relevance == "relevant" and first_sentiment != second_sentiment:
                final_sentiment = _validated_label(
                    row.get("sentiment_adjudication" if joint_schema else "adjudicated_label", ""),
                    field="sentiment_adjudication" if joint_schema else "adjudicated_label",
                    revision_id=revision_id,
                )
        if final_relevance != "relevant":
            continue
        sentence = f"{row.get('headline', '').strip()}\n\n{row.get('clean_text', '').strip()}".strip()
        benchmark_rows.append(
            {
                "Sentence": sentence,
                "Sentiment": final_sentiment,
                "article_id": row["article_id"],
                "revision_id": revision_id,
                "symbol": row["sample_symbol"],
                "news_date": row["news_date"],
            }
        )
    if not relevance_pairs:
        raise LsegNewsError("secondary annotation file contains no revisions from the primary sample")
    percent = sum(a == b for a, b in sentiment_pairs) / len(sentiment_pairs) if sentiment_pairs else 0.0
    kappa = cohen_kappa(
        [a for a, _ in sentiment_pairs],
        [b for _, b in sentiment_pairs],
    ) if sentiment_pairs else 0.0
    relevance_percent = sum(a == b for a, b in relevance_pairs) / len(relevance_pairs)
    relevance_kappa = cohen_kappa(
        [a for a, _ in relevance_pairs],
        [b for _, b in relevance_pairs],
    )
    unique_dates = sorted({row["news_date"] for row in benchmark_rows})
    if len(unique_dates) >= 2:
        development_dates, holdout_dates = chronological_date_split(unique_dates)
    else:
        development_dates, holdout_dates = tuple(unique_dates), ()
    holdout_set = set(holdout_dates)
    for row in benchmark_rows:
        row["chronological_split"] = "holdout" if row["news_date"] in holdout_set else "development"
    destination.mkdir(parents=True, exist_ok=False)
    labeled_path = _write_csv(
        destination / "labeled_lseg_sample.csv",
        benchmark_rows,
        ["Sentence", "Sentiment", "article_id", "revision_id", "symbol", "news_date", "chronological_split"],
    )
    metrics_path = destination / "annotation_agreement.json"
    atomic_write_json(
        metrics_path,
        {
            "schema_version": 1,
            "created_at": utc_now(),
            "double_coded_items": len(relevance_pairs),
            "percent_agreement": percent,
            "cohen_kappa": kappa,
            "relevance_percent_agreement": relevance_percent,
            "relevance_cohen_kappa": relevance_kappa,
            "adjudication_required_for_disagreements": True,
            "chronological_split": {
                "method": "sorted_unique_dates_70_30",
                "development_fraction": 0.7,
                "development_dates": development_dates,
                "holdout_dates": holdout_dates,
                "freeze_before_holdout": [
                    "model_tags_and_digests",
                    "prompt_hash",
                    "cleaner_version",
                    "scoring_representation",
                    "decision_policy",
                ],
            },
            "inputs": {
                str(primary_file): sha256_file(primary_file),
                str(secondary_file): sha256_file(secondary_file),
            },
            "output": {str(labeled_path): sha256_file(labeled_path)},
            "sharing": {"redistribute": False, "licensed_full_text": True},
        },
    )
    return LsegAnnotationEvaluation(
        destination,
        labeled_path,
        metrics_path,
        percent,
        kappa,
        len(relevance_pairs),
        relevance_percent,
        relevance_kappa,
    )


def chronological_date_split(dates: list[str], development_fraction: float = 0.7) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split sorted unique dates once, keeping every date entirely in development or holdout."""
    if not 0 < development_fraction < 1:
        raise ValueError("development_fraction must be between zero and one")
    ordered = sorted(set(dates))
    if len(ordered) < 2:
        raise ValueError("at least two distinct dates are required")
    boundary = max(1, min(len(ordered) - 1, int(len(ordered) * development_fraction)))
    return tuple(ordered[:boundary]), tuple(ordered[boundary:])
