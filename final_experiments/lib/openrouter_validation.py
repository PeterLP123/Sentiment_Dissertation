"""Frozen OpenRouter validation for the public sentiment benchmark.

This module deliberately accepts benchmark rows, not LSEG records.  LSEG headline
text is licensed local data and must not be sent to a hosted provider unless a
separate licence review explicitly permits that processing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.constants import ALLOWED_LABELS
from sentiment_benchmark.dataset import load_dataset
from sentiment_benchmark.env import load_env_file
from sentiment_benchmark.metrics import bootstrap_metric_ci, evaluate_responses
from sentiment_benchmark.models import DatasetRow
from sentiment_benchmark.parser import parse_model_response
from sentiment_benchmark.prompts import load_prompts, render_messages

MODEL_ID = "google/gemma-4-26b-a4b-it"
PROVIDER_NAME = "DeepInfra"
PROVIDER_SLUG = "deepinfra"
PROMPT_ID = "investor_headline_soft_label_v1"
PROMPT_HASH = "81596538d99b29b8"
SAMPLE_METHOD = "first rows in SHA-256(sentence) order; labels excluded from ordering"
DEFAULT_LIMIT = 1_000
DEFAULT_CONCURRENCY = 25
DEFAULT_MAX_COMPLETION_TOKENS = 128
DEFAULT_RETRIES = 3
INPUT_PRICE_PER_MILLION = 0.07
OUTPUT_PRICE_PER_MILLION = 0.34
SCHEMA_VERSION = 1
LOG_LOSS_CLIP = 1e-15


@dataclass(frozen=True)
class ValidationPaths:
    output_dir: Path
    responses: Path
    manifest: Path
    metrics: Path


def validation_paths(output_dir: str | Path) -> ValidationPaths:
    root = Path(output_dir)
    return ValidationPaths(
        output_dir=root,
        responses=root / "responses.jsonl",
        manifest=root / "manifest.json",
        metrics=root / "metrics.json",
    )


def _sentence_sha256(sentence: str) -> str:
    return hashlib.sha256(sentence.strip().encode("utf-8")).hexdigest()


def select_validation_rows(dataset_path: str | Path, limit: int = DEFAULT_LIMIT) -> list[DatasetRow]:
    """Select a deterministic, label-blind benchmark sample."""
    if limit < 1:
        raise ValueError("limit must be positive")
    rows = [row for row in load_dataset(dataset_path) if not row.has_conflicting_duplicate]
    if limit > len(rows):
        raise ValueError(f"limit {limit:,} exceeds the {len(rows):,}-row primary population")
    return sorted(rows, key=lambda row: (_sentence_sha256(row.sentence), row.row_number))[:limit]


def sample_sha256(rows: list[DatasetRow]) -> str:
    payload = "\n".join(f"{row.row_number}|{_sentence_sha256(row.sentence)}" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_contract(*, require_fp8: bool = False) -> dict[str, Any]:
    """Return the exact paid-request settings frozen for this experiment."""
    contract: dict[str, Any] = {
        "temperature": 0.0,
        # Endpoint metadata exposes this capability as ``max_tokens``.  Using
        # that wire name keeps ``require_parameters`` fail-closed and avoids a
        # false unsupported-parameter rejection of its newer API alias.
        "max_tokens": DEFAULT_MAX_COMPLETION_TOKENS,
        "reasoning": {"effort": "none", "exclude": True},
        "provider": {
            "only": [PROVIDER_SLUG],
            "allow_fallbacks": False,
            "require_parameters": True,
            "data_collection": "deny",
            "zdr": True,
            "max_price": {
                "prompt": INPUT_PRICE_PER_MILLION,
                "completion": OUTPUT_PRICE_PER_MILLION,
            },
        },
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "financial_headline_sentiment",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "positive": {"type": "number", "minimum": 0, "maximum": 1},
                        "negative": {"type": "number", "minimum": 0, "maximum": 1},
                        "neutral": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["positive", "negative", "neutral"],
                    "additionalProperties": False,
                },
            },
        },
    }
    if require_fp8:
        contract["provider"]["quantizations"] = ["fp8"]
    return contract


def _config_hash(dataset_sha256: str, selected_sha256: str, limit: int) -> str:
    prompt = load_prompts()[PROMPT_ID]
    if prompt.prompt_hash != PROMPT_HASH:
        raise ValueError(f"frozen prompt hash changed: expected {PROMPT_HASH}, got {prompt.prompt_hash}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "sample_sha256": selected_sha256,
        "sample_method": SAMPLE_METHOD,
        "limit": limit,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "prompt_id": PROMPT_ID,
        "prompt_hash": PROMPT_HASH,
        "request": request_contract(),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_request(row: DatasetRow, *, require_fp8: bool = False) -> dict[str, Any]:
    prompt = load_prompts()[PROMPT_ID]
    if prompt.prompt_hash != PROMPT_HASH:
        raise ValueError(f"frozen prompt hash changed: expected {PROMPT_HASH}, got {prompt.prompt_hash}")
    return {
        "model": MODEL_ID,
        "messages": render_messages(prompt, row.blind()),
        **request_contract(require_fp8=require_fp8),
        "stream": False,
    }


def _retry_delay_seconds(response: httpx.Response | None, attempt: int) -> float:
    """Respect OpenRouter's numeric Retry-After header, with bounded fallback backoff."""
    if response is not None:
        raw_delay = response.headers.get("Retry-After")
        if raw_delay:
            try:
                delay = float(raw_delay)
            except ValueError:
                delay = -1.0
            if math.isfinite(delay) and delay >= 0:
                return delay
    return float(min(2**attempt, 8))


def _load_checkpoint(path: Path, config_hash: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return attempts, latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid checkpoint JSON at {path}:{line_number}") from exc
            if row.get("config_hash") != config_hash:
                raise ValueError("existing checkpoint belongs to a different frozen configuration")
            content_hash = str(row.get("sentence_sha256") or "")
            if not content_hash:
                raise ValueError(f"invalid checkpoint hash at {path}:{line_number}")
            attempts.append(row)
            latest[content_hash] = row
    return attempts, latest


class PinnedOpenRouterClient:
    """Small client that exposes the routing controls absent from the core client."""

    def __init__(self, api_key: str | None = None, timeout: float = 90.0, *, require_fp8: bool = False) -> None:
        load_env_file()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for paid validation requests")
        self.require_fp8 = require_fp8
        self._prompt = load_prompts()[PROMPT_ID]
        if self._prompt.prompt_hash != PROMPT_HASH:
            raise ValueError(f"frozen prompt hash changed: expected {PROMPT_HASH}, got {self._prompt.prompt_hash}")
        self._request_contract = request_contract(require_fp8=require_fp8)
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    def _build_request(self, row: DatasetRow) -> dict[str, Any]:
        """Build a request without reparsing the prompt file for every headline."""
        return {
            "model": MODEL_ID,
            "messages": render_messages(self._prompt, row.blind()),
            **self._request_contract,
            "stream": False,
        }

    async def classify(self, row: DatasetRow, retries: int = DEFAULT_RETRIES) -> dict[str, Any]:
        started = time.monotonic()
        response: httpx.Response | None = None
        error: Exception | None = None
        attempts = 0
        for attempt in range(retries + 1):
            attempts = attempt + 1
            try:
                response = await self._client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-OpenRouter-Title": "Sentiment Dissertation Gemma 4 validation",
                    },
                    json=self._build_request(row),
                )
                if response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                    break
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                error = exc
                if attempt == retries:
                    break
            await asyncio.sleep(_retry_delay_seconds(response, attempt))

        latency_ms = (time.monotonic() - started) * 1_000
        base = {
            "row_number": row.row_number,
            "sentence_sha256": _sentence_sha256(row.sentence),
            "true_label": row.hidden_label,
            "latency_ms": latency_ms,
            "attempt_count": attempts,
        }
        if response is None:
            return {**base, "status": "transport_error", "parse_status": "error", "error": str(error)}
        if response.status_code >= 400:
            return {
                **base,
                "status": "api_error",
                "parse_status": "error",
                "http_status": response.status_code,
                "error": response.text[:1_000],
            }
        try:
            raw = response.json()
            choice = (raw.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
            parsed = parse_model_response(content, "soft_label", ALLOWED_LABELS)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return {**base, "status": "malformed_response", "parse_status": "error", "error": str(exc)}

        usage = raw.get("usage") or {}
        returned_provider = raw.get("provider")
        returned_model = raw.get("model")
        probabilities = parsed.label_probabilities
        valid = parsed.parse_status == "valid" and probabilities is not None
        if valid and not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            valid = False
        route_errors: list[str] = []
        if returned_provider and returned_provider.casefold() not in {PROVIDER_NAME.casefold(), PROVIDER_SLUG.casefold()}:
            valid = False
            route_errors.append(f"unexpected provider {returned_provider!r}")
        if returned_model and returned_model != MODEL_ID:
            valid = False
            route_errors.append(f"unexpected model {returned_model!r}")
        return {
            **base,
            "status": "success" if valid else "invalid_output",
            "parse_status": "valid" if valid else "invalid",
            "normalized_label": parsed.normalized_label if valid else None,
            "probabilities": probabilities if valid else None,
            "generation_id": raw.get("id"),
            "returned_model": returned_model,
            "returned_provider": returned_provider,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reported_cost_usd": usage.get("cost"),
            "finish_reason": choice.get("finish_reason"),
            "error": "; ".join(route_errors) or None,
        }


def _response_for_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_number": row["row_number"],
        "status": "success" if row.get("status") == "success" else "api_error",
        "parse_status": row.get("parse_status", "error"),
        "normalized_label": row.get("normalized_label"),
        "label_probabilities": row.get("probabilities"),
        "latency_ms": row.get("latency_ms"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "total_tokens": row.get("total_tokens"),
    }


def evaluate_checkpoint(
    selected: list[DatasetRow],
    responses: list[dict[str, Any]],
    all_attempts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    response_records = [_response_for_metrics(row) for row in responses]
    evaluation = asdict(evaluate_responses(selected, response_records, MODEL_ID, scope="frozen_hash_order_sample"))
    valid = [row for row in responses if row.get("status") == "success"]
    if valid:
        y_true = [str(row["true_label"]) for row in valid]
        y_pred = [str(row["normalized_label"]) for row in valid]
        true_probabilities = np.asarray(
            [float(row["probabilities"][str(row["true_label"])]) for row in valid],
            dtype=float,
        )
        evaluation["multiclass_log_loss"] = float(-np.log(np.clip(true_probabilities, LOG_LOSS_CLIP, 1.0)).mean())
        evaluation["log_loss_probability_clip"] = LOG_LOSS_CLIP
        evaluation["accuracy_ci"] = bootstrap_metric_ci(y_true, y_pred, "accuracy", n_resamples=2_000, seed=42)
        evaluation["macro_f1_ci"] = bootstrap_metric_ci(y_true, y_pred, "macro_f1", n_resamples=2_000, seed=42)
    else:
        evaluation["multiclass_log_loss"] = None
        evaluation["log_loss_probability_clip"] = LOG_LOSS_CLIP
        evaluation["accuracy_ci"] = None
        evaluation["macro_f1_ci"] = None
    evaluation["attempted_rows"] = len(responses)
    evaluation["valid_rows"] = len(valid)
    evaluation["coverage_success_share"] = len(valid) / len(selected) if selected else 0.0
    provider_completions = [row for row in responses if row.get("status") in {"success", "invalid_output"}]
    evaluation["format_valid_share"] = len(valid) / len(provider_completions) if provider_completions else 0.0
    billed_attempts = all_attempts if all_attempts is not None else responses
    evaluation["attempt_records"] = len(billed_attempts)
    reported_costs = [float(row["reported_cost_usd"]) for row in billed_attempts if row.get("reported_cost_usd") is not None]
    evaluation["reported_cost_usd"] = sum(reported_costs) if reported_costs else None
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in billed_attempts)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in billed_attempts)
    evaluation["listed_price_estimate_usd"] = (
        prompt_tokens * INPUT_PRICE_PER_MILLION + completion_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000
    return evaluation


def paired_classification_comparison(
    y_true: list[str],
    candidate: list[str],
    reference: list[str],
    *,
    n_resamples: int = 2_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired accuracy/macro-F1 comparison on an identical ordered sample."""
    if not y_true or not (len(y_true) == len(candidate) == len(reference)):
        raise ValueError("paired comparison requires three non-empty, equal-length vectors")
    true = np.asarray(y_true, dtype=object)
    candidate_array = np.asarray(candidate, dtype=object)
    reference_array = np.asarray(reference, dtype=object)
    candidate_correct = candidate_array == true
    reference_correct = reference_array == true
    candidate_only = int(np.sum(candidate_correct & ~reference_correct))
    reference_only = int(np.sum(~candidate_correct & reference_correct))
    rng = np.random.default_rng(seed)
    accuracy_differences = np.empty(n_resamples, dtype=float)
    macro_f1_differences = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, len(true), size=len(true))
        accuracy_differences[index] = candidate_correct[sample].mean() - reference_correct[sample].mean()
        macro_f1_differences[index] = f1_score(
            true[sample], candidate_array[sample], labels=list(ALLOWED_LABELS), average="macro", zero_division=0
        ) - f1_score(true[sample], reference_array[sample], labels=list(ALLOWED_LABELS), average="macro", zero_division=0)
    candidate_accuracy = float(accuracy_score(true, candidate_array))
    reference_accuracy = float(accuracy_score(true, reference_array))
    candidate_macro_f1 = float(f1_score(true, candidate_array, labels=list(ALLOWED_LABELS), average="macro", zero_division=0))
    reference_macro_f1 = float(f1_score(true, reference_array, labels=list(ALLOWED_LABELS), average="macro", zero_division=0))
    discordant = candidate_only + reference_only
    return {
        "n": len(true),
        "seed": seed,
        "n_resamples": n_resamples,
        "candidate_accuracy": candidate_accuracy,
        "reference_accuracy": reference_accuracy,
        "accuracy_difference": candidate_accuracy - reference_accuracy,
        "accuracy_difference_ci": [float(value) for value in np.quantile(accuracy_differences, [0.025, 0.975])],
        "candidate_macro_f1": candidate_macro_f1,
        "reference_macro_f1": reference_macro_f1,
        "macro_f1_difference": candidate_macro_f1 - reference_macro_f1,
        "macro_f1_difference_ci": [float(value) for value in np.quantile(macro_f1_differences, [0.025, 0.975])],
        "candidate_only_correct": candidate_only,
        "reference_only_correct": reference_only,
        "mcnemar_exact_p": float(binomtest(min(candidate_only, reference_only), discordant, p=0.5).pvalue) if discordant else 1.0,
    }


def prepare_manifest(dataset_path: str | Path, output_dir: str | Path, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    dataset = Path(dataset_path)
    paths = validation_paths(output_dir)
    selected = select_validation_rows(dataset, limit)
    dataset_hash = sha256_file(dataset)
    selected_hash = sample_sha256(selected)
    config_hash = _config_hash(dataset_hash, selected_hash, limit)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "prepared",
        "prepared_at": datetime.now(UTC).isoformat(),
        "estimand": "three-class financial-headline sentiment accuracy and calibration on a frozen public benchmark sample",
        "unit": "unique public benchmark headline/sentence",
        "selection": {
            "method": SAMPLE_METHOD,
            "label_blind": True,
            "limit": limit,
            "population_rows": len(load_dataset(dataset)),
            "sample_label_counts": dict(Counter(row.hidden_label for row in selected)),
            "sample_sha256": selected_hash,
        },
        "input": {"dataset_path": str(dataset), "dataset_sha256": dataset_hash, "licensed_lseg_text": False},
        "model": {"id": MODEL_ID, "provider": PROVIDER_NAME},
        "prompt": {"id": PROMPT_ID, "hash": PROMPT_HASH},
        "request": request_contract(),
        "config_hash": config_hash,
        "outputs": {"responses": str(paths.responses), "metrics": str(paths.metrics)},
        "gate": {
            "format_valid_share_minimum": 0.99,
            "coverage_success_share_minimum": 1.0,
            "interpretation": "Model suitability requires human-labelled benchmark performance; LLM-FinBERT agreement is not accuracy.",
        },
        "licence_boundary": "This harness may send only the public benchmark. Licensed LSEG headlines remain local.",
    }


async def run_validation(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    limit: int = DEFAULT_LIMIT,
    concurrency: int = DEFAULT_CONCURRENCY,
    retries: int = DEFAULT_RETRIES,
    retry_failures: bool = False,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    paths = validation_paths(output_dir)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = prepare_manifest(dataset_path, output_dir, limit)
    config_hash = str(manifest["config_hash"])
    selected = select_validation_rows(dataset_path, limit)
    prior_attempts, completed = _load_checkpoint(paths.responses, config_hash)
    pending = [
        row
        for row in selected
        if _sentence_sha256(row.sentence) not in completed
        or (retry_failures and completed[_sentence_sha256(row.sentence)].get("status") != "success")
    ]
    started_at = datetime.now(UTC)
    manifest.update(
        {
            "status": "running",
            "started_at": started_at.isoformat(),
            "resume_unique_rows": len(completed),
            "prior_attempt_records": len(prior_attempts),
            "retry_failures": retry_failures,
        }
    )
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    client = PinnedOpenRouterClient()

    async def score(row: DatasetRow) -> None:
        async with semaphore:
            result = await client.classify(row, retries=retries)
        result["config_hash"] = config_hash
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
        async with write_lock:
            with paths.responses.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())

    try:
        await asyncio.gather(*(score(row) for row in pending))
    finally:
        await client.close()

    attempts, completed = _load_checkpoint(paths.responses, config_hash)
    ordered = [completed[_sentence_sha256(row.sentence)] for row in selected if _sentence_sha256(row.sentence) in completed]
    successful_rows = sum(row.get("status") == "success" for row in ordered)
    metrics = evaluate_checkpoint(selected, ordered, attempts)
    paths.metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    completed_at = datetime.now(UTC)
    manifest.update(
        {
            "status": "completed" if successful_rows == limit else "incomplete",
            "completed_at": completed_at.isoformat(),
            "runtime_seconds": (completed_at - started_at).total_seconds(),
            "counts": {
                "selected": limit,
                "attempted_this_run": len(pending),
                "checkpoint_unique_rows": len(ordered),
                "checkpoint_successful_rows": successful_rows,
                "checkpoint_attempt_records": len(attempts),
            },
            "results": {
                "coverage_success_share": metrics["coverage_success_share"],
                "format_valid_share": metrics["format_valid_share"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "brier_score": (metrics.get("calibration") or {}).get("brier_score"),
                "ece": (metrics.get("calibration") or {}).get("ece"),
                "multiclass_log_loss": metrics["multiclass_log_loss"],
                "reported_cost_usd": metrics["reported_cost_usd"],
                "listed_price_estimate_usd": metrics["listed_price_estimate_usd"],
            },
            "outputs": {
                "responses": str(paths.responses),
                "responses_sha256": sha256_file(paths.responses),
                "metrics": str(paths.metrics),
                "metrics_sha256": sha256_file(paths.metrics),
            },
        }
    )
    paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_LIMIT",
    "MODEL_ID",
    "PROMPT_HASH",
    "PROMPT_ID",
    "PROVIDER_NAME",
    "PROVIDER_SLUG",
    "build_request",
    "evaluate_checkpoint",
    "paired_classification_comparison",
    "prepare_manifest",
    "request_contract",
    "run_validation",
    "select_validation_rows",
]
