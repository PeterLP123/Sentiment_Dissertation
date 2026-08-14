"""Strict no-newness LSEG first-reaction scoring via DeepInfra/OpenRouter.

The scorer reuses Notebook 83's immutable return-blind event population but
sends only the selected current headline and target-company identity. Earlier
headlines, prices and returns never enter the hosted request.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from final_experiments.lib.lseg_materiality import (
    INPUT_PRICE_PER_MILLION,
    MODEL_ID,
    OUTPUT_PRICE_PER_MILLION,
    PROVIDER_NAME,
    PROVIDER_SLUG,
)
from final_experiments.lib.lseg_prompt_value import (
    EXPECTED_CANDIDATE_EVENTS,
    OPERATIONAL_SPEND_CEILING_USD,
    prompt_value_paths,
    total_reported_experiment_cost,
)
from sentiment_benchmark.artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from sentiment_benchmark.env import load_env_file
from sentiment_benchmark.prompts import prompt_hash

DEFAULT_CONFIG = Path(
    "configs/strategy_research/lseg_structured_reaction_score_v1.toml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "final_experiments/outputs/84_lseg_structured_reaction_score"
)
SOURCE_OUTPUT_ROOT = Path(
    "final_experiments/outputs/83_lseg_prompt_economic_value"
)
PROMPT_ID = "structured_first_reaction_v1"
HORIZON_SESSIONS = 1
REQUEST_WALL_TIMEOUT_SECONDS = 45.0
MAX_TOKENS = 100
SCORE_VALUES = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTATION_VALUES = ("worse_than_expected", "unclear", "better_than_expected")
PERSISTENCE_VALUES = ("temporary", "unclear", "persistent")
REACTION_VALUES = ("down", "flat_or_unclear", "up")

REACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_specific": {"type": "boolean"},
        "expectation_revision": {"type": "string", "enum": list(EXPECTATION_VALUES)},
        "persistence": {"type": "string", "enum": list(PERSISTENCE_VALUES)},
        "first_tradable_reaction": {"type": "string", "enum": list(REACTION_VALUES)},
        "reaction_score": {"type": "number", "enum": list(SCORE_VALUES)},
    },
    "required": [
        "target_specific",
        "expectation_revision",
        "persistence",
        "first_tradable_reaction",
        "reaction_score",
    ],
    "additionalProperties": False,
}


class StructuredReactionError(RuntimeError):
    """Raised when the reaction-score experiment cannot be trusted."""


@dataclass(frozen=True)
class StructuredReactionPrompt:
    prompt_id: str
    system_prompt: str
    user_template: str
    prompt_hash: str

    def render(self, event: Mapping[str, Any]) -> str:
        return self.user_template.format(
            target_company=str(event["target_company_name"]),
            symbol=str(event["symbol"]),
            headline=str(event["headline"]),
        )


@dataclass(frozen=True)
class StructuredReactionPaths:
    root: Path
    attempts: Path
    scores: Path
    manifest: Path


def structured_reaction_paths(
    root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> StructuredReactionPaths:
    base = Path(root)
    return StructuredReactionPaths(
        root=base,
        attempts=base / "structured_first_reaction_v1_attempts.jsonl",
        scores=base / "structured_first_reaction_v1_scores.jsonl",
        manifest=base / "structured_first_reaction_v1_manifest.json",
    )


def load_structured_reaction_prompt(
    path: str | Path = DEFAULT_CONFIG,
) -> StructuredReactionPrompt:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    experiment = payload.get("experiment") or {}
    prompt = payload.get("prompt") or {}
    expected = {
        "id": "lseg_structured_reaction_score_v1",
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "horizon_sessions": HORIZON_SESSIONS,
    }
    if any(experiment.get(key) != value for key, value in expected.items()):
        raise StructuredReactionError("structured-reaction experiment identity changed")
    if prompt.get("id") != PROMPT_ID:
        raise StructuredReactionError("structured-reaction prompt ID changed")
    system = str(prompt.get("system_prompt") or "").strip()
    user = str(prompt.get("user_template") or "").strip()
    required = (
        "do not decide whether the information is new or repeated",
        "target_specific",
        "expectation_revision",
        "persistence",
        "first_tradable_reaction",
        "reaction_score",
        "not a percentage return",
    )
    normalized = " ".join(system.casefold().split())
    if any(text not in normalized for text in required):
        raise StructuredReactionError("structured-reaction prompt contract changed")
    if "prior_context" in user or "earlier" in user.casefold():
        raise StructuredReactionError("no-newness prompt must not render prior headlines")
    return StructuredReactionPrompt(
        prompt_id=PROMPT_ID,
        system_prompt=system,
        user_template=user,
        prompt_hash=prompt_hash(PROMPT_ID, system, user, "structured_json"),
    )


def parse_reaction_labels(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = set(REACTION_SCHEMA["required"])
    if set(payload) != required:
        raise ValueError("model output does not contain exactly the required fields")
    target_specific = payload["target_specific"]
    expectation = payload["expectation_revision"]
    persistence = payload["persistence"]
    reaction = payload["first_tradable_reaction"]
    raw_score = payload["reaction_score"]
    if not isinstance(target_specific, bool):
        raise ValueError("target_specific must be boolean")
    if expectation not in EXPECTATION_VALUES:
        raise ValueError("expectation_revision is invalid")
    if persistence not in PERSISTENCE_VALUES:
        raise ValueError("persistence is invalid")
    if reaction not in REACTION_VALUES:
        raise ValueError("first_tradable_reaction is invalid")
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise ValueError("reaction_score must be numeric")
    score = float(raw_score)
    if not math.isfinite(score) or score not in SCORE_VALUES:
        raise ValueError("reaction_score is outside the fixed score grid")
    if not target_specific and (
        expectation != "unclear"
        or persistence != "unclear"
        or reaction != "flat_or_unclear"
        or score != 0.0
    ):
        raise ValueError("non-target output must fail closed to unclear and zero")
    if reaction == "flat_or_unclear" and score != 0.0:
        raise ValueError("flat_or_unclear requires a zero reaction_score")
    if reaction == "down" and score >= 0.0:
        raise ValueError("down requires a negative reaction_score")
    if reaction == "up" and score <= 0.0:
        raise ValueError("up requires a positive reaction_score")
    if expectation == "better_than_expected" and reaction == "down":
        raise ValueError("better_than_expected cannot imply a down reaction")
    if expectation == "worse_than_expected" and reaction == "up":
        raise ValueError("worse_than_expected cannot imply an up reaction")
    return {
        "target_specific": target_specific,
        "expectation_revision": expectation,
        "persistence": persistence,
        "first_tradable_reaction": reaction,
        "reaction_score": score,
        "trade_signal": score if target_specific else 0.0,
    }


def openrouter_request_contract() -> dict[str, Any]:
    return {
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "reasoning": {"effort": "none", "exclude": True},
        "provider": {
            "only": [PROVIDER_SLUG],
            "quantizations": ["fp8"],
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
                "name": "lseg_structured_first_reaction",
                "strict": True,
                "schema": REACTION_SCHEMA,
            },
        },
    }


def conservative_pending_cost_upper(
    events: Sequence[Mapping[str, Any]],
    prompt: StructuredReactionPrompt,
) -> float:
    input_tokens = sum(
        len(prompt.system_prompt) + len(prompt.render(event)) for event in events
    )
    output_tokens = len(events) * MAX_TOKENS
    return (
        input_tokens * INPUT_PRICE_PER_MILLION
        + output_tokens * OUTPUT_PRICE_PER_MILLION
    ) / 1_000_000.0


def _config_hash(
    candidate_manifest: Path,
    candidate_events: Path,
    prompt: StructuredReactionPrompt,
) -> tuple[str, dict[str, Any]]:
    identity = {
        "schema_version": 1,
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "candidate_events_sha256": sha256_file(candidate_events),
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "horizon_sessions": HORIZON_SESSIONS,
        "prior_headlines_rendered": False,
        "newness_assessed": False,
        "schema_sha256": sha256_text(canonical_json(REACTION_SCHEMA)),
        "request": openrouter_request_contract(),
    }
    return sha256_text(canonical_json(identity)), identity


def _load_attempts(
    path: Path,
    *,
    config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    successes: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return attempts, successes
    for line_number, row in enumerate(read_jsonl(path), start=1):
        if row.get("config_hash") != config_hash:
            raise StructuredReactionError(
                f"score checkpoint config mismatch at line {line_number}"
            )
        event_id = str(row.get("prompt_event_id") or "")
        if not event_id:
            raise StructuredReactionError(f"missing event ID at line {line_number}")
        attempts.append(row)
        if row.get("status") == "success":
            if event_id in successes:
                raise StructuredReactionError(f"duplicate success for {event_id}")
            successes[event_id] = row
    return attempts, successes


def reported_new_experiment_cost(paths: StructuredReactionPaths) -> float:
    if not paths.attempts.is_file():
        return 0.0
    return sum(
        float(row["reported_cost_usd"])
        for row in read_jsonl(paths.attempts)
        if row.get("reported_cost_usd") is not None
    )


async def _score_one(
    client: httpx.AsyncClient,
    api_key: str,
    event: Mapping[str, Any],
    prompt: StructuredReactionPrompt,
    config_hash: str,
    *,
    retries: int,
) -> dict[str, Any]:
    rendered = prompt.render(event)
    request = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": prompt.system_prompt},
            {"role": "user", "content": rendered},
        ],
        **openrouter_request_contract(),
        "stream": False,
    }
    response: httpx.Response | None = None
    error: Exception | None = None
    started = time.monotonic()
    attempts = 0
    for attempt in range(retries + 1):
        attempts = attempt + 1
        try:
            response = await asyncio.wait_for(
                client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-OpenRouter-Title": (
                            "Sentiment Dissertation structured reaction score"
                        ),
                    },
                    json=request,
                ),
                timeout=REQUEST_WALL_TIMEOUT_SECONDS,
            )
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == retries:
                break
        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            error = exc
            if attempt == retries:
                break
        await asyncio.sleep(float(min(2**attempt, 8)))
    base = {
        "prompt_event_id": str(event["prompt_event_id"]),
        "headline_sha256": str(event["headline_sha256"]),
        "config_hash": config_hash,
        "model_id": MODEL_ID,
        "provider": PROVIDER_NAME,
        "quantization": "fp8",
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "horizon_sessions": HORIZON_SESSIONS,
        "prior_headlines_rendered": False,
        "newness_assessed": False,
        "input_sha256": sha256_text(rendered),
        "attempt_count": attempts,
        "latency_ms": (time.monotonic() - started) * 1_000.0,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if response is None:
        return {**base, "status": "transport_error", "error": str(error)}
    if response.status_code >= 400:
        return {
            **base,
            "status": "api_error",
            "http_status": response.status_code,
            "error": response.text[:1_000],
        }
    try:
        raw = response.json()
        choice = (raw.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        labels = parse_reaction_labels(json.loads(content))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return {**base, "status": "invalid_output", "error": str(exc)}
    returned_provider = str(raw.get("provider") or "")
    returned_model = str(raw.get("model") or "")
    route_errors: list[str] = []
    if returned_provider and returned_provider.casefold() not in {
        PROVIDER_NAME.casefold(),
        PROVIDER_SLUG.casefold(),
    }:
        route_errors.append(f"unexpected provider {returned_provider!r}")
    if returned_model and returned_model != MODEL_ID:
        route_errors.append(f"unexpected model {returned_model!r}")
    usage = raw.get("usage") or {}
    return {
        **base,
        **labels,
        "status": "success" if not route_errors else "route_mismatch",
        "generation_id": raw.get("id"),
        "returned_model": returned_model or None,
        "returned_provider": returned_provider or None,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reported_cost_usd": usage.get("cost"),
        "finish_reason": choice.get("finish_reason"),
        "error": "; ".join(route_errors) or None,
    }


async def score_structured_reaction(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    source_output_root: str | Path = SOURCE_OUTPUT_ROOT,
    prompt_config: str | Path = DEFAULT_CONFIG,
    concurrency: int = 25,
    retries: int = 3,
    retry_failures: bool = False,
    max_new_calls: int | None = None,
    end_entry_session_exclusive: str | None = None,
    max_combined_reported_cost_usd: float = OPERATIONAL_SPEND_CEILING_USD,
    confirm_authorized_transfer: bool = False,
) -> dict[str, Any]:
    """Score the no-newness prompt under the original cumulative spend cap."""

    if not confirm_authorized_transfer:
        raise StructuredReactionError("explicit external-transfer approval is required")
    if not 0 < max_combined_reported_cost_usd <= 20.0:
        raise StructuredReactionError("combined spend ceiling must be in (0, 20]")
    if concurrency < 1 or retries < 0:
        raise ValueError("concurrency must be positive and retries non-negative")
    source = prompt_value_paths(source_output_root)
    if not source.candidate_manifest.is_file() or not source.candidate_events.is_file():
        raise StructuredReactionError("Notebook 83 candidate population is required")
    candidate_manifest = read_json(source.candidate_manifest)
    if (
        candidate_manifest.get("status") != "completed"
        or candidate_manifest["counts"]["candidate_events"] != EXPECTED_CANDIDATE_EVENTS
    ):
        raise StructuredReactionError("candidate population identity changed")
    expected_sha = candidate_manifest["output"]["sha256"]
    if sha256_file(source.candidate_events) != expected_sha:
        raise StructuredReactionError("candidate event file changed")
    parent_events = read_jsonl(source.candidate_events)
    if end_entry_session_exclusive is None:
        events = parent_events
    else:
        events = [
            event
            for event in parent_events
            if str(event["entry_session"]) < end_entry_session_exclusive
        ]
        if not events:
            raise StructuredReactionError("chronological scoring scope is empty")
    prompt = load_structured_reaction_prompt(prompt_config)
    config_hash, identity = _config_hash(
        source.candidate_manifest,
        source.candidate_events,
        prompt,
    )
    paths = structured_reaction_paths(output_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    attempts, successes = _load_attempts(paths.attempts, config_hash=config_hash)
    pending = [
        event
        for event in events
        if str(event["prompt_event_id"]) not in successes
    ]
    if not retry_failures and attempts:
        attempted_ids = {str(row["prompt_event_id"]) for row in attempts}
        pending = [
            event
            for event in pending
            if str(event["prompt_event_id"]) not in attempted_ids
        ]
    if max_new_calls is not None:
        if max_new_calls < 1:
            raise ValueError("max_new_calls must be positive")
        pending = pending[:max_new_calls]

    prior_cost = total_reported_experiment_cost(source)
    new_cost_before = reported_new_experiment_cost(paths)
    pending_upper = conservative_pending_cost_upper(pending, prompt)
    if prior_cost + new_cost_before + pending_upper > max_combined_reported_cost_usd:
        raise StructuredReactionError(
            "conservative combined-cost upper bound exceeds the cumulative ceiling: "
            f"{prior_cost + new_cost_before + pending_upper:.6f} > "
            f"{max_combined_reported_cost_usd:.6f}"
        )
    load_env_file()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise StructuredReactionError("OPENROUTER_API_KEY is required")

    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    async with httpx.AsyncClient(timeout=120.0) as client:

        async def run_one(event: Mapping[str, Any]) -> None:
            async with semaphore:
                result = await _score_one(
                    client,
                    api_key,
                    event,
                    prompt,
                    config_hash,
                    retries=retries,
                )
            encoded = json.dumps(result, sort_keys=True, ensure_ascii=False) + "\n"
            async with write_lock:
                with paths.attempts.open("a", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())

        await asyncio.gather(*(run_one(event) for event in pending))

    all_attempts, successes = _load_attempts(paths.attempts, config_hash=config_hash)
    ordered = [
        successes[str(event["prompt_event_id"])]
        for event in events
        if str(event["prompt_event_id"]) in successes
    ]
    atomic_write_jsonl(paths.scores, ordered)
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in all_attempts)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in all_attempts)
    new_cost = reported_new_experiment_cost(paths)
    combined_cost = prior_cost + new_cost
    if combined_cost > max_combined_reported_cost_usd:
        raise StructuredReactionError("reported combined spend exceeded its ceiling")
    manifest = {
        "schema_version": 1,
        "status": "completed" if len(ordered) == len(events) else "incomplete",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "identity": identity,
        "config_hash": config_hash,
        "counts": {
            "parent_candidate_events": len(parent_events),
            "scoring_scope_events": len(events),
            "attempt_records": len(all_attempts),
            "attempted_this_run": len(pending),
            "successful_unique": len(ordered),
            "missing_success": len(events) - len(ordered),
            "status_counts": dict(Counter(str(row.get("status")) for row in all_attempts)),
        },
        "scoring_scope": {
            "end_entry_session_exclusive": end_entry_session_exclusive,
            "chronological": end_entry_session_exclusive is not None,
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "reported_new_experiment_cost_usd": new_cost,
            "reported_notebook83_cost_usd": prior_cost,
            "reported_combined_cost_usd": combined_cost,
            "combined_spend_ceiling_usd": max_combined_reported_cost_usd,
            "conservative_pending_cost_upper_usd": pending_upper,
        },
        "outputs": {
            "attempts": {
                "path": str(paths.attempts),
                "sha256": sha256_file(paths.attempts),
                "licensed": True,
                "gitignored": True,
            },
            "scores": {
                "path": str(paths.scores),
                "sha256": sha256_file(paths.scores),
                "rows": len(ordered),
                "licensed": True,
                "gitignored": True,
            },
        },
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT_ROOT",
    "HORIZON_SESSIONS",
    "PROMPT_ID",
    "REACTION_SCHEMA",
    "SCORE_VALUES",
    "StructuredReactionError",
    "StructuredReactionPaths",
    "StructuredReactionPrompt",
    "conservative_pending_cost_upper",
    "load_structured_reaction_prompt",
    "openrouter_request_contract",
    "parse_reaction_labels",
    "reported_new_experiment_cost",
    "score_structured_reaction",
    "structured_reaction_paths",
]
