"""Frozen independent mid-cap replication of the negative-news h2 signal."""

from __future__ import annotations

import csv
import html
import io
import math
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import fmean, median
from typing import Any

import numpy as np

from ..artifact_io import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    canonical_json,
    read_json,
    read_jsonl,
    sha256_file,
    sha256_text,
)
from .directional_event_gate import SelectedSessionEvent, select_strongest_session_events
from .market import load_adjusted_opens_csv
from .model_only_development import (
    DIRECTION_SCORE,
    MODEL_ONLY_CONTRACT,
    _filtered_score,
    _validate_completed_score_contract,
    validate_model_only_scoring_universe,
)
from .schemas import load_strategy_events, write_strategy_events
from .signal_quality import AuditItem, JointSignalRecord, SignalAuditSpec, build_prior_context
from .sources import LsegEventSettings, build_strategy_events

REPLICATION_CONTRACT = "independent_midcap_negative_h2_replication_v1"


class IndependentReplicationError(RuntimeError):
    """Raised when the frozen replication cannot be prepared or evaluated safely."""


@dataclass(frozen=True)
class FrozenReplication:
    config_path: Path
    payload: Mapping[str, Any]
    config_sha256: str

    @property
    def experiment_id(self) -> str:
        return str(self.payload["experiment"]["id"])

    @property
    def derived_root(self) -> Path:
        return Path(str(self.payload["outputs"]["derived_root"]))

    @property
    def results_root(self) -> Path:
        return Path(str(self.payload["outputs"]["results_root"]))


@dataclass(frozen=True)
class PreparedReplication:
    replication_id: str
    output_dir: Path
    events_path: Path
    universe_dir: Path
    event_count: int
    reused: bool


@dataclass(frozen=True)
class FinalizedReplicationScores:
    strategy_id: str
    output_dir: Path
    scores_path: Path
    successful_scores: int
    excluded_scores: int
    reused: bool


@dataclass(frozen=True)
class ReplicationTrade:
    start_session: str
    end_session: str
    symbol: str
    gross_short_return: float
    net_short_return: float
    panel_hedged_net_short_return: float


@dataclass(frozen=True)
class FoldResult:
    fold_index: int
    first_session: str
    last_session: str
    trades: int
    supported_stocks: int
    mean_net_short_return: float | None
    mean_panel_hedged_net_short_return: float | None


@dataclass(frozen=True)
class BootstrapMean:
    observed_mean: float
    ci_low: float
    ci_high: float
    one_sided_p_value: float
    valid_replications: int
    replications: int
    block_length: int
    seed: int


@dataclass(frozen=True)
class ReplicationDecision:
    passed: bool
    rejection_reasons: tuple[str, ...]
    trades: int
    supported_stocks: int
    positive_folds: int
    one_sided_p_value: float
    confidence_interval_low: float


@dataclass(frozen=True)
class ReplicationRun:
    replication_id: str
    output_dir: Path
    report_path: Path
    manifest_path: Path
    decision: ReplicationDecision
    reused: bool


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise IndependentReplicationError(f"replication config is missing [{key}]")
    return value


def load_frozen_replication(path: str | Path) -> FrozenReplication:
    """Load and verify the pre-return frozen protocol and its declared inputs."""

    config_path = Path(path)
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    experiment = _require_mapping(payload, "experiment")
    news = _require_mapping(payload, "news")
    model = _require_mapping(payload, "model")
    prices = _require_mapping(payload, "prices")
    strategy = _require_mapping(payload, "strategy")
    folds = _require_mapping(payload, "folds")
    inference = _require_mapping(payload, "inference")
    acceptance = _require_mapping(payload, "acceptance")
    _require_mapping(payload, "eligibility")
    _require_mapping(payload, "outputs")
    expected = {
        "status": "frozen",
        "schema_version": 1,
        "text_unit": "headline_only",
        "provider": "ollama",
        "position": "short",
        "holding_sessions": 2,
        "fold_count": 3,
        "block_length": 5,
        "replications": 2000,
        "seed": 20260715,
        "p_max": 0.01,
        "confidence_level": 0.99,
    }
    actual = {
        "status": experiment.get("status"),
        "schema_version": experiment.get("schema_version"),
        "text_unit": news.get("text_unit"),
        "provider": model.get("provider"),
        "position": strategy.get("position"),
        "holding_sessions": strategy.get("holding_sessions"),
        "fold_count": folds.get("count"),
        "block_length": inference.get("block_length_sessions"),
        "replications": inference.get("replications"),
        "seed": inference.get("seed"),
        "p_max": inference.get("one_sided_p_value_maximum"),
        "confidence_level": inference.get("confidence_level"),
    }
    if actual != expected:
        raise IndependentReplicationError(f"replication contract changed: expected {expected}, got {actual}")
    if float(strategy.get("cost_bps_per_side", -1)) != 10.0:
        raise IndependentReplicationError("replication cost must remain 10 bps per side")
    if int(acceptance.get("required_positive_folds", -1)) != 3:
        raise IndependentReplicationError("replication must require three positive folds")
    declared_files = (
        (Path(str(news["corpus_manifest"])), str(news["corpus_manifest_sha256"])),
        (Path(str(model["prompt_path"])), str(model["prompt_file_sha256"])),
        (Path(str(prices["manifest_path"])), str(prices["manifest_sha256"])),
    )
    for declared_path, expected_hash in declared_files:
        if not declared_path.is_file() or sha256_file(declared_path) != expected_hash:
            raise IndependentReplicationError(f"frozen replication input is missing or changed: {declared_path}")
    price_manifest = read_json(Path(str(prices["manifest_path"])))
    if price_manifest.get("status") != "completed":
        raise IndependentReplicationError("price manifest is not completed")
    if price_manifest.get("file", {}).get("sha256") != prices.get("panel_sha256"):
        raise IndependentReplicationError("price panel hash disagrees with the frozen price manifest")
    return FrozenReplication(config_path, payload, sha256_file(config_path))


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _replication_assumption(replication_id: str, event_count: int) -> str:
    return f"""# Independent replication assumption: {replication_id}

- Frozen full event universe: {event_count} events.
- Local model: Gemma 4 12B at the declared immutable digest.
- Text unit: Reuters headline only; no return or price enters any prompt.
- Eligibility: non-neutral, materiality and novelty at least moderate.
- Selection: strongest eligible event per stock-session before retaining negative events only.
- Strategy: short at the eligible adjusted open and exit two sessions later, with 10 bps charged per side.
- No parameter may be changed after mid-cap returns are opened.
- Model labels are a researcher-accepted modelling assumption, not human ground truth.
"""


def prepare_independent_replication(config_path: str | Path) -> PreparedReplication:
    """Build immutable events and a complete local-scoring universe without reading prices."""

    frozen = load_frozen_replication(config_path)
    payload = frozen.payload
    news = payload["news"]
    model = payload["model"]
    identity = {
        "schema_version": 1,
        "contract": REPLICATION_CONTRACT,
        "experiment_id": frozen.experiment_id,
        "frozen_config_sha256": frozen.config_sha256,
        "corpus_manifest_sha256": news["corpus_manifest_sha256"],
        "prompt_file_sha256": model["prompt_file_sha256"],
        "model": model["model"],
        "model_digest": model["model_digest"],
        "analysis_start_session": news["analysis_start_session"],
        "analysis_end_session": news["analysis_end_session"],
    }
    identity_hash = sha256_text(canonical_json(identity))
    replication_id = f"{frozen.experiment_id}-{identity_hash[:12]}"
    output_dir = frozen.derived_root / replication_id
    events_dir = output_dir / "events"
    events_path = events_dir / "events.jsonl"
    events_manifest_path = events_dir / "manifest.json"
    universe_dir = output_dir / "universe"
    universe_manifest_path = universe_dir / "sample_manifest.json"
    if events_manifest_path.is_file() and universe_manifest_path.is_file():
        events_manifest = read_json(events_manifest_path)
        if events_manifest.get("identity_sha256") != identity_hash:
            raise IndependentReplicationError("existing independent-replication event identity differs")
        expected_events_hash = events_manifest.get("outputs", {}).get("events.jsonl", {}).get("sha256")
        if not events_path.is_file() or sha256_file(events_path) != expected_events_hash:
            raise IndependentReplicationError("completed independent-replication events changed")
        validate_model_only_scoring_universe(universe_dir)
        return PreparedReplication(
            replication_id,
            output_dir,
            events_path,
            universe_dir,
            int(events_manifest["row_counts"]["events"]),
            True,
        )
    if output_dir.exists():
        raise IndependentReplicationError(f"refusing to reuse incomplete replication preparation: {output_dir}")

    settings = LsegEventSettings(
        processing_buffer_minutes=int(news["processing_buffer_minutes"]),
        calendar_name=str(news["exchange_calendar"]),
        exchange_timezone=str(news["exchange_timezone"]),
    )
    built = build_strategy_events(Path(str(news["corpus_manifest"])), settings=settings)
    start = str(news["analysis_start_session"])
    end = str(news["analysis_end_session"])
    events = tuple(event for event in built.events if start <= event.eligible_execution_session.isoformat() <= end)
    if not events:
        raise IndependentReplicationError("frozen replication event universe is empty")
    price_manifest = read_json(Path(str(payload["prices"]["manifest_path"])))
    expected_symbols = {str(value).upper() for value in price_manifest.get("symbols") or []}
    event_symbols = {event.symbol for event in events}
    unexpected = event_symbols - expected_symbols
    if unexpected:
        raise IndependentReplicationError(f"events contain symbols absent from frozen price manifest: {sorted(unexpected)}")

    events_dir.mkdir(parents=True, exist_ok=False)
    write_strategy_events(events_path, events)
    atomic_write_json(
        events_manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "identity_sha256": identity_hash,
            "identity": identity,
            "row_counts": {"events": len(events), "symbols": len(event_symbols)},
            "attrition": built.attrition,
            "event_rules": built.event_rules,
            "outputs": {
                events_path.name: {
                    "path": events_path.as_posix(),
                    "sha256": sha256_file(events_path),
                    "size_bytes": events_path.stat().st_size,
                }
            },
        },
    )

    event_rows = [event.to_payload() for event in events]
    event_ids = [event.event_id for event in events]
    boundary = (date.fromisoformat(str(news["analysis_end_session"])) + timedelta(days=1)).isoformat()
    spec = SignalAuditSpec(
        sample_per_symbol=1,
        prior_window_days=int(news["prior_context_days"]),
        maximum_prior_headlines=int(news["maximum_prior_headlines"]),
    )
    contexts = build_prior_context(event_rows, event_ids, boundary, spec)
    items: list[AuditItem] = []
    symbol_counts: Counter[str] = Counter()
    for index, event in enumerate(events, start=1):
        current_text = "\n\n".join(value.strip() for value in (event.headline, event.lead_or_body) if value.strip())
        if sha256_text(current_text) != event.text_sha256:
            raise IndependentReplicationError(f"event text hash changed: {event.event_id}")
        context = contexts[event.event_id]
        item = AuditItem(
            audit_id=f"IR-{index:05d}",
            event_id=event.event_id,
            symbol=event.symbol,
            target_company_name=event.target_company_name or event.symbol,
            available_at_utc=event.available_at_utc.isoformat(),
            eligible_execution_session=event.eligible_execution_session.isoformat(),
            headline=event.headline,
            lead_or_body=event.lead_or_body,
            prior_context=context,
            content_hash=event.text_sha256,
            context_hash=sha256_text(context),
        )
        items.append(item)
        symbol_counts[event.symbol] += 1
    universe_identity = {
        "schema_version": 1,
        "contract": MODEL_ONLY_CONTRACT,
        "replication_contract": REPLICATION_CONTRACT,
        "replication_identity_sha256": identity_hash,
        "source_run_id": replication_id,
        "source_run_identity": identity_hash,
        "evaluation_start": boundary,
        "events_path": events_path.as_posix(),
        "events_sha256": sha256_file(events_path),
        "model": model["model"],
        "model_digest": model["model_digest"],
        "human_validation_status": model["human_validation_status"],
        "label_status": "model_generated_research_assumption",
        "prior_window_days": spec.prior_window_days,
        "maximum_prior_headlines": spec.maximum_prior_headlines,
        "selection": "all frozen independent-replication events",
    }
    universe_hash = sha256_text(canonical_json(universe_identity))
    universe_dir.mkdir(parents=True, exist_ok=False)
    items_path = universe_dir / "audit_items.csv"
    assumption_path = universe_dir / "model_only_assumption.md"
    atomic_write_text(
        items_path,
        _csv_text([item.to_row() for item in items], list(AuditItem.__dataclass_fields__)),
    )
    atomic_write_text(assumption_path, _replication_assumption(replication_id, len(items)))
    atomic_write_json(
        universe_manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "experiment_id": replication_id,
            "identity_sha256": universe_hash,
            "identity": universe_identity,
            "row_counts": {"audit_items": len(items), "symbols": len(symbol_counts)},
            "symbol_counts": dict(sorted(symbol_counts.items())),
            "outputs": {
                path.name: {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in (items_path, assumption_path)
            },
            "warning": "Independent replication; local model labels are not human validated.",
        },
    )
    validate_model_only_scoring_universe(universe_dir)
    return PreparedReplication(replication_id, output_dir, events_path, universe_dir, len(events), False)


def _load_audit_items(path: Path) -> tuple[AuditItem, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        items = tuple(AuditItem.from_row(row) for row in csv.DictReader(handle))
    if not items or len({item.audit_id for item in items}) != len(items):
        raise IndependentReplicationError("replication scoring items are empty or duplicated")
    return items


def finalize_independent_replication_scores(
    universe_dir: str | Path,
    score_dir: str | Path,
) -> FinalizedReplicationScores:
    """Finalize successes and explicit exhausted invalid-score exclusions.

    This never converts an invalid response to neutral.  A missing score is
    accepted only after the frozen number of attempts has been exhausted.
    """

    universe_root = Path(universe_dir)
    scoring_root = Path(score_dir)
    validate_model_only_scoring_universe(universe_root)
    universe_manifest = read_json(universe_root / "sample_manifest.json")
    universe_identity = universe_manifest.get("identity", {})
    if universe_identity.get("replication_contract") != REPLICATION_CONTRACT:
        raise IndependentReplicationError("scoring universe is not the independent replication")
    items_path = universe_root / "audit_items.csv"
    items = _load_audit_items(items_path)

    score_manifest_path = scoring_root / "scores_manifest.json"
    score_manifest = read_json(score_manifest_path)
    score_identity = score_manifest.get("identity", {})
    if score_identity.get("sample_identity_sha256") != universe_manifest.get("identity_sha256"):
        raise IndependentReplicationError("score cache does not belong to the independent replication universe")
    if score_manifest.get("identity_sha256") != sha256_text(canonical_json(score_identity)):
        raise IndependentReplicationError("score cache identity hash is invalid")

    completed_files = tuple(sorted((scoring_root / "cache" / "completed").rglob("*.json")))
    records = tuple(JointSignalRecord.from_payload(read_json(path)) for path in completed_files)
    if len({record.audit_id for record in records}) != len(records):
        raise IndependentReplicationError("completed score cache contains duplicate audit IDs")
    if any(record.status != "success" for record in records):
        raise IndependentReplicationError("completed cache contains a non-success record")
    _validate_completed_score_contract(universe_manifest, items_path, score_manifest, records, items)
    record_by_id = {record.audit_id: record for record in records}
    item_by_id = {item.audit_id: item for item in items}
    if not set(record_by_id) <= set(item_by_id):
        raise IndependentReplicationError("completed cache contains unknown audit IDs")

    missing_ids = sorted(set(item_by_id) - set(record_by_id))
    attempt_payloads: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for attempt_path in sorted((scoring_root / "cache" / "attempts").rglob("*.json")):
        attempt = read_json(attempt_path)
        audit_id = str(attempt.get("audit_id") or "")
        if audit_id in missing_ids:
            attempt_payloads[audit_id].append(attempt)
    required_attempts = int(score_identity.get("request", {}).get("retries", 0))
    exclusions: list[dict[str, Any]] = []
    for audit_id in missing_ids:
        attempts = attempt_payloads.get(audit_id, [])
        if not attempts:
            raise IndependentReplicationError(f"missing score has no attempt journal: {audit_id}")
        latest = max(attempts, key=lambda row: int(row.get("attempt_count") or 0))
        attempt_count = int(latest.get("attempt_count") or 0)
        if attempt_count < required_attempts or latest.get("status") not in {"invalid", "error"}:
            raise IndependentReplicationError(f"missing score has not exhausted its frozen attempts: {audit_id}")
        exclusions.append(
            {
                "audit_id": audit_id,
                "event_id": item_by_id[audit_id].event_id,
                "status": str(latest["status"]),
                "attempt_count": attempt_count,
                "error": str(latest.get("error") or "unspecified invalid response"),
            }
        )

    identity = {
        "schema_version": 1,
        "contract": REPLICATION_CONTRACT,
        "universe_identity_sha256": universe_manifest["identity_sha256"],
        "score_cache_identity_sha256": score_manifest["identity_sha256"],
        "audit_items_sha256": sha256_file(items_path),
        "completed_cache_records": len(records),
        "excluded_cache_records": len(exclusions),
        "missing_or_invalid_score_policy": "exclude after frozen attempts; never coerce to neutral",
        "minimum_materiality": "moderate",
        "minimum_novelty": "moderate",
        "direction_mapping": DIRECTION_SCORE,
    }
    identity_hash = sha256_text(canonical_json(identity))
    strategy_id = f"replication-filtered-scores-{identity_hash[:12]}"
    output_dir = universe_root / "filtered_strategy" / strategy_id
    scores_path = output_dir / "scores.jsonl"
    exclusions_path = output_dir / "score_exclusions.json"
    manifest_path = output_dir / "manifest.json"
    expected = (scores_path, exclusions_path)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash or existing.get("status") != "completed":
            raise IndependentReplicationError("existing finalized replication scores differ or are incomplete")
        for path in expected:
            if sha256_file(path) != existing.get("outputs", {}).get(path.name, {}).get("sha256"):
                raise IndependentReplicationError(f"completed finalized score artifact changed: {path}")
        return FinalizedReplicationScores(
            strategy_id,
            output_dir,
            scores_path,
            len(records),
            len(exclusions),
            True,
        )
    if output_dir.exists():
        raise IndependentReplicationError(f"refusing to overwrite incomplete finalized scores: {output_dir}")

    output_rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for item in items:
        record = record_by_id.get(item.audit_id)
        if record is None:
            continue
        score, reason = _filtered_score(record)
        reasons[reason] += 1
        output_rows.append(
            {
                "event_id": item.event_id,
                "status": "success",
                "score": score,
                "raw_label": "negative" if score < 0 else "positive" if score > 0 else "neutral",
                "model_direction_severity": record.direction_severity,
                "model_materiality": record.materiality,
                "model_novelty": record.novelty,
                "eligible": reason == "eligible",
                "filter_reason": reason,
                "model_id": record.model_id,
                "model_digest": record.model_digest,
                "prompt_id": record.prompt_id,
                "prompt_hash": record.prompt_hash,
                "source_cache_key": record.cache_key,
            }
        )
    output_rows.sort(key=lambda row: str(row["event_id"]))
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_jsonl(scores_path, output_rows)
    atomic_write_json(exclusions_path, {"count": len(exclusions), "exclusions": exclusions})
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "strategy_id": strategy_id,
            "identity_sha256": identity_hash,
            "identity": identity,
            "row_counts": {
                "events": len(items),
                "successful_scores": len(records),
                "excluded_scores": len(exclusions),
                "eligible_events": reasons["eligible"],
            },
            "filter_reasons": dict(sorted(reasons.items())),
            "outputs": {
                path.name: {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in expected
            },
            "warning": "Invalid scores are excluded after frozen attempts and are never converted to neutral.",
        },
    )
    return FinalizedReplicationScores(strategy_id, output_dir, scores_path, len(records), len(exclusions), False)


def _equal_folds(sessions: Sequence[str], count: int = 3) -> tuple[tuple[str, ...], ...]:
    if count != 3 or len(sessions) < count:
        raise IndependentReplicationError("replication requires three non-empty folds")
    base, remainder = divmod(len(sessions), count)
    folds: list[tuple[str, ...]] = []
    offset = 0
    for index in range(count):
        length = base + (1 if index < remainder else 0)
        folds.append(tuple(sessions[offset : offset + length]))
        offset += length
    return tuple(folds)


def calculate_negative_h2_trades(
    selected: Sequence[SelectedSessionEvent],
    price_lookup: Mapping[tuple[str, str], float],
    sessions: Sequence[str],
    symbols: Sequence[str],
    folds: Sequence[Sequence[str]],
    *,
    cost_bps_per_side: float = 10.0,
) -> tuple[ReplicationTrade, ...]:
    """Calculate independent two-session short trades, excluding fold crossings."""

    index = {session: position for position, session in enumerate(sessions)}
    fold_by_session = {session: fold for fold, values in enumerate(folds) for session in values}
    cost = 2 * cost_bps_per_side / 10_000
    trades: list[ReplicationTrade] = []
    for event in selected:
        if event.direction != -1 or event.session not in index:
            continue
        start_index = index[event.session]
        if start_index + 2 >= len(sessions):
            continue
        end_session = sessions[start_index + 2]
        if fold_by_session.get(event.session) != fold_by_session.get(end_session):
            continue
        start = price_lookup.get((event.symbol, event.session))
        end = price_lookup.get((event.symbol, end_session))
        if start is None or end is None or start <= 0 or end <= 0:
            continue
        panel_returns = []
        for symbol in symbols:
            panel_start = price_lookup.get((symbol, event.session))
            panel_end = price_lookup.get((symbol, end_session))
            if panel_start is not None and panel_end is not None and panel_start > 0 and panel_end > 0:
                panel_returns.append(panel_end / panel_start - 1)
        if not panel_returns:
            continue
        stock_return = end / start - 1
        gross_short = -stock_return
        trades.append(
            ReplicationTrade(
                event.session,
                end_session,
                event.symbol,
                gross_short,
                gross_short - cost,
                fmean(panel_returns) - stock_return - cost,
            )
        )
    return tuple(sorted(trades, key=lambda row: (row.start_session, row.symbol, row.end_session)))


def bootstrap_session_mean(
    trades: Sequence[ReplicationTrade],
    sessions: Sequence[str],
    *,
    block_length: int,
    replications: int,
    seed: int,
    confidence_level: float,
) -> BootstrapMean:
    if not trades or block_length < 1 or block_length > len(sessions):
        raise IndependentReplicationError("bootstrap inputs are empty or incompatible")
    by_session: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        by_session[trade.start_session].append(trade.net_short_return)
    generator = np.random.default_rng(seed)
    block_count = math.ceil(len(sessions) / block_length)
    maximum_start = len(sessions) - block_length
    samples: list[float] = []
    for _ in range(replications):
        values: list[float] = []
        sampled_sessions: list[str] = []
        for _block in range(block_count):
            start = int(generator.integers(0, maximum_start + 1))
            sampled_sessions.extend(sessions[start : start + block_length])
        for session in sampled_sessions[: len(sessions)]:
            values.extend(by_session.get(session, ()))
        if values:
            samples.append(fmean(values))
    if len(samples) / replications < 0.99:
        raise IndependentReplicationError("fewer than 99% of bootstrap draws contained a trade")
    tail = (1 - confidence_level) / 2
    array = np.asarray(samples, dtype=float)
    observed = fmean(trade.net_short_return for trade in trades)
    return BootstrapMean(
        observed,
        float(np.quantile(array, tail)),
        float(np.quantile(array, 1 - tail)),
        (1 + int(np.sum(array <= 0))) / (len(samples) + 1),
        len(samples),
        replications,
        block_length,
        seed,
    )


def evaluate_replication(
    trades: Sequence[ReplicationTrade],
    folds: Sequence[FoldResult],
    bootstrap: BootstrapMean,
    acceptance: Mapping[str, Any],
    inference: Mapping[str, Any],
) -> ReplicationDecision:
    supported = len({trade.symbol for trade in trades})
    positive_folds = sum(row.mean_net_short_return is not None and row.mean_net_short_return > 0 for row in folds)
    reasons: list[str] = []
    if len(trades) < int(acceptance["minimum_total_trades"]):
        reasons.append("insufficient_total_trades")
    if supported < int(acceptance["minimum_supported_stocks"]):
        reasons.append("insufficient_supported_stocks")
    minimum_per_fold = int(acceptance.get("minimum_trades_per_fold", 0))
    if any(row.trades < minimum_per_fold for row in folds):
        reasons.append("insufficient_trades_in_one_or_more_folds")
    if positive_folds != int(acceptance["required_positive_folds"]):
        reasons.append("mean_net_return_not_positive_in_all_folds")
    if bootstrap.observed_mean <= 0:
        reasons.append("mean_net_short_return_not_positive")
    if bootstrap.one_sided_p_value > float(inference["one_sided_p_value_maximum"]):
        reasons.append("one_sided_p_value_failed")
    if bootstrap.ci_low <= 0:
        reasons.append("99pct_confidence_lower_bound_not_positive")
    return ReplicationDecision(
        not reasons,
        tuple(reasons),
        len(trades),
        supported,
        positive_folds,
        bootstrap.one_sided_p_value,
        bootstrap.ci_low,
    )


def _fold_results(trades: Sequence[ReplicationTrade], folds: Sequence[Sequence[str]]) -> tuple[FoldResult, ...]:
    output = []
    for index, sessions in enumerate(folds):
        allowed = set(sessions)
        rows = [row for row in trades if row.start_session in allowed]
        output.append(
            FoldResult(
                index,
                sessions[0],
                sessions[-1],
                len(rows),
                len({row.symbol for row in rows}),
                fmean(row.net_short_return for row in rows) if rows else None,
                fmean(row.panel_hedged_net_short_return for row in rows) if rows else None,
            )
        )
    return tuple(output)


def _stock_rows(trades: Sequence[ReplicationTrade]) -> list[dict[str, Any]]:
    grouped: dict[str, list[ReplicationTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.symbol].append(trade)
    rows = []
    for symbol in sorted(grouped):
        values = grouped[symbol]
        rows.append(
            {
                "symbol": symbol,
                "trades": len(values),
                "mean_gross_short_return": fmean(row.gross_short_return for row in values),
                "mean_net_short_return": fmean(row.net_short_return for row in values),
                "median_net_short_return": median(row.net_short_return for row in values),
                "win_rate": sum(row.net_short_return > 0 for row in values) / len(values),
                "mean_panel_hedged_net_short_return": fmean(row.panel_hedged_net_short_return for row in values),
            }
        )
    return rows


def _stock_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    width, height = 900, 460
    left, right, top, bottom = 70, 25, 45, 110
    chart_width, chart_height = width - left - right, height - top - bottom
    values = [float(row["mean_net_short_return"]) for row in rows]
    bound = max(max((abs(value) for value in values), default=0), 0.001)
    zero = top + chart_height / 2
    step = chart_width / max(len(rows), 1)
    bar_width = step * 0.7
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="18">Mean net two-session short return by stock</text>',
        f'<line x1="{left}" y1="{zero:.2f}" x2="{width-right}" y2="{zero:.2f}" stroke="#444"/>',
    ]
    for index, row in enumerate(rows):
        value = float(row["mean_net_short_return"])
        scaled = value / bound * chart_height / 2
        x = left + index * step + (step - bar_width) / 2
        y = zero - max(scaled, 0)
        color = "#2a7f62" if value > 0 else "#b64949"
        label_x = x + bar_width / 2
        label_y = height - bottom + 18
        elements.extend(
            [
                (
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                    f'height="{abs(scaled):.2f}" fill="{color}"/>'
                ),
                (
                    f'<text x="{label_x:.2f}" y="{label_y}" transform="rotate(55 {label_x:.2f} {label_y})" '
                    f'font-family="sans-serif" font-size="11">{html.escape(str(row["symbol"]))}</text>'
                ),
            ]
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _report(
    replication_id: str,
    decision: ReplicationDecision,
    trades: Sequence[ReplicationTrade],
    folds: Sequence[FoldResult],
    bootstrap: BootstrapMean,
    stocks: Sequence[Mapping[str, Any]],
    selection_counts: Mapping[str, int],
) -> str:
    decision_text = "PASS" if decision.passed else "STOP"
    lines = [
        f"# Independent mid-cap negative-news replication: {replication_id}",
        "",
        "> Frozen independent replication. No parameter was selected from these returns.",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: 2026-07-17",
        "- Verification Status: ANALYZED pending deterministic replay and full tests",
        "- Version Label: `exp_result_v1`",
        "",
        "## Decision",
        "",
        f"**{decision_text}**",
        "",
        f"- Trades: `{decision.trades}` across `{decision.supported_stocks}` stocks.",
        f"- Mean net short return: `{bootstrap.observed_mean:.3%}`.",
        f"- 99% bootstrap interval: `{bootstrap.ci_low:.3%}` to `{bootstrap.ci_high:.3%}`.",
        f"- One-sided block-bootstrap p-value: `{bootstrap.one_sided_p_value:.6f}`.",
        f"- Positive folds: `{decision.positive_folds}/3`.",
    ]
    if decision.rejection_reasons:
        lines.append(f"- Failed requirements: `{', '.join(decision.rejection_reasons)}`.")
    lines.extend(
        [
            "",
            "## Frozen design",
            "",
            "- One strongest eligible model event per stock-session, then negative events only.",
            "- Short at the eligible split-adjusted open; exit two sessions later.",
            "- 10 bps per side; dividends are not adjusted.",
            "- Three equal contiguous time folds; no trade crosses a fold boundary.",
            "- Ordinary non-wrapping five-session block bootstrap, 2,000 replications, seed `20260715`.",
            "- Primary outcome is the raw net short return; panel-hedged return is diagnostic only.",
            "",
            "## Attrition",
            "",
            f"- Frozen events: `{selection_counts['input_events']}`.",
            f"- Exhausted invalid-score exclusions: `{selection_counts['scoring_exclusions']}`.",
            f"- Model-eligible events: `{selection_counts['eligible_events']}`.",
            f"- Selected strongest stock-sessions: `{selection_counts['selected_sessions']}`.",
            f"- Selected negative stock-sessions: `{selection_counts['negative_sessions']}`.",
            f"- Fold-contained, price-supported trades: `{len(trades)}`.",
            "",
            "## Time-fold results",
            "",
            "| Fold | Sessions | Trades | Stocks | Mean net short | Mean panel-hedged |",
            "| ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold in folds:
        net = "n/a" if fold.mean_net_short_return is None else f"{fold.mean_net_short_return:.3%}"
        hedged = (
            "n/a" if fold.mean_panel_hedged_net_short_return is None else f"{fold.mean_panel_hedged_net_short_return:.3%}"
        )
        lines.append(
            f"| {fold.fold_index + 1} | {fold.first_session} to {fold.last_session} | {fold.trades} | "
            f"{fold.supported_stocks} | {net} | {hedged} |"
        )
    lines.extend(
        [
            "",
            "## Stock-level results",
            "",
            "| Symbol | Trades | Mean gross | Mean net | Median net | Win rate | Mean panel-hedged |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in stocks:
        lines.append(
            f"| {row['symbol']} | {row['trades']} | {float(row['mean_gross_short_return']):.3%} | "
            f"{float(row['mean_net_short_return']):.3%} | {float(row['median_net_short_return']):.3%} | "
            f"{float(row['win_rate']):.1%} | {float(row['mean_panel_hedged_net_short_return']):.3%} |"
        )
    lines.extend(
        [
            "",
            "![Mean net return by stock](figures/stock_mean_net_return.svg)",
            "",
            "## Interpretation",
            "",
            "This is an association test using model-defined labels, not evidence of causality or deployable alpha. "
            "The panel is an independent stock universe but overlaps the same historical regime. A pass would justify "
            "prospective evaluation of the frozen rule, not further historical tuning. A failure triggers the predeclared stop.",
            "",
        ]
    )
    return "\n".join(lines)


def run_independent_replication(
    config_path: str | Path,
    universe_dir: str | Path,
    strategy_dir: str | Path,
) -> ReplicationRun:
    """Open the frozen prices once and materialize the predeclared aggregate result."""

    frozen = load_frozen_replication(config_path)
    universe_root = Path(universe_dir)
    strategy_root = Path(strategy_dir)
    validate_model_only_scoring_universe(universe_root)
    universe_manifest = read_json(universe_root / "sample_manifest.json")
    universe_identity = universe_manifest.get("identity", {})
    if universe_identity.get("replication_contract") != REPLICATION_CONTRACT:
        raise IndependentReplicationError("universe is not the frozen independent replication")
    if universe_identity.get("replication_identity_sha256") is None:
        raise IndependentReplicationError("universe has no replication identity")
    events_path = Path(str(universe_identity["events_path"]))
    if not events_path.is_file() or sha256_file(events_path) != universe_identity.get("events_sha256"):
        raise IndependentReplicationError("frozen replication events are missing or changed")
    scores_path = strategy_root / "scores.jsonl"
    strategy_manifest_path = strategy_root / "manifest.json"
    strategy_manifest = read_json(strategy_manifest_path)
    if strategy_manifest.get("identity", {}).get("universe_identity_sha256") != universe_manifest.get("identity_sha256"):
        raise IndependentReplicationError("filtered scores do not belong to the replication universe")
    if sha256_file(scores_path) != strategy_manifest.get("outputs", {}).get("scores.jsonl", {}).get("sha256"):
        raise IndependentReplicationError("filtered replication scores are missing or changed")

    prices_config = frozen.payload["prices"]
    price_path = Path(str(prices_config["panel_path"]))
    if not price_path.is_file() or sha256_file(price_path) != prices_config["panel_sha256"]:
        raise IndependentReplicationError("frozen price panel is missing or changed")
    price_manifest_path = Path(str(prices_config["manifest_path"]))
    price_manifest = read_json(price_manifest_path)
    semantic_field = str(prices_config["field"])
    if (
        semantic_field == "adjusted_open"
        and price_manifest.get("provider") == "lseg"
        and price_manifest.get("return_convention") == "split-adjusted price returns; dividends not back-adjusted"
    ):
        physical_price_column = "open"
    else:
        physical_price_column = semantic_field
    opens = load_adjusted_opens_csv(price_path, price_column=physical_price_column)
    start = str(frozen.payload["news"]["analysis_start_session"])
    end = str(frozen.payload["news"]["analysis_end_session"])
    all_sessions = tuple(sorted({row.session for row in opens}))
    sessions = tuple(session for session in all_sessions if start <= session <= end)
    symbols = tuple(sorted({row.symbol for row in opens}))
    folds = _equal_folds(sessions, int(frozen.payload["folds"]["count"]))
    events = [event.to_payload() for event in load_strategy_events(events_path)]
    scores = read_jsonl(scores_path)
    score_event_ids = {str(row.get("event_id") or "") for row in scores}
    scored_events = [event for event in events if str(event.get("event_id") or "") in score_event_ids]
    scoring_exclusions = len(events) - len(scored_events)
    if scoring_exclusions != int(strategy_manifest.get("row_counts", {}).get("excluded_scores", -1)):
        raise IndependentReplicationError("score exclusions disagree with the finalized score manifest")
    selection = select_strongest_session_events(scored_events, scores, sessions)
    negative = tuple(row for row in selection.selected_events if row.direction == -1)
    price_lookup = {(row.symbol, row.session): row.adjusted_open for row in opens}
    trades = calculate_negative_h2_trades(
        negative,
        price_lookup,
        sessions,
        symbols,
        folds,
        cost_bps_per_side=float(frozen.payload["strategy"]["cost_bps_per_side"]),
    )
    fold_rows = _fold_results(trades, folds)
    inference = frozen.payload["inference"]
    bootstrap = bootstrap_session_mean(
        trades,
        sessions,
        block_length=int(inference["block_length_sessions"]),
        replications=int(inference["replications"]),
        seed=int(inference["seed"]),
        confidence_level=float(inference["confidence_level"]),
    )
    acceptance = dict(frozen.payload["acceptance"])
    acceptance["minimum_trades_per_fold"] = int(frozen.payload["folds"]["minimum_trades_per_fold"])
    decision = evaluate_replication(trades, fold_rows, bootstrap, acceptance, inference)
    stock_rows = _stock_rows(trades)
    selection_counts = {
        "input_events": len(events),
        "scoring_exclusions": scoring_exclusions,
        "eligible_events": selection.eligible_events,
        "selected_sessions": len(selection.selected_events),
        "negative_sessions": len(negative),
    }
    identity = {
        "schema_version": 1,
        "contract": REPLICATION_CONTRACT,
        "frozen_config_sha256": frozen.config_sha256,
        "universe_identity_sha256": universe_manifest["identity_sha256"],
        "events_sha256": sha256_file(events_path),
        "filtered_scores_sha256": sha256_file(scores_path),
        "price_panel_sha256": sha256_file(price_path),
        "price_manifest_sha256": sha256_file(price_manifest_path),
        "price_field_resolution": {
            "semantic_field": semantic_field,
            "physical_column": physical_price_column,
            "manifest_return_convention": price_manifest.get("return_convention"),
        },
        "decision_rule": dict(frozen.payload["acceptance"]),
        "inference": dict(inference),
        "output_policy": "aggregate-and-stock-level only; no event identifiers or article text",
    }
    identity_hash = sha256_text(canonical_json(identity))
    replication_id = f"negative-h2-replication-{identity_hash[:12]}"
    output_dir = frozen.results_root / replication_id
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    fold_path = output_dir / "fold_summary.csv"
    stock_path = output_dir / "stock_summary.csv"
    bootstrap_path = output_dir / "bootstrap.json"
    acceptance_path = output_dir / "acceptance.json"
    figure_path = output_dir / "figures" / "stock_mean_net_return.svg"
    expected = (report_path, fold_path, stock_path, bootstrap_path, acceptance_path, figure_path)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash or existing.get("status") != "completed":
            raise IndependentReplicationError("existing replication result identity differs or is incomplete")
        for path in expected:
            if sha256_file(path) != existing.get("outputs", {}).get(path.relative_to(output_dir).as_posix(), {}).get("sha256"):
                raise IndependentReplicationError(f"completed replication output changed: {path}")
        stored = existing.get("decision", {})
        if canonical_json(stored) != canonical_json(asdict(decision)):
            raise IndependentReplicationError("completed replication decision disagrees with deterministic replay")
        return ReplicationRun(replication_id, output_dir, report_path, manifest_path, decision, True)
    if output_dir.exists():
        raise IndependentReplicationError(f"refusing to overwrite incomplete replication output: {output_dir}")
    figure_path.parent.mkdir(parents=True, exist_ok=False)
    atomic_write_text(fold_path, _csv_text([asdict(row) for row in fold_rows], list(FoldResult.__dataclass_fields__)))
    atomic_write_text(stock_path, _csv_text(stock_rows, list(stock_rows[0]) if stock_rows else ("symbol",)))
    atomic_write_json(bootstrap_path, asdict(bootstrap))
    atomic_write_json(acceptance_path, asdict(decision))
    atomic_write_text(figure_path, _stock_svg(stock_rows))
    atomic_write_text(
        report_path,
        _report(replication_id, decision, trades, fold_rows, bootstrap, stock_rows, selection_counts),
    )
    outputs = {
        path.relative_to(output_dir).as_posix(): {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in expected
    }
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "completed",
            "replication_id": replication_id,
            "identity_sha256": identity_hash,
            "identity": identity,
            "row_counts": {"trades": len(trades), "stocks": len(stock_rows)},
            "selection_counts": selection_counts,
            "bootstrap": asdict(bootstrap),
            "decision": asdict(decision),
            "outputs": outputs,
            "warning": "Independent historical association test; not evidence of deployable or profitable alpha.",
        },
    )
    return ReplicationRun(replication_id, output_dir, report_path, manifest_path, decision, False)
