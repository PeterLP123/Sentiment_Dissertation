"""Development-only local audit of direction, materiality, and novelty labels.

This module deliberately stops before strategy construction. It samples only
pre-evaluation events, keeps licensed text under ``Data/derived``, scores three
strict five-level enums in one structured loopback-Ollama response, and exposes
a separate gate for genuinely human labels.
"""

from __future__ import annotations

import csv
import io
import ipaddress
import tomllib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

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
from ..models import StructuredJSONResponseRecord
from ..ollama_client import OllamaClient
from ..prompts import prompt_hash

SignalQualityTask = Literal["direction_severity", "materiality", "novelty"]

DIRECTION_LABELS = ("very_negative", "negative", "neutral", "positive", "very_positive")
LEVEL_LABELS = ("none", "low", "moderate", "high", "very_high")
TASKS: tuple[SignalQualityTask, ...] = ("direction_severity", "materiality", "novelty")
TASK_LABELS: dict[SignalQualityTask, tuple[str, ...]] = {
    "direction_severity": DIRECTION_LABELS,
    "materiality": LEVEL_LABELS,
    "novelty": LEVEL_LABELS,
}
JOINT_PROMPT_ID = "strategy_target_signal_quality_joint_v2"
DEFAULT_JOINT_PROMPT_PATH = Path("configs/strategy_research/signal_quality_joint_prompt.toml")
SOURCE_STRATA = ("negative", "neutral", "positive")
JOINT_SIGNAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "direction_severity": {"type": "string", "enum": list(DIRECTION_LABELS)},
        "materiality": {"type": "string", "enum": list(LEVEL_LABELS)},
        "novelty": {"type": "string", "enum": list(LEVEL_LABELS)},
    },
    "required": list(TASKS),
    "additionalProperties": False,
}


class SignalQualityError(RuntimeError):
    """Raised when the signal-quality audit cannot proceed safely."""


@dataclass(frozen=True)
class JointSignalPrompt:
    prompt_id: str
    system_prompt: str
    user_template: str
    output_mode: str
    prompt_hash: str

    def render(self, item: AuditItem) -> str:
        return self.user_template.format(
            target_company=item.target_company_name,
            symbol=item.symbol,
            current_news=_current_text(item.headline, item.lead_or_body),
            prior_context=item.prior_context,
        )


@dataclass(frozen=True)
class SignalAuditSpec:
    sample_per_symbol: int = 3
    seed: int = 20_260_716
    prior_window_days: int = 30
    maximum_prior_headlines: int = 5
    require_complete_human_sample: bool = True
    minimum_weighted_kappa: float = 0.40
    maximum_severe_disagreement_rate: float = 0.10
    minimum_high_level_precision: float = 0.70

    def __post_init__(self) -> None:
        if self.sample_per_symbol < 1:
            raise ValueError("sample_per_symbol must be positive")
        if self.prior_window_days < 1 or self.maximum_prior_headlines < 1:
            raise ValueError("prior context limits must be positive")
        if not -1 <= self.minimum_weighted_kappa <= 1:
            raise ValueError("minimum_weighted_kappa must be in [-1, 1]")
        if not 0 <= self.maximum_severe_disagreement_rate <= 1:
            raise ValueError("maximum_severe_disagreement_rate must be in [0, 1]")
        if not 0 <= self.minimum_high_level_precision <= 1:
            raise ValueError("minimum_high_level_precision must be in [0, 1]")


@dataclass(frozen=True)
class AuditItem:
    audit_id: str
    event_id: str
    symbol: str
    target_company_name: str
    available_at_utc: str
    eligible_execution_session: str
    headline: str
    lead_or_body: str
    prior_context: str
    content_hash: str
    context_hash: str

    def to_row(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> AuditItem:
        item = cls(**{field: str(row.get(field) or "") for field in cls.__dataclass_fields__})
        if item.content_hash != sha256_text(_current_text(item.headline, item.lead_or_body)):
            raise SignalQualityError(f"audit item content hash mismatch: {item.audit_id}")
        if item.context_hash != sha256_text(item.prior_context):
            raise SignalQualityError(f"audit item context hash mismatch: {item.audit_id}")
        return item


@dataclass(frozen=True)
class PreparedSignalAudit:
    experiment_id: str
    output_dir: Path
    items_path: Path
    human_template_path: Path
    codebook_path: Path
    manifest_path: Path
    item_count: int
    reused: bool


@dataclass(frozen=True)
class JointSignalRecord:
    audit_id: str
    event_id: str
    symbol: str
    direction_severity: str | None
    materiality: str | None
    novelty: str | None
    status: str
    model_id: str
    model_digest: str
    endpoint: str
    prompt_id: str
    prompt_hash: str
    input_hash: str
    cache_key: str
    attempt_count: int
    latency_ms: float | None
    created_at_utc: str
    error: str | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> JointSignalRecord:
        return cls(
            audit_id=str(payload.get("audit_id") or ""),
            event_id=str(payload.get("event_id") or ""),
            symbol=str(payload.get("symbol") or "").upper(),
            direction_severity=(
                str(payload["direction_severity"]) if payload.get("direction_severity") is not None else None
            ),
            materiality=str(payload["materiality"]) if payload.get("materiality") is not None else None,
            novelty=str(payload["novelty"]) if payload.get("novelty") is not None else None,
            status=str(payload.get("status") or ""),
            model_id=str(payload.get("model_id") or ""),
            model_digest=str(payload.get("model_digest") or ""),
            endpoint=str(payload.get("endpoint") or ""),
            prompt_id=str(payload.get("prompt_id") or ""),
            prompt_hash=str(payload.get("prompt_hash") or ""),
            input_hash=str(payload.get("input_hash") or ""),
            cache_key=str(payload.get("cache_key") or ""),
            attempt_count=int(payload.get("attempt_count") or 0),
            latency_ms=float(payload["latency_ms"]) if payload.get("latency_ms") is not None else None,
            created_at_utc=str(payload.get("created_at_utc") or ""),
            error=str(payload["error"]) if payload.get("error") is not None else None,
        )

    @property
    def labels(self) -> dict[str, str | None]:
        return {
            "direction_severity": self.direction_severity,
            "materiality": self.materiality,
            "novelty": self.novelty,
        }


@dataclass(frozen=True)
class SignalAuditScoreRun:
    score_id: str
    output_dir: Path
    manifest_path: Path
    scores_path: Path
    expected_calls: int
    successful_scores: int
    calls_made: int
    complete: bool
    reused: bool


@dataclass(frozen=True)
class ValidationMetric:
    task: SignalQualityTask
    rows: int
    exact_agreement: float
    within_one_level: float
    severe_disagreement_rate: float
    quadratic_weighted_kappa: float | None
    high_level_precision: float | None
    high_level_recall: float | None


@dataclass(frozen=True)
class SignalAuditAcceptance:
    check: str
    passed: bool
    observed: str
    requirement: str


@dataclass(frozen=True)
class SignalAuditValidationRun:
    validation_id: str
    output_dir: Path
    report_path: Path
    manifest_path: Path
    metrics: tuple[ValidationMetric, ...]
    acceptance: tuple[SignalAuditAcceptance, ...]
    reused: bool


def _parse_timestamp(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SignalQualityError(f"naive audit timestamp: {value!r}")
    return parsed.astimezone(UTC)


def _current_text(headline: str, lead_or_body: str) -> str:
    return "\n\n".join(value.strip() for value in (headline, lead_or_body) if value.strip())


def _stable_rank(seed: int, namespace: str, event_id: str) -> str:
    return sha256_text(f"{seed}|{namespace}|{event_id}")


def select_audit_events(
    events: Sequence[Mapping[str, Any]],
    scores: Sequence[Mapping[str, Any]],
    evaluation_start: str,
    spec: SignalAuditSpec,
) -> tuple[tuple[dict[str, Any], str], ...]:
    """Choose a deterministic pre-evaluation sample, balanced by stock and v1 label."""

    score_by_event = {str(row.get("event_id") or ""): row for row in scores}
    by_symbol: dict[str, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for raw_event in events:
        event = dict(raw_event)
        if str(event.get("eligible_execution_session") or "") >= evaluation_start:
            continue
        event_id = str(event.get("event_id") or "")
        score = score_by_event.get(event_id)
        if score is None or score.get("status") != "success":
            continue
        label = str(score.get("raw_label") or "")
        if label not in SOURCE_STRATA:
            raise SignalQualityError(f"unexpected source score label for audit sampling: {label!r}")
        symbol = str(event.get("symbol") or "").upper()
        by_symbol[symbol].append((event, label))
    if not by_symbol:
        raise SignalQualityError("no scored development events are available for audit sampling")

    selected: list[tuple[dict[str, Any], str]] = []
    for symbol in sorted(by_symbol):
        candidates = by_symbol[symbol]
        if len(candidates) < spec.sample_per_symbol:
            raise SignalQualityError(f"{symbol} has fewer than {spec.sample_per_symbol} development events")
        symbol_selected: list[tuple[dict[str, Any], str]] = []
        for label in SOURCE_STRATA[: spec.sample_per_symbol]:
            stratum = [row for row in candidates if row[1] == label]
            if stratum:
                symbol_selected.append(
                    min(stratum, key=lambda row: _stable_rank(spec.seed, f"{symbol}|{label}", str(row[0]["event_id"])))
                )
        selected_ids = {str(row[0]["event_id"]) for row in symbol_selected}
        remaining = [row for row in candidates if str(row[0]["event_id"]) not in selected_ids]
        remaining.sort(key=lambda row: _stable_rank(spec.seed, f"{symbol}|fallback", str(row[0]["event_id"])))
        symbol_selected.extend(remaining[: spec.sample_per_symbol - len(symbol_selected)])
        if len(symbol_selected) != spec.sample_per_symbol:
            raise SignalQualityError(f"could not fill the audit allocation for {symbol}")
        selected.extend(symbol_selected)
    selected.sort(key=lambda row: _stable_rank(spec.seed, "audit-order", str(row[0]["event_id"])))
    return tuple(selected)


def build_prior_context(
    events: Sequence[Mapping[str, Any]],
    selected_event_ids: Sequence[str],
    evaluation_start: str,
    spec: SignalAuditSpec,
) -> dict[str, str]:
    """Return only strictly earlier same-stock headlines from the development block."""

    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in events:
        event = dict(raw)
        if str(event.get("eligible_execution_session") or "") < evaluation_start:
            by_symbol[str(event.get("symbol") or "").upper()].append(event)
    for rows in by_symbol.values():
        rows.sort(key=lambda row: (_parse_timestamp(row.get("available_at_utc")), str(row.get("event_id") or "")))
    selected = set(selected_event_ids)
    contexts: dict[str, str] = {}
    window = timedelta(days=spec.prior_window_days)
    for rows in by_symbol.values():
        for current in rows:
            event_id = str(current.get("event_id") or "")
            if event_id not in selected:
                continue
            current_time = _parse_timestamp(current.get("available_at_utc"))
            prior = [
                row
                for row in rows
                if _parse_timestamp(row.get("available_at_utc")) < current_time
                and current_time - _parse_timestamp(row.get("available_at_utc")) <= window
            ][-spec.maximum_prior_headlines :]
            if not prior:
                contexts[event_id] = "(No strictly earlier same-company headline in the frozen context window.)"
                continue
            contexts[event_id] = "\n".join(
                f"[{_parse_timestamp(row.get('available_at_utc')).date().isoformat()}] {str(row.get('headline') or '').strip()}"
                for row in prior
            )
    missing = selected - set(contexts)
    if missing:
        raise SignalQualityError(f"failed to construct prior context for {len(missing)} audit events")
    return contexts


def load_joint_signal_prompt(path: str | Path = DEFAULT_JOINT_PROMPT_PATH) -> JointSignalPrompt:
    prompt_path = Path(path)
    with prompt_path.open("rb") as handle:
        payload = tomllib.load(handle)
    raw = payload.get("prompt")
    if not isinstance(raw, dict):
        raise SignalQualityError(f"joint signal prompt file has no [prompt] table: {prompt_path}")
    prompt_id = str(raw.get("id") or "")
    system_prompt = str(raw.get("system_prompt") or "").strip()
    user_template = str(raw.get("user_template") or "").strip()
    output_mode = str(raw.get("output_mode") or "")
    if prompt_id != JOINT_PROMPT_ID:
        raise SignalQualityError(f"unexpected joint signal prompt id: {prompt_id!r}")
    if output_mode != "structured_json":
        raise SignalQualityError("joint signal prompt must use structured_json output")
    required_fields = ("{target_company}", "{symbol}", "{current_news}", "{prior_context}")
    if any(field not in user_template for field in required_fields):
        raise SignalQualityError("joint signal prompt user template is missing a required field")
    required_contract_text = (
        "Do not use outside knowledge",
        "predict a stock return",
        "materiality=none requires direction_severity=neutral",
        "use only for novelty",
    )
    combined = f"{system_prompt}\n{user_template}"
    if any(text not in combined for text in required_contract_text):
        raise SignalQualityError("joint signal prompt is missing a required leakage or consistency rule")
    for label in (*DIRECTION_LABELS, *LEVEL_LABELS):
        if label not in system_prompt:
            raise SignalQualityError(f"joint signal prompt omits enum label: {label}")
    return JointSignalPrompt(
        prompt_id=prompt_id,
        system_prompt=system_prompt,
        user_template=user_template,
        output_mode=output_mode,
        prompt_hash=prompt_hash(prompt_id, system_prompt, user_template, output_mode),
    )


def _csv_rows(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _load_audit_items(path: Path) -> tuple[AuditItem, ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        items = tuple(AuditItem.from_row(row) for row in csv.DictReader(handle))
    if not items or len({item.audit_id for item in items}) != len(items):
        raise SignalQualityError("audit items are empty or contain duplicate audit IDs")
    if len({item.event_id for item in items}) != len(items):
        raise SignalQualityError("audit items contain duplicate event IDs")
    return items


def prepare_signal_audit(
    run_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    spec: SignalAuditSpec | None = None,
) -> PreparedSignalAudit:
    """Freeze an outcome-blind audit packet without making a model call."""

    root = Path(run_dir)
    report_manifest_path = root / "manifests" / "report.json"
    if not report_manifest_path.is_file():
        raise SignalQualityError("completed source run report manifest is missing")
    report_manifest = read_json(report_manifest_path)
    if report_manifest.get("status") != "completed":
        raise SignalQualityError("source run report stage is not completed")
    config = report_manifest["config"]
    evaluation_start = str(config["run"]["evaluation_start"])
    derived_run = Path(str(config["outputs"]["derived_root"])) / root.name
    events_path = derived_run / "events" / "events.jsonl"
    scores_path = derived_run / "scores" / "scores.jsonl"
    for required in (events_path, scores_path):
        if not required.is_file():
            raise SignalQualityError(f"required audit input is missing: {required}")
    audit_spec = spec or SignalAuditSpec()
    identity = {
        "schema_version": 2,
        "source_run_id": root.name,
        "source_run_identity": report_manifest.get("run_identity_sha256"),
        "evaluation_start": evaluation_start,
        "events_sha256": sha256_file(events_path),
        "source_scores_sha256": sha256_file(scores_path),
        "model": config["scoring"]["model"],
        "model_digest": config["scoring"]["model_digest"],
        "spec": asdict(audit_spec),
        "sampling_contract": "per stock: one deterministic v1 negative/neutral/positive when available, then hash-ranked fallback",
        "context_contract": "strictly earlier same-stock development headlines only; no prices, returns, or evaluation events",
        "prompt_independence": "sample identity is independent of later scorer prompt revisions",
        "label_contracts": {task: list(TASK_LABELS[task]) for task in TASKS},
    }
    identity_hash = sha256_text(canonical_json(identity))
    experiment_id = f"signal-quality-audit-{identity_hash[:12]}"
    output_dir = Path(output_root) if output_root is not None else derived_run / "signal_quality" / experiment_id
    items_path = output_dir / "audit_items.csv"
    template_path = output_dir / "human_labels_template.csv"
    codebook_path = output_dir / "audit_codebook.md"
    plan_path = output_dir / "experiment_plan.md"
    manifest_path = output_dir / "sample_manifest.json"
    expected = (items_path, template_path, codebook_path, plan_path)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash:
            raise SignalQualityError(f"existing signal audit identity mismatch: {output_dir}")
        for path in expected:
            expected_hash = existing.get("outputs", {}).get(path.name, {}).get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise SignalQualityError(f"completed signal audit output is missing or changed: {path}")
        return PreparedSignalAudit(
            experiment_id,
            output_dir,
            items_path,
            template_path,
            codebook_path,
            manifest_path,
            int(existing["row_counts"]["audit_items"]),
            True,
        )
    if output_dir.exists():
        raise SignalQualityError(f"refusing to reuse incomplete signal audit directory: {output_dir}")

    events = read_jsonl(events_path)
    scores = read_jsonl(scores_path)
    selected = select_audit_events(events, scores, evaluation_start, audit_spec)
    contexts = build_prior_context(events, [str(row[0]["event_id"]) for row in selected], evaluation_start, audit_spec)
    audit_items: list[AuditItem] = []
    source_strata: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    for index, (event, source_label) in enumerate(selected, start=1):
        event_id = str(event["event_id"])
        context = contexts[event_id]
        item = AuditItem(
            audit_id=f"SQ-{index:03d}",
            event_id=event_id,
            symbol=str(event["symbol"]).upper(),
            target_company_name=str(event.get("target_company_name") or event["symbol"]),
            available_at_utc=str(event["available_at_utc"]),
            eligible_execution_session=str(event["eligible_execution_session"]),
            headline=str(event.get("headline") or ""),
            lead_or_body=str(event.get("lead_or_body") or ""),
            prior_context=context,
            content_hash=str(event["text_sha256"]),
            context_hash=sha256_text(context),
        )
        audit_items.append(item)
        source_strata[source_label] += 1
        symbol_counts[item.symbol] += 1
    fields = list(AuditItem.__dataclass_fields__)
    item_rows = [item.to_row() for item in audit_items]
    template_fields = (
        "audit_id",
        "event_id",
        "human_direction_severity",
        "human_materiality",
        "human_novelty",
        "human_notes",
    )
    template_rows = [
        {
            "audit_id": item.audit_id,
            "event_id": item.event_id,
            "human_direction_severity": "",
            "human_materiality": "",
            "human_novelty": "",
            "human_notes": "",
        }
        for item in audit_items
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_text(items_path, _csv_rows(item_rows, fields))
    atomic_write_text(template_path, _csv_rows(template_rows, template_fields))
    atomic_write_text(codebook_path, _audit_codebook(experiment_id, len(audit_items), audit_spec))
    atomic_write_text(plan_path, _experiment_plan(experiment_id, root, output_dir, len(audit_items), audit_spec))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "experiment_id": experiment_id,
        "identity_sha256": identity_hash,
        "identity": identity,
        "row_counts": {
            "audit_items": len(audit_items),
            "symbols": len(symbol_counts),
            "source_negative": source_strata["negative"],
            "source_neutral": source_strata["neutral"],
            "source_positive": source_strata["positive"],
        },
        "symbol_counts": dict(sorted(symbol_counts.items())),
        "outputs": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in expected
        },
        "warning": "Development-only audit sample; source labels used for sampling are intentionally absent from audit_items.csv.",
    }
    atomic_write_json(manifest_path, manifest)
    return PreparedSignalAudit(
        experiment_id,
        output_dir,
        items_path,
        template_path,
        codebook_path,
        manifest_path,
        len(audit_items),
        False,
    )


def _audit_codebook(experiment_id: str, item_count: int, spec: SignalAuditSpec) -> str:
    return f"""# Signal-quality human audit: {experiment_id}

This is a blind, development-only audit of {item_count} target-company news events. Read `audit_items.csv`, copy
`human_labels_template.csv` to `human_labels.csv`, and enter exactly one label in each of the three human columns.
Do not inspect model predictions or market returns while labelling. Label every row; use `human_notes` only for concise
ambiguity or evidence notes.

## Direction and severity

- `very_negative`: clearly large adverse target-company economic implication.
- `negative`: adverse but not clearly extreme.
- `neutral`: mixed, immaterial, incidental, unclear, or no directional target-company implication.
- `positive`: beneficial but not clearly extreme.
- `very_positive`: clearly large beneficial target-company economic implication.

## Materiality

- `none`: no target-specific economic consequence.
- `low`: minor consequence.
- `moderate`: meaningful but non-central consequence.
- `high`: clearly significant company consequence.
- `very_high`: potentially transformative, existential, or company-wide consequence.

## Novelty

Compare the current article only with `prior_context`.

- `none`: repetition or no new fact.
- `low`: minor detail or timing update.
- `moderate`: meaningful incremental fact in an existing development.
- `high`: substantially new event, outcome, or disclosure.
- `very_high`: first disclosure or major break not indicated by the supplied context.

## Frozen gate

- All {item_count} rows must be labelled when `require_complete_human_sample={str(spec.require_complete_human_sample).lower()}`.
- Quadratic weighted kappa must be at least {spec.minimum_weighted_kappa:.2f} for each task.
- A disagreement of three or more ordinal levels must be at most {spec.maximum_severe_disagreement_rate:.0%} for each task.
- Precision for model `high`/`very_high` materiality and novelty must be at least {spec.minimum_high_level_precision:.0%}.
- One human annotator audits model validity; this does not estimate human-human reliability.
"""


def _experiment_plan(
    experiment_id: str,
    run_dir: Path,
    output_dir: Path,
    item_count: int,
    spec: SignalAuditSpec,
) -> str:
    return f"""## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: {datetime.now(UTC).date().isoformat()}
- Verification Status: UNVERIFIED
- Version Label: signal_quality_audit_v1

## Experiment Overview

- **Title**: Development-only local signal-quality audit
- **ID**: `{experiment_id}`
- **Objective**: Determine whether a local model can reproduce human direction/severity, materiality, and novelty
  labels before any further backtest.
- **Hypothesis**: Materiality and novelty labels may identify noise that the three-class direction score retained.
- **Type**: classification audit

## Inputs

- Completed source run: `{run_dir}`
- Frozen sample: `{output_dir / 'audit_items.csv'}`
- Sample size: `{item_count}` events (`{spec.sample_per_symbol}` per stock)
- Boundary: events are strictly before the source run's evaluation start; no return data enters sampling or prompts.

## Execution sequence

1. `sentiment-bench strategy score-signal-audit --audit-dir {output_dir}`
2. Complete a blind copy named `human_labels.csv` from the template.
3. `sentiment-bench strategy validate-signal-audit --score-dir <score-dir> --human-labels {output_dir / 'human_labels.csv'}`
4. Do not score the full development universe or run another backtest unless the frozen human gate passes.
"""


def _assert_loopback_endpoint(endpoint: str) -> str:
    normalized = endpoint.strip().rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SignalQualityError("Ollama endpoint must be an HTTP(S) loopback URL")
    hostname = parsed.hostname.lower()
    if hostname != "localhost":
        try:
            if not ipaddress.ip_address(hostname).is_loopback:
                raise SignalQualityError("signal-quality scoring is local-only and requires a loopback Ollama endpoint")
        except ValueError as exc:
            raise SignalQualityError("signal-quality scoring is local-only and requires a loopback Ollama endpoint") from exc
    return normalized


def parse_joint_signal_labels(payload: Mapping[str, Any]) -> dict[SignalQualityTask, str]:
    """Validate the exact enums and semantic consistency of one joint response."""

    if set(payload) != set(TASKS):
        raise ValueError("joint signal response must contain exactly the three required keys")
    labels: dict[SignalQualityTask, str] = {}
    for task in TASKS:
        raw_label = payload.get(task)
        if not isinstance(raw_label, str) or raw_label not in TASK_LABELS[task]:
            raise ValueError(f"joint signal response has an invalid {task} label")
        labels[task] = raw_label
    direction = labels["direction_severity"]
    materiality = labels["materiality"]
    if materiality == "none" and direction != "neutral":
        raise ValueError("materiality=none requires direction_severity=neutral")
    if direction in {"very_negative", "very_positive"} and materiality not in {"high", "very_high"}:
        raise ValueError("extreme direction requires high or very_high materiality")
    return labels


def _score_key(
    item: AuditItem,
    prompt: JointSignalPrompt,
    *,
    model_id: str,
    model_digest: str,
    endpoint: str,
) -> str:
    return sha256_text(
        canonical_json(
            {
                "audit_id": item.audit_id,
                "event_id": item.event_id,
                "input_hash": sha256_text(prompt.render(item)),
                "model_id": model_id,
                "model_digest": model_digest,
                "endpoint": endpoint,
                "prompt_id": prompt.prompt_id,
                "prompt_hash": prompt.prompt_hash,
                "schema_sha256": sha256_text(canonical_json(JOINT_SIGNAL_SCHEMA)),
                "temperature": 0.0,
                "max_completion_tokens": 64,
                "ollama_think": False,
            }
        )
    )


def _completed_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / "completed" / key[:2] / f"{key}.json"


def _attempt_dir(cache_dir: Path, key: str) -> Path:
    return cache_dir / "attempts" / key[:2] / key


def _validate_joint_record(record: JointSignalRecord, expected: Mapping[str, Any], path: Path) -> None:
    actual = {field: getattr(record, field) for field in expected}
    if actual != dict(expected):
        raise SignalQualityError(f"incompatible or corrupt signal-quality cache record: {path}")
    if record.status == "success":
        try:
            parse_joint_signal_labels(record.labels)
        except ValueError as exc:
            raise SignalQualityError(f"invalid completed joint signal labels: {path}: {exc}") from exc


def _expected_record_identity(
    item: AuditItem,
    prompt: JointSignalPrompt,
    *,
    model_id: str,
    model_digest: str,
    endpoint: str,
) -> dict[str, Any]:
    key = _score_key(item, prompt, model_id=model_id, model_digest=model_digest, endpoint=endpoint)
    return {
        "audit_id": item.audit_id,
        "event_id": item.event_id,
        "symbol": item.symbol,
        "model_id": model_id,
        "model_digest": model_digest,
        "endpoint": endpoint,
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "input_hash": sha256_text(prompt.render(item)),
        "cache_key": key,
    }


def _load_cached_record(
    cache_dir: Path,
    item: AuditItem,
    prompt: JointSignalPrompt,
    *,
    model_id: str,
    model_digest: str,
    endpoint: str,
) -> JointSignalRecord | None:
    expected = _expected_record_identity(
        item,
        prompt,
        model_id=model_id,
        model_digest=model_digest,
        endpoint=endpoint,
    )
    path = _completed_path(cache_dir, str(expected["cache_key"]))
    if path.is_file():
        record = JointSignalRecord.from_payload(read_json(path))
        _validate_joint_record(record, expected, path)
        if record.status != "success":
            raise SignalQualityError(f"non-success record found in completed cache: {path}")
        return record
    attempts = _attempt_dir(cache_dir, str(expected["cache_key"]))
    recovered: JointSignalRecord | None = None
    for attempt_path in sorted(attempts.glob("*.json")) if attempts.is_dir() else ():
        record = JointSignalRecord.from_payload(read_json(attempt_path))
        _validate_joint_record(record, expected, attempt_path)
        if record.status == "success":
            recovered = record
    if recovered is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, recovered.to_payload())
    return recovered


def _previous_attempts(cache_dir: Path, key: str) -> int:
    attempts = _attempt_dir(cache_dir, key)
    count = 0
    for path in sorted(attempts.glob("*.json")) if attempts.is_dir() else ():
        record = JointSignalRecord.from_payload(read_json(path))
        if record.cache_key != key:
            raise SignalQualityError(f"signal-quality attempt cache mismatch: {path}")
        count = max(count, record.attempt_count)
    return count


def _write_joint_attempt(cache_dir: Path, record: JointSignalRecord) -> None:
    attempts = _attempt_dir(cache_dir, record.cache_key)
    attempts.mkdir(parents=True, exist_ok=True)
    path = attempts / f"{len(list(attempts.glob('*.json'))) + 1:06d}.json"
    if path.exists():
        raise SignalQualityError(f"refusing to overwrite signal-quality attempt: {path}")
    atomic_write_json(path, record.to_payload())
    if record.status == "success":
        completed = _completed_path(cache_dir, record.cache_key)
        if completed.exists():
            existing = JointSignalRecord.from_payload(read_json(completed))
            if existing != record:
                raise SignalQualityError(f"refusing to overwrite a different completed score: {completed}")
        else:
            completed.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(completed, record.to_payload())


async def _verify_ollama_model(client: Any, model_id: str, model_digest: str) -> None:
    models = await client.list_models()
    matching = [model for model in models if model.model_id == model_id]
    if not matching:
        raise SignalQualityError(f"Ollama does not expose the frozen model: {model_id}")
    observed = str(matching[0].raw_metadata.get("digest") or "")
    if observed != model_digest:
        raise SignalQualityError(
            f"Ollama model digest mismatch for {model_id}: expected {model_digest}, observed {observed or 'missing'}"
        )


def _progress_manifest(
    identity_hash: str,
    identity: Mapping[str, Any],
    *,
    score_id: str,
    expected: int,
    successful: int,
    calls_made: int,
    status: str,
    scores_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": status,
        "score_id": score_id,
        "identity_sha256": identity_hash,
        "identity": dict(identity),
        "progress": {
            "expected_event_scores": expected,
            "successful_event_scores": successful,
            "missing_event_scores": expected - successful,
            "calls_made_this_invocation": calls_made,
        },
    }
    if status == "completed":
        payload["outputs"] = {
            scores_path.name: {
                "path": scores_path.as_posix(),
                "sha256": sha256_file(scores_path),
                "size_bytes": scores_path.stat().st_size,
            }
        }
    return payload


async def score_signal_audit(
    audit_dir: str | Path,
    *,
    endpoint: str | None = None,
    max_new_calls: int | None = None,
    allow_calls: bool = False,
    client: Any | None = None,
    verify_model: bool = True,
) -> SignalAuditScoreRun:
    """Score the frozen audit sample through an explicitly authorized local model."""

    if max_new_calls is not None and max_new_calls < 1:
        raise ValueError("max_new_calls must be positive when supplied")
    audit_root = Path(audit_dir)
    sample_manifest_path = audit_root / "sample_manifest.json"
    items_path = audit_root / "audit_items.csv"
    if not sample_manifest_path.is_file() or not items_path.is_file():
        raise SignalQualityError("audit sample is incomplete")
    sample_manifest = read_json(sample_manifest_path)
    expected_items_hash = sample_manifest.get("outputs", {}).get(items_path.name, {}).get("sha256")
    if sample_manifest.get("status") != "completed" or sha256_file(items_path) != expected_items_hash:
        raise SignalQualityError("audit sample manifest or item hash is invalid")
    items = _load_audit_items(items_path)
    source_identity = sample_manifest["identity"]
    model_id = str(source_identity["model"])
    model_digest = str(source_identity["model_digest"])
    resolved_endpoint = _assert_loopback_endpoint(endpoint or "http://127.0.0.1:11435")
    prompt_path = DEFAULT_JOINT_PROMPT_PATH
    if not prompt_path.is_file():
        raise SignalQualityError(f"joint signal-quality prompt is missing: {prompt_path}")
    prompt = load_joint_signal_prompt(prompt_path)
    identity = {
        "schema_version": 2,
        "sample_identity_sha256": sample_manifest["identity_sha256"],
        "audit_items_sha256": sha256_file(items_path),
        "model": model_id,
        "model_digest": model_digest,
        "endpoint": resolved_endpoint,
        "prompt_path": prompt_path.as_posix(),
        "prompt_file_sha256": sha256_file(prompt_path),
        "prompt_id": prompt.prompt_id,
        "prompt_hash": prompt.prompt_hash,
        "schema_sha256": sha256_text(canonical_json(JOINT_SIGNAL_SCHEMA)),
        "labels": {task: list(TASK_LABELS[task]) for task in TASKS},
        "call_contract": "one joint structured response per event",
        "request": {
            "temperature": 0.0,
            "max_completion_tokens": 64,
            "ollama_think": False,
            "structured_json_output": True,
            "retries": 3,
        },
    }
    identity_hash = sha256_text(canonical_json(identity))
    score_id = f"signal-quality-scores-{identity_hash[:12]}"
    output_dir = audit_root / "scoring" / score_id
    cache_dir = output_dir / "cache"
    scores_path = output_dir / "model_scores.jsonl"
    manifest_path = output_dir / "scores_manifest.json"
    expected_calls = len(items)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash:
            raise SignalQualityError(f"signal-quality score identity mismatch: {output_dir}")
        if existing.get("status") == "completed":
            expected_hash = existing.get("outputs", {}).get(scores_path.name, {}).get("sha256")
            if not scores_path.is_file() or sha256_file(scores_path) != expected_hash:
                raise SignalQualityError("completed signal-quality scores are missing or changed")
            return SignalAuditScoreRun(
                score_id,
                output_dir,
                manifest_path,
                scores_path,
                expected_calls,
                expected_calls,
                0,
                True,
                True,
            )
    if not allow_calls:
        raise SignalQualityError("model calls require the explicit score-signal-audit command")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        manifest_path,
        _progress_manifest(
            identity_hash,
            identity,
            score_id=score_id,
            expected=expected_calls,
            successful=0,
            calls_made=0,
            status="in_progress",
            scores_path=scores_path,
        ),
    )

    owns_client = client is None
    scoring_client = client or OllamaClient(
        host=resolved_endpoint,
        keep_alive=-1,
        default_think=False,
    )
    records: list[JointSignalRecord] = []
    calls_made = 0
    try:
        if verify_model:
            await _verify_ollama_model(scoring_client, model_id, model_digest)
        for row_number, item in enumerate(items, start=1):
            cached = _load_cached_record(
                cache_dir,
                item,
                prompt,
                model_id=model_id,
                model_digest=model_digest,
                endpoint=resolved_endpoint,
            )
            if cached is not None:
                records.append(cached)
                continue
            if max_new_calls is not None and calls_made >= max_new_calls:
                continue
            expected = _expected_record_identity(
                item,
                prompt,
                model_id=model_id,
                model_digest=model_digest,
                endpoint=resolved_endpoint,
            )
            previous = _previous_attempts(cache_dir, str(expected["cache_key"]))
            response: StructuredJSONResponseRecord = await scoring_client.generate_structured_json(
                model_id,
                row_number=row_number,
                prompt_hash=prompt.prompt_hash,
                system_prompt=prompt.system_prompt,
                user_prompt=prompt.render(item),
                schema=JOINT_SIGNAL_SCHEMA,
                temperature=0.0,
                max_completion_tokens=64,
                retries=3,
                ollama_think=False,
            )
            calls_made += 1
            labels: dict[SignalQualityTask, str] | None = None
            validation_error: str | None = None
            if response.status == "success" and response.parse_status == "valid" and response.parsed_json is not None:
                try:
                    labels = parse_joint_signal_labels(response.parsed_json)
                except ValueError as exc:
                    validation_error = str(exc)
            valid = labels is not None
            record = JointSignalRecord(
                **expected,
                direction_severity=labels["direction_severity"] if labels is not None else None,
                materiality=labels["materiality"] if labels is not None else None,
                novelty=labels["novelty"] if labels is not None else None,
                status="success" if valid else "invalid",
                attempt_count=previous + response.attempt_count,
                latency_ms=response.latency_ms,
                created_at_utc=datetime.now(UTC).isoformat(),
                error=None
                if valid
                else (validation_error or response.error or "model output did not match the joint signal schema"),
            )
            _write_joint_attempt(cache_dir, record)
            if valid:
                records.append(record)
            atomic_write_json(
                manifest_path,
                _progress_manifest(
                    identity_hash,
                    identity,
                    score_id=score_id,
                    expected=expected_calls,
                    successful=len(records),
                    calls_made=calls_made,
                    status="in_progress",
                    scores_path=scores_path,
                ),
            )
    finally:
        if owns_client:
            await scoring_client.close()

    # A bounded invocation may skip uncached rows. Re-scan so recovered entries
    # and successful rows from earlier invocations are all counted exactly once.
    complete_records: list[JointSignalRecord] = []
    for item in items:
        cached_record = _load_cached_record(
            cache_dir,
            item,
            prompt,
            model_id=model_id,
            model_digest=model_digest,
            endpoint=resolved_endpoint,
        )
        if cached_record is not None:
            complete_records.append(cached_record)
    complete_records.sort(key=lambda record: record.audit_id)
    complete = len(complete_records) == expected_calls
    if complete:
        serialized = [record.to_payload() for record in complete_records]
        if scores_path.exists():
            existing_rows = read_jsonl(scores_path)
            if existing_rows != serialized:
                raise SignalQualityError(f"refusing to overwrite different completed model scores: {scores_path}")
        else:
            atomic_write_jsonl(scores_path, serialized)
    atomic_write_json(
        manifest_path,
        _progress_manifest(
            identity_hash,
            identity,
            score_id=score_id,
            expected=expected_calls,
            successful=len(complete_records),
            calls_made=calls_made,
            status="completed" if complete else "in_progress",
            scores_path=scores_path,
        ),
    )
    return SignalAuditScoreRun(
        score_id,
        output_dir,
        manifest_path,
        scores_path,
        expected_calls,
        len(complete_records),
        calls_made,
        complete,
        False,
    )


def _quadratic_weighted_kappa(human: Sequence[int], model: Sequence[int], levels: int = 5) -> float | None:
    if len(human) != len(model) or not human:
        raise ValueError("kappa inputs must be non-empty and equal length")
    observed = np.zeros((levels, levels), dtype=float)
    for left, right in zip(human, model, strict=True):
        observed[left, right] += 1
    human_hist = observed.sum(axis=1)
    model_hist = observed.sum(axis=0)
    expected = np.outer(human_hist, model_hist) / observed.sum()
    weights = np.fromfunction(lambda i, j: ((i - j) / (levels - 1)) ** 2, (levels, levels), dtype=float)
    expected_disagreement = float(np.sum(weights * expected))
    observed_disagreement = float(np.sum(weights * observed))
    if expected_disagreement == 0:
        return None
    return 1 - observed_disagreement / expected_disagreement


def _binary_precision_recall(human: Sequence[int], model: Sequence[int], threshold: int) -> tuple[float | None, float | None]:
    predicted = [value >= threshold for value in model]
    actual = [value >= threshold for value in human]
    true_positive = sum(left and right for left, right in zip(actual, predicted, strict=True))
    predicted_positive = sum(predicted)
    actual_positive = sum(actual)
    precision = true_positive / predicted_positive if predicted_positive else None
    recall = true_positive / actual_positive if actual_positive else None
    return precision, recall


def _metric(
    task: SignalQualityTask,
    human_labels: Sequence[str],
    model_labels: Sequence[str],
) -> ValidationMetric:
    labels = TASK_LABELS[task]
    ordinal = {label: index for index, label in enumerate(labels)}
    human = [ordinal[label] for label in human_labels]
    model = [ordinal[label] for label in model_labels]
    differences = [abs(left - right) for left, right in zip(human, model, strict=True)]
    precision: float | None = None
    recall: float | None = None
    if task in {"materiality", "novelty"}:
        precision, recall = _binary_precision_recall(human, model, threshold=3)
    return ValidationMetric(
        task=task,
        rows=len(human),
        exact_agreement=sum(value == 0 for value in differences) / len(differences),
        within_one_level=sum(value <= 1 for value in differences) / len(differences),
        severe_disagreement_rate=sum(value >= 3 for value in differences) / len(differences),
        quadratic_weighted_kappa=_quadratic_weighted_kappa(human, model),
        high_level_precision=precision,
        high_level_recall=recall,
    )


def validate_signal_audit(
    score_dir: str | Path,
    human_labels_path: str | Path,
    *,
    output_root: str | Path | None = None,
) -> SignalAuditValidationRun:
    """Compare completed local scores with a complete, independently human-labelled sample."""

    scores_root = Path(score_dir)
    score_manifest = read_json(scores_root / "scores_manifest.json")
    scores_path = scores_root / "model_scores.jsonl"
    if score_manifest.get("status") != "completed" or not scores_path.is_file():
        raise SignalQualityError("signal-quality model scoring is not complete")
    expected_scores_hash = score_manifest.get("outputs", {}).get(scores_path.name, {}).get("sha256")
    if sha256_file(scores_path) != expected_scores_hash:
        raise SignalQualityError("completed signal-quality model scores changed")
    audit_root = scores_root.parent.parent
    sample_manifest = read_json(audit_root / "sample_manifest.json")
    spec = SignalAuditSpec(**sample_manifest["identity"]["spec"])
    items = _load_audit_items(audit_root / "audit_items.csv")
    human_path = Path(human_labels_path)
    if not human_path.is_file():
        raise SignalQualityError(f"human labels file is missing: {human_path}")
    with human_path.open(encoding="utf-8", newline="") as handle:
        human_rows = list(csv.DictReader(handle))
    if len(human_rows) != len(items) or len({row.get("audit_id") for row in human_rows}) != len(human_rows):
        raise SignalQualityError("human labels must contain exactly one row for every audit item")
    expected_pairs = {(item.audit_id, item.event_id) for item in items}
    observed_pairs = {(str(row.get("audit_id") or ""), str(row.get("event_id") or "")) for row in human_rows}
    if observed_pairs != expected_pairs:
        raise SignalQualityError("human label audit IDs or event IDs do not match the frozen sample")
    for row in human_rows:
        for task in TASKS:
            field = f"human_{task}"
            if str(row.get(field) or "") not in TASK_LABELS[task]:
                raise SignalQualityError(f"invalid or missing {field} for {row.get('audit_id')}")
    human_by_id = {str(row["audit_id"]): row for row in human_rows}
    model_records = tuple(JointSignalRecord.from_payload(row) for row in read_jsonl(scores_path))
    if len(model_records) != len(items) or len({record.audit_id for record in model_records}) != len(model_records):
        raise SignalQualityError("model scores must contain exactly one joint record per audit item")
    model_lookup = {record.audit_id: record for record in model_records}
    if set(model_lookup) != {item.audit_id for item in items}:
        raise SignalQualityError("joint model score identities do not match the frozen audit sample")
    metrics: list[ValidationMetric] = []
    for task in TASKS:
        human_values = [str(human_by_id[item.audit_id][f"human_{task}"]) for item in items]
        model_values: list[str] = []
        for item in items:
            label = model_lookup[item.audit_id].labels[task]
            if label is None:
                raise SignalQualityError(f"completed joint model score is missing {task}: {item.audit_id}")
            model_values.append(label)
        metrics.append(_metric(task, human_values, model_values))
    acceptance: list[SignalAuditAcceptance] = [
        SignalAuditAcceptance(
            "complete_human_sample",
            len(human_rows) == len(items),
            f"{len(human_rows)}/{len(items)}",
            f"{len(items)}/{len(items)}",
        )
    ]
    for metric in metrics:
        acceptance.extend(
            [
                SignalAuditAcceptance(
                    f"{metric.task}_weighted_kappa",
                    metric.quadratic_weighted_kappa is not None
                    and metric.quadratic_weighted_kappa >= spec.minimum_weighted_kappa,
                    (
                        "undefined"
                        if metric.quadratic_weighted_kappa is None
                        else f"{metric.quadratic_weighted_kappa:.3f}"
                    ),
                    f">= {spec.minimum_weighted_kappa:.2f}",
                ),
                SignalAuditAcceptance(
                    f"{metric.task}_severe_disagreement",
                    metric.severe_disagreement_rate <= spec.maximum_severe_disagreement_rate,
                    f"{metric.severe_disagreement_rate:.1%}",
                    f"<= {spec.maximum_severe_disagreement_rate:.0%}",
                ),
            ]
        )
        if metric.task in {"materiality", "novelty"}:
            precision = metric.high_level_precision
            acceptance.append(
                SignalAuditAcceptance(
                    f"{metric.task}_high_level_precision",
                    precision is not None and precision >= spec.minimum_high_level_precision,
                    "undefined" if precision is None else f"{precision:.1%}",
                    f">= {spec.minimum_high_level_precision:.0%}",
                )
            )
    identity = {
        "schema_version": 1,
        "sample_identity_sha256": sample_manifest["identity_sha256"],
        "score_identity_sha256": score_manifest["identity_sha256"],
        "model_scores_sha256": sha256_file(scores_path),
        "human_labels_sha256": sha256_file(human_path),
        "spec": asdict(spec),
    }
    identity_hash = sha256_text(canonical_json(identity))
    validation_id = f"signal-quality-validation-{identity_hash[:12]}"
    source_run_id = str(sample_manifest["identity"]["source_run_id"])
    output_dir = (
        Path(output_root)
        if output_root is not None
        else Path("results/strategy_research") / source_run_id / "signal_quality_validation" / validation_id
    )
    report_path = output_dir / "report.md"
    metrics_path = output_dir / "metrics.csv"
    acceptance_path = output_dir / "acceptance_checks.csv"
    manifest_path = output_dir / "manifest.json"
    expected_outputs = (report_path, metrics_path, acceptance_path)
    if manifest_path.is_file():
        existing = read_json(manifest_path)
        if existing.get("identity_sha256") != identity_hash:
            raise SignalQualityError(f"existing validation identity mismatch: {output_dir}")
        for path in expected_outputs:
            expected_hash = existing.get("outputs", {}).get(path.name, {}).get("sha256")
            if not path.is_file() or sha256_file(path) != expected_hash:
                raise SignalQualityError(f"completed validation output is missing or changed: {path}")
        return SignalAuditValidationRun(
            validation_id,
            output_dir,
            report_path,
            manifest_path,
            tuple(metrics),
            tuple(acceptance),
            True,
        )
    if output_dir.exists():
        raise SignalQualityError(f"refusing to reuse incomplete validation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    metric_fields = list(ValidationMetric.__dataclass_fields__)
    acceptance_fields = list(SignalAuditAcceptance.__dataclass_fields__)
    atomic_write_text(metrics_path, _csv_rows([asdict(row) for row in metrics], metric_fields))
    atomic_write_text(acceptance_path, _csv_rows([asdict(row) for row in acceptance], acceptance_fields))
    atomic_write_text(report_path, _validation_report(validation_id, metrics, acceptance))
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "validation_id": validation_id,
        "identity_sha256": identity_hash,
        "identity": identity,
        "metrics": [asdict(row) for row in metrics],
        "acceptance": [asdict(row) for row in acceptance],
        "outputs": {
            path.name: {"path": path.as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in expected_outputs
        },
    }
    atomic_write_json(manifest_path, manifest)
    return SignalAuditValidationRun(
        validation_id,
        output_dir,
        report_path,
        manifest_path,
        tuple(metrics),
        tuple(acceptance),
        False,
    )


def _validation_report(
    validation_id: str,
    metrics: Sequence[ValidationMetric],
    acceptance: Sequence[SignalAuditAcceptance],
) -> str:
    passed = sum(row.passed for row in acceptance)
    lines = [
        f"# Human signal-quality validation: {validation_id}",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: validate",
        f"- Origin Date: {datetime.now(UTC).date().isoformat()}",
        "- Verification Status: ANALYZED",
        "- Version Label: signal_quality_validation_v1",
        "",
        f"The frozen model passed **{passed}/{len(acceptance)}** pre-declared checks.",
        "",
        "| Task | Rows | Exact | Within one | Severe | Weighted kappa | High precision | High recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in metrics:
        precision = "n/a" if metric.high_level_precision is None else f"{metric.high_level_precision:.1%}"
        recall = "n/a" if metric.high_level_recall is None else f"{metric.high_level_recall:.1%}"
        kappa = "undefined" if metric.quadratic_weighted_kappa is None else f"{metric.quadratic_weighted_kappa:.3f}"
        lines.append(
            f"| {metric.task} | {metric.rows} | {metric.exact_agreement:.1%} | {metric.within_one_level:.1%} | "
            f"{metric.severe_disagreement_rate:.1%} | {kappa} | {precision} | {recall} |"
        )
    lines.extend(
        [
            "",
            "| Check | Result | Observed | Requirement |",
            "| --- | --- | --- | --- |",
        ]
    )
    for check in acceptance:
        lines.append(
            f"| {check.check} | {'PASS' if check.passed else 'FAIL'} | {check.observed} | {check.requirement} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- One human annotator provides a validity audit, not an inter-annotator reliability estimate.",
            "- The sample is balanced by stock and prior three-class model output, not population-weighted.",
            "- Agreement supports label fidelity only; it does not establish predictiveness or profitable alpha.",
            "- No evaluation-period event or return is used by this audit.",
            "",
            "### Statistical fallacy scan — 11/11 checked",
            "",
            (
                "- Simpson's paradox is untested at stock level because three sampled events per stock cannot support "
                "stable stock-specific agreement estimates."
            ),
            "- No individual-person inference is made from company-level news, avoiding an ecological claim.",
            (
                "- Berkson/selection bias remains possible because sampling is conditioned on the prior model's "
                "three-class strata and fixed stock universe."
            ),
            "- No post-label variable is controlled, so this audit does not introduce a statistical collider.",
            "- Population base rates are not claimed; the balanced audit precision is explicitly sample-specific.",
            "- There is no extreme-group pre/post comparison, so regression to the mean is not an operative explanation.",
            (
                "- Only successfully constructed and previously scored events are eligible; this selection boundary "
                "is disclosed as survivorship risk."
            ),
            (
                "- All three tasks and every frozen gate are retained, limiting selective reporting while not "
                "eliminating look-elsewhere concerns."
            ),
            (
                "- The protocol is frozen before new scores and human labels but is not prospectively registered, "
                "so forking-path caution remains."
            ),
            "- Agreement is descriptive and no causal language is used.",
            "- Temporal direction between labels and outcomes is not asserted, so reverse causality is outside the audit claim.",
            "",
        ]
    )
    return "\n".join(lines)
