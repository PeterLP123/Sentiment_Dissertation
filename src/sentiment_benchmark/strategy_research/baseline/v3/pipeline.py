"""Immutable execution for the exploratory FinBERT rank-reversal strategy."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....artifact_io import canonical_json, read_json, sha256_file, sha256_text
from ...artifacts import write_immutable_json, write_immutable_jsonl, write_immutable_text
from ...inference import paired_block_bootstrap
from ...ledger import DailyLedgerRow, assert_no_boundary_crossing, run_open_to_open_ledger
from ...market import AdjustedOpen, OpenToOpenReturn
from ...schemas import StrategyEvent
from ...sources import LsegEventSettings, build_strategy_events
from ..pipeline import (
    BaselineRunError,
    _bootstrap_payload,
    _event_screening_metadata,
    _ledger_payload,
    _load_prices,
    _metrics_payload,
    _number,
    _percent,
    _require_file,
    _resolve,
    _runtime_dependencies,
    _statistical_rows,
)
from ..signals import BaselineScoreError, load_headline_scores
from ..v2.signals import ArmSignal, build_v2_targets
from .config import FinbertV3Config
from .signals import RankPortfolioBuildResult, build_finbert_labels, build_rank_reversal_targets


class FinbertV3RunError(BaselineRunError):
    """Raised when v3 inputs or outputs violate the frozen contract."""


@dataclass(frozen=True)
class FinbertV3Inspection:
    resolved_run_id: str
    event_count: int
    selected_event_count: int
    firm_session_signal_count: int
    symbols: tuple[str, ...]
    decision_sessions: int
    evaluation_sessions: int
    active_sessions_by_arm: dict[str, int]
    median_ranked_firms: int
    derived_dir: Path
    results_dir: Path
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FinbertV3RunResult:
    resolved_run_id: str
    derived_dir: Path
    results_dir: Path
    report_path: Path
    metrics_path: Path
    reused: bool


@dataclass(frozen=True)
class _LoadedInputs:
    identities: dict[str, str]
    events: tuple[StrategyEvent, ...]
    event_attrition: dict[str, int]
    labels: Any
    adjusted_opens: tuple[AdjustedOpen, ...]
    returns: tuple[OpenToOpenReturn, ...]
    symbols: tuple[str, ...]
    decision_sessions: tuple[str, ...]
    run_id: str
    derived_dir: Path
    results_dir: Path
    score_revision_enforced: bool


@dataclass(frozen=True)
class _ArmRun:
    arm: str
    session_rows: tuple[dict[str, object], ...]
    ledger: tuple[DailyLedgerRow, ...]
    grid_ledgers: dict[float, tuple[DailyLedgerRow, ...]]


def _implementation_sha256() -> str:
    v3_root = Path(__file__).parent
    baseline_root = v3_root.parent
    strategy_root = baseline_root.parent
    package_root = strategy_root.parent
    files = [v3_root / name for name in ("config.py", "signals.py", "pipeline.py")]
    files.extend(baseline_root / name for name in ("signals.py", "pipeline.py"))
    files.extend((baseline_root / "v2") / name for name in ("signals.py",))
    files.extend(
        strategy_root / name
        for name in ("artifacts.py", "inference.py", "ledger.py", "market.py", "portfolio.py", "schemas.py", "sources.py")
    )
    files.extend(package_root / name for name in ("artifact_io.py", "headline_value.py", "lseg_corpus.py", "lseg_source.py"))
    entries = [f"{path.relative_to(package_root).as_posix()}:{sha256_file(path)}" for path in sorted(files)]
    return sha256_text("\n".join(entries))


def _verify_score_manifest(config: FinbertV3Config, score_path: Path, manifest_path: Path, root: Path) -> bool:
    _require_file(score_path, "v3 FinBERT score artifact")
    _require_file(manifest_path, "v3 FinBERT score manifest")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise FinbertV3RunError("FinBERT score manifest is not completed")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("sha256") != sha256_file(score_path):
        raise FinbertV3RunError("FinBERT score manifest hash does not match the artifact")
    declared_path = str(output.get("path") or "")
    if declared_path:
        declared = Path(declared_path)
        declared = declared if declared.is_absolute() else root / declared
        if declared.resolve() != score_path.resolve():
            raise FinbertV3RunError("FinBERT score manifest points to a different artifact")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or int(counts.get("remaining", -1)) != 0:
        raise FinbertV3RunError("FinBERT score manifest does not establish complete scoring")
    inference = manifest.get("inference")
    if not isinstance(inference, dict) or inference.get("local_only") is not True:
        raise FinbertV3RunError("FinBERT score manifest does not establish local-only inference")
    model = (manifest.get("models") or {}).get("finbert")
    if not isinstance(model, dict):
        raise FinbertV3RunError("FinBERT score manifest has no model identity")
    if model.get("model_id") != config.scorer.model_id or model.get("revision") != config.scorer.revision:
        raise FinbertV3RunError("FinBERT model ID/revision differs from the frozen v3 config")
    enforced = model.get("revision_enforced") is True
    if model.get("revision_enforced") is False:
        raise FinbertV3RunError("FinBERT manifest explicitly says the revision was not enforced")
    if not enforced and not config.data.allow_legacy_revision_enforcement_metadata:
        raise FinbertV3RunError("FinBERT manifest lacks revision-enforcement evidence")
    sharing = manifest.get("sharing")
    if not isinstance(sharing, dict) or sharing.get("contains_licensed_headline_text") is not True:
        raise FinbertV3RunError("FinBERT manifest must identify the licensed-text artifact")
    return enforced


def _input_identities(config: FinbertV3Config, root: Path) -> dict[str, str]:
    paths = {
        "corpus_manifest_sha256": _resolve(root, config.data.corpus_manifest),
        "score_artifact_sha256": _resolve(root, config.data.score_path),
        "score_manifest_sha256": _resolve(root, config.data.score_manifest),
        "price_panel_sha256": _resolve(root, config.prices.panel_path),
        "price_manifest_sha256": _resolve(root, config.prices.manifest_path),
    }
    for name, path in paths.items():
        _require_file(path, name.removesuffix("_sha256"))
    return {
        "config_sha256": config.config_sha256,
        "implementation_sha256": _implementation_sha256(),
        "runtime_dependencies_sha256": sha256_text(canonical_json(_runtime_dependencies())),
        **{name: sha256_file(path) for name, path in paths.items()},
    }


def _load_inputs(config: FinbertV3Config, repo_root: Path) -> _LoadedInputs:
    root = repo_root.resolve()
    collection_root = _resolve(root, config.data.collection_root).resolve()
    corpus_manifest = _resolve(root, config.data.corpus_manifest).resolve()
    score_path = _resolve(root, config.data.score_path).resolve()
    score_manifest = _resolve(root, config.data.score_manifest).resolve()
    price_panel = _resolve(root, config.prices.panel_path).resolve()
    price_manifest = _resolve(root, config.prices.manifest_path).resolve()
    if not collection_root.is_dir():
        raise FinbertV3RunError(f"collection root does not exist: {collection_root}")
    for label, path in (("corpus manifest", corpus_manifest), ("score artifact", score_path), ("score manifest", score_manifest)):
        if not path.is_relative_to(collection_root):
            raise FinbertV3RunError(f"licensed {label} must remain inside the collection root")
    identities = _input_identities(config, root)
    revision_enforced = _verify_score_manifest(config, score_path, score_manifest, root)
    event_result = build_strategy_events(
        corpus_manifest,
        settings=LsegEventSettings(
            processing_buffer_minutes=config.data.processing_buffer_minutes,
            calendar_name=config.data.calendar,
            exchange_timezone=config.data.timezone,
        ),
    )
    scores = load_headline_scores(score_path, ["finbert"])
    metadata = _event_screening_metadata(corpus_manifest, event_result.events)
    try:
        labels = build_finbert_labels(event_result.events, scores, metadata, config.screening)
    except BaselineScoreError as exc:
        raise FinbertV3RunError(str(exc)) from exc
    opens, returns, symbols = _load_prices(price_panel, price_manifest)
    event_symbols = {event.symbol for event in event_result.events}
    if not event_symbols.issubset(symbols):
        raise FinbertV3RunError(f"events contain symbols absent from prices: {sorted(event_symbols - set(symbols))}")
    next_by_session: dict[str, str] = {}
    for row in returns:
        previous = next_by_session.setdefault(row.session, row.next_session)
        if previous != row.next_session:
            raise FinbertV3RunError(f"inconsistent next price session after {row.session}")
    all_sessions = tuple(sorted(next_by_session))
    selected_sessions = sorted({row.eligible_execution_session for row in labels.labels})
    if not selected_sessions:
        raise FinbertV3RunError("v3 screening left no eligible event sessions")
    missing = sorted(set(selected_sessions) - set(all_sessions))
    if missing:
        raise FinbertV3RunError(f"events map outside executable price history: {missing[:5]}")
    decision_sessions = tuple(session for session in all_sessions if session >= selected_sessions[0])
    boundary = config.run.evaluation_start.isoformat()
    if boundary not in decision_sessions:
        raise FinbertV3RunError("evaluation_start is absent from the executable price-session spine")
    run_id = f"{config.run.id}-{sha256_text(canonical_json(identities))[:12]}"
    return _LoadedInputs(
        identities=identities,
        events=event_result.events,
        event_attrition=event_result.attrition,
        labels=labels,
        adjusted_opens=opens,
        returns=returns,
        symbols=symbols,
        decision_sessions=decision_sessions,
        run_id=run_id,
        derived_dir=_resolve(root, config.outputs.derived_root) / run_id,
        results_dir=_resolve(root, config.outputs.results_root) / run_id,
        score_revision_enforced=revision_enforced,
    )


def _comparator_targets(inputs: _LoadedInputs, minimum_names: int) -> RankPortfolioBuildResult:
    arm_signals = tuple(ArmSignal(row.session, row.symbol, row.action, abs(row.mean_continuous_score)) for row in inputs.labels.signals)
    result = build_v2_targets(
        arm_signals,
        arm="finbert_sign_1d",
        sessions=inputs.decision_sessions,
        symbols=inputs.symbols,
        gross_exposure=1.0,
        minimum_names_per_side=minimum_names,
        weighting="equal",
    )
    rows = tuple({**row, "arm": "finbert_sign_1d"} for row in result.session_rows)
    return RankPortfolioBuildResult(targets=result.targets, session_rows=rows)


def _run_arms(config: FinbertV3Config, inputs: _LoadedInputs) -> dict[str, _ArmRun]:
    portfolios = {
        "finbert_rank_reversal": build_rank_reversal_targets(
            inputs.labels.signals,
            sessions=inputs.decision_sessions,
            symbols=inputs.symbols,
            gross_exposure=config.portfolio.gross_exposure,
            config=config.signal,
        ),
        "finbert_sign_1d": _comparator_targets(inputs, config.signal.minimum_names_per_side),
    }
    boundary = config.run.evaluation_start.isoformat()
    runs: dict[str, _ArmRun] = {}
    for arm, portfolio in portfolios.items():
        grid: dict[float, tuple[DailyLedgerRow, ...]] = {}
        for cost in config.portfolio.cost_grid_bps:
            ledger = tuple(
                run_open_to_open_ledger(
                    portfolio.targets,
                    inputs.returns,
                    cost_rate_per_side=cost / 10_000,
                    initial_nav_usd=config.portfolio.initial_nav_usd,
                    accounting_reset_session=boundary,
                    force_final_liquidation=config.portfolio.force_final_liquidation,
                )
            )
            assert_no_boundary_crossing(ledger, boundary)
            grid[cost] = ledger
        runs[arm] = _ArmRun(
            arm=arm,
            session_rows=portfolio.session_rows,
            ledger=grid[config.portfolio.cost_bps_per_side],
            grid_ledgers=grid,
        )
    return runs


def _period_metrics(rows: Sequence[DailyLedgerRow], boundary: str) -> dict[str, Any]:
    statistical = _statistical_rows(rows)
    return {
        "all": _metrics_payload(statistical),
        "development": _metrics_payload([row for row in statistical if row.session < boundary]),
        "evaluation": _metrics_payload([row for row in statistical if row.session >= boundary]),
    }


def _evaluate(config: FinbertV3Config, runs: Mapping[str, _ArmRun]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    boundary = config.run.evaluation_start.isoformat()
    metrics = {arm: _period_metrics(run.ledger, boundary) for arm, run in runs.items()}
    cost_grid = {
        arm: {f"{cost:g}": _period_metrics(ledger, boundary)["evaluation"] for cost, ledger in sorted(run.grid_ledgers.items())}
        for arm, run in runs.items()
    }

    def evaluation_returns(arm: str) -> dict[str, float]:
        return {row.session: row.net_return for row in _statistical_rows(runs[arm].ledger) if row.session >= boundary}

    primary = evaluation_returns("finbert_rank_reversal")
    if len(primary) < config.inference.block_length_sessions:
        raise FinbertV3RunError("evaluation block is shorter than the bootstrap block")
    bootstrap = {
        "rank_reversal_vs_cash": _bootstrap_payload(
            paired_block_bootstrap(
                primary,
                {session: 0.0 for session in primary},
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=config.inference.seed,
            )
        ),
        "rank_reversal_vs_sign_1d": _bootstrap_payload(
            paired_block_bootstrap(
                primary,
                evaluation_returns("finbert_sign_1d"),
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=config.inference.seed + 1,
            )
        ),
    }
    return metrics, bootstrap, cost_grid


def _report(
    config: FinbertV3Config,
    inputs: _LoadedInputs,
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    cost_grid: Mapping[str, Any],
) -> str:
    boundary = config.run.evaluation_start.isoformat()
    lines = [
        "# FinBERT rank-reversal strategy v3 result",
        "",
        f"- Resolved run: `{inputs.run_id}`",
        f"- Status: `{config.run.sample_status}`; post-selection diagnostic, not an untouched test",
        f"- Evaluation boundary retained from the earlier pilot: `{boundary}`",
        f"- Input events: {len(inputs.events):,}; selected target-specific events: {inputs.labels.attrition['selected_events']:,}",
        f"- Universe: {len(inputs.symbols)} stocks; decision sessions: {len(inputs.decision_sessions)}",
        f"- Score revision-enforcement metadata present: `{inputs.score_revision_enforced}`",
        "",
        "## Frozen construction",
        "",
        "For each formation session, target-specific firm-level mean continuous FinBERT scores are ranked",
        "cross-sectionally. The decision signal averages each firm's ranks over the previous five sessions,",
        "with a one-session lag, then goes long the lowest-sentiment 20% and short the highest-sentiment",
        "20%. Each leg is equal-weighted, gross exposure is 1.0, and at least ten ranked firms and two",
        "names per side are required. This converts generic polarity into a relative, diversified reversal",
        "signal while preventing current-session news from entering the current target.",
        "",
        "The comparator is the v2-style same-direction one-session FinBERT sign strategy on the identical",
        "events, prices, evaluation boundary and costs. The headline assumption remains 10 bps per dollar",
        "traded, with predeclared 5/10/20 bps sensitivity.",
        "",
        "## Evaluation results (10 bps per side)",
        "",
        "| Arm | Active days | Gross cumulative | Net cumulative | Net Sharpe | Max drawdown | Turnover |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("finbert_rank_reversal", "finbert_sign_1d"):
        row = metrics[arm]["evaluation"]
        lines.append(
            f"| `{arm}` | {row['active_day_count_excluding_liquidation']} | "
            f"{_percent(row['cumulative_gross_return'])} | {_percent(row['cumulative_net_return'])} | "
            f"{_number(row['after_cost_sharpe'])} | {_percent(row['maximum_drawdown'])} | "
            f"{_number(row['total_turnover'])} |"
        )
    lines.extend(["", "## Moving-block bootstrap", ""])
    for key, label in (
        ("rank_reversal_vs_cash", "Rank reversal minus cash"),
        ("rank_reversal_vs_sign_1d", "Rank reversal minus same-direction FinBERT"),
    ):
        row = bootstrap[key]
        lines.append(
            f"- {label}: mean daily difference {_percent(row['mean_daily_difference'])}; 95% interval "
            f"[{_percent(row['confidence_interval_low'])}, {_percent(row['confidence_interval_high'])}] "
            f"(`{row['confidence_direction']}`)."
        )
    lines.extend(
        [
            "",
            "## Cost sensitivity (evaluation net cumulative / Sharpe)",
            "",
            "| Arm | 5 bps | 10 bps | 20 bps |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in ("finbert_rank_reversal", "finbert_sign_1d"):
        cells = []
        for cost in config.portfolio.cost_grid_bps:
            row = cost_grid[arm][f"{cost:g}"]
            cells.append(f"{_percent(row['cumulative_net_return'])} / {_number(row['after_cost_sharpe'])}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Claim boundary and next gate",
            "",
            "- This design was chosen after the 33-stock pilot showed an inverse five-session FinBERT level relation.",
            "  Its result on this same cohort cannot establish out-of-sample improvement, regardless of Sharpe.",
            "- No threshold, lookback, lag, tail fraction, cost or direction may be retuned on this cohort.",
            "- A better-strategy claim requires the frozen v3 protocol to beat both cash and the identical-cohort",
            "  same-direction comparator on a genuinely independent corpus, with a positive block-bootstrap lower bound.",
            "- The July 15 score manifest predates explicit `revision_enforced` metadata. Exact replay is secured by",
            "  the artifact hash, but confirmatory scoring must use the current enforced-revision manifest contract.",
            "- Licensed Reuters text and headline-bearing score artifacts remain local and ignored.",
            "",
        ]
    )
    return "\n".join(lines)


def inspect_finbert_v3(config: FinbertV3Config, *, repo_root: str | Path = ".") -> FinbertV3Inspection:
    inputs = _load_inputs(config, Path(repo_root))
    runs = _run_arms(config, inputs)
    boundary = config.run.evaluation_start.isoformat()
    ranked_values = [row["ranked_firms"] for row in runs["finbert_rank_reversal"].session_rows]
    if not all(isinstance(value, int) for value in ranked_values):
        raise FinbertV3RunError("rank construction emitted a non-integer breadth count")
    ranked = [value for value in ranked_values if isinstance(value, int)]
    active = {arm: sum(row.active_names > 0 and not row.final_liquidation for row in run.ledger) for arm, run in runs.items()}
    return FinbertV3Inspection(
        resolved_run_id=inputs.run_id,
        event_count=len(inputs.events),
        selected_event_count=inputs.labels.attrition["selected_events"],
        firm_session_signal_count=len(inputs.labels.signals),
        symbols=inputs.symbols,
        decision_sessions=len(inputs.decision_sessions),
        evaluation_sessions=sum(session >= boundary for session in inputs.decision_sessions),
        active_sessions_by_arm=active,
        median_ranked_firms=sorted(ranked)[len(ranked) // 2],
        derived_dir=inputs.derived_dir,
        results_dir=inputs.results_dir,
        warnings=(
            "The v3 direction and horizon were chosen after results on this cohort; this run is diagnostic only.",
            "Confirmation requires a new corpus and current enforced-revision score provenance.",
            "Prices omit dividend adjustment, borrow, variable spreads and market impact.",
        ),
    )


def run_finbert_v3(config: FinbertV3Config, *, repo_root: str | Path = ".") -> FinbertV3RunResult:
    root = Path(repo_root).resolve()
    inputs = _load_inputs(config, root)
    runs = _run_arms(config, inputs)
    metrics, bootstrap, cost_grid = _evaluate(config, runs)
    derived = inputs.derived_dir
    results = inputs.results_dir
    label_rows = [row.to_payload() for row in inputs.labels.labels]
    signal_rows = [row.to_payload() for row in inputs.labels.signals]
    construction_rows = [dict(row) for arm in sorted(runs) for row in runs[arm].session_rows]
    construction_rows.sort(key=lambda row: (str(row.get("session")), str(row.get("arm"))))
    pnl_rows = [_ledger_payload(arm, row) for arm in sorted(runs) for row in runs[arm].ledger]
    pnl_rows.sort(key=lambda row: (str(row["session"]), str(row["scorer"])))
    attrition = {
        "event_builder": inputs.event_attrition,
        "screening": inputs.labels.attrition,
        "label_counts": dict(sorted(Counter(row.label for row in inputs.labels.labels).items())),
        "firm_session_actions": dict(sorted(Counter(row.action for row in inputs.labels.signals).items())),
    }
    report = _report(config, inputs, metrics, bootstrap, cost_grid)
    reused_flags: list[bool] = []
    paths: dict[str, Path] = {}
    for name, path, payload in (
        ("event_labels", derived / "event_labels.jsonl", label_rows),
        ("firm_session_signals", derived / "firm_session_signals.jsonl", signal_rows),
        ("session_construction", results / "session_construction.jsonl", construction_rows),
        ("daily_pnl", results / "daily_pnl.jsonl", pnl_rows),
    ):
        written, reused = write_immutable_jsonl(path, payload)
        paths[name] = written
        reused_flags.append(reused)
    for name, path, json_payload in (
        ("attrition", derived / "attrition.json", attrition),
        ("metrics", results / "metrics.json", metrics),
        ("bootstrap", results / "bootstrap.json", bootstrap),
        ("cost_grid", results / "cost_grid.json", cost_grid),
    ):
        written, reused = write_immutable_json(path, json_payload)
        paths[name] = written
        reused_flags.append(reused)
    report_path, reused = write_immutable_text(results / "report.md", report)
    paths["report"] = report_path
    reused_flags.append(reused)
    manifest = {
        "schema_version": 3,
        "status": "completed",
        "run_id": inputs.run_id,
        "sample_status": config.run.sample_status,
        "config": config.to_payload(),
        "input_identities": inputs.identities,
        "runtime_dependencies": _runtime_dependencies(),
        "method": {
            "event_identity": "earliest_story_family_symbol",
            "score_join": "sha256(normalize_headline(event.headline))",
            "signal": "mean continuous FinBERT firm-session score -> daily cross-sectional percentile rank",
            "primary_arm": "one-session lagged mean of five formation-session ranks; long bottom 20%, short top 20%",
            "comparator": "same-direction one-session sign(mean hard FinBERT label)",
            "execution": "split-adjusted open-to-open; 15-minute availability buffer",
            "cost_bps_per_side": config.portfolio.cost_bps_per_side,
            "turnover": "sum(abs(target_weight - current_weight))",
        },
        "row_counts": {
            "events": len(inputs.events),
            "event_labels": len(label_rows),
            "firm_session_signals": len(signal_rows),
            "daily_pnl": len(pnl_rows),
        },
        "outputs": {
            name: {
                "path": path.resolve().relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in sorted(paths.items())
        },
        "sharing": {
            "contains_licensed_headline_text": False,
            "licensed_score_input_remains_local": True,
            "source_control": False,
        },
        "interpretation": {
            "confirmatory": False,
            "specified_after_level_reversal_results": True,
            "independent_confirmation_required": True,
            "retuning_after_result_permitted": False,
            "legacy_score_revision_enforcement_metadata": not inputs.score_revision_enforced,
        },
    }
    _manifest_path, reused = write_immutable_json(results / "manifest.json", manifest)
    reused_flags.append(reused)
    return FinbertV3RunResult(
        resolved_run_id=inputs.run_id,
        derived_dir=derived,
        results_dir=results,
        report_path=report_path,
        metrics_path=paths["metrics"],
        reused=all(reused_flags),
    )
