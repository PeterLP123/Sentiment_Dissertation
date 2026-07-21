"""Read-only inspection and immutable execution of the v2 sentiment baseline.

The v2 pipeline reuses the v1 point-in-time event builder, price loader,
ledger, and accounting verbatim. It differs only in the predeclared protocol:
a canonical compound-threshold VADER comparator from its own verified
artifact, a two-name minimum per side, magnitude-weighted and
agreement-conditioned arms, a firm-session event-level bootstrap as the
primary inference, and a fixed transaction-cost reporting grid.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ....artifact_io import canonical_json, read_json, sha256_file, sha256_text
from ...artifacts import write_immutable_json, write_immutable_jsonl, write_immutable_text
from ...diagnostics import calculate_stock_diagnostics
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
    _verify_score_manifest,
)
from ..signals import BaselineScoreError, load_headline_scores
from .config import LiteratureBaselineV2Config
from .signals import (
    ArmSignal,
    V2LabelBuildResult,
    build_arm_signals,
    build_v2_event_labels,
    build_v2_targets,
    load_vader_compound_scores,
)


class BaselineV2RunError(BaselineRunError):
    """Raised when a frozen v2 input or output violates the baseline contract."""


@dataclass(frozen=True)
class BaselineV2Inspection:
    resolved_run_id: str
    input_identities: dict[str, str]
    event_count: int
    selected_event_count: int
    firm_session_signal_count: int
    symbols: tuple[str, ...]
    price_sessions: int
    decision_sessions: int
    development_sessions: int
    evaluation_sessions: int
    active_sessions_by_arm: dict[str, int]
    derived_dir: Path
    results_dir: Path
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BaselineV2RunResult:
    resolved_run_id: str
    derived_dir: Path
    results_dir: Path
    report_path: Path
    metrics_path: Path
    reused: bool


@dataclass(frozen=True)
class _LoadedV2Inputs:
    identities: dict[str, str]
    events: tuple[StrategyEvent, ...]
    event_attrition: dict[str, int]
    labels: V2LabelBuildResult
    adjusted_opens: tuple[AdjustedOpen, ...]
    returns: tuple[OpenToOpenReturn, ...]
    symbols: tuple[str, ...]
    decision_sessions: tuple[str, ...]
    run_id: str
    derived_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class _ArmRun:
    arm: str
    signals: tuple[ArmSignal, ...]
    session_rows: tuple[dict[str, object], ...]
    ledger: tuple[DailyLedgerRow, ...]
    grid_ledgers: dict[float, tuple[DailyLedgerRow, ...]]


def _verify_compound_manifest(
    *,
    compound_path: Path,
    manifest_path: Path,
    base_score_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require_file(compound_path, "VADER compound score artifact")
    _require_file(manifest_path, "VADER compound score manifest")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise BaselineV2RunError("VADER compound score manifest is not completed")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("sha256") != sha256_file(compound_path):
        raise BaselineV2RunError("VADER compound manifest hash does not match the score artifact")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or int(counts.get("remaining", -1)) != 0 or int(counts.get("failed", -1)) != 0:
        raise BaselineV2RunError("VADER compound manifest does not establish complete successful scoring")
    inference = manifest.get("inference")
    if (
        not isinstance(inference, dict)
        or inference.get("local_only") is not True
        or inference.get("local_files_only_enforced") is not True
    ):
        raise BaselineV2RunError("VADER compound manifest does not prove cache-only inference was enforced")
    models = manifest.get("models")
    model = models.get("vader_compound") if isinstance(models, dict) else None
    if (
        not isinstance(model, dict)
        or model.get("implementation") != "nltk.sentiment.vader"
        or model.get("classification") != "compound_threshold"
        or float(model.get("threshold", 0.0)) != 0.05
    ):
        raise BaselineV2RunError("VADER compound manifest does not pin the canonical compound convention")
    lexicon = str(model.get("lexicon_sha256") or "")
    if len(lexicon) != 64 or any(value not in "0123456789abcdef" for value in lexicon):
        raise BaselineV2RunError("VADER compound manifest does not pin the VADER lexicon")
    base_models = base_score_manifest.get("models") or {}
    base_lexicon = str((base_models.get("vader") or {}).get("lexicon_sha256") or "")
    if base_lexicon and lexicon != base_lexicon:
        raise BaselineV2RunError("VADER compound lexicon differs from the base score artifact's lexicon")
    sharing = manifest.get("sharing")
    if not isinstance(sharing, dict) or sharing.get("contains_licensed_headline_text") is not True:
        raise BaselineV2RunError("VADER compound manifest must identify the licensed-text artifact")
    return manifest


def _implementation_sha256_v2() -> str:
    """Hash the v2 files plus the shared v1 implementation they reuse."""

    v2_root = Path(__file__).parent
    baseline_root = v2_root.parent
    strategy_root = baseline_root.parent
    package_root = strategy_root.parent
    files = [v2_root / name for name in ("config.py", "signals.py", "pipeline.py")]
    files.extend(baseline_root / name for name in ("config.py", "signals.py", "pipeline.py"))
    files.extend(
        strategy_root / name
        for name in (
            "artifacts.py",
            "diagnostics.py",
            "inference.py",
            "ledger.py",
            "market.py",
            "portfolio.py",
            "schemas.py",
            "sources.py",
            "tuning.py",
        )
    )
    files.extend(
        package_root / name
        for name in ("artifact_io.py", "headline_value.py", "lseg_corpus.py", "lseg_source.py")
    )
    entries = [f"{path.relative_to(package_root).as_posix()}:{sha256_file(path)}" for path in sorted(files)]
    return sha256_text("\n".join(entries))


def _input_identities_v2(config: LiteratureBaselineV2Config, repo_root: Path) -> dict[str, str]:
    paths = {
        "corpus_manifest_sha256": _resolve(repo_root, config.data.base.corpus_manifest),
        "score_artifact_sha256": _resolve(repo_root, config.data.base.score_path),
        "score_manifest_sha256": _resolve(repo_root, config.data.base.score_manifest),
        "vader_compound_artifact_sha256": _resolve(repo_root, config.data.vader_compound_path),
        "vader_compound_manifest_sha256": _resolve(repo_root, config.data.vader_compound_manifest),
        "price_panel_sha256": _resolve(repo_root, config.prices.panel_path),
        "price_manifest_sha256": _resolve(repo_root, config.prices.manifest_path),
    }
    for name, path in paths.items():
        _require_file(path, name.removesuffix("_sha256"))
    return {
        "config_sha256": config.config_sha256,
        "implementation_sha256": _implementation_sha256_v2(),
        "runtime_dependencies_sha256": sha256_text(canonical_json(_runtime_dependencies())),
        **{name: sha256_file(path) for name, path in paths.items()},
    }


def _load_inputs_v2(config: LiteratureBaselineV2Config, repo_root: Path) -> _LoadedV2Inputs:
    root = repo_root.resolve()
    collection_root = _resolve(root, config.data.base.collection_root).resolve()
    corpus_manifest = _resolve(root, config.data.base.corpus_manifest).resolve()
    score_path = _resolve(root, config.data.base.score_path).resolve()
    score_manifest = _resolve(root, config.data.base.score_manifest).resolve()
    compound_path = _resolve(root, config.data.vader_compound_path).resolve()
    compound_manifest = _resolve(root, config.data.vader_compound_manifest).resolve()
    price_panel = _resolve(root, config.prices.panel_path).resolve()
    price_manifest = _resolve(root, config.prices.manifest_path).resolve()
    if not collection_root.is_dir():
        raise BaselineV2RunError(f"collection root does not exist: {collection_root}")
    for name, path in (
        ("corpus manifest", corpus_manifest),
        ("score artifact", score_path),
        ("score manifest", score_manifest),
        ("VADER compound artifact", compound_path),
        ("VADER compound manifest", compound_manifest),
    ):
        if not path.is_relative_to(collection_root):
            raise BaselineV2RunError(f"licensed {name} must remain inside the declared collection root")

    identities = _input_identities_v2(config, root)
    base_manifest = _verify_score_manifest(
        config,
        repo_root=root,
        score_path=score_path,
        manifest_path=score_manifest,
    )
    _verify_compound_manifest(
        compound_path=compound_path,
        manifest_path=compound_manifest,
        base_score_manifest=base_manifest,
    )
    event_result = build_strategy_events(
        corpus_manifest,
        settings=LsegEventSettings(
            processing_buffer_minutes=config.data.base.processing_buffer_minutes,
            calendar_name=config.data.base.calendar,
            exchange_timezone=config.data.base.timezone,
        ),
    )
    finbert_scores = load_headline_scores(score_path, ["finbert"])
    compound_scores = load_vader_compound_scores(compound_path)
    event_metadata = _event_screening_metadata(corpus_manifest, event_result.events)
    try:
        labels = build_v2_event_labels(
            event_result.events, finbert_scores, compound_scores, event_metadata, config
        )
    except BaselineScoreError as exc:
        raise BaselineV2RunError(str(exc)) from exc
    opens, returns, symbols = _load_prices(price_panel, price_manifest)
    event_symbols = {event.symbol for event in event_result.events}
    if not event_symbols.issubset(symbols):
        raise BaselineV2RunError(f"events contain symbols absent from prices: {sorted(event_symbols - set(symbols))}")
    next_by_session: dict[str, str] = {}
    for row in returns:
        previous = next_by_session.setdefault(row.session, row.next_session)
        if previous != row.next_session:
            raise BaselineV2RunError(f"inconsistent next price session after {row.session}")
    all_decision_sessions = tuple(sorted(next_by_session))
    selected_event_sessions = sorted({row.eligible_execution_session for row in labels.labels})
    if not selected_event_sessions:
        raise BaselineV2RunError("baseline screening left no eligible event sessions")
    first_event_session, last_event_session = selected_event_sessions[0], selected_event_sessions[-1]
    decision_sessions = tuple(
        session for session in all_decision_sessions if first_event_session <= session <= last_event_session
    )
    if config.run.evaluation_start.isoformat() not in decision_sessions:
        raise BaselineV2RunError("evaluation_start is absent from the executable price-session spine")
    missing_event_sessions = sorted(set(selected_event_sessions) - set(all_decision_sessions))
    if missing_event_sessions:
        raise BaselineV2RunError(f"events map outside executable price history: {missing_event_sessions[:5]}")
    identity_hash = sha256_text(canonical_json(identities))
    run_id = f"{config.run.id}-{identity_hash[:12]}"
    return _LoadedV2Inputs(
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
    )


def _run_arms(config: LiteratureBaselineV2Config, inputs: _LoadedV2Inputs) -> dict[str, _ArmRun]:
    boundary = config.run.evaluation_start.isoformat()
    runs: dict[str, _ArmRun] = {}
    for arm in config.signal.arms:
        arm_signals = build_arm_signals(inputs.labels.signals, arm)
        weighting = "magnitude" if arm == "finbert_magnitude" else "equal"
        portfolio = build_v2_targets(
            arm_signals,
            arm=arm,
            sessions=inputs.decision_sessions,
            symbols=inputs.symbols,
            gross_exposure=config.portfolio.gross_exposure,
            minimum_names_per_side=config.portfolio.minimum_names_per_side,
            weighting=weighting,
        )
        grid_ledgers: dict[float, tuple[DailyLedgerRow, ...]] = {}
        for cost_bps in config.portfolio.cost_grid_bps:
            ledger = tuple(
                run_open_to_open_ledger(
                    portfolio.targets,
                    inputs.returns,
                    cost_rate_per_side=cost_bps / 10_000,
                    initial_nav_usd=config.portfolio.initial_nav_usd,
                    accounting_reset_session=boundary,
                    force_final_liquidation=config.portfolio.force_final_liquidation,
                )
            )
            assert_no_boundary_crossing(ledger, boundary)
            grid_ledgers[cost_bps] = ledger
        runs[arm] = _ArmRun(
            arm=arm,
            signals=arm_signals,
            session_rows=portfolio.session_rows,
            ledger=grid_ledgers[config.portfolio.cost_bps_per_side],
            grid_ledgers=grid_ledgers,
        )
    return runs


def _event_level_payload(
    arm: str,
    arm_signals: Sequence[ArmSignal],
    return_lookup: Mapping[tuple[str, str], float],
    *,
    boundary: str,
    block_length: int,
    replications: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the session means of signed gross firm-session returns.

    This is the predeclared primary v2 test: it conditions on the firm-sessions
    the signal actually selects (unlike the portfolio spine, which mixes in
    cash days) and uses many more observations than the two-leg portfolio.
    Gross open-to-open returns are used so the test measures predictiveness,
    not the cost assumption; the portfolio ledger carries the costed economics.
    """

    selected = [row for row in arm_signals if row.session >= boundary and row.action != 0]
    signed: dict[str, list[float]] = {}
    for row in selected:
        value = return_lookup.get((row.session, row.symbol))
        if value is None:
            raise BaselineV2RunError(f"missing return for firm-session {row.symbol} on {row.session}")
        signed.setdefault(row.session, []).append(row.action * value)
    session_means = {session: sum(values) / len(values) for session, values in sorted(signed.items())}
    firm_count = len(selected)
    payload: dict[str, Any] = {
        "arm": arm,
        "period": "evaluation",
        "return_basis": "gross_signed_open_to_open",
        "firm_session_count": firm_count,
        "long_firm_sessions": sum(row.action > 0 for row in selected),
        "short_firm_sessions": sum(row.action < 0 for row in selected),
        "active_session_count": len(session_means),
        "mean_signed_gross_return_firm_level": (
            sum(row.action * return_lookup[(row.session, row.symbol)] for row in selected) / firm_count
            if firm_count
            else None
        ),
    }
    if len(session_means) < block_length:
        payload["status"] = "insufficient_active_sessions"
        return payload
    result = paired_block_bootstrap(
        session_means,
        {session: 0.0 for session in session_means},
        block_length=block_length,
        replications=replications,
        seed=seed,
    )
    payload["status"] = "ok"
    bootstrap = _bootstrap_payload(result)
    bootstrap.pop("effective_dates", None)
    payload["session_mean_bootstrap"] = bootstrap
    return payload


def _evaluate_v2(
    config: LiteratureBaselineV2Config,
    inputs: _LoadedV2Inputs,
    runs: Mapping[str, _ArmRun],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    boundary = config.run.evaluation_start.isoformat()
    metrics: dict[str, Any] = {}
    stock_metrics: dict[str, Any] = {}
    cost_grid: dict[str, Any] = {}
    for arm, run in sorted(runs.items()):
        all_rows = _statistical_rows(run.ledger)
        metrics[arm] = {
            "all": _metrics_payload(all_rows),
            "development": _metrics_payload([row for row in all_rows if row.session < boundary]),
            "evaluation": _metrics_payload([row for row in all_rows if row.session >= boundary]),
        }
        raw_evaluation = [row for row in run.ledger if row.session >= boundary]
        stock_metrics[arm] = [dataclasses.asdict(row) for row in calculate_stock_diagnostics(raw_evaluation)]
        cost_grid[arm] = {}
        for cost_bps, ledger in sorted(run.grid_ledgers.items()):
            evaluation = [row for row in _statistical_rows(ledger) if row.session >= boundary]
            payload = _metrics_payload(evaluation)
            cost_grid[arm][f"{cost_bps:g}"] = {
                "cost_bps_per_side": cost_bps,
                "cumulative_net_return": payload["cumulative_net_return"],
                "after_cost_sharpe": payload["after_cost_sharpe"],
                "breakeven_cost_bps_per_side_approx": payload["breakeven_cost_bps_per_side_approx"],
            }

    def evaluation_returns(arm: str) -> dict[str, float]:
        return {
            row.session: row.net_return
            for row in _statistical_rows(runs[arm].ledger)
            if row.session >= boundary
        }

    primary = evaluation_returns("finbert")
    if len(primary) < config.inference.block_length_sessions:
        raise BaselineV2RunError("evaluation block is shorter than the frozen bootstrap block length")
    seed = config.inference.seed
    daily = {
        "finbert_vs_cash": _bootstrap_payload(
            paired_block_bootstrap(
                primary,
                {session: 0.0 for session in primary},
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=seed,
            )
        ),
        "finbert_vs_vader_compound": _bootstrap_payload(
            paired_block_bootstrap(
                primary,
                evaluation_returns("vader_compound"),
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=seed + 1,
            )
        ),
        "finbert_magnitude_vs_finbert": _bootstrap_payload(
            paired_block_bootstrap(
                evaluation_returns("finbert_magnitude"),
                primary,
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=seed + 2,
            )
        ),
        "agreement_vs_finbert": _bootstrap_payload(
            paired_block_bootstrap(
                evaluation_returns("agreement"),
                primary,
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=seed + 3,
            )
        ),
    }
    return_lookup = {(row.session, row.symbol): row.value for row in inputs.returns}
    event_level = {
        arm: _event_level_payload(
            arm,
            runs[arm].signals,
            return_lookup,
            boundary=boundary,
            block_length=config.inference.block_length_sessions,
            replications=config.inference.replications,
            seed=seed + 10 + offset,
        )
        for offset, arm in enumerate(("finbert", "vader_compound", "agreement"))
    }
    bootstrap = {"event_level_primary": event_level, "daily_portfolio": daily}
    return metrics, stock_metrics, bootstrap, cost_grid


def _report_v2(
    config: LiteratureBaselineV2Config,
    inputs: _LoadedV2Inputs,
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    cost_grid: Mapping[str, Any],
) -> str:
    boundary = config.run.evaluation_start.isoformat()
    lines = [
        "# Sentiment trading baseline v2 result",
        "",
        f"- Resolved run: `{inputs.run_id}`",
        f"- Status: retrospective comparator (`{config.run.sample_status}`), not an untouched holdout",
        "- Protocol provenance: the v2 design was specified after the v1 result on this same cohort was",
        "  observed. Every v2 result on this sample is exploratory/diagnostic; the protocol is intended",
        "  for confirmatory reuse only on an independent cohort.",
        f"- Reference design: {config.replication.paper_id}, `{config.replication.reference_version}`, "
        f"DOI `{config.replication.doi}`",
        f"- Fidelity: `{config.replication.fidelity}`",
        f"- Chronological evaluation start: `{boundary}`",
        f"- Input events: {len(inputs.events):,}; selected non-technical single-target events: "
        f"{inputs.labels.attrition['selected_events']:,}",
        f"- Universe: {len(inputs.symbols)} stocks; executable decision sessions: {len(inputs.decision_sessions)}",
        "",
        "## Frozen v2 design",
        "",
        "Pinned ProsusAI FinBERT remains the primary scorer. The comparator is canonical VADER: the",
        "compound score with the published +/-0.05 thresholds, from its own completed cache-only artifact.",
        "Directions are the sign of the mean hard label within stock and execution session; positions",
        "enter at the first XNYS open strictly after the 15-minute buffer and hold one adjusted-open",
        "session with no carry. Every arm requires at least two names on each side (single-name weight",
        "is therefore capped at 25% of NAV) and stays in zero-return cash otherwise. Four predeclared",
        "arms are reported: `finbert` (equal weight), `vader_compound` (equal weight),",
        "`finbert_magnitude` (legs weighted by absolute mean continuous score), and `agreement`",
        "(FinBERT direction kept only when canonical VADER agrees and is non-zero). Costs remain 10",
        "basis points per dollar traded, with a fixed 5/10/20 reporting grid.",
        "",
        "## Primary inference: firm-session event-level bootstrap",
        "",
        "The predeclared primary test bootstraps session means of signed gross open-to-open",
        "firm-session returns (moving blocks of 5, 2,000 replications) in the evaluation period:",
        "",
    ]
    event_level = bootstrap["event_level_primary"]
    for arm in ("finbert", "vader_compound", "agreement"):
        payload = event_level[arm]
        if payload["status"] != "ok":
            lines.append(
                f"- `{arm}`: {payload['firm_session_count']} firm-sessions across "
                f"{payload['active_session_count']} sessions - insufficient active sessions for the frozen block bootstrap."
            )
            continue
        result = payload["session_mean_bootstrap"]
        lines.append(
            f"- `{arm}`: {payload['firm_session_count']} firm-sessions "
            f"({payload['long_firm_sessions']} long / {payload['short_firm_sessions']} short) across "
            f"{payload['active_session_count']} sessions; mean signed gross session return "
            f"{_percent(result['mean_daily_difference'])}; 95% moving-block interval "
            f"[{_percent(result['confidence_interval_low'])}, {_percent(result['confidence_interval_high'])}] "
            f"(`{result['confidence_direction']}`)."
        )
    lines.extend(
        [
            "",
            "## Portfolio evaluation results (10 bps per side)",
            "",
            "| Arm | Active days | Gross cumulative | Net cumulative | Net Sharpe | Max drawdown | Avg gross traded weight |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for arm in config.signal.arms:
        row = metrics[arm]["evaluation"]
        lines.append(
            f"| `{arm}` | {row['active_day_count_excluding_liquidation']} | "
            f"{_percent(row['cumulative_gross_return'])} | {_percent(row['cumulative_net_return'])} | "
            f"{_number(row['after_cost_sharpe'])} | {_percent(row['maximum_drawdown'])} | "
            f"{_number(row['average_turnover'])} |"
        )
    daily = bootstrap["daily_portfolio"]
    lines.extend(
        [
            "",
            "## Predeclared portfolio comparisons (daily net, evaluation period)",
            "",
        ]
    )
    for name, label in (
        ("finbert_vs_cash", "FinBERT minus zero-return cash"),
        ("finbert_vs_vader_compound", "FinBERT minus canonical VADER"),
        ("finbert_magnitude_vs_finbert", "Magnitude-weighted FinBERT minus equal-weight FinBERT"),
        ("agreement_vs_finbert", "Agreement-conditioned minus unconditional FinBERT"),
    ):
        payload = daily[name]
        lines.append(
            f"- {label}: mean daily difference {_percent(payload['mean_daily_difference'])}; "
            f"95% moving-block interval [{_percent(payload['confidence_interval_low'])}, "
            f"{_percent(payload['confidence_interval_high'])}] (`{payload['confidence_direction']}`)."
        )
    lines.extend(
        [
            "",
            "## Transaction-cost sensitivity (evaluation net cumulative / Sharpe)",
            "",
            "| Arm | 5 bps | 10 bps | 20 bps |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in config.signal.arms:
        cells = []
        for cost_bps in config.portfolio.cost_grid_bps:
            payload = cost_grid[arm][f"{cost_bps:g}"]
            cells.append(f"{_percent(payload['cumulative_net_return'])} / {_number(payload['after_cost_sharpe'])}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "A null or adverse result is a completed baseline and must not trigger threshold, horizon,",
            "cohort, or model selection on this sample.",
            "",
            "## Interpretation limits",
            "",
            "- The v2 protocol was frozen after v1 results on this cohort were observed; nothing here is",
            "  confirmatory. The next defensible step is running this frozen protocol on an independent cohort.",
            "- This 22-stock period has already been examined in prior exploratory work.",
            "- FinBERT and VADER classify generic headline polarity, not target-specific expected return.",
            "- Open-to-open returns adapt, not replicate, the reference paper's timing windows.",
            "- The selected cohort may contain coverage or survivorship bias.",
            "- Opens are split-adjusted but dividends are not back-adjusted.",
            "- The cost model omits borrow availability and fees, spread variation, market impact, and intraday liquidity.",
            "- The cash comparator earns zero; cash yield and financing are omitted.",
            "- Licensed Reuters/LSEG text remains local and is absent from these result artifacts.",
            "",
            "## Declared deviations from the reference paper",
            "",
            *[f"- {item}" for item in config.replication.deviations],
            "",
        ]
    )
    return "\n".join(lines)


def inspect_baseline_v2(
    config: LiteratureBaselineV2Config, *, repo_root: str | Path = "."
) -> BaselineV2Inspection:
    """Validate every local input and calculate identities without writing."""

    inputs = _load_inputs_v2(config, Path(repo_root))
    runs = _run_arms(config, inputs)
    boundary = config.run.evaluation_start.isoformat()
    evaluation_sessions = sum(session >= boundary for session in inputs.decision_sessions)
    active = {
        arm: sum(row.active_names > 0 and not row.final_liquidation for row in run.ledger)
        for arm, run in runs.items()
    }
    return BaselineV2Inspection(
        resolved_run_id=inputs.run_id,
        input_identities=inputs.identities,
        event_count=len(inputs.events),
        selected_event_count=inputs.labels.attrition["selected_events"],
        firm_session_signal_count=len(inputs.labels.signals),
        symbols=inputs.symbols,
        price_sessions=len({row.session for row in inputs.adjusted_opens}),
        decision_sessions=len(inputs.decision_sessions),
        development_sessions=len(inputs.decision_sessions) - evaluation_sessions,
        evaluation_sessions=evaluation_sessions,
        active_sessions_by_arm=active,
        derived_dir=inputs.derived_dir,
        results_dir=inputs.results_dir,
        warnings=(
            "The v2 protocol was specified after v1 results on this cohort were observed; "
            "all v2 results on this sample are exploratory, not confirmatory.",
            "This sample has been examined previously; results are retrospective comparator evidence.",
            "Prices are split-adjusted but not dividend-adjusted; borrow, spread, and impact are not fully modelled.",
        ),
    )


def run_baseline_v2(
    config: LiteratureBaselineV2Config, *, repo_root: str | Path = "."
) -> BaselineV2RunResult:
    """Execute the frozen v2 baseline and write aggregate immutable evidence."""

    root = Path(repo_root).resolve()
    inputs = _load_inputs_v2(config, root)
    runs = _run_arms(config, inputs)
    metrics, stock_metrics, bootstrap, cost_grid = _evaluate_v2(config, inputs, runs)
    derived = inputs.derived_dir
    results = inputs.results_dir
    label_rows = [row.to_payload() for row in inputs.labels.labels]
    signal_rows = [row.to_payload() for row in inputs.labels.signals]
    construction_rows = [dict(row) for run in runs.values() for row in run.session_rows]
    construction_rows.sort(key=lambda row: (str(row["session"]), str(row["scorer"])))
    pnl_rows = [_ledger_payload(arm, row) for arm, run in sorted(runs.items()) for row in run.ledger]
    pnl_rows.sort(key=lambda row: (str(row["session"]), str(row["scorer"])))
    attrition = {
        "event_builder": inputs.event_attrition,
        "baseline_screening": inputs.labels.attrition,
        "label_counts_by_scorer": {
            scorer: dict(sorted(Counter(row.label for row in inputs.labels.labels if row.scorer == scorer).items()))
            for scorer in ("finbert", "vader_compound")
        },
        "firm_session_action_counts_by_scorer": {
            scorer: dict(sorted(Counter(row.action for row in inputs.labels.signals if row.scorer == scorer).items()))
            for scorer in ("finbert", "vader_compound")
        },
        "arm_firm_session_counts": {
            arm: {
                "total": len(run.signals),
                "long": sum(row.action > 0 for row in run.signals),
                "short": sum(row.action < 0 for row in run.signals),
            }
            for arm, run in sorted(runs.items())
        },
    }
    report = _report_v2(config, inputs, metrics, bootstrap, cost_grid)

    writes: list[bool] = []
    paths: dict[str, Path] = {}
    for name, path, jsonl_payload in (
        ("event_labels", derived / "event_labels.jsonl", label_rows),
        ("firm_session_signals", derived / "firm_session_signals.jsonl", signal_rows),
        ("daily_pnl", results / "daily_pnl.jsonl", pnl_rows),
        ("session_construction", results / "session_construction.jsonl", construction_rows),
    ):
        written, reused = write_immutable_jsonl(path, jsonl_payload)
        paths[name] = written
        writes.append(reused)
    for name, path, json_payload in (
        ("attrition", derived / "attrition.json", attrition),
        ("metrics", results / "metrics.json", metrics),
        ("stock_metrics", results / "stock_metrics.json", stock_metrics),
        ("bootstrap", results / "bootstrap.json", bootstrap),
        ("cost_grid", results / "cost_grid.json", cost_grid),
    ):
        written, reused = write_immutable_json(path, json_payload)
        paths[name] = written
        writes.append(reused)
    report_path, reused = write_immutable_text(results / "report.md", report)
    paths["report"] = report_path
    writes.append(reused)

    manifest_payload = {
        "schema_version": 2,
        "status": "completed",
        "run_id": inputs.run_id,
        "sample_status": config.run.sample_status,
        "config": config.to_payload(),
        "input_identities": inputs.identities,
        "runtime_dependencies": _runtime_dependencies(),
        "method": {
            "reference_design": {
                "paper": config.replication.paper_id,
                "version": config.replication.reference_version,
                "url": config.replication.reference_url,
                "fidelity": config.replication.fidelity,
            },
            "event_identity": "earliest_story_family_symbol",
            "text_unit": "headline_only",
            "score_join": "sha256(normalize_headline(event.headline))",
            "screening_metadata": "chosen corpus occurrence, not headline-level unioned score metadata",
            "scorer_labels": {
                "finbert": "argmax of pinned three-class ProsusAI/finbert probabilities",
                "vader_compound": "canonical VADER compound thresholds at +/-0.05",
            },
            "signal": "sign(mean(hard_label)) within scorer-symbol-execution_session",
            "arms": {
                "finbert": "equal-weight dollar-neutral primary",
                "vader_compound": "equal-weight dollar-neutral comparator",
                "finbert_magnitude": "legs weighted by absolute mean continuous score; equal-weight fallback on all-zero side",
                "agreement": "FinBERT direction kept only when canonical VADER direction agrees and is non-zero",
            },
            "execution": "first XNYS open strictly after availability plus 15-minute processing buffer",
            "holding": "one split-adjusted open-to-open session; no carry",
            "portfolio": "dollar-neutral, minimum two names per side, cash otherwise",
            "cost_bps_per_side": config.portfolio.cost_bps_per_side,
            "cost_grid_bps": list(config.portfolio.cost_grid_bps),
            "turnover_definition": "sum(abs(target_weight - current_weight))",
            "cost_definition": "cost_bps_per_side times gross traded weight; charged on every buy and sell",
            "primary_inference": (
                "moving-block bootstrap of evaluation-period session means of signed gross "
                "open-to-open firm-session returns"
            ),
            "terminal_liquidation": (
                "separate audit row; exit cost and turnover folded into the final holding interval for statistics"
            ),
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
            "fidelity": "conceptual_adaptation",
            "confirmatory": False,
            "specified_after_v1_results": True,
            "retuning_after_result_permitted": False,
        },
    }
    manifest_path, manifest_reused = write_immutable_json(results / "manifest.json", manifest_payload)
    writes.append(manifest_reused)
    return BaselineV2RunResult(
        resolved_run_id=inputs.run_id,
        derived_dir=derived,
        results_dir=results,
        report_path=report_path,
        metrics_path=paths["metrics"],
        reused=all(writes),
    )
