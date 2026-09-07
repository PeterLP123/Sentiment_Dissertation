"""Read-only inspection and immutable execution of the sentiment baseline."""

from __future__ import annotations

import dataclasses
import importlib.metadata
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...artifact_io import canonical_json, read_json, sha256_file, sha256_text
from ...headline_value import classify_headline
from ...lseg_corpus import load_verified_lseg_corpus
from ..artifacts import write_immutable_json, write_immutable_jsonl, write_immutable_text
from ..diagnostics import calculate_portfolio_metrics, calculate_stock_diagnostics
from ..inference import PairedBootstrapResult, paired_block_bootstrap
from ..ledger import DailyLedgerRow, assert_no_boundary_crossing, run_open_to_open_ledger
from ..market import (
    AdjustedOpen,
    OpenToOpenReturn,
    calculate_open_to_open_returns,
    load_adjusted_opens_csv,
    validate_exchange_sessions,
)
from ..schemas import StrategyEvent
from ..sources import LsegEventSettings, build_strategy_events
from .config import LiteratureBaselineConfig
from .signals import (
    BaselineScoreError,
    EventScreeningMetadata,
    LabelBuildResult,
    PortfolioBuildResult,
    build_equal_weight_targets,
    build_event_labels,
    load_headline_scores,
)

if TYPE_CHECKING:
    from .v2.config import LiteratureBaselineV2Config


class BaselineRunError(RuntimeError):
    """Raised when a frozen input or output violates the baseline contract."""


@dataclass(frozen=True)
class BaselineInspection:
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
    active_sessions_by_scorer: dict[str, int]
    derived_dir: Path
    results_dir: Path
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class BaselineRunResult:
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
    labels: LabelBuildResult
    adjusted_opens: tuple[AdjustedOpen, ...]
    returns: tuple[OpenToOpenReturn, ...]
    symbols: tuple[str, ...]
    decision_sessions: tuple[str, ...]
    run_id: str
    derived_dir: Path
    results_dir: Path


@dataclass(frozen=True)
class _ScorerRun:
    scorer: str
    portfolio: PortfolioBuildResult
    ledger: tuple[DailyLedgerRow, ...]


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise BaselineRunError(f"{name} does not exist: {path}")


def _verify_score_manifest(
    config: LiteratureBaselineConfig | LiteratureBaselineV2Config,
    *,
    repo_root: Path,
    score_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    _require_file(score_path, "headline score artifact")
    _require_file(manifest_path, "headline score manifest")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise BaselineRunError("headline score manifest is not completed")
    output = manifest.get("output")
    if not isinstance(output, dict) or output.get("sha256") != sha256_file(score_path):
        raise BaselineRunError("headline score manifest hash does not match the score artifact")
    declared_path = str(output.get("path") or "")
    if declared_path:
        resolved_declared = Path(declared_path)
        resolved_declared = resolved_declared if resolved_declared.is_absolute() else repo_root / resolved_declared
        if resolved_declared.resolve() != score_path.resolve():
            raise BaselineRunError("headline score manifest points to a different output path")
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or int(counts.get("remaining", -1)) != 0:
        raise BaselineRunError("headline score manifest does not establish complete local scoring")
    inference = manifest.get("inference")
    if (
        not isinstance(inference, dict)
        or inference.get("local_only") is not True
        or inference.get("local_files_only_enforced") is not True
    ):
        raise BaselineRunError("headline score manifest does not prove cache-only inference was enforced")
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise BaselineRunError("headline score manifest has no model identities")
    configured_finbert = next(item for item in config.scorers if item.id == "finbert")
    finbert = models.get("finbert")
    if not isinstance(finbert, dict):
        raise BaselineRunError("headline score manifest has no FinBERT identity")
    if (
        finbert.get("model_id") != configured_finbert.model_id
        or finbert.get("revision") != configured_finbert.revision
        or finbert.get("revision_enforced") is not True
        or finbert.get("local_model_path") is not None
    ):
        raise BaselineRunError("FinBERT model ID/revision differs from the frozen config")
    vader = models.get("vader")
    if not isinstance(vader, dict) or vader.get("implementation") != "nltk.sentiment.vader":
        raise BaselineRunError("headline score manifest has an unexpected VADER implementation")
    vader_lexicon_hash = str(vader.get("lexicon_sha256") or "")
    if len(vader_lexicon_hash) != 64 or any(value not in "0123456789abcdef" for value in vader_lexicon_hash):
        raise BaselineRunError("headline score manifest does not pin the VADER lexicon")
    sharing = manifest.get("sharing")
    if not isinstance(sharing, dict) or sharing.get("contains_licensed_headline_text") is not True:
        raise BaselineRunError("headline score manifest must identify the licensed-text artifact")
    return manifest


def _verify_price_manifest(panel_path: Path, manifest_path: Path) -> dict[str, Any]:
    _require_file(panel_path, "strategy price panel")
    _require_file(manifest_path, "strategy price manifest")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise BaselineRunError("strategy price manifest is not completed")
    if (
        manifest.get("execution_field") != "adjusted_open"
        or manifest.get("return_convention") != "open_to_open"
        or manifest.get("calendar") != "XNYS"
        or manifest.get("adjustment_supported") is not True
        or manifest.get("split_adjusted") is not True
        or manifest.get("dividend_adjusted") is not False
    ):
        raise BaselineRunError("strategy price manifest does not establish split-adjusted open-to-open semantics")
    files = manifest.get("files")
    price_entry = files.get("price_panel") if isinstance(files, dict) else None
    if not isinstance(price_entry, dict) or price_entry.get("sha256") != sha256_file(panel_path):
        raise BaselineRunError("strategy price manifest hash does not match the panel")
    if price_entry.get("path") and Path(str(price_entry["path"])).name != panel_path.name:
        raise BaselineRunError("strategy price manifest names a different panel")
    return manifest


def _load_prices(panel_path: Path, manifest_path: Path) -> tuple[tuple[AdjustedOpen, ...], tuple[OpenToOpenReturn, ...], tuple[str, ...]]:
    manifest = _verify_price_manifest(panel_path, manifest_path)
    opens = tuple(load_adjusted_opens_csv(panel_path, price_column="adjusted_open", adjustment_supported=True))
    validate_exchange_sessions(opens, calendar_name=str(manifest.get("calendar") or "XNYS"))
    symbols = tuple(sorted({row.symbol for row in opens}))
    declared_symbols = tuple(sorted(str(item).strip().upper() for item in manifest.get("symbols") or []))
    if symbols != declared_symbols or int(manifest.get("symbol_count", -1)) != len(symbols):
        raise BaselineRunError("strategy price manifest symbols do not match the panel")
    sessions_by_symbol: dict[str, tuple[str, ...]] = {}
    for symbol in symbols:
        sessions_by_symbol[symbol] = tuple(row.session for row in opens if row.symbol == symbol)
    spines = set(sessions_by_symbol.values())
    if len(spines) != 1:
        raise BaselineRunError("strategy price panel is not a complete symbol-session grid")
    spine = next(iter(spines))
    if int(manifest.get("session_count", -1)) != len(spine) or int(manifest.get("row_count", -1)) != len(opens):
        raise BaselineRunError("strategy price manifest counts do not match the panel")
    returns = tuple(calculate_open_to_open_returns(opens))
    return opens, returns, symbols


def _input_identities(config: LiteratureBaselineConfig, repo_root: Path) -> dict[str, str]:
    paths = {
        "corpus_manifest_sha256": _resolve(repo_root, config.data.corpus_manifest),
        "score_artifact_sha256": _resolve(repo_root, config.data.score_path),
        "score_manifest_sha256": _resolve(repo_root, config.data.score_manifest),
        "price_panel_sha256": _resolve(repo_root, config.prices.panel_path),
        "price_manifest_sha256": _resolve(repo_root, config.prices.manifest_path),
    }
    for name, path in paths.items():
        _require_file(path, name.removesuffix("_sha256"))
    return {
        "config_sha256": config.config_sha256,
        "implementation_sha256": _implementation_sha256(),
        "runtime_dependencies_sha256": sha256_text(canonical_json(_runtime_dependencies())),
        **{name: sha256_file(path) for name, path in paths.items()},
    }


def _implementation_sha256() -> str:
    """Hash the baseline and shared point-in-time/accounting implementation."""

    baseline_root = Path(__file__).parent
    strategy_root = baseline_root.parent
    package_root = strategy_root.parent
    files = [baseline_root / name for name in ("config.py", "signals.py", "pipeline.py")]
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


def _runtime_dependencies() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            name: importlib.metadata.version(name)
            for name in ("exchange-calendars", "numpy", "pandas")
        },
    }


def _load_inputs(config: LiteratureBaselineConfig, repo_root: Path) -> _LoadedInputs:
    root = repo_root.resolve()
    collection_root = _resolve(root, config.data.collection_root).resolve()
    corpus_manifest = _resolve(root, config.data.corpus_manifest).resolve()
    score_path = _resolve(root, config.data.score_path).resolve()
    score_manifest = _resolve(root, config.data.score_manifest).resolve()
    price_panel = _resolve(root, config.prices.panel_path).resolve()
    price_manifest = _resolve(root, config.prices.manifest_path).resolve()
    if not collection_root.is_dir():
        raise BaselineRunError(f"collection root does not exist: {collection_root}")
    if not corpus_manifest.is_relative_to(collection_root):
        raise BaselineRunError("corpus manifest must be inside the declared collection root")
    if not score_path.is_relative_to(collection_root) or not score_manifest.is_relative_to(collection_root):
        raise BaselineRunError("licensed score artifacts must remain inside the collection root")

    identities = _input_identities(config, root)
    _verify_score_manifest(
        config,
        repo_root=root,
        score_path=score_path,
        manifest_path=score_manifest,
    )
    event_result = build_strategy_events(
        corpus_manifest,
        settings=LsegEventSettings(
            processing_buffer_minutes=config.data.processing_buffer_minutes,
            calendar_name=config.data.calendar,
            exchange_timezone=config.data.timezone,
        ),
    )
    scores = load_headline_scores(score_path, [item.id for item in config.scorers])
    event_metadata = _event_screening_metadata(corpus_manifest, event_result.events)
    try:
        labels = build_event_labels(event_result.events, scores, event_metadata, config)
    except BaselineScoreError as exc:
        raise BaselineRunError(str(exc)) from exc
    opens, returns, symbols = _load_prices(price_panel, price_manifest)
    event_symbols = {event.symbol for event in event_result.events}
    if not event_symbols.issubset(symbols):
        raise BaselineRunError(f"events contain symbols absent from prices: {sorted(event_symbols - set(symbols))}")
    next_by_session: dict[str, str] = {}
    for row in returns:
        previous = next_by_session.setdefault(row.session, row.next_session)
        if previous != row.next_session:
            raise BaselineRunError(f"inconsistent next price session after {row.session}")
    all_decision_sessions = tuple(sorted(next_by_session))
    selected_event_sessions = sorted({row.eligible_execution_session for row in labels.labels})
    if not selected_event_sessions:
        raise BaselineRunError("baseline screening left no eligible event sessions")
    first_event_session, last_event_session = selected_event_sessions[0], selected_event_sessions[-1]
    decision_sessions = tuple(
        session for session in all_decision_sessions if first_event_session <= session <= last_event_session
    )
    if config.run.evaluation_start.isoformat() not in decision_sessions:
        raise BaselineRunError("evaluation_start is absent from the executable price-session spine")
    executable = set(all_decision_sessions)
    missing_event_sessions = sorted(
        set(selected_event_sessions) - executable
    )
    if missing_event_sessions:
        raise BaselineRunError(f"events map outside executable price history: {missing_event_sessions[:5]}")
    identity_hash = sha256_text(canonical_json(identities))
    run_id = f"{config.run.id}-{identity_hash[:12]}"
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
    )


def _event_screening_metadata(
    corpus_manifest: Path,
    events: Sequence[StrategyEvent],
) -> dict[str, EventScreeningMetadata]:
    """Derive screening flags from each chosen revision, never a future-unioned headline row."""

    manifest, records = load_verified_lseg_corpus(corpus_manifest)
    companies = ((manifest.get("config") or {}).get("companies") or [])
    aliases_by_symbol: dict[str, tuple[str, ...]] = {}
    for item in companies:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        aliases = [symbol]
        aliases.extend(str(value).strip() for value in item.get("aliases") or [] if str(value).strip())
        for field_name in ("name", "ric"):
            if str(item.get(field_name) or "").strip():
                aliases.append(str(item[field_name]).strip())
        aliases_by_symbol[symbol] = tuple(dict.fromkeys(aliases))
    record_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        revision_key = (str(record.get("story_family_id") or ""), str(record.get("revision_id") or ""))
        if revision_key in record_lookup:
            raise BaselineRunError(f"corpus contains duplicate family/revision identity: {revision_key}")
        record_lookup[revision_key] = record
    result: dict[str, EventScreeningMetadata] = {}
    for event in events:
        chosen_record = record_lookup.get((event.story_family_id, event.revision_id))
        if chosen_record is None:
            raise BaselineRunError(f"chosen event revision is absent from verified corpus: {event.event_id}")
        matched_symbols = tuple(
            sorted(
                {
                    str(value).strip().upper()
                    for value in chosen_record.get("matched_symbols") or []
                    if str(value).strip()
                }
            )
        )
        classification = classify_headline(
            event.headline,
            aliases=aliases_by_symbol.get(event.symbol, (event.symbol,)),
            matched_symbol_count=len(matched_symbols),
        )
        result[event.event_id] = EventScreeningMetadata(
            matched_symbols=matched_symbols,
            explicit_target=classification.alias_match,
            market_price_technical=classification.event_type == "market_price_technical",
        )
    return result


def _run_ledgers(config: LiteratureBaselineConfig, inputs: _LoadedInputs) -> dict[str, _ScorerRun]:
    boundary = config.run.evaluation_start.isoformat()
    runs: dict[str, _ScorerRun] = {}
    for scorer in (item.id for item in config.scorers):
        portfolio = build_equal_weight_targets(
            inputs.labels.signals,
            scorer=scorer,
            sessions=inputs.decision_sessions,
            symbols=inputs.symbols,
            gross_exposure=config.portfolio.gross_exposure,
            minimum_names_per_side=config.portfolio.minimum_names_per_side,
        )
        ledger = tuple(
            run_open_to_open_ledger(
                portfolio.targets,
                inputs.returns,
                cost_rate_per_side=config.portfolio.cost_rate_per_side,
                initial_nav_usd=config.portfolio.initial_nav_usd,
                accounting_reset_session=boundary,
                force_final_liquidation=config.portfolio.force_final_liquidation,
            )
        )
        assert_no_boundary_crossing(ledger, boundary)
        runs[scorer] = _ScorerRun(scorer, portfolio, ledger)
    return runs


def inspect_baseline(config: LiteratureBaselineConfig, *, repo_root: str | Path = ".") -> BaselineInspection:
    """Validate every local input and calculate identities without writing."""

    inputs = _load_inputs(config, Path(repo_root))
    runs = _run_ledgers(config, inputs)
    boundary = config.run.evaluation_start.isoformat()
    evaluation_sessions = sum(session >= boundary for session in inputs.decision_sessions)
    active = {
        scorer: sum(row.active_names > 0 and not row.final_liquidation for row in run.ledger)
        for scorer, run in runs.items()
    }
    return BaselineInspection(
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
        active_sessions_by_scorer=active,
        derived_dir=inputs.derived_dir,
        results_dir=inputs.results_dir,
        warnings=(
            "This sample has been examined previously; results are retrospective comparator evidence, not an untouched holdout.",
            "FinBERT and VADER share-argmax measure headline polarity, not expected returns.",
            "Prices are split-adjusted but not dividend-adjusted; borrow fees, spread, and market impact are not fully modelled.",
        ),
    )


def _ledger_payload(scorer: str, row: DailyLedgerRow) -> dict[str, object]:
    return {
        "scorer": scorer,
        "session": row.session,
        "next_session": row.next_session,
        "start_nav_usd": row.start_nav_usd,
        "gross_return": row.gross_return,
        "transaction_cost": row.transaction_cost,
        "net_return": row.net_return,
        "end_nav_usd": row.end_nav_usd,
        "turnover": row.turnover,
        "gross_exposure": row.gross_exposure,
        "long_exposure": row.long_exposure,
        "short_exposure": row.short_exposure,
        "net_exposure": row.net_exposure,
        "active_names": row.active_names,
        "final_liquidation": row.final_liquidation,
    }


def _metrics_payload(rows: Sequence[DailyLedgerRow]) -> dict[str, Any]:
    metrics = dataclasses.asdict(calculate_portfolio_metrics(rows))
    active = [row for row in rows if row.active_names > 0 and not row.final_liquidation]
    total_turnover = sum(row.turnover for row in rows)
    gross_sum = sum(row.gross_return for row in rows)
    metrics["mean_daily_gross_return"] = sum(row.gross_return for row in rows) / len(rows) if rows else 0.0
    metrics["mean_daily_net_return"] = sum(row.net_return for row in rows) / len(rows) if rows else 0.0
    metrics["total_turnover"] = total_turnover
    metrics["breakeven_cost_bps_per_side_approx"] = (
        gross_sum / total_turnover * 10_000 if total_turnover > 0 else None
    )
    metrics["active_day_count_excluding_liquidation"] = len(active)
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for row in rows:
        wealth *= 1 + row.net_return
        peak = max(peak, wealth)
        maximum_drawdown = min(maximum_drawdown, wealth / peak - 1)
    metrics["maximum_drawdown"] = maximum_drawdown
    return metrics


def _statistical_rows(rows: Sequence[DailyLedgerRow]) -> list[DailyLedgerRow]:
    """Fold the terminal exit cost into its holding interval for daily statistics."""

    ordered = list(rows)
    if not ordered or not ordered[-1].final_liquidation:
        return ordered
    if len(ordered) < 2:
        raise BaselineRunError("a liquidation audit row has no preceding holding interval")
    liquidation = ordered.pop()
    previous = ordered[-1]
    if previous.next_session != liquidation.session:
        raise BaselineRunError("terminal liquidation does not match the preceding return endpoint")
    combined_net = (1 + previous.net_return) * (1 + liquidation.net_return) - 1
    ordered[-1] = dataclasses.replace(
        previous,
        transaction_cost=previous.gross_return - combined_net,
        net_return=combined_net,
        end_nav_usd=liquidation.end_nav_usd,
        turnover=previous.turnover + liquidation.turnover,
        orders=previous.orders + liquidation.orders,
        contributions=previous.contributions + liquidation.contributions,
        final_liquidation=False,
    )
    return ordered


def _bootstrap_payload(result: PairedBootstrapResult) -> dict[str, Any]:
    return {
        "mean_daily_difference": result.mean_daily_difference,
        "confidence_level": result.confidence_level,
        "confidence_interval_low": result.confidence_interval_low,
        "confidence_interval_high": result.confidence_interval_high,
        "mean_direction": result.mean_direction,
        "confidence_direction": result.confidence_direction,
        "effective_dates": list(result.effective_dates),
        "block_length": result.block_length,
        "replications": result.replications,
        "seed": result.seed,
    }


def _evaluate(
    config: LiteratureBaselineConfig,
    runs: Mapping[str, _ScorerRun],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    boundary = config.run.evaluation_start.isoformat()
    metrics: dict[str, Any] = {}
    stock_metrics: dict[str, Any] = {}
    for scorer, run in sorted(runs.items()):
        all_rows = _statistical_rows(run.ledger)
        development = [row for row in all_rows if row.session < boundary]
        evaluation = [row for row in all_rows if row.session >= boundary]
        metrics[scorer] = {
            "all": _metrics_payload(all_rows),
            "development": _metrics_payload(development),
            "evaluation": _metrics_payload(evaluation),
        }
        raw_evaluation = [row for row in run.ledger if row.session >= boundary]
        stock_metrics[scorer] = [dataclasses.asdict(row) for row in calculate_stock_diagnostics(raw_evaluation)]
    primary = config.primary_scorer.id
    primary_returns = {
        row.session: row.net_return
        for row in _statistical_rows(runs[primary].ledger)
        if row.session >= boundary
    }
    if len(primary_returns) < config.inference.block_length_sessions:
        raise BaselineRunError("evaluation block is shorter than the frozen bootstrap block length")
    zero_return_cash = {session: 0.0 for session in primary_returns}
    bootstrap: dict[str, Any] = {
        "primary_vs_cash": _bootstrap_payload(
            paired_block_bootstrap(
                primary_returns,
                zero_return_cash,
                block_length=config.inference.block_length_sessions,
                replications=config.inference.replications,
                seed=config.inference.seed,
            )
        )
    }
    comparator = next(item.id for item in config.scorers if item.role == "comparator")
    comparator_returns = {
        row.session: row.net_return
        for row in _statistical_rows(runs[comparator].ledger)
        if row.session >= boundary
    }
    bootstrap["primary_vs_comparator"] = _bootstrap_payload(
        paired_block_bootstrap(
            primary_returns,
            comparator_returns,
            block_length=config.inference.block_length_sessions,
            replications=config.inference.replications,
            seed=config.inference.seed + 1,
        )
    )
    return metrics, stock_metrics, bootstrap


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.3f}%"


def _number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _report(
    config: LiteratureBaselineConfig,
    inputs: _LoadedInputs,
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> str:
    boundary = config.run.evaluation_start.isoformat()
    lines = [
        "# Sentiment trading baseline result",
        "",
        f"- Resolved run: `{inputs.run_id}`",
        f"- Status: retrospective comparator (`{config.run.sample_status}`), not an untouched holdout",
        f"- Reference design: {config.replication.paper_id}, `{config.replication.reference_version}`, "
        f"DOI `{config.replication.doi}`",
        f"- Fidelity: `{config.replication.fidelity}`",
        f"- Chronological evaluation start: `{boundary}`",
        f"- Input events: {len(inputs.events):,}; selected non-technical single-target events: "
        f"{inputs.labels.attrition['selected_events']:,}",
        f"- Universe: {len(inputs.symbols)} stocks; executable decision sessions: {len(inputs.decision_sessions)}",
        "",
        "## Frozen design",
        "",
        "The primary model is pinned ProsusAI FinBERT. VADER share-argmax with `pos - neg` as its continuous diagnostic is a "
        "predeclared derived comparator; it is not canonical VADER compound classification. "
        "Hard labels map to -1/0/+1, are averaged within stock and execution session, and the sign of that mean is traded. "
        "Positions enter at the first XNYS open strictly after the 15-minute processing buffer, hold for one adjusted-open session, "
        "and do not carry without new eligible news. A session trades only when both legs meet the frozen minimum; each leg receives "
        "half of the gross exposure. Costs equal 10 basis points times gross traded weight, defined as the sum of absolute weight "
        "changes; this is charged on every buy and sell.",
        "",
        "## Evaluation results",
        "",
        "| Scorer | Active days | Gross cumulative | Net cumulative | Net Sharpe | Max drawdown | Avg gross traded weight |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scorer in (item.id for item in config.scorers):
        row = metrics[scorer]["evaluation"]
        lines.append(
            f"| `{scorer}` | {row['active_day_count_excluding_liquidation']} | "
            f"{_percent(row['cumulative_gross_return'])} | {_percent(row['cumulative_net_return'])} | "
            f"{_number(row['after_cost_sharpe'])} | {_percent(row['maximum_drawdown'])} | "
            f"{_number(row['average_turnover'])} |"
        )
    if metrics["vader"]["evaluation"]["active_day_count_excluding_liquidation"] == 0:
        lines.extend(
            [
                "",
                "The frozen VADER share-argmax labels produced no evaluation session with both a long and a short leg, "
                "so the VADER comparator remained in zero-return cash. This sparse-label result is retained rather than repaired "
                "with a return-selected threshold.",
            ]
        )
    primary_cash = bootstrap["primary_vs_cash"]
    primary_comparator = bootstrap["primary_vs_comparator"]
    lines.extend(
        [
            "",
            "## Predeclared comparisons",
            "",
            f"- FinBERT minus zero-return cash mean daily net return: "
            f"{_percent(primary_cash['mean_daily_difference'])}; "
            f"{int(100 * primary_cash['confidence_level'])}% moving-block interval "
            f"[{_percent(primary_cash['confidence_interval_low'])}, {_percent(primary_cash['confidence_interval_high'])}] "
            f"(`{primary_cash['confidence_direction']}`).",
            f"- FinBERT minus VADER share-argmax mean daily net return: "
            f"{_percent(primary_comparator['mean_daily_difference'])}; "
            f"{int(100 * primary_comparator['confidence_level'])}% moving-block interval "
            f"[{_percent(primary_comparator['confidence_interval_low'])}, {_percent(primary_comparator['confidence_interval_high'])}] "
            f"(`{primary_comparator['confidence_direction']}`).",
            "",
            "A null or adverse result is a completed baseline and must not trigger threshold, horizon, cohort, "
            "or model selection on this sample.",
            "",
            "## Interpretation limits",
            "",
            f"- This {len(inputs.symbols)}-stock period has already been examined in prior exploratory work; "
            "it cannot support a fresh confirmatory claim.",
            "- FinBERT and VADER share-argmax classify generic headline polarity, not target-specific expected return.",
            "- The design adapts a modern headline long/short template to next-open, open-to-open data; "
            "it is not an exact paper replication.",
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


def run_baseline(config: LiteratureBaselineConfig, *, repo_root: str | Path = ".") -> BaselineRunResult:
    """Execute the frozen local baseline and write aggregate immutable evidence."""

    root = Path(repo_root).resolve()
    inputs = _load_inputs(config, root)
    runs = _run_ledgers(config, inputs)
    metrics, stock_metrics, bootstrap = _evaluate(config, runs)
    derived = inputs.derived_dir
    results = inputs.results_dir
    label_rows = [row.to_payload() for row in inputs.labels.labels]
    signal_rows = [row.to_payload() for row in inputs.labels.signals]
    construction_rows = [row for run in runs.values() for row in run.portfolio.session_rows]
    construction_rows.sort(key=lambda row: (str(row["session"]), str(row["scorer"])))
    pnl_rows = [_ledger_payload(scorer, row) for scorer, run in sorted(runs.items()) for row in run.ledger]
    pnl_rows.sort(key=lambda row: (str(row["session"]), str(row["scorer"])))
    attrition = {
        "event_builder": inputs.event_attrition,
        "baseline_screening": inputs.labels.attrition,
        "label_counts_by_scorer": {
            scorer: dict(sorted(Counter(row.label for row in inputs.labels.labels if row.scorer == scorer).items()))
            for scorer in sorted(runs)
        },
        "firm_session_action_counts_by_scorer": {
            scorer: dict(sorted(Counter(row.action for row in inputs.labels.signals if row.scorer == scorer).items()))
            for scorer in sorted(runs)
        },
    }
    report = _report(config, inputs, metrics, bootstrap)

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
    ):
        written, reused = write_immutable_json(path, json_payload)
        paths[name] = written
        writes.append(reused)
    report_path, reused = write_immutable_text(results / "report.md", report)
    paths["report"] = report_path
    writes.append(reused)

    manifest_payload = {
        "schema_version": 1,
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
                "vader": "derived share-argmax of pos/neg/neu; not canonical VADER compound thresholds",
            },
            "signal": "sign(mean(hard_label)) within scorer-symbol-execution_session",
            "execution": "first XNYS open strictly after availability plus 15-minute processing buffer",
            "holding": "one split-adjusted open-to-open session; no carry",
            "portfolio": "equal-weight dollar-neutral with both legs required",
            "cost_bps_per_side": config.portfolio.cost_bps_per_side,
            "turnover_definition": "sum(abs(target_weight - current_weight))",
            "cost_definition": "cost_bps_per_side times gross traded weight; charged on every buy and sell",
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
            "retuning_after_result_permitted": False,
        },
    }
    manifest_path, manifest_reused = write_immutable_json(results / "manifest.json", manifest_payload)
    writes.append(manifest_reused)
    return BaselineRunResult(
        resolved_run_id=inputs.run_id,
        derived_dir=derived,
        results_dir=results,
        report_path=report_path,
        metrics_path=paths["metrics"],
        reused=all(writes),
    )
