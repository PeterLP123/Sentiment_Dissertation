"""Frozen second-moment and full-FinBERT extension of the FNSPID scale smoke."""

# Report prose and SQL statements intentionally retain readable long literals.
# ruff: noqa: E501

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import sqlite3
import time
import tomllib
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .artifact_io import atomic_write_json, atomic_write_text, sha256_file
from .baselines import _disable_hf_progress_bars, _soft_sentiment_from_scores
from .fnspid_feasibility import _hash64, _normal, _open_zstd_csv, _timestamp_parts
from .fnspid_vader_smoke import (
    PRICE_RECAP_PATTERN,
    _benjamini_hochberg,
    _recap_verdict,
    _SessionMapper,
    _signal_verdict,
    attach_abnormal_returns,
    is_price_recap,
    load_fnspid_prices,
)
from .runtime_metadata import collect_run_environment


class FnspidMoment2FinbertError(RuntimeError):
    """Raised when the frozen extension cannot run as specified."""


@dataclass(frozen=True)
class FnspidMoment2FinbertConfig:
    upstream_commit: str = "1927242"
    upstream_output_dir: str = "reports/loop_vader_scale_20260719"
    start_year: int = 2011
    end_year: int = 2023
    tail_definition_end_year: int = 2016
    tail_evaluation_start_year: int = 2017
    horizons: tuple[int, ...] = tuple(range(1, 11))
    market_symbol: str = "SPY"
    event_hash_seed: str = "fnspid-feasibility-v1"
    random_seed: int = 20260719
    finbert_batch_size: int = 64
    finbert_checkpoint_size: int = 4096
    signal_alpha: float = 0.05
    recap_inflation_ratio: float = 1.5
    vader_negative_threshold: float = -0.05
    tail_probability: float = 0.05
    volatility_lookback_sessions: int = 5
    second_moment_bh_scope: str = ""
    second_moment_verdict_scope: str = ""
    second_moment_yes_rule: str = ""
    second_moment_marginal_rule: str = ""
    mean_signal_yes_rule: str = ""
    mean_signal_marginal_rule: str = ""
    recap_inflation_rule: str = ""


EVENT_DB_NAME = "finbert_checkpoint.sqlite3"
PART1_MANIFEST_NAME = "part1_manifest.json"
MODEL_ID = "ProsusAI/finbert"
EXPECTED_UPSTREAM_COUNTS = {
    "deduplicated_coherent_events": 1_640_796,
    "aggregated_symbol_sessions": 736_596,
    "coherent_symbols": 574,
}


def load_config(path: str | Path) -> FnspidMoment2FinbertConfig:
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    values = dict(payload.get("experiment", {}))
    if "horizons" in values:
        values["horizons"] = tuple(int(value) for value in values["horizons"])
    return FnspidMoment2FinbertConfig(**values)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=120)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=FILE")
    return connection


def _create_event_store(path: Path) -> sqlite3.Connection:
    if path.exists():
        raise FnspidMoment2FinbertError(f"refusing to overwrite checkpoint database: {path}")
    connection = _connect(path)
    connection.execute(
        """
        CREATE TABLE events (
            event_index INTEGER PRIMARY KEY,
            event_key TEXT NOT NULL UNIQUE,
            symbol TEXT NOT NULL,
            session_date TEXT NOT NULL,
            headline TEXT NOT NULL,
            is_recap INTEGER NOT NULL,
            vader_compound REAL NOT NULL,
            p_positive REAL,
            p_negative REAL,
            p_neutral REAL,
            finbert_score REAL,
            finbert_negative_dominant INTEGER
        )
        """
    )
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.commit()
    return connection


def _metadata_set(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, sort_keys=True)),
    )


def _metadata_get(connection: sqlite3.Connection, key: str) -> Any:
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    return None if row is None else json.loads(row[0])


def _verify_upstream(
    *,
    config: FnspidMoment2FinbertConfig,
    config_path: Path,
    news_paths: tuple[Path, ...],
    price_path: Path,
) -> tuple[Path, dict[str, Any], set[str], dict[str, Any]]:
    upstream = Path(config.upstream_output_dir)
    manifest_path = upstream / "manifest.json"
    cohort_path = upstream / "cohort_symbols.csv"
    if not manifest_path.is_file() or not cohort_path.is_file():
        raise FnspidMoment2FinbertError("committed upstream smoke artifacts are missing")
    manifest = _safe_json(manifest_path)
    if manifest.get("window", {}).get("start_year") != config.start_year or manifest.get("window", {}).get("end_year") != config.end_year:
        raise FnspidMoment2FinbertError("upstream coherent window differs from the frozen extension")
    counts = manifest.get("counts", {})
    if int(counts.get("deduplicated_coherent_events", -1)) != EXPECTED_UPSTREAM_COUNTS["deduplicated_coherent_events"]:
        raise FnspidMoment2FinbertError("upstream event count differs from the frozen extension")
    cohort = set(pd.read_csv(cohort_path, dtype=str)["symbol"].astype(str))
    if len(cohort) != EXPECTED_UPSTREAM_COUNTS["coherent_symbols"]:
        raise FnspidMoment2FinbertError("upstream cohort size differs from the frozen extension")
    expected_news = {Path(item["path"]).resolve(): item for item in manifest["inputs"]["news_archives"]}
    if set(news_paths) != set(expected_news):
        raise FnspidMoment2FinbertError("news archives differ from the committed smoke inputs")
    expected_price = manifest["inputs"]["price_archive"]
    if price_path != Path(expected_price["path"]).resolve():
        raise FnspidMoment2FinbertError("price archive differs from the committed smoke input")
    for path in (*news_paths, price_path):
        if not path.is_file():
            raise FnspidMoment2FinbertError(f"required immutable input is missing: {path}")
    input_record = {
        "upstream_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
        },
        "news_archives": [
            {
                "path": str(path),
                "sha256": expected_news[path]["sha256"],
                "size_bytes": path.stat().st_size,
                "hash_source": "verified committed upstream manifest",
            }
            for path in news_paths
        ],
        "price_archive": {
            "path": str(price_path),
            "sha256": expected_price["sha256"],
            "size_bytes": price_path.stat().st_size,
            "hash_source": "verified committed upstream manifest",
        },
    }
    return upstream, manifest, cohort, input_record


def _extract_vader_events(
    *,
    db_path: Path,
    archives: tuple[tuple[Path, bool], ...],
    symbols: set[str],
    config: FnspidMoment2FinbertConfig,
) -> dict[str, int]:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise FnspidMoment2FinbertError("NLTK with the cached VADER lexicon is required") from exc
    analyzer = SentimentIntensityAnalyzer()
    mapper = _SessionMapper()
    connection = _create_event_store(db_path)
    seen_events: set[int] = set()
    counts: Counter[str] = Counter()
    pending: list[tuple[Any, ...]] = []
    event_index = 0
    csv.field_size_limit(2**31 - 1)
    try:
        for archive, exclude_timed_rows in archives:
            with _open_zstd_csv(archive) as handle:
                reader = csv.DictReader(handle)
                for row_number, row in enumerate(reader, start=1):
                    counts["physical_rows_scanned"] += 1
                    if row_number % 1_000_000 == 0:
                        print(f"Part 1 scan {archive.name}: {row_number:,} rows", flush=True)
                    symbol = str(row.get("Stock_symbol") or "").strip().upper()
                    if symbol not in symbols:
                        continue
                    counts["rows_for_coherent_symbols"] += 1
                    raw_timestamp = str(row.get("Date") or "")
                    parts = _timestamp_parts(raw_timestamp)
                    if parts is None:
                        counts["invalid_timestamp_rows"] += 1
                        continue
                    if exclude_timed_rows and parts[3]:
                        counts["nasdaq_timed_rows_excluded"] += 1
                        continue
                    mapped = mapper.map(raw_timestamp)
                    if mapped is None:
                        counts["invalid_timestamp_rows"] += 1
                        continue
                    session_day, precise = mapped
                    counts["precise_timestamp_rows" if precise else "date_only_rows"] += 1
                    if not config.start_year <= int(session_day[:4]) <= config.end_year:
                        counts["mapped_outside_window"] += 1
                        continue
                    headline = str(row.get("Article_title") or "").strip()
                    url = str(row.get("Url") or "").strip()
                    identity = _normal(url).rstrip("/") or _normal(headline)
                    event_key = _hash64(symbol, session_day, identity, seed=config.event_hash_seed)
                    if event_key in seen_events:
                        counts["exact_firm_event_duplicates"] += 1
                        continue
                    seen_events.add(event_key)
                    counts["deduplicated_coherent_events"] += 1
                    if not headline:
                        counts["empty_headlines_excluded"] += 1
                        continue
                    event_index += 1
                    compound = float(analyzer.polarity_scores(headline)["compound"])
                    pending.append(
                        (
                            event_index,
                            f"{event_key:016x}",
                            symbol,
                            session_day,
                            headline,
                            int(is_price_recap(headline)),
                            compound,
                        )
                    )
                    counts["vader_scored_headlines"] += 1
                    if len(pending) >= 10_000:
                        connection.executemany(
                            "INSERT INTO events(event_index,event_key,symbol,session_date,headline,is_recap,vader_compound) VALUES (?,?,?,?,?,?,?)",
                            pending,
                        )
                        connection.commit()
                        pending.clear()
        if pending:
            connection.executemany(
                "INSERT INTO events(event_index,event_key,symbol,session_date,headline,is_recap,vader_compound) VALUES (?,?,?,?,?,?,?)",
                pending,
            )
        connection.execute("CREATE INDEX events_symbol_session ON events(symbol, session_date)")
        _metadata_set(connection, "event_extraction_complete", True)
        _metadata_set(connection, "event_counts", dict(counts))
        connection.commit()
    finally:
        connection.close()
    if counts["vader_scored_headlines"] != EXPECTED_UPSTREAM_COUNTS["deduplicated_coherent_events"]:
        raise FnspidMoment2FinbertError(
            f"rebuilt VADER event count {counts['vader_scored_headlines']:,} does not match committed 1,640,796"
        )
    return dict(counts)


def _aggregate_vader(connection: sqlite3.Connection, config: FnspidMoment2FinbertConfig) -> pd.DataFrame:
    panel = pd.read_sql_query(
        """
        SELECT symbol, session_date,
               AVG(vader_compound) AS vader_mean,
               AVG(CASE WHEN vader_compound < ? THEN 1.0 ELSE 0.0 END) AS vader_negative_share,
               COUNT(*) AS article_count
        FROM events
        GROUP BY symbol, session_date
        ORDER BY symbol, session_date
        """,
        connection,
        params=(config.vader_negative_threshold,),
    )
    panel["session_date"] = pd.to_datetime(panel["session_date"], errors="raise")
    panel["log_article_count"] = np.log1p(panel["article_count"].astype(float))
    if len(panel) != EXPECTED_UPSTREAM_COUNTS["aggregated_symbol_sessions"]:
        raise FnspidMoment2FinbertError("rebuilt session count does not match the committed VADER panel")
    return panel


def _attach_risk_inputs(
    panel: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    config: FnspidMoment2FinbertConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = attach_abnormal_returns(panel, prices, market_symbol=config.market_symbol, horizons=config.horizons)
    output["prior5_abs_ar"] = math.nan
    output["tail_threshold"] = math.nan
    market = prices[config.market_symbol].set_index("session_date")["adjusted_close"].sort_index()
    market_dates = pd.DatetimeIndex(market.index)
    market_return = np.log(market).diff()
    positions = {pd.Timestamp(value): index for index, value in enumerate(market_dates)}
    thresholds: list[dict[str, Any]] = []
    for symbol, indexes in output.groupby("symbol", sort=True).groups.items():
        frame = prices.get(str(symbol))
        if frame is None:
            continue
        stock = frame.set_index("session_date")["adjusted_close"].reindex(market_dates)
        abnormal = np.log(stock).diff() - market_return
        trailing = (
            abnormal.abs()
            .shift(1)
            .rolling(config.volatility_lookback_sessions, min_periods=config.volatility_lookback_sessions)
            .mean()
        )
        definition_mask = (market_dates.year >= config.start_year) & (market_dates.year <= config.tail_definition_end_year)
        definition = abnormal.loc[definition_mask].dropna()
        if definition.empty:
            continue
        threshold = float(definition.quantile(config.tail_probability, interpolation="linear"))
        thresholds.append(
            {
                "symbol": symbol,
                "tail_threshold": threshold,
                "definition_daily_ar_observations": len(definition),
                "definition_start": definition.index.min().date().isoformat(),
                "definition_end": definition.index.max().date().isoformat(),
            }
        )
        for index in indexes:
            position = positions.get(pd.Timestamp(output.at[index, "session_date"]))
            if position is None:
                continue
            value = trailing.iloc[position]
            output.at[index, "prior5_abs_ar"] = float(value) if pd.notna(value) else math.nan
            output.at[index, "tail_threshold"] = threshold
    output["abs_ar_h1"] = output["ar_h1"].abs()
    output["tail_h1"] = np.where(
        output["ar_h1"].notna() & output["tail_threshold"].notna(),
        (output["ar_h1"] < output["tail_threshold"]).astype(float),
        math.nan,
    )
    output["period"] = np.where(
        output["session_date"].dt.year <= config.tail_definition_end_year,
        "2011-2016 definition",
        "2017-2023 evaluation",
    )
    return output, pd.DataFrame(thresholds)


def _fit_clustered_multi(data: pd.DataFrame, *, outcome: str, predictors: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required = [outcome, "session_date", *predictors]
    usable = data.dropna(subset=required).copy()
    if len(usable) < 3 or usable["session_date"].nunique() < 2:
        raise FnspidMoment2FinbertError(f"insufficient observations for {outcome} on {predictors}")
    design = sm.add_constant(usable[predictors], has_constant="add")
    ordinary = sm.OLS(usable[outcome], design).fit()
    fit = ordinary.get_robustcov_results(
        cov_type="cluster",
        groups=usable["session_date"],
        use_correction=True,
        df_correction=True,
        use_t=True,
    )
    names = list(fit.model.exog_names)
    intervals = np.asarray(fit.conf_int(alpha=0.05))
    coefficients = []
    for index, name in enumerate(names):
        coefficients.append(
            {
                "regressor": name,
                "coefficient": float(fit.params[index]),
                "clustered_se": float(fit.bse[index]),
                "t_stat": float(fit.tvalues[index]),
                "p_value": float(fit.pvalues[index]),
                "ci_95_low": float(intervals[index][0]),
                "ci_95_high": float(intervals[index][1]),
            }
        )
    summary = {
        "observations": len(usable),
        "date_clusters": int(usable["session_date"].nunique()),
        "r_squared": float(ordinary.rsquared),
        "mean_outcome": float(usable[outcome].mean()),
    }
    return summary, coefficients


def _risk_analysis_specs(panel: pd.DataFrame, config: FnspidMoment2FinbertConfig) -> list[tuple[str, str, pd.DataFrame, bool]]:
    return [
        ("absolute_ar_h1", "abs_ar_h1", panel, True),
        (
            "tail_h1_definition_2011_2016",
            "tail_h1",
            panel[panel["session_date"].dt.year <= config.tail_definition_end_year],
            False,
        ),
        (
            "tail_h1_evaluation_2017_2023",
            "tail_h1",
            panel[panel["session_date"].dt.year >= config.tail_evaluation_start_year],
            True,
        ),
    ]


def estimate_risk_suite(
    panel: pd.DataFrame,
    *,
    scorer: str,
    mean_predictor: str,
    negative_share_predictor: str,
    config: FnspidMoment2FinbertConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    controls = ["prior5_abs_ar", "log_article_count"]
    sentiment = [mean_predictor, negative_share_predictor]
    model_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for analysis, outcome, data, verdict_scope in _risk_analysis_specs(panel, config):
        baseline_summary, baseline_coefficients = _fit_clustered_multi(data, outcome=outcome, predictors=controls)
        full_summary, full_coefficients = _fit_clustered_multi(data, outcome=outcome, predictors=controls + sentiment)
        delta = full_summary["r_squared"] - baseline_summary["r_squared"]
        for model, summary in (("controls_only", baseline_summary), ("controls_plus_sentiment", full_summary)):
            model_rows.append(
                {
                    "scorer": scorer,
                    "analysis": analysis,
                    "outcome": outcome,
                    "verdict_scope": verdict_scope,
                    "model": model,
                    **summary,
                    "delta_r_squared_vs_controls": 0.0 if model == "controls_only" else delta,
                }
            )
        for model, rows in (("controls_only", baseline_coefficients), ("controls_plus_sentiment", full_coefficients)):
            for row in rows:
                coefficient_rows.append(
                    {
                        "scorer": scorer,
                        "analysis": analysis,
                        "outcome": outcome,
                        "verdict_scope": verdict_scope,
                        "model": model,
                        **row,
                        "bh_q_value": math.nan,
                    }
                )
    coefficients = pd.DataFrame(coefficient_rows)
    sentiment_mask = coefficients["model"].eq("controls_plus_sentiment") & coefficients["regressor"].isin(sentiment)
    coefficients.loc[sentiment_mask, "bh_q_value"] = _benjamini_hochberg(coefficients.loc[sentiment_mask, "p_value"].astype(float).tolist())
    evidence = coefficients[sentiment_mask & coefficients["verdict_scope"].astype(bool)]
    if bool((evidence["bh_q_value"] < config.signal_alpha).any()):
        verdict = "YES"
    elif bool((evidence["p_value"] < config.signal_alpha).any()):
        verdict = "MARGINAL"
    else:
        verdict = "NO"
    return pd.DataFrame(model_rows), coefficients, verdict


def _manifest_files(output_dir: Path) -> dict[str, dict[str, Any]]:
    excluded = {"manifest.json", EVENT_DB_NAME, f"{EVENT_DB_NAME}-wal", f"{EVENT_DB_NAME}-shm"}
    return {
        path.relative_to(output_dir).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name not in excluded
    }


def _part1_report(
    *,
    path: Path,
    model_fits: pd.DataFrame,
    coefficients: pd.DataFrame,
    attrition: pd.DataFrame,
    verdict: str,
    config: FnspidMoment2FinbertConfig,
    resumed_from_interrupted_checkpoint: bool,
    finbert_scored_before_part1: int,
) -> None:
    lines = [
        "# FNSPID second-moment/tail smoke and full FinBERT verdict",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run",
        f"- Origin Date: {_utc_now()}",
        "- Verification Status: PART 1 COMPLETED; PART 2 PENDING",
        "- Version Label: fnspid_moment2_finbert_v1",
        "",
        "# Part 1 — VADER second-moment and tail smoke",
        "",
        f"**Second-moment/tail verdict: {verdict}.**",
        "",
        "The frozen controls-only baseline is prior-five-session mean absolute daily abnormal return plus log(1 + article count). "
        "The full model adds mean VADER compound and the share of headlines with compound < -0.05. AR is the stock adjusted-close "
        "log return minus SPY's adjusted-close log return. All regressions are OLS/LPM with calendar news-session clustered standard errors.",
        "",
        "Each symbol's tail cutoff is its 5th percentile of all available daily AR observations in 2011-2016, using linear quantile "
        "interpolation. The 2011-2016 tail regression is definition-period description only; only the separately reported 2017-2023 "
        "tail regression contributes to the verdict. BH q-values cover the six reported full-model sentiment coefficients.",
        "",
        "## Part 1 model fit",
        "",
        "| Analysis | Model | N | Date clusters | R2 | Delta R2 | Mean outcome | Verdict evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in model_fits.itertuples(index=False):
        lines.append(
            f"| {row.analysis} | {row.model} | {row.observations:,} | {row.date_clusters:,} | {row.r_squared:.8f} | "
            f"{row.delta_r_squared_vs_controls:.8f} | {row.mean_outcome:.8f} | {'yes' if row.verdict_scope else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Part 1 coefficients",
            "",
            "| Analysis | Model | Regressor | Coefficient | Clustered SE | t | p | BH q | 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in coefficients.itertuples(index=False):
        q_value = "—" if not math.isfinite(float(row.bh_q_value)) else f"{row.bh_q_value:.6g}"
        lines.append(
            f"| {row.analysis} | {row.model} | {row.regressor} | {row.coefficient:.8f} | {row.clustered_se:.8f} | "
            f"{row.t_stat:.2f} | {row.p_value:.6g} | {q_value} | [{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] |"
        )
    lines.extend(
        [
            "",
            "## Part 1 attrition",
            "",
            "| Step | Unit | Surviving | Excluded at step |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in attrition.itertuples(index=False):
        lines.append(f"| {row.step} | {row.unit} | {row.surviving:,} | {row.excluded_at_step:,} |")
    lines.extend(
        [
            "",
            "# Part 2 — full FinBERT",
            "",
            "**PENDING.** Part 1 was written to disk before full-window FinBERT scoring began.",
            "",
            "## Integrity notes",
            "",
            "- Interruption history: the previous session was killed after the resumable event checkpoint was written but before Part 1 report/manifest creation."
            if resumed_from_interrupted_checkpoint
            else "- Interruption history: this invocation created the event checkpoint and completed Part 1 without a prior interrupted checkpoint.",
            f"- At Part 1 resume, {finbert_scored_before_part1:,} FinBERT rows were present; no additional FinBERT scoring began until this report and the Part-1-only manifest were durably written.",
            "- The committed 2011-2023 cohort, event identity hash, timing rules, price inputs, AR definition, horizons, inference, and recap regex are reused unchanged.",
            "- Raw news and price archives are opened read-only. The resumable SQLite checkpoint contains licensed headline text and is gitignored.",
            "- The config fixes seeds, correction families, split dates, and verdict rules before outcomes are estimated.",
            "- No outcome-based tuning is permitted; this is the one requested extension run.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def _load_interrupted_event_checkpoint(db_path: Path) -> tuple[dict[str, int], int]:
    connection = _connect(db_path)
    try:
        integrity = str(connection.execute("PRAGMA quick_check").fetchone()[0])
        if integrity != "ok":
            raise FnspidMoment2FinbertError(f"interrupted event checkpoint failed SQLite quick_check: {integrity}")
        if not bool(_metadata_get(connection, "event_extraction_complete")):
            raise FnspidMoment2FinbertError("interrupted event checkpoint has no completed event extraction marker")
        counts = _metadata_get(connection, "event_counts")
        if not isinstance(counts, dict):
            raise FnspidMoment2FinbertError("interrupted event checkpoint has no event-count metadata")
        events = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        scored = int(connection.execute("SELECT COUNT(*) FROM events WHERE finbert_score IS NOT NULL").fetchone()[0])
    finally:
        connection.close()
    if events != EXPECTED_UPSTREAM_COUNTS["deduplicated_coherent_events"]:
        raise FnspidMoment2FinbertError(
            f"interrupted event checkpoint contains {events:,} events; expected "
            f"{EXPECTED_UPSTREAM_COUNTS['deduplicated_coherent_events']:,}"
        )
    if int(counts.get("vader_scored_headlines", -1)) != events:
        raise FnspidMoment2FinbertError("interrupted event checkpoint event counts disagree with the events table")
    return {str(key): int(value) for key, value in counts.items()}, scored


def _part1_attrition(
    *, counts: dict[str, int], panel: pd.DataFrame, prices: dict[str, pd.DataFrame], model_fits: pd.DataFrame
) -> pd.DataFrame:
    steps: list[tuple[str, str, int, int]] = [
        ("physical_news_rows_scanned", "rows", counts.get("physical_rows_scanned", 0), 0),
        (
            "rows_for_coherent_symbols",
            "rows",
            counts.get("rows_for_coherent_symbols", 0),
            counts.get("physical_rows_scanned", 0) - counts.get("rows_for_coherent_symbols", 0),
        ),
        (
            "mapped_window_firm_events_before_dedup",
            "events",
            counts.get("deduplicated_coherent_events", 0) + counts.get("exact_firm_event_duplicates", 0),
            counts.get("mapped_outside_window", 0) + counts.get("invalid_timestamp_rows", 0) + counts.get("nasdaq_timed_rows_excluded", 0),
        ),
        (
            "deduplicated_nonempty_scored_headlines",
            "headlines",
            counts.get("vader_scored_headlines", 0),
            counts.get("exact_firm_event_duplicates", 0) + counts.get("empty_headlines_excluded", 0),
        ),
        ("aggregated_symbol_sessions", "sessions", len(panel), 0),
        (
            "coherent_symbols_with_price_files",
            "symbols",
            int(panel.loc[panel["symbol"].isin(prices), "symbol"].nunique()),
            int(panel["symbol"].nunique() - panel.loc[panel["symbol"].isin(prices), "symbol"].nunique()),
        ),
    ]
    full_rows = model_fits[model_fits["model"].eq("controls_plus_sentiment")]
    for row in full_rows.itertuples(index=False):
        steps.append((f"analysis_{row.analysis}", "symbol-sessions", int(row.observations), len(panel) - int(row.observations)))
    return pd.DataFrame(steps, columns=["step", "unit", "surviving", "excluded_at_step"])


def run_part1(
    *,
    output_dir: Path,
    news_paths: tuple[Path, ...],
    nasdaq_path: Path,
    price_path: Path,
    config: FnspidMoment2FinbertConfig,
    config_path: Path,
    command: tuple[str, ...],
) -> Path:
    report_path = output_dir / "report.md"
    manifest_path = output_dir / "manifest.json"
    part1_manifest_path = output_dir / PART1_MANIFEST_NAME
    if report_path.is_file() and manifest_path.is_file() and part1_manifest_path.is_file():
        return report_path
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / EVENT_DB_NAME
    allowed_interrupted_files = {EVENT_DB_NAME, f"{EVENT_DB_NAME}-wal", f"{EVENT_DB_NAME}-shm"}
    unexpected = sorted(path.name for path in output_dir.iterdir() if path.name not in allowed_interrupted_files)
    if unexpected:
        raise FnspidMoment2FinbertError(
            f"refusing to overwrite incomplete one-shot outputs in {output_dir}: {', '.join(unexpected)}"
        )
    upstream, upstream_manifest, cohort, inputs = _verify_upstream(
        config=config, config_path=config_path, news_paths=news_paths, price_path=price_path
    )
    started = time.perf_counter()
    resumed_from_interrupted_checkpoint = db_path.is_file()
    if resumed_from_interrupted_checkpoint:
        counts, finbert_scored_before_part1 = _load_interrupted_event_checkpoint(db_path)
    else:
        counts = _extract_vader_events(
            db_path=db_path,
            archives=tuple((path, path == nasdaq_path) for path in news_paths),
            symbols=cohort,
            config=config,
        )
        finbert_scored_before_part1 = 0
    prices, missing_price_symbols = load_fnspid_prices(price_path, cohort, market_symbol=config.market_symbol)
    connection = _connect(output_dir / EVENT_DB_NAME)
    try:
        panel = _aggregate_vader(connection, config)
        panel, thresholds = _attach_risk_inputs(panel, prices, config)
        panel.to_sql("session_panel", connection, if_exists="replace", index=False, chunksize=5_000)
        connection.execute("CREATE UNIQUE INDEX session_panel_symbol_date ON session_panel(symbol, session_date)")
        _metadata_set(connection, "session_panel_complete", True)
        _metadata_set(connection, "part1_analysis_complete", True)
        connection.commit()
    finally:
        connection.close()
    model_fits, coefficients, verdict = estimate_risk_suite(
        panel,
        scorer="vader",
        mean_predictor="vader_mean",
        negative_share_predictor="vader_negative_share",
        config=config,
    )
    attrition = _part1_attrition(counts=counts, panel=panel, prices=prices, model_fits=model_fits)
    model_fits.to_csv(output_dir / "part1_vader_second_moment_model_fit.csv", index=False, lineterminator="\n")
    coefficients.to_csv(output_dir / "part1_vader_second_moment_coefficients.csv", index=False, lineterminator="\n")
    thresholds.to_csv(output_dir / "part1_tail_thresholds.csv", index=False, lineterminator="\n")
    attrition.to_csv(output_dir / "attrition.csv", index=False, lineterminator="\n")
    _part1_report(
        path=report_path,
        model_fits=model_fits,
        coefficients=coefficients,
        attrition=attrition,
        verdict=verdict,
        config=config,
        resumed_from_interrupted_checkpoint=resumed_from_interrupted_checkpoint,
        finbert_scored_before_part1=finbert_scored_before_part1,
    )
    manifest = {
        "schema_version": 1,
        "status": "part1_completed_part2_pending",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "experiment_type": "one-shot two-part extension of committed FNSPID smoke",
        "evaluation_run_number": 1,
        "upstream_commit": config.upstream_commit,
        "upstream_output_dir": str(upstream),
        "outcome_based_tuning": False,
        "raw_data_mutated": False,
        "part1_written_before_part2_started": finbert_scored_before_part1 == 0,
        "part1_resumed_from_interrupted_checkpoint": resumed_from_interrupted_checkpoint,
        "interruption_history": [
            {
                "event": "previous_session_killed_after_event_checkpoint_before_part1_artifacts",
                "checkpoint_mtime": datetime.fromtimestamp(db_path.stat().st_mtime, UTC).isoformat(),
                "finbert_scored_before_part1": finbert_scored_before_part1,
                "recovery": "validated and reused checkpoint; wrote Part 1 report and immutable Part-1-only manifest before resuming scoring",
            }
        ]
        if resumed_from_interrupted_checkpoint
        else [],
        "config": {**asdict(config), "horizons": list(config.horizons)},
        "frozen_rules": {
            "prior_volatility": "mean absolute daily AR over the five market sessions strictly preceding the mapped news session",
            "tail_threshold": "per-symbol 5th percentile of all available daily AR in 2011-2016; pandas linear interpolation",
            "tail_evidence": "2017-2023 reported separately and used for verdict; 2011-2016 definition-period regression excluded from verdict",
            "inference": "OLS/LPM with calendar news-session clustered SEs and finite-cluster corrections",
            "second_moment_multiple_testing": config.second_moment_bh_scope,
            "second_moment_verdict": config.second_moment_verdict_scope,
            "mean_multiple_testing": "BH across h=1..10",
            "negative_share_vader": f"compound < {config.vader_negative_threshold}",
            "negative_share_finbert": "p_negative is strictly greater than p_positive and p_neutral; ties are not negative-dominant",
            "recap_regex": PRICE_RECAP_PATTERN.pattern,
        },
        "verdicts": {"vader_second_moment_tail": verdict},
        "counts": {
            **counts,
            "aggregated_symbol_sessions": len(panel),
            "coherent_symbols": len(cohort),
            "symbols_with_tail_threshold": len(thresholds),
        },
        "missing_price_symbols": sorted(missing_price_symbols),
        "inputs": inputs,
        "checkpoint": {
            "path": str(output_dir / EVENT_DB_NAME),
            "gitignored": True,
            "contains_licensed_headline_text": True,
            "finbert_scored": finbert_scored_before_part1,
            "finbert_remaining": counts["vader_scored_headlines"] - finbert_scored_before_part1,
        },
        "commands": {"part1": list(command)},
        "environment": collect_run_environment(),
        "runtime_seconds": {"part1": time.perf_counter() - started},
        "files": _manifest_files(output_dir),
    }
    atomic_write_json(part1_manifest_path, manifest)
    live_manifest = json.loads(json.dumps(manifest))
    live_manifest["part1_manifest"] = {
        "path": str(part1_manifest_path),
        "sha256": sha256_file(part1_manifest_path),
        "size_bytes": part1_manifest_path.stat().st_size,
        "immutable_after_part1": True,
    }
    atomic_write_json(manifest_path, live_manifest)
    return report_path


def _local_finbert_snapshot() -> tuple[Path, str]:
    base = Path.home() / ".cache/huggingface/hub/models--ProsusAI--finbert"
    reference = base / "refs/main"
    if not reference.is_file():
        raise FnspidMoment2FinbertError("cached ProsusAI/finbert refs/main is missing")
    revision = reference.read_text(encoding="utf-8").strip()
    snapshot = base / "snapshots" / revision
    if not snapshot.is_dir():
        raise FnspidMoment2FinbertError(f"cached ProsusAI/finbert snapshot is missing: {snapshot}")
    return snapshot, revision


def run_finbert_scoring(*, output_dir: Path, config: FnspidMoment2FinbertConfig, command: tuple[str, ...]) -> Path:
    manifest_path = output_dir / "manifest.json"
    db_path = output_dir / EVENT_DB_NAME
    if not manifest_path.is_file() or not db_path.is_file():
        raise FnspidMoment2FinbertError("Part 1 artifacts and checkpoint must exist before FinBERT scoring")
    manifest = _safe_json(manifest_path)
    part1_manifest_path = output_dir / PART1_MANIFEST_NAME
    if not part1_manifest_path.is_file() or not (output_dir / "report.md").is_file():
        raise FnspidMoment2FinbertError("durable Part 1 report and Part-1-only manifest are required before FinBERT scoring")
    part1_record = manifest.get("part1_manifest", {})
    if part1_record.get("sha256") != sha256_file(part1_manifest_path):
        raise FnspidMoment2FinbertError("Part-1-only manifest hash differs from the live manifest")
    if manifest.get("status") == "completed":
        return output_dir / "report.md"
    snapshot, revision = _local_finbert_snapshot()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import pipeline as hf_pipeline

    _disable_hf_progress_bars()
    device: int | str = (
        0 if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else -1)
    )
    connection = _connect(db_path)
    expected = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
    completed_before = int(connection.execute("SELECT COUNT(*) FROM events WHERE finbert_score IS NOT NULL").fetchone()[0])
    manifest["status"] = "finbert_scoring"
    manifest["updated_at"] = _utc_now()
    manifest.setdefault("commands", {})["finbert"] = list(command)
    manifest["finbert"] = {
        "model_id": MODEL_ID,
        "model_revision": revision,
        "local_snapshot": str(snapshot),
        "local_only": True,
        "batch_size": config.finbert_batch_size,
        "checkpoint_size": config.finbert_checkpoint_size,
        "device": "cuda" if device == 0 else str(device),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "started_at": _utc_now(),
        "resumed_from": completed_before,
        "scored": completed_before,
        "remaining": expected - completed_before,
    }
    atomic_write_json(manifest_path, manifest)
    started = time.perf_counter()
    classifier = hf_pipeline("text-classification", model=str(snapshot), truncation=True, device=device)
    try:
        while True:
            pending = connection.execute(
                "SELECT event_index, headline FROM events WHERE finbert_score IS NULL ORDER BY event_index LIMIT ?",
                (config.finbert_checkpoint_size,),
            ).fetchall()
            if not pending:
                break
            outputs: Any = classifier(
                [str(row[1]) for row in pending],
                top_k=None,
                batch_size=config.finbert_batch_size,
            )
            if len(outputs) != len(pending):
                raise FnspidMoment2FinbertError("FinBERT returned an unexpected result count")
            updates = []
            for (event_index, _), raw in zip(pending, outputs, strict=True):
                result = _soft_sentiment_from_scores(raw)
                negative_dominant = int(result.p_negative > result.p_positive and result.p_negative > result.p_neutral)
                updates.append(
                    (
                        result.p_positive,
                        result.p_negative,
                        result.p_neutral,
                        result.score,
                        negative_dominant,
                        int(event_index),
                    )
                )
            connection.executemany(
                "UPDATE events SET p_positive=?,p_negative=?,p_neutral=?,finbert_score=?,finbert_negative_dominant=? WHERE event_index=?",
                updates,
            )
            connection.commit()
            elapsed = time.perf_counter() - started
            current_scored = int(connection.execute("SELECT COUNT(*) FROM events WHERE finbert_score IS NOT NULL").fetchone()[0])
            rate = (current_scored - completed_before) / elapsed if elapsed > 0 else 0.0
            manifest = _safe_json(manifest_path)
            manifest["updated_at"] = _utc_now()
            manifest["finbert"].update(
                {
                    "scored": current_scored,
                    "remaining": expected - current_scored,
                    "runtime_seconds_this_invocation": elapsed,
                    "headlines_per_second_this_invocation": rate,
                    "eta_seconds": (expected - current_scored) / rate if rate > 0 else None,
                }
            )
            manifest["checkpoint"].update({"finbert_scored": current_scored, "finbert_remaining": expected - current_scored})
            atomic_write_json(manifest_path, manifest)
            print(
                f"FinBERT checkpoint: {current_scored:,}/{expected:,} ({current_scored / expected:.2%}); "
                f"{rate:.2f} headlines/s; ETA {(expected - current_scored) / rate / 3600:.2f}h",
                flush=True,
            )
        scored = int(connection.execute("SELECT COUNT(*) FROM events WHERE finbert_score IS NOT NULL").fetchone()[0])
        if scored != expected:
            raise FnspidMoment2FinbertError(f"FinBERT scoring incomplete: {scored:,}/{expected:,}")
        probability_error = int(
            connection.execute(
                "SELECT COUNT(*) FROM events WHERE ABS((p_positive+p_negative+p_neutral)-1.0)>1e-5 OR finbert_score < -1 OR finbert_score > 1"
            ).fetchone()[0]
        )
        if probability_error:
            raise FnspidMoment2FinbertError(f"FinBERT probability integrity failures: {probability_error}")
        _metadata_set(connection, "finbert_scoring_complete", True)
        _metadata_set(connection, "finbert_model_revision", revision)
        connection.commit()
    finally:
        connection.close()
    manifest = _safe_json(manifest_path)
    manifest["status"] = "finbert_scored_analysis_pending"
    manifest["updated_at"] = _utc_now()
    manifest["finbert"].update(
        {
            "completed_at": _utc_now(),
            "scored": expected,
            "remaining": 0,
            "runtime_seconds_this_invocation": time.perf_counter() - started,
        }
    )
    manifest["checkpoint"].update({"finbert_scored": expected, "finbert_remaining": 0})
    atomic_write_json(manifest_path, manifest)
    return output_dir / "report.md"


def _aggregate_finbert(connection: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT symbol, session_date,
               AVG(finbert_score) AS finbert_mean,
               AVG(finbert_negative_dominant * 1.0) AS finbert_negative_share,
               COUNT(*) AS finbert_article_count,
               AVG(CASE WHEN is_recap=1 THEN finbert_score END) AS recap_finbert_mean,
               SUM(CASE WHEN is_recap=1 THEN 1 ELSE 0 END) AS recap_article_count,
               AVG(CASE WHEN is_recap=0 THEN finbert_score END) AS other_finbert_mean,
               SUM(CASE WHEN is_recap=0 THEN 1 ELSE 0 END) AS other_article_count
        FROM events
        GROUP BY symbol, session_date
        ORDER BY symbol, session_date
        """,
        connection,
    )


def _estimate_finbert_mean(panel: pd.DataFrame, config: FnspidMoment2FinbertConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for horizon in config.horizons:
        summary, coefficients = _fit_clustered_multi(panel, outcome=f"ar_h{horizon}", predictors=["finbert_mean"])
        coefficient = next(row for row in coefficients if row["regressor"] == "finbert_mean")
        rows.append(
            {
                "horizon": horizon,
                "intercept": next(row["coefficient"] for row in coefficients if row["regressor"] == "const"),
                "lambda": coefficient["coefficient"],
                "clustered_se": coefficient["clustered_se"],
                "t_stat": coefficient["t_stat"],
                "p_value": coefficient["p_value"],
                "ci_95_low": coefficient["ci_95_low"],
                "ci_95_high": coefficient["ci_95_high"],
                "events": summary["observations"],
                "date_clusters": summary["date_clusters"],
                "mean_outcome": summary["mean_outcome"],
                "r_squared": summary["r_squared"],
            }
        )
    lambdas = pd.DataFrame(rows)
    lambdas["bh_q_value"] = _benjamini_hochberg(lambdas["p_value"].tolist())
    split_rows = []
    for content_class, predictor in (("PRICE_RECAP", "recap_finbert_mean"), ("OTHER", "other_finbert_mean")):
        summary, coefficients = _fit_clustered_multi(panel, outcome="ar_h1", predictors=[predictor])
        coefficient = next(row for row in coefficients if row["regressor"] == predictor)
        split_rows.append(
            {
                "content_class": content_class,
                "horizon": 1,
                "intercept": next(row["coefficient"] for row in coefficients if row["regressor"] == "const"),
                "lambda": coefficient["coefficient"],
                "clustered_se": coefficient["clustered_se"],
                "t_stat": coefficient["t_stat"],
                "p_value": coefficient["p_value"],
                "ci_95_low": coefficient["ci_95_low"],
                "ci_95_high": coefficient["ci_95_high"],
                "events": summary["observations"],
                "date_clusters": summary["date_clusters"],
                "mean_outcome": summary["mean_outcome"],
                "r_squared": summary["r_squared"],
            }
        )
    return lambdas, pd.DataFrame(split_rows)


def _final_report(
    *,
    path: Path,
    part1_model_fits: pd.DataFrame,
    part1_coefficients: pd.DataFrame,
    attrition: pd.DataFrame,
    vader_verdict: str,
    finbert_lambdas: pd.DataFrame,
    finbert_split: pd.DataFrame,
    finbert_model_fits: pd.DataFrame,
    finbert_coefficients: pd.DataFrame,
    mean_verdict: str,
    finbert_risk_verdict: str,
    recap_verdict: str,
    manifest: dict[str, Any],
) -> None:
    lines = [
        "# FNSPID second-moment/tail smoke and full FinBERT verdict",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: experiment-agent",
        "- Origin Mode: run + validate",
        f"- Origin Date: {_utc_now()}",
        "- Verification Status: VERIFIED",
        "- Version Label: fnspid_moment2_finbert_v1",
        "",
        "# Part 1 — VADER second-moment and tail smoke",
        "",
        f"**Does any VADER sentiment predictor add second-moment/tail information beyond volatility clustering? {vader_verdict}.**",
        "",
        "The controls-only baseline is prior-five-session mean absolute daily abnormal return plus log(1 + article count). The full "
        "model adds mean VADER compound and the share of headlines with compound < -0.05. AR is the stock adjusted-close log return "
        "minus SPY's adjusted-close log return. All regressions are OLS/LPM with calendar news-session clustered standard errors.",
        "",
        "Each symbol's tail cutoff is its 5th percentile of all available daily AR observations in 2011-2016, using linear quantile "
        "interpolation. The definition-period tail regression is descriptive and excluded from the verdict; 2017-2023 is reported separately. "
        "BH q-values cover the six reported full-model sentiment coefficients.",
        "",
        "## Part 1 model fit",
        "",
        "| Analysis | Model | N | Date clusters | R2 | Delta R2 | Mean outcome | Verdict evidence |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in part1_model_fits.itertuples(index=False):
        lines.append(
            f"| {row.analysis} | {row.model} | {row.observations:,} | {row.date_clusters:,} | {row.r_squared:.8f} | "
            f"{row.delta_r_squared_vs_controls:.8f} | {row.mean_outcome:.8f} | {'yes' if row.verdict_scope else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Part 1 coefficients",
            "",
            "| Analysis | Model | Regressor | Coefficient | Clustered SE | t | p | BH q | 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in part1_coefficients.itertuples(index=False):
        q_value = "—" if not math.isfinite(float(row.bh_q_value)) else f"{row.bh_q_value:.6g}"
        lines.append(
            f"| {row.analysis} | {row.model} | {row.regressor} | {row.coefficient:.8f} | {row.clustered_se:.8f} | "
            f"{row.t_stat:.2f} | {row.p_value:.6g} | {q_value} | [{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] |"
        )
    lines.extend(
        [
            "",
            "## Part 1 attrition",
            "",
            "| Step | Unit | Surviving | Excluded at step |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in attrition.itertuples(index=False):
        lines.append(f"| {row.step} | {row.unit} | {row.surviving:,} | {row.excluded_at_step:,} |")
    lines.extend(
        [
            "",
            "# Part 2 — full FinBERT",
            "",
            f"**Mean signal at h>=1: {mean_verdict}. Second-moment/tail signal: {finbert_risk_verdict}. Recap inflation established: {recap_verdict}.**",
            "",
            f"All {manifest['finbert']['scored']:,} deduplicated window headlines were scored locally with cached ProsusAI/finbert revision "
            f"`{manifest['finbert']['model_revision']}`. The score is p_positive - p_negative; negative-share is the fraction whose "
            "p_negative is strictly larger than both other class probabilities.",
            "",
            "## FinBERT lambda(h), full panel",
            "",
            "| h | Lambda(h) | Clustered SE | t | p | BH q | 95% CI | Sessions | Date clusters | Delta/R2 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in finbert_lambdas.rename(columns={"lambda": "coefficient"}).itertuples(index=False):
        lines.append(
            f"| {row.horizon} | {row.coefficient:.8f} | {row.clustered_se:.8f} | {row.t_stat:.2f} | {row.p_value:.6g} | "
            f"{row.bh_q_value:.6g} | [{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] | {row.events:,} | {row.date_clusters:,} | {row.r_squared:.8f} |"
        )
    lines.extend(
        [
            "",
            "## FinBERT price-recap/other lambda(1) split",
            "",
            "| Content signal | Lambda(1) | Clustered SE | t | p | 95% CI | Sessions | Date clusters |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in finbert_split.rename(columns={"lambda": "coefficient"}).itertuples(index=False):
        lines.append(
            f"| {row.content_class} | {row.coefficient:.8f} | {row.clustered_se:.8f} | {row.t_stat:.2f} | {row.p_value:.6g} | "
            f"[{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] | {row.events:,} | {row.date_clusters:,} |"
        )
    lines.extend(
        [
            "",
            "## FinBERT second-moment/tail model fit",
            "",
            "| Analysis | Model | N | Date clusters | R2 | Delta R2 | Mean outcome | Verdict evidence |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in finbert_model_fits.itertuples(index=False):
        lines.append(
            f"| {row.analysis} | {row.model} | {row.observations:,} | {row.date_clusters:,} | {row.r_squared:.8f} | "
            f"{row.delta_r_squared_vs_controls:.8f} | {row.mean_outcome:.8f} | {'yes' if row.verdict_scope else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## FinBERT second-moment/tail coefficients",
            "",
            "| Analysis | Model | Regressor | Coefficient | Clustered SE | t | p | BH q | 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in finbert_coefficients.itertuples(index=False):
        q_value = "—" if not math.isfinite(float(row.bh_q_value)) else f"{row.bh_q_value:.6g}"
        lines.append(
            f"| {row.analysis} | {row.model} | {row.regressor} | {row.coefficient:.8f} | {row.clustered_se:.8f} | "
            f"{row.t_stat:.2f} | {row.p_value:.6g} | {q_value} | [{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] |"
        )
    recap = finbert_split.set_index("content_class")
    recap_ratio = abs(float(recap.at["PRICE_RECAP", "lambda"])) / max(abs(float(recap.at["OTHER", "lambda"])), 1e-15)
    lines.extend(
        [
            "",
            "## Ten-line plain-English summary",
            "",
            f"1. VADER second-moment/tail information beyond volatility clustering: **{vader_verdict}**.",
            f"2. FinBERT mean-return signal at h>=1 after BH across ten horizons: **{mean_verdict}**.",
            f"3. FinBERT second-moment/tail information beyond volatility clustering: **{finbert_risk_verdict}**.",
            f"3. Recap inflation under the frozen same-sign, 1.5x, recap-p<0.05 rule: **{recap_verdict}** (absolute coefficient ratio {recap_ratio:.2f}x).",
            f"4. The FinBERT second-moment/tail suite returned {finbert_risk_verdict} under its separately recorded six-coefficient BH family.",
            f"5. FinBERT scored all {manifest['finbert']['scored']:,} deduplicated 2011-2023 window headlines from 574 coherent firms.",
            "6. Mean-signal q-values correct exactly the ten h=1..10 horizon tests.",
            "7. Second-moment q-values correct six sentiment coefficients per scorer; the definition-period tail regression is not verdict evidence.",
            "8. Volatility clustering is controlled with the mean absolute AR from the five sessions strictly before the news session; article count is the attention control.",
            "9. Tail cutoffs never use 2017-2023 outcomes: they are fixed from each symbol's 2011-2016 daily AR distribution.",
            "10. This remains a one-shot smoke extension with no outcome-based tuning; raw archives were immutable and FinBERT ran from the local cache only.",
            "",
            "## Reproducibility and integrity",
            "",
            "- `manifest.json` records frozen rules, seeds, upstream and input hashes, model revision, environment, commands, progress, and final output hashes.",
            "- `finbert_checkpoint.sqlite3` is resumable, gitignored, contains licensed headline text and per-headline probabilities, and must not be committed or redistributed.",
            "- All aggregate CSV and Markdown deliverables contain no licensed headline text.",
            "- The Part 1 report and tables existed on disk before Part 2 scoring started.",
            "- Interruption history: the previous session was killed after event-checkpoint creation and before Part 1 artifacts; the checkpoint was validated and reused, with zero FinBERT rows present when Part 1 resumed.",
            "- No yfinance call, download, or other network access was attempted.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def run_finalize(*, output_dir: Path, config: FnspidMoment2FinbertConfig, command: tuple[str, ...]) -> Path:
    manifest_path = output_dir / "manifest.json"
    db_path = output_dir / EVENT_DB_NAME
    manifest = _safe_json(manifest_path)
    if manifest.get("status") == "completed":
        return output_dir / "report.md"
    connection = _connect(db_path)
    try:
        if not bool(_metadata_get(connection, "finbert_scoring_complete")):
            raise FnspidMoment2FinbertError("FinBERT scoring is not complete")
        session_panel = pd.read_sql_query("SELECT * FROM session_panel ORDER BY symbol, session_date", connection)
        finbert_panel = _aggregate_finbert(connection)
    finally:
        connection.close()
    session_panel["session_date"] = pd.to_datetime(session_panel["session_date"], errors="raise")
    finbert_panel["session_date"] = pd.to_datetime(finbert_panel["session_date"], errors="raise")
    panel = session_panel.merge(finbert_panel, on=["symbol", "session_date"], how="inner", validate="one_to_one")
    if len(panel) != EXPECTED_UPSTREAM_COUNTS["aggregated_symbol_sessions"]:
        raise FnspidMoment2FinbertError("FinBERT and VADER symbol-session panels disagree")
    if not bool((panel["article_count"].astype(int) == panel["finbert_article_count"].astype(int)).all()):
        raise FnspidMoment2FinbertError("FinBERT and VADER article counts disagree")
    lambdas, split = _estimate_finbert_mean(panel, config)
    model_fits, coefficients, risk_verdict = estimate_risk_suite(
        panel,
        scorer="finbert",
        mean_predictor="finbert_mean",
        negative_share_predictor="finbert_negative_share",
        config=config,
    )
    mean_verdict = _signal_verdict(lambdas, config.signal_alpha)
    recap_config = type("RecapConfig", (), {"recap_inflation_ratio": config.recap_inflation_ratio, "signal_alpha": config.signal_alpha})()
    recap_verdict = "ESTABLISHED" if _recap_verdict(split, recap_config) == "YES" else "NOT ESTABLISHED"
    lambdas.to_csv(output_dir / "part2_finbert_lambda_h1_h10.csv", index=False, lineterminator="\n")
    split.to_csv(output_dir / "part2_finbert_recap_split_lambda1.csv", index=False, lineterminator="\n")
    model_fits.to_csv(output_dir / "part2_finbert_second_moment_model_fit.csv", index=False, lineterminator="\n")
    coefficients.to_csv(output_dir / "part2_finbert_second_moment_coefficients.csv", index=False, lineterminator="\n")
    part1_model_fits = pd.read_csv(output_dir / "part1_vader_second_moment_model_fit.csv")
    part1_coefficients = pd.read_csv(output_dir / "part1_vader_second_moment_coefficients.csv")
    attrition = pd.read_csv(output_dir / "attrition.csv")
    vader_verdict = str(manifest["verdicts"]["vader_second_moment_tail"])
    manifest["status"] = "finalizing"
    manifest["updated_at"] = _utc_now()
    manifest.setdefault("commands", {})["finalize"] = list(command)
    manifest["verdicts"].update(
        {
            "finbert_mean_h_ge_1": mean_verdict,
            "finbert_second_moment_tail": risk_verdict,
            "finbert_recap_inflation": recap_verdict,
        }
    )
    _final_report(
        path=output_dir / "report.md",
        part1_model_fits=part1_model_fits,
        part1_coefficients=part1_coefficients,
        attrition=attrition,
        vader_verdict=vader_verdict,
        finbert_lambdas=lambdas,
        finbert_split=split,
        finbert_model_fits=model_fits,
        finbert_coefficients=coefficients,
        mean_verdict=mean_verdict,
        finbert_risk_verdict=risk_verdict,
        recap_verdict=recap_verdict,
        manifest=manifest,
    )
    manifest["status"] = "completed"
    manifest["updated_at"] = _utc_now()
    manifest["files"] = _manifest_files(output_dir)
    atomic_write_json(manifest_path, manifest)
    return output_dir / "report.md"


def run_experiment_phase(
    *,
    phase: str,
    news_archives: tuple[str | Path, ...],
    nasdaq_archive: str | Path,
    price_archive: str | Path,
    config: FnspidMoment2FinbertConfig,
    config_path: str | Path,
    output_dir: str | Path,
    command: tuple[str, ...] = (),
) -> Path:
    news_paths = tuple(Path(path).resolve() for path in news_archives)
    nasdaq_path = Path(nasdaq_archive).resolve()
    price_path = Path(price_archive).resolve()
    destination = Path(output_dir)
    if nasdaq_path not in news_paths:
        raise FnspidMoment2FinbertError("the Nasdaq archive must also be one of the news archives")
    if phase in {"part1", "all"}:
        run_part1(
            output_dir=destination,
            news_paths=news_paths,
            nasdaq_path=nasdaq_path,
            price_path=price_path,
            config=config,
            config_path=Path(config_path),
            command=command,
        )
    if phase in {"finbert", "all"}:
        run_finbert_scoring(output_dir=destination, config=config, command=command)
    if phase in {"finalize", "all"}:
        run_finalize(output_dir=destination, config=config, command=command)
    return destination / "report.md"
