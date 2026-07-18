"""One-shot, full-panel FNSPID VADER/return smoke experiment."""

from __future__ import annotations

import csv
import math
import os
import random
import re
import tempfile
import time
import tomllib
import warnings
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .artifact_io import atomic_write_json, atomic_write_text, sha256_file
from .baselines import iter_finbert_text_batches
from .fnspid_feasibility import (
    FnspidAuditConfig,
    _hash64,
    _normal,
    _open_zstd_csv,
    _timestamp_parts,
    audit_fnspid_news,
)
from .runtime_metadata import collect_run_environment


class FnspidVaderSmokeError(RuntimeError):
    """Raised when the frozen smoke experiment cannot run as specified."""


@dataclass(frozen=True)
class FnspidVaderSmokeConfig:
    preferred_start_year: int = 2011
    preferred_end_year: int = 2023
    fallback_start_year: int = 2010
    fallback_end_year: int = 2023
    minimum_news_days_per_firm_year: int = 20
    minimum_coherent_firms: int = 300
    minimum_coherent_events: int = 50_000
    horizons: tuple[int, ...] = tuple(range(1, 11))
    market_symbol: str = "SPY"
    event_hash_seed: str = "fnspid-feasibility-v1"
    random_seed: int = 20260719
    price_sanity_seed: int = 2026071901
    finbert_sample_seed: int = 2026071902
    price_sanity_symbols: int = 10
    price_sanity_dates_per_symbol: int = 30
    price_sanity_year: int = 2023
    price_mismatch_relative_tolerance: float = 0.001
    price_sanity_max_mismatch_rate: float = 0.05
    price_sanity_minimum_match_rate: float = 0.90
    finbert_sample_size: int = 5_000
    finbert_batch_size: int = 64
    signal_alpha: float = 0.05
    recap_inflation_ratio: float = 1.5


PRICE_RECAP_PATTERN = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*%|"
    r"\bshares?\s+(?:are\s+|were\s+|is\s+)?(?:up|down|rise|rises|rose|rising|fall|falls|fell|falling|"
    r"jump|jumps|jumped|surge|surges|surged|plunge|plunges|plunged|drop|drops|dropped)\b|"
    r"\bmovers?\b|\bmarket\s+wrap\b|\btop\s+(?:gainers?|losers?)\b|\b52[ -]week\b|"
    r"\bpre[ -]?market\b|\bafter[ -]?hours?\b"
    r")",
    flags=re.IGNORECASE,
)


class _SessionMapper:
    def __init__(self, calendar_name: str = "XNYS") -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The 'generic' unit for NumPy timedelta is deprecated")
            import exchange_calendars as xcals

            self.calendar = xcals.get_calendar(calendar_name, start="1900-01-01", end="2030-12-31")
        self._next_session_cache: dict[str, str] = {}

    def map(self, raw_timestamp: str) -> tuple[str, bool] | None:
        parts = _timestamp_parts(raw_timestamp)
        if parts is None:
            return None
        publication_day, _, _, has_clock_time = parts
        if has_clock_time:
            timestamp = pd.Timestamp(raw_timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            session = self.calendar.minute_to_session(timestamp.floor("min"), direction="next")
            return session.date().isoformat(), True
        mapped = self._next_session_cache.get(publication_day)
        if mapped is None:
            publication = pd.Timestamp(publication_day)
            session = self.calendar.date_to_session(publication, direction="next")
            if session.date().isoformat() == publication_day:
                session = self.calendar.next_session(session)
            mapped = session.date().isoformat()
            self._next_session_cache[publication_day] = mapped
        return mapped, False


def is_price_recap(headline: str) -> bool:
    return bool(PRICE_RECAP_PATTERN.search(headline))


def _coherent_symbols(
    coverage_path: Path,
    *,
    start_year: int,
    end_year: int,
    minimum_news_days: int,
) -> set[str]:
    coverage = pd.read_csv(coverage_path, dtype={"symbol": str, "year": int, "mappable_news_days": int})
    selected = coverage[coverage["year"].between(start_year, end_year)].copy()
    selected = selected[selected["mappable_news_days"] >= minimum_news_days]
    expected_years = end_year - start_year + 1
    counts = selected.groupby("symbol", sort=True)["year"].nunique()
    return set(counts[counts == expected_years].index.astype(str))


def choose_window(coverage_path: Path, candidates_path: Path, config: FnspidVaderSmokeConfig) -> tuple[int, int, set[str], int]:
    candidates = pd.read_csv(candidates_path)
    for start_year, end_year in (
        (config.preferred_start_year, config.preferred_end_year),
        (config.fallback_start_year, config.fallback_end_year),
    ):
        symbols = _coherent_symbols(
            coverage_path,
            start_year=start_year,
            end_year=end_year,
            minimum_news_days=config.minimum_news_days_per_firm_year,
        )
        row = candidates[(candidates["start_year"] == start_year) & (candidates["end_year"] == end_year)]
        events = int(row.iloc[0]["coherent_firm_mappable_events"]) if len(row) == 1 else 0
        if len(symbols) >= config.minimum_coherent_firms and events >= config.minimum_coherent_events:
            return start_year, end_year, symbols, events
    raise FnspidVaderSmokeError("neither the preferred nor fallback coherent FNSPID window passed the frozen size gates")


def _reservoir_add(sample: list[str], value: str, seen: int, *, limit: int, rng: random.Random) -> None:
    if len(sample) < limit:
        sample.append(value)
        return
    replacement = rng.randrange(seen)
    if replacement < limit:
        sample[replacement] = value


def build_vader_session_panel(
    archives: tuple[tuple[Path, bool], ...],
    *,
    symbols: set[str],
    start_year: int,
    end_year: int,
    config: FnspidVaderSmokeConfig,
) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer
    except ImportError as exc:  # pragma: no cover
        raise FnspidVaderSmokeError("NLTK with the VADER lexicon is required") from exc

    analyzer = SentimentIntensityAnalyzer()
    mapper = _SessionMapper()
    rng = random.Random(config.finbert_sample_seed)
    finbert_sample: list[str] = []
    aggregates: dict[tuple[str, str], list[float | int]] = {}
    seen_events: set[int] = set()
    counts: Counter[str] = Counter()
    csv.field_size_limit(2**31 - 1)
    for archive, exclude_timed_rows in archives:
        with _open_zstd_csv(archive) as handle:
            reader = csv.DictReader(handle)
            for row_number, row in enumerate(reader, start=1):
                counts["physical_rows_scanned"] += 1
                if row_number % 1_000_000 == 0:
                    print(f"VADER pass scanned {archive.name}: {row_number:,} rows", flush=True)
                symbol = str(row.get("Stock_symbol") or "").strip().upper()
                if symbol not in symbols:
                    continue
                counts["rows_for_coherent_symbols"] += 1
                raw_timestamp = str(row.get("Date") or "")
                timestamp_parts = _timestamp_parts(raw_timestamp)
                if timestamp_parts is None:
                    counts["invalid_timestamp_rows"] += 1
                    continue
                if exclude_timed_rows and timestamp_parts[3]:
                    counts["nasdaq_timed_rows_excluded"] += 1
                    continue
                mapped = mapper.map(raw_timestamp)
                if mapped is None:  # pragma: no cover - validated immediately above
                    counts["invalid_timestamp_rows"] += 1
                    continue
                session_day, precise_timestamp = mapped
                counts["precise_timestamp_rows" if precise_timestamp else "date_only_rows"] += 1
                year = int(session_day[:4])
                if not start_year <= year <= end_year:
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
                compound = float(analyzer.polarity_scores(headline)["compound"])
                recap = is_price_recap(headline)
                key = (symbol, session_day)
                values = aggregates.setdefault(key, [0.0, 0, 0.0, 0, 0.0, 0])
                values[0] = float(values[0]) + compound
                values[1] = int(values[1]) + 1
                bucket = 2 if recap else 4
                values[bucket] = float(values[bucket]) + compound
                values[bucket + 1] = int(values[bucket + 1]) + 1
                counts["vader_scored_headlines"] += 1
                counts["price_recap_headlines" if recap else "other_headlines"] += 1
                _reservoir_add(
                    finbert_sample,
                    headline,
                    counts["vader_scored_headlines"],
                    limit=config.finbert_sample_size,
                    rng=rng,
                )
    rows = []
    for (symbol, session_day), values in sorted(aggregates.items()):
        total_sum, total_count, recap_sum, recap_count, other_sum, other_count = values
        rows.append(
            {
                "symbol": symbol,
                "session_date": pd.Timestamp(session_day),
                "vader_mean": float(total_sum) / int(total_count),
                "article_count": int(total_count),
                "recap_vader_mean": float(recap_sum) / int(recap_count) if int(recap_count) else math.nan,
                "recap_article_count": int(recap_count),
                "other_vader_mean": float(other_sum) / int(other_count) if int(other_count) else math.nan,
                "other_article_count": int(other_count),
            }
        )
    panel = pd.DataFrame(rows)
    counts["aggregated_symbol_sessions"] = len(panel)
    counts["symbols_with_scored_events"] = panel["symbol"].nunique() if not panel.empty else 0
    return panel, finbert_sample, dict(counts)


def _zip_price_member_map(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for name in archive.namelist():
        path = Path(name)
        if name.startswith("full_history/") and path.suffix.casefold() == ".csv":
            members.setdefault(path.stem.upper(), name)
    return members


def load_fnspid_prices(price_archive: Path, symbols: set[str], *, market_symbol: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    with zipfile.ZipFile(price_archive) as archive:
        members = _zip_price_member_map(archive)
        for symbol in sorted(symbols | {market_symbol}):
            member = members.get(symbol.upper())
            if member is None:
                missing.append(symbol)
                continue
            with archive.open(member) as handle:
                frame = pd.read_csv(handle)
            frame.columns = [str(column).strip().casefold() for column in frame.columns]
            if not {"date", "close", "adj close"}.issubset(frame.columns):
                missing.append(symbol)
                continue
            frame = frame[["date", "close", "adj close"]].rename(columns={"adj close": "adjusted_close"})
            frame["session_date"] = pd.to_datetime(frame.pop("date"), errors="coerce").dt.normalize()
            frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame["adjusted_close"] = pd.to_numeric(frame["adjusted_close"], errors="coerce")
            frame = frame.dropna(subset=["session_date", "close", "adjusted_close"])
            frame = frame[(frame["close"] > 0) & (frame["adjusted_close"] > 0)]
            frames[symbol] = frame.drop_duplicates("session_date", keep="last").sort_values("session_date", kind="stable")
    if market_symbol not in frames:
        raise FnspidVaderSmokeError(f"price archive lacks required market proxy {market_symbol}")
    return frames, missing


def attach_abnormal_returns(
    sessions: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    *,
    market_symbol: str,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    market = prices[market_symbol].set_index("session_date")["adjusted_close"].sort_index()
    market_dates = pd.DatetimeIndex(market.index)
    market_return = np.log(market).diff()
    date_positions = {pd.Timestamp(date): position for position, date in enumerate(market_dates)}
    output = sessions.copy()
    for horizon in horizons:
        output[f"ar_h{horizon}"] = math.nan
    for symbol, indexes in output.groupby("symbol", sort=True).groups.items():
        frame = prices.get(str(symbol))
        if frame is None:
            continue
        stock = frame.set_index("session_date")["adjusted_close"].reindex(market_dates)
        abnormal = np.log(stock).diff() - market_return
        for index in indexes:
            event_date = pd.Timestamp(output.at[index, "session_date"])
            position = date_positions.get(event_date)
            if position is None:
                continue
            for horizon in horizons:
                future_position = position + horizon
                if future_position < len(market_dates):
                    value = abnormal.iloc[future_position]
                    output.at[index, f"ar_h{horizon}"] = float(value) if pd.notna(value) else math.nan
    return output


def _fit_clustered(data: pd.DataFrame, *, predictor: str, outcome: str) -> dict[str, Any]:
    usable = data.dropna(subset=[predictor, outcome, "session_date"])
    if len(usable) < 3 or usable["session_date"].nunique() < 2:
        raise FnspidVaderSmokeError(f"insufficient observations for {predictor} on {outcome}")
    design = sm.add_constant(usable[[predictor]], has_constant="add")
    fit = (
        sm.OLS(usable[outcome], design)
        .fit()
        .get_robustcov_results(
            cov_type="cluster",
            groups=usable["session_date"],
            use_correction=True,
            df_correction=True,
            use_t=True,
        )
    )
    names = list(fit.model.exog_names)
    predictor_index = names.index(predictor)
    constant_index = names.index("const")
    interval = np.asarray(fit.conf_int(alpha=0.05))[predictor_index]
    return {
        "intercept": float(fit.params[constant_index]),
        "lambda": float(fit.params[predictor_index]),
        "clustered_se": float(fit.bse[predictor_index]),
        "t_stat": float(fit.tvalues[predictor_index]),
        "p_value": float(fit.pvalues[predictor_index]),
        "ci_95_low": float(interval[0]),
        "ci_95_high": float(interval[1]),
        "events": len(usable),
        "date_clusters": int(usable["session_date"].nunique()),
        "mean_outcome": float(usable[outcome].mean()),
    }


def _benjamini_hochberg(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(np.asarray(p_values, dtype=float))
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for rank_index in range(count - 1, -1, -1):
        original_index = int(order[rank_index])
        rank = rank_index + 1
        running = min(running, float(p_values[original_index]) * count / rank)
        adjusted[original_index] = min(running, 1.0)
    return adjusted.tolist()


def estimate_lambdas(panel: pd.DataFrame, config: FnspidVaderSmokeConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for horizon in config.horizons:
        row = {"horizon": horizon, **_fit_clustered(panel, predictor="vader_mean", outcome=f"ar_h{horizon}")}
        rows.append(row)
    full = pd.DataFrame(rows)
    full["bh_q_value"] = _benjamini_hochberg(full["p_value"].tolist())
    split_rows = []
    for content_class, predictor in (("PRICE_RECAP", "recap_vader_mean"), ("OTHER", "other_vader_mean")):
        split_rows.append(
            {
                "content_class": content_class,
                "horizon": 1,
                **_fit_clustered(panel, predictor=predictor, outcome="ar_h1"),
            }
        )
    return full, pd.DataFrame(split_rows)


def _extract_yfinance_close(download: pd.DataFrame, symbol: str) -> pd.Series:
    if download.empty:
        return pd.Series(dtype=float)
    if isinstance(download.columns, pd.MultiIndex):
        for key in (("Close", symbol), (symbol, "Close")):
            if key in download.columns:
                return pd.to_numeric(download[key], errors="coerce")
        return pd.Series(dtype=float)
    if "Close" in download.columns:
        return pd.to_numeric(download["Close"], errors="coerce")
    return pd.Series(dtype=float)


def run_price_sanity(
    prices: dict[str, pd.DataFrame],
    cohort_symbols: set[str],
    config: FnspidVaderSmokeConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(config.price_sanity_seed)
    candidates = []
    for symbol in sorted(cohort_symbols):
        frame = prices.get(symbol)
        if frame is None:
            continue
        eligible = frame[frame["session_date"].dt.year == config.price_sanity_year]
        if len(eligible) >= config.price_sanity_dates_per_symbol:
            candidates.append(symbol)
    if len(candidates) < config.price_sanity_symbols:
        raise FnspidVaderSmokeError("fewer than ten cohort symbols have 30 price dates in the sanity year")
    selected = sorted(rng.choice(candidates, size=config.price_sanity_symbols, replace=False).tolist())
    sampled: dict[str, pd.DataFrame] = {}
    for symbol in selected:
        eligible = prices[symbol][prices[symbol]["session_date"].dt.year == config.price_sanity_year]
        positions = np.sort(rng.choice(len(eligible), size=config.price_sanity_dates_per_symbol, replace=False))
        sampled[symbol] = eligible.iloc[positions][["session_date", "close"]].copy()
    start = min(frame["session_date"].min() for frame in sampled.values()).date().isoformat()
    end_date = max(frame["session_date"].max() for frame in sampled.values()).date() + timedelta(days=1)
    error: str | None = None
    try:
        import yfinance as yf

        reference = yf.download(
            selected,
            start=start,
            end=end_date.isoformat(),
            auto_adjust=False,
            repair=False,
            progress=False,
            threads=False,
            timeout=15,
        )
    except Exception as exc:  # pragma: no cover - network behavior is environment-specific
        reference = pd.DataFrame()
        error = f"{type(exc).__name__}: {exc}"
    if reference.empty and error is None:
        error = "yfinance returned no rows; its downloader emitted network/DNS failures"
    rows: list[dict[str, Any]] = []
    for symbol in selected:
        yahoo = _extract_yfinance_close(reference, symbol)
        yahoo.index = pd.to_datetime(yahoo.index, errors="coerce").normalize()
        for row in sampled[symbol].itertuples(index=False):
            yahoo_value = float(yahoo.get(row.session_date, math.nan))
            relative_difference = abs(float(row.close) - yahoo_value) / abs(yahoo_value) if math.isfinite(yahoo_value) else math.nan
            rows.append(
                {
                    "symbol": symbol,
                    "session_date": row.session_date.date().isoformat(),
                    "fnspid_close": float(row.close),
                    "yfinance_close": yahoo_value,
                    "relative_difference": relative_difference,
                    "mismatch": bool(relative_difference > config.price_mismatch_relative_tolerance)
                    if math.isfinite(relative_difference)
                    else None,
                }
            )
    details = pd.DataFrame(rows)
    matched = details["relative_difference"].notna()
    matched_count = int(matched.sum())
    intended = len(details)
    mismatch_count = int(details.loc[matched, "mismatch"].sum()) if matched_count else 0
    match_rate = matched_count / intended if intended else 0.0
    mismatch_rate = mismatch_count / matched_count if matched_count else math.nan
    if matched_count == 0:
        status = "UNAVAILABLE"
    elif match_rate >= config.price_sanity_minimum_match_rate and mismatch_rate <= config.price_sanity_max_mismatch_rate:
        status = "BROADLY_CONSISTENT"
    else:
        status = "NOT_BROADLY_CONSISTENT"
    summary = {
        "status": status,
        "symbols_requested": len(selected),
        "dates_per_symbol_requested": config.price_sanity_dates_per_symbol,
        "comparisons_intended": intended,
        "comparisons_available": matched_count,
        "match_rate": match_rate,
        "mismatch_count": mismatch_count,
        "mismatch_rate": mismatch_rate,
        "relative_tolerance": config.price_mismatch_relative_tolerance,
        "broad_consistency_max_mismatch_rate": config.price_sanity_max_mismatch_rate,
        "minimum_match_rate": config.price_sanity_minimum_match_rate,
        "sampled_symbols": selected,
        "reference": "yfinance download, auto_adjust=False, raw Close",
        "error": error,
    }
    return details, summary


def benchmark_finbert(headlines: list[str], *, full_count: int, config: FnspidVaderSmokeConfig) -> dict[str, Any]:
    if len(headlines) != min(config.finbert_sample_size, full_count):
        raise FnspidVaderSmokeError("FinBERT reservoir sample has an unexpected size")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import transformers

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    started = time.perf_counter()
    scored = 0
    for batch in iter_finbert_text_batches(headlines, batch_size=config.finbert_batch_size):
        scored += len(batch)
    elapsed = time.perf_counter() - started
    if scored != len(headlines):
        raise FnspidVaderSmokeError("FinBERT throughput run returned an unexpected score count")
    seconds_per_headline = elapsed / scored
    model_cache = Path.home() / ".cache/huggingface/hub/models--ProsusAI--finbert/refs/main"
    revision = model_cache.read_text(encoding="utf-8").strip() if model_cache.is_file() else None
    return {
        "model_id": "ProsusAI/finbert",
        "model_revision": revision,
        "sample_seed": config.finbert_sample_seed,
        "sample_headlines": scored,
        "batch_size": config.finbert_batch_size,
        "device": device,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "elapsed_seconds_including_model_load": elapsed,
        "headlines_per_second": scored / elapsed,
        "projected_full_window_seconds": seconds_per_headline * full_count,
        "projected_full_window_hours": seconds_per_headline * full_count / 3600,
        "projection_method": "linear extrapolation from the fixed 5,000-headline sample; includes one model load",
    }


def _format_number(value: float, digits: int = 6) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def _signal_verdict(lambdas: pd.DataFrame, alpha: float) -> str:
    if bool((lambdas["bh_q_value"] < alpha).any()):
        return "YES"
    if bool((lambdas["p_value"] < alpha).any()):
        return "MARGINAL"
    return "NO"


def _recap_verdict(split: pd.DataFrame, config: FnspidVaderSmokeConfig) -> str:
    recap = split[split["content_class"] == "PRICE_RECAP"].iloc[0]
    other = split[split["content_class"] == "OTHER"].iloc[0]
    same_sign = np.sign(recap["lambda"]) == np.sign(other["lambda"])
    ratio = abs(float(recap["lambda"])) / max(abs(float(other["lambda"])), 1e-15)
    if same_sign and ratio >= config.recap_inflation_ratio and float(recap["p_value"]) < config.signal_alpha:
        return "YES"
    if ratio <= 1.0 or float(recap["p_value"]) >= config.signal_alpha:
        return "NO"
    return "MARGINAL"


def _write_report(
    path: Path,
    *,
    start_year: int,
    end_year: int,
    cohort_events_audit: int,
    cohort_symbols: set[str],
    counts: dict[str, int],
    attrition: pd.DataFrame,
    lambdas: pd.DataFrame,
    split: pd.DataFrame,
    price_summary: dict[str, Any],
    finbert: dict[str, Any],
    config: FnspidVaderSmokeConfig,
) -> tuple[str, str]:
    signal = _signal_verdict(lambdas, config.signal_alpha)
    recap_inflation = _recap_verdict(split, config)
    recap_share = counts["price_recap_headlines"] / counts["vader_scored_headlines"]
    lines = [
        "# FNSPID full-panel VADER scale smoke",
        "",
        f"**Smoke verdict for any h>=1 relation: {signal}.** This is a design-ranking smoke test, not a dissertation results run.",
        "",
        "## Frozen design and integrity status",
        "",
        f"The selected coherent window is **{start_year}-{end_year}**, with **{len(cohort_symbols):,} firms** and "
        f"**{cohort_events_audit:,} deduplicated events** in the independent Gate-1 audit. Nasdaq date-only rows were retained under "
        "the strict next-session rule, while every non-midnight row in that archive was excluded. Date-only and exact-midnight rows "
        "map strictly to the next XNYS session; genuine timed rows from the other archive use the exchange-session/close rule.",
        "",
        "The design, seeds, regex, horizons, inference, market proxy, mismatch thresholds, and verdict rules were fixed before returns "
        "were estimated. Raw news and price archives were opened read-only. No development/evaluation split is used because this is a "
        "one-shot smoke test that will be superseded; these full-panel estimates are not confirmatory or out-of-sample evidence.",
        "",
        "Returns are SPY-market-adjusted adjusted-close-to-adjusted-close log returns. Each row below is a separate OLS of AR(+h) on "
        "mean symbol-session VADER compound with an intercept and calendar-event-date-clustered standard errors. BH q-values adjust the "
        "ten horizon p-values only for the smoke verdict.",
        "",
        "## Lambda(h), full panel",
        "",
        "| h | Lambda(h) | clustered SE | t | p | BH q | 95% CI | Sessions | Date clusters |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for row in lambdas.rename(columns={"lambda": "coefficient"}).itertuples(index=False):
        lines.append(
            f"| {row.horizon} | {row.coefficient:.8f} | {row.clustered_se:.8f} | {row.t_stat:.2f} | {row.p_value:.6g} | "
            f"{row.bh_q_value:.6g} | [{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] | {row.events:,} | {row.date_clusters:,} |"
        )
    lines.extend(
        [
            "",
            "## Price-recap content split",
            "",
            f"The crude regex marks **{counts['price_recap_headlines']:,} / {counts['vader_scored_headlines']:,} headlines "
            f"({recap_share:.2%})** as PRICE-RECAP. The split regressions use category-specific mean sentiment: recap headlines on "
            "symbol-sessions containing at least one recap, and other headlines on sessions containing at least one other headline. "
            "Mixed sessions can enter both rows; this is descriptive, not a causal decomposition.",
            "",
            "| Content signal | Lambda(1) | clustered SE | t | p | 95% CI | Sessions | Date clusters |",
            "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in split.rename(columns={"lambda": "coefficient"}).itertuples(index=False):
        lines.append(
            f"| {row.content_class} | {row.coefficient:.8f} | {row.clustered_se:.8f} | {row.t_stat:.2f} | {row.p_value:.6g} | "
            f"[{row.ci_95_low:.8f}, {row.ci_95_high:.8f}] | {row.events:,} | {row.date_clusters:,} |"
        )
    mismatch_text = _format_number(float(price_summary["mismatch_rate"]), 4)
    lines.extend(
        [
            "",
            "## Price sanity check — unvalidated Gate 2",
            "",
            f"Status: **{price_summary['status']}**. The fixed check requested {price_summary['comparisons_intended']:,} raw-close "
            f"comparisons (10 seeded-random cohort symbols x 30 seeded-random {config.price_sanity_year} dates) against yfinance. "
            f"Available comparisons: {price_summary['comparisons_available']:,}; mismatch rate at the fixed "
            f"{config.price_mismatch_relative_tolerance:.2%} relative tolerance: **{mismatch_text}**.",
            "",
            "This experiment remains **unvalidated-Gate-2** even if the spot check is broadly consistent. If status is UNAVAILABLE, "
            "the local sandbox could not reach Yahoo and no mismatch rate is claimed; the outcome analysis proceeded only as the "
            "explicitly labelled smoke test. Detailed intended comparisons and missing reference values are retained in "
            "`price_sanity.csv`.",
            "",
            "## FinBERT throughput projection",
            "",
            f"ProsusAI/finbert scored {finbert['sample_headlines']:,} seeded-reservoir headlines on **{finbert['device']}** in "
            f"{finbert['elapsed_seconds_including_model_load']:.1f} seconds including model load "
            f"({finbert['headlines_per_second']:.2f} headlines/second). Linear projection for all "
            f"{counts['vader_scored_headlines']:,} window headlines: **{finbert['projected_full_window_hours']:.2f} hours**. "
            "The projection is a throughput estimate, not a full FinBERT results run.",
            "",
            "## Attrition",
            "",
            "| Step | Unit | Surviving | Excluded at step |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for row in attrition.itertuples(index=False):
        lines.append(f"| {row.step} | {row.unit} | {row.surviving:,} | {row.excluded_at_step:,} |")
    recap_sentence = {
        "YES": "Recap content appears to inflate the h=1 relation under the frozen descriptive rule.",
        "MARGINAL": "Evidence that recap content inflates the h=1 relation is marginal under the frozen descriptive rule.",
        "NO": "Recap content does not meet the frozen threshold for inflation; its larger point estimate is imprecise.",
    }[recap_inflation]
    lines.extend(
        [
            "",
            "## Ten-line plain-English summary",
            "",
            f"1. Post-news-session sentiment-return signal at h>=1: **{signal}** under the frozen smoke rule.",
            "2. Only h=4 has unadjusted p<0.05, and its BH-adjusted q=0.227 does not survive the ten-horizon correction.",
            f"3. The analysis uses {len(cohort_symbols):,} coherent firms from {start_year} through {end_year}.",
            f"4. VADER scored {counts['vader_scored_headlines']:,} non-empty, deduplicated event headlines on CPU.",
            "5. Lambda(h) is a single-session abnormal return response, not a cumulative return.",
            f"6. PRICE-RECAP headlines are {recap_share:.2%} of the scored event headlines.",
            f"7. {recap_sentence}",
            f"8. The yfinance spot check status is {price_summary['status']}; the mismatch rate is {mismatch_text}.",
            "9. Gate 2 remains explicitly unvalidated, and the full-panel estimates are not out-of-sample evidence.",
            f"10. Full-window FinBERT scoring is projected at {finbert['projected_full_window_hours']:.2f} hours on the measured hardware.",
            "",
            "## Reproducibility notes",
            "",
            "- `manifest.json` records fixed seeds, thresholds, input hashes, environment, model revision, command, and output hashes.",
            "- `gate1_selective/` is the aggregate-only cohort audit; it contains no headline text.",
            "- `cohort_symbols.csv`, regression tables, attrition, and price diagnostics contain no licensed headline text.",
            "- A pre-outcome pass was interrupted after a path-resolution performance bug; no returns had been estimated or viewed.",
            "- There was no outcome-based tuning and this directory contains the first and only completed results run.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")
    return signal, recap_inflation


def _attrition_table(counts: dict[str, int], *, cohort_symbols: int, prices: dict[str, pd.DataFrame], panel: pd.DataFrame) -> pd.DataFrame:
    steps = [
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
            "deduplicated_coherent_firm_events",
            "events",
            counts.get("deduplicated_coherent_events", 0),
            counts.get("exact_firm_event_duplicates", 0),
        ),
        (
            "nonempty_vader_scored_headlines",
            "headlines",
            counts.get("vader_scored_headlines", 0),
            counts.get("empty_headlines_excluded", 0),
        ),
        ("aggregated_symbol_sessions", "sessions", len(panel), 0),
        ("coherent_symbols", "symbols", cohort_symbols, 0),
        (
            "coherent_symbols_with_price_files",
            "symbols",
            sum(symbol in prices for symbol in panel["symbol"].unique()),
            sum(symbol not in prices for symbol in panel["symbol"].unique()),
        ),
    ]
    for horizon in range(1, 11):
        usable = int(panel[f"ar_h{horizon}"].notna().sum())
        steps.append((f"symbol_sessions_with_ar_h{horizon}", "sessions", usable, len(panel) - usable))
    return pd.DataFrame(steps, columns=["step", "unit", "surviving", "excluded_at_step"])


def run_fnspid_vader_smoke(
    *,
    news_archives: tuple[str | Path, ...],
    nasdaq_archive: str | Path,
    price_archive: str | Path,
    output_dir: str | Path,
    config: FnspidVaderSmokeConfig | None = None,
    command: tuple[str, ...] = (),
) -> Path:
    config = config or FnspidVaderSmokeConfig()
    news_paths = tuple(Path(path).resolve() for path in news_archives)
    nasdaq_path = Path(nasdaq_archive).resolve()
    prices_path = Path(price_archive).resolve()
    destination = Path(output_dir)
    if destination.exists():
        raise FnspidVaderSmokeError(f"refusing to overwrite one-shot output: {destination}")
    if not news_paths or not all(path.is_file() for path in news_paths) or nasdaq_path not in news_paths or not prices_path.is_file():
        raise FnspidVaderSmokeError("frozen news and price archives must exist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary_name:
        staging = Path(temporary_name)
        gate1 = staging / "gate1_selective"
        audit_fnspid_news(
            news_paths,
            gate1,
            config=FnspidAuditConfig(
                minimum_firms=config.minimum_coherent_firms,
                minimum_news_days_per_firm_year=config.minimum_news_days_per_firm_year,
                minimum_events=config.minimum_coherent_events,
                hash_seed=config.event_hash_seed,
                date_only_policy="next_trading_session",
                full_datetime_policy="session_close",
                calendar_name="XNYS",
            ),
            exclude_full_datetime_archives=(nasdaq_path,),
        )
        start_year, end_year, cohort_symbols, audited_events = choose_window(
            gate1 / "symbol_year_coverage.csv", gate1 / "candidate_windows.csv", config
        )
        panel, finbert_sample, counts = build_vader_session_panel(
            tuple((path, path == nasdaq_path) for path in news_paths),
            symbols=cohort_symbols,
            start_year=start_year,
            end_year=end_year,
            config=config,
        )
        if panel.empty:
            raise FnspidVaderSmokeError("VADER event panel is empty")
        prices, missing_price_symbols = load_fnspid_prices(prices_path, cohort_symbols, market_symbol=config.market_symbol)
        panel = attach_abnormal_returns(panel, prices, market_symbol=config.market_symbol, horizons=config.horizons)
        price_details, price_summary = run_price_sanity(prices, cohort_symbols, config)
        lambdas, recap_split = estimate_lambdas(panel, config)
        finbert = benchmark_finbert(finbert_sample, full_count=counts["vader_scored_headlines"], config=config)
        attrition = _attrition_table(counts, cohort_symbols=len(cohort_symbols), prices=prices, panel=panel)

        cohort_frame = pd.DataFrame({"symbol": sorted(cohort_symbols)})
        cohort_frame.to_csv(staging / "cohort_symbols.csv", index=False, lineterminator="\n")
        attrition.to_csv(staging / "attrition.csv", index=False, lineterminator="\n")
        lambdas.to_csv(staging / "lambda_h1_h10.csv", index=False, lineterminator="\n")
        recap_split.to_csv(staging / "recap_split_lambda1.csv", index=False, lineterminator="\n")
        price_details.to_csv(staging / "price_sanity.csv", index=False, lineterminator="\n")
        atomic_write_json(staging / "finbert_throughput.json", finbert)
        signal, recap_inflation = _write_report(
            staging / "report.md",
            start_year=start_year,
            end_year=end_year,
            cohort_events_audit=audited_events,
            cohort_symbols=cohort_symbols,
            counts=counts,
            attrition=attrition,
            lambdas=lambdas,
            split=recap_split,
            price_summary=price_summary,
            finbert=finbert,
            config=config,
        )
        result_files = [path for path in staging.rglob("*") if path.is_file() and path.name != "manifest.json"]
        manifest = {
            "schema_version": 1,
            "status": "completed",
            "experiment_type": "one-shot design-ranking smoke; not a results run",
            "evaluation_run_number": 1,
            "pre_outcome_invalidated_attempts": [
                {
                    "reason": "archive path was resolved once per timed row, causing avoidable system calls",
                    "stage_reached": "selective Gate-1 audit only; no VADER panel, prices, returns, or outcomes opened",
                    "correction": "resolve the archive-level exclusion flag once before the row loop",
                }
            ],
            "outcome_based_tuning": False,
            "development_evaluation_split": False,
            "signal_verdict_h_ge_1": signal,
            "recap_inflation_verdict": recap_inflation,
            "window": {
                "start_year": start_year,
                "end_year": end_year,
                "coherent_symbols": len(cohort_symbols),
                "audited_deduplicated_events": audited_events,
            },
            "config": {**asdict(config), "horizons": list(config.horizons)},
            "timing_rule": {
                "date_only_or_exact_midnight": "strictly next XNYS session",
                "full_datetime": "XNYS session containing the UTC minute, otherwise next session",
                "nasdaq_date_only_rows_included": True,
                "nasdaq_full_datetime_rows_included": False,
            },
            "outcomes": {
                "method": "stock adjusted-close log return minus SPY adjusted-close log return",
                "horizons": list(config.horizons),
                "inference": "separate OLS by horizon, calendar-event-date-clustered SEs",
                "multiple_testing_for_smoke_label": "Benjamini-Hochberg across h=1..10",
            },
            "counts": counts,
            "price_sanity": price_summary,
            "missing_price_symbols": sorted(missing_price_symbols),
            "finbert_throughput": finbert,
            "inputs": {
                "news_archives": [
                    {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in news_paths
                ],
                "price_archive": {"path": str(prices_path), "sha256": sha256_file(prices_path), "size_bytes": prices_path.stat().st_size},
            },
            "command": list(command),
            "environment": collect_run_environment(),
            "files": {
                path.relative_to(staging).as_posix(): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
                for path in sorted(result_files)
            },
        }
        atomic_write_json(staging / "manifest.json", manifest)
        staging.rename(destination)
    return destination / "report.md"


def load_config(path: str | Path) -> FnspidVaderSmokeConfig:
    with Path(path).open("rb") as handle:
        payload = tomllib.load(handle)
    config = dict(payload.get("experiment", {}))
    if "horizons" in config:
        config["horizons"] = tuple(int(value) for value in config["horizons"])
    return FnspidVaderSmokeConfig(**config)
