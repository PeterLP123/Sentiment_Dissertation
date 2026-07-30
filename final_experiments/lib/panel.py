"""FNSPID firm-day panel builder for final_experiments (Workstream 1 / S1).

Joins scored news (Moment-2 FinBERT checkpoint), FNSPID adjusted opens, SPY,
and the LSEG earnings calendar onto ``(symbol, session_date)``.

Timing for news rows is inherited from the frozen Gate-1 next-session mapping
already applied in the checkpoint (date-only → first XNYS session strictly
after the stated date). Earnings use the BMO/AMC session rules recorded in
``final_experiments/data/earnings/README.md``.

Publisher / ``story_family_id`` / intraday ``availability_timestamp`` are not
on the checkpoint grain; they are left null with an explicit note rather than
pretending the LSEG schema exists on FNSPID.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COHORT = REPO_ROOT / "reports" / "loop_vader_scale_20260719" / "cohort_symbols.csv"
DEFAULT_EVENTS_DB = REPO_ROOT / "reports" / "loop_moment2_finbert_20260719" / "finbert_checkpoint.sqlite3"
DEFAULT_PRICE_ZIP = Path(
    "/Users/peterprendergast/Documents/Sentiment_Dissertation_data"
    "/FNSPID/bf9189c41527198897d1af3e17b1a0095279fc45/raw/Stock_price/full_history.zip"
)
DEFAULT_EARNINGS = (
    REPO_ROOT / "final_experiments" / "data" / "earnings" / "fnspid_earnings_calendar_2011_2023_quarterly.csv"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "final_experiments" / "outputs" / "01_panel"

# Frozen chronological split (plan.html, 2026-07-30). Do not fit on evaluation.
FROZEN_DEV_END = "2019-12-31"
FROZEN_EVAL_START = "2020-01-01"
# Back-compat aliases for any early callers.
PROPOSED_DEV_END = FROZEN_DEV_END
PROPOSED_EVAL_START = FROZEN_EVAL_START


@dataclass(frozen=True)
class PanelPaths:
    cohort: Path = DEFAULT_COHORT
    events_db: Path = DEFAULT_EVENTS_DB
    price_zip: Path = DEFAULT_PRICE_ZIP
    earnings: Path = DEFAULT_EARNINGS
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _zip_price_member_map(archive: zipfile.ZipFile) -> dict[str, str]:
    members: dict[str, str] = {}
    for name in archive.namelist():
        if "/._" in name or not name.endswith(".csv"):
            continue
        if not name.startswith("full_history/"):
            continue
        stem = Path(name).stem.upper()
        members[stem] = name
    return members


def load_cohort_symbols(path: Path) -> list[str]:
    frame = pd.read_csv(path)
    return sorted(frame["symbol"].astype(str).str.strip().unique().tolist())


def aggregate_news_events(db_path: Path, symbols: set[str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collapse checkpoint events to one news-bearing firm-day row."""
    attrition: dict[str, int] = {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n_events = int(con.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        attrition["events_in_checkpoint"] = n_events
        # Read only needed columns; headlines stay in DB, not in the panel.
        events = pd.read_sql_query(
            """
            SELECT symbol, session_date, is_recap,
                   vader_compound, finbert_score, finbert_negative_dominant
            FROM events
            """,
            con,
        )
    finally:
        con.close()

    events["symbol"] = events["symbol"].astype(str)
    events["session_date"] = pd.to_datetime(events["session_date"]).dt.normalize()
    attrition["events_after_parse"] = int(len(events))

    events = events.loc[events["symbol"].isin(symbols)].copy()
    attrition["events_in_cohort"] = int(len(events))

    grouped = (
        events.groupby(["symbol", "session_date"], sort=False)
        .agg(
            article_count=("finbert_score", "size"),
            recap_count=("is_recap", "sum"),
            vader_mean=("vader_compound", "mean"),
            finbert_mean=("finbert_score", "mean"),
            finbert_negative_share=("finbert_negative_dominant", "mean"),
        )
        .reset_index()
    )
    grouped["recap_share"] = grouped["recap_count"] / grouped["article_count"].clip(lower=1)
    grouped["log_article_count"] = np.log(grouped["article_count"].astype(float))
    grouped["text_kind"] = "headline"
    grouped["timing_policy"] = "date_only_next_session_or_precise_as_in_checkpoint"
    grouped["publisher"] = pd.NA
    grouped["story_family_id"] = pd.NA
    grouped["availability_timestamp"] = pd.NA
    attrition["firm_days_news"] = int(len(grouped))
    attrition["symbols_with_news"] = int(grouped["symbol"].nunique())
    return grouped, attrition


def load_adjusted_open_prices(
    price_zip: Path,
    symbols: set[str],
    *,
    market_symbol: str = "SPY",
    start: str = "2011-01-01",
    end: str = "2023-12-31",
) -> tuple[pd.DataFrame, list[str]]:
    """Load FNSPID daily bars and derive split-adjusted opens.

    ``adjusted_open = open * (adj_close / close)``. Dividends are not folded in
    (same price-return convention as the rest of the FNSPID work).
    """
    wanted = {s.upper() for s in symbols | {market_symbol}}
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    with zipfile.ZipFile(price_zip) as archive:
        members = _zip_price_member_map(archive)
        for symbol in sorted(wanted):
            member = members.get(symbol)
            if member is None:
                missing.append(symbol)
                continue
            with archive.open(member) as handle:
                frame = pd.read_csv(handle)
            frame.columns = [str(c).strip().casefold() for c in frame.columns]
            need = {"date", "open", "close", "adj close"}
            if not need.issubset(frame.columns):
                missing.append(symbol)
                continue
            out = frame[["date", "open", "close", "adj close"]].copy()
            out["session_date"] = pd.to_datetime(out.pop("date"), errors="coerce").dt.normalize()
            for col in ("open", "close", "adj close"):
                out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out.dropna(subset=["session_date", "open", "close", "adj close"])
            out = out[(out["open"] > 0) & (out["close"] > 0) & (out["adj close"] > 0)]
            out["adjusted_open"] = out["open"] * (out["adj close"] / out["close"])
            out["adjusted_close"] = out["adj close"]
            out["symbol"] = symbol
            out = out.drop(columns=["adj close"])
            out = out.drop_duplicates("session_date", keep="last")
            start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
            out = out.loc[out["session_date"].between(start_ts, end_ts)]
            frames.append(out)
    if not frames:
        raise RuntimeError(f"no prices loaded from {price_zip}")
    prices = pd.concat(frames, ignore_index=True)
    if market_symbol.upper() not in set(prices["symbol"]):
        raise RuntimeError(f"price archive lacks market proxy {market_symbol}")
    # Restore original cohort casing where possible.
    return prices, sorted(s for s in missing if s != market_symbol.upper())


def map_earnings_to_sessions(
    earnings: pd.DataFrame,
    *,
    calendar_name: str = "XNYS",
) -> pd.DataFrame:
    """Map earnings report_date → XNYS session_date using frozen BMO/AMC rules."""
    try:
        import exchange_calendars as xcals
    except ImportError as exc:  # pragma: no cover
        raise ImportError("exchange_calendars is required for earnings session mapping") from exc

    if earnings.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "session_date",
                "earnings_report_date",
                "earnings_timing_flag",
                "earnings_session_rule",
                "earnings_fiscal_period",
                "is_earnings_session",
            ]
        )

    frame = earnings.copy()
    frame["symbol"] = frame["symbol"].astype(str).str.upper()
    frame["report_date"] = pd.to_datetime(frame["report_date"]).dt.normalize()
    # Dedup contaminated same-day multiples (documented in earnings README).
    frame = frame.sort_values(["symbol", "report_date"]).drop_duplicates(
        subset=["symbol", "report_date"], keep="first"
    )

    calendar = xcals.get_calendar(calendar_name, start="2000-01-01", end="2030-12-31")
    cache: dict[tuple[str, str], str | None] = {}

    def _map_one(report_date: pd.Timestamp, rule: str) -> str | None:
        key = (report_date.date().isoformat(), str(rule))
        if key in cache:
            return cache[key]
        try:
            if rule == "same_session":
                if calendar.is_session(report_date):
                    mapped = report_date.date().isoformat()
                else:
                    mapped = calendar.date_to_session(report_date, direction="next").date().isoformat()
            else:  # next_session (AMC / unknown)
                session = calendar.date_to_session(report_date, direction="next")
                if session.date().isoformat() == report_date.date().isoformat():
                    session = calendar.next_session(session)
                mapped = session.date().isoformat()
        except Exception:
            mapped = None
        cache[key] = mapped
        return mapped

    rules = frame.get("session_rule", pd.Series(["next_session"] * len(frame))).fillna("next_session")
    mapped_dates = [
        _map_one(rd, rule) for rd, rule in zip(frame["report_date"], rules, strict=False)
    ]
    out = pd.DataFrame(
        {
            "symbol": frame["symbol"].to_numpy(),
            "session_date": pd.to_datetime(mapped_dates),
            "earnings_report_date": frame["report_date"].to_numpy(),
            "earnings_timing_flag": frame.get("timing_flag"),
            "earnings_session_rule": rules.to_numpy(),
            "earnings_fiscal_period": frame.get("fiscal_period"),
            "is_earnings_session": True,
        }
    )
    out = out.dropna(subset=["session_date"])
    out["session_date"] = out["session_date"].dt.normalize()
    # If two report dates collapse to one session, keep first.
    out = out.drop_duplicates(subset=["symbol", "session_date"], keep="first")
    return out


def attach_open_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach next-session open-to-open simple returns for stock and SPY."""
    out = panel.sort_values(["symbol", "session_date"]).copy()
    out["next_adjusted_open"] = out.groupby("symbol", sort=False)["adjusted_open"].shift(-1)
    out["ret_open_h1"] = out["next_adjusted_open"] / out["adjusted_open"] - 1.0
    out["spy_next_adjusted_open"] = out.groupby("symbol", sort=False)["spy_adjusted_open"].shift(-1)
    # SPY series is constant across symbols on a date; shift within symbol still works
    # because spy_adjusted_open is aligned per row. Prefer date-level lead for market:
    spy_lead = (
        out[["session_date", "spy_adjusted_open"]]
        .drop_duplicates("session_date")
        .sort_values("session_date")
        .assign(
            spy_next_adjusted_open=lambda d: d["spy_adjusted_open"].shift(-1),
            spy_ret_open_h1=lambda d: d["spy_adjusted_open"].shift(-1) / d["spy_adjusted_open"] - 1.0,
        )[["session_date", "spy_next_adjusted_open", "spy_ret_open_h1"]]
    )
    out = out.drop(columns=["spy_next_adjusted_open"], errors="ignore")
    out = out.merge(spy_lead, on="session_date", how="left")
    out["ar_open_h1"] = out["ret_open_h1"] - out["spy_ret_open_h1"]
    return out


def build_fnspid_firm_day_panel(
    paths: PanelPaths | None = None,
    *,
    market_symbol: str = "SPY",
    start: str = "2011-01-01",
    end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the S1 firm-day panel and an attrition table.

    Grain: news-bearing ``(symbol, session_date)`` for the coherent cohort,
    inner-joined to a tradable adjusted open (and SPY open on that session).
    """
    paths = paths or PanelPaths()
    cohort = load_cohort_symbols(paths.cohort)
    cohort_set = set(cohort)
    attrition_rows: list[dict[str, Any]] = []

    def _step(name: str, surviving: int, note: str = "") -> None:
        attrition_rows.append({"step": name, "surviving": int(surviving), "note": note})

    _step("cohort_symbols", len(cohort_set), str(paths.cohort))

    news, news_attr = aggregate_news_events(paths.events_db, cohort_set)
    for key, value in news_attr.items():
        _step(key, value)
    _step("firm_days_after_news_aggregate", len(news))

    prices, missing_prices = load_adjusted_open_prices(
        paths.price_zip,
        cohort_set,
        market_symbol=market_symbol,
        start=start,
        end=end,
    )
    _step(
        "cohort_symbols_with_prices",
        len(cohort_set) - len([s for s in missing_prices if s in {x.upper() for x in cohort_set}]),
        f"missing={missing_prices[:20]}{'…' if len(missing_prices) > 20 else ''}",
    )

    # Normalise news symbols to upper for join with price zip stems.
    news = news.copy()
    news["symbol"] = news["symbol"].astype(str).str.upper()
    news["session_date"] = pd.to_datetime(news["session_date"]).dt.normalize()

    stock_px = prices.loc[prices["symbol"] != market_symbol.upper()].copy()
    spy = prices.loc[prices["symbol"] == market_symbol.upper(), ["session_date", "adjusted_open"]].rename(
        columns={"adjusted_open": "spy_adjusted_open"}
    )

    panel = news.merge(
        stock_px[
            [
                "symbol",
                "session_date",
                "open",
                "close",
                "adjusted_open",
                "adjusted_close",
            ]
        ],
        on=["symbol", "session_date"],
        how="inner",
    )
    _step("firm_days_after_price_inner_join", len(panel), "news ∩ adjusted_open")

    panel = panel.merge(spy, on="session_date", how="inner")
    _step("firm_days_after_spy_join", len(panel), "session must have SPY open")

    earnings_raw = pd.read_csv(paths.earnings) if paths.earnings.exists() else pd.DataFrame()
    earnings = map_earnings_to_sessions(earnings_raw)
    _step("earnings_sessions_mapped", len(earnings), str(paths.earnings.name) if paths.earnings.exists() else "missing")

    panel = panel.merge(earnings, on=["symbol", "session_date"], how="left")
    panel["is_earnings_session"] = panel["is_earnings_session"].fillna(False)
    panel["is_earnings_session"] = panel["is_earnings_session"].astype(bool)
    _step(
        "firm_days_with_earnings_flag",
        int(panel["is_earnings_session"].sum()),
        "left join; not a filter",
    )

    panel = attach_open_returns(panel)
    _step("firm_days_final", len(panel))

    # Chronological split labels (frozen; no fitting on evaluation).
    panel["split"] = np.where(
        panel["session_date"] <= pd.Timestamp(FROZEN_DEV_END),
        "development",
        np.where(panel["session_date"] >= pd.Timestamp(FROZEN_EVAL_START), "evaluation", "gap"),
    )

    attrition = pd.DataFrame(attrition_rows)
    meta: dict[str, Any] = {
        "built_at": _utc_now(),
        "grain": "(symbol, session_date) news-bearing ∩ tradable adjusted open ∩ SPY",
        "cohort_path": str(paths.cohort),
        "events_db": str(paths.events_db),
        "price_zip": str(paths.price_zip),
        "earnings_path": str(paths.earnings),
        "market_symbol": market_symbol,
        "window": {"start": start, "end": end},
        "timing": {
            "news": (
                "Inherited from loop_moment2_finbert checkpoint / Gate-1 revised "
                "next-session mapping. Date-only stamps → first XNYS session strictly "
                "after stated date. Not the LSEG +15min open rule."
            ),
            "earnings": "BMO/during_market→same_session; AMC/unknown→next_session",
            "availability_timestamp": "null on FNSPID checkpoint grain — unverifiable",
            "publisher_story_family": "null — require zstd rescan to populate",
        },
        "returns": {
            "ret_open_h1": "next session adjusted_open / adjusted_open - 1",
            "ar_open_h1": "ret_open_h1 - spy_ret_open_h1",
            "price_convention": "split-adjusted open via open*(adj_close/close); not dividend-adjusted",
        },
        "primary_spine": {
            "name": "FNSPID",
            "cohort": "2011-2023 coherent",
            "decided": "2026-07-30",
            "lseg_role": "out-of-regime robustness only; never pooled",
        },
        "frozen_chronological_split": {
            "development_end": FROZEN_DEV_END,
            "evaluation_start": FROZEN_EVAL_START,
            "status": "frozen_2026-07-30",
            "language": "chronological evaluation block — not a pristine confirmatory holdout",
            "rule": "no fitting, threshold search, aggregator selection, or multiplicity peeking on evaluation",
        },
        # Alias retained so older notebook cells keep working until re-synced.
        "proposed_chronological_split": {
            "development_end": FROZEN_DEV_END,
            "evaluation_start": FROZEN_EVAL_START,
            "status": "frozen_2026-07-30",
        },
        "missing_price_symbols": missing_prices,
        "n_rows": int(len(panel)),
        "n_symbols": int(panel["symbol"].nunique()) if len(panel) else 0,
        "n_sessions": int(panel["session_date"].nunique()) if len(panel) else 0,
        "earnings_session_share": float(panel["is_earnings_session"].mean()) if len(panel) else 0.0,
        "split_counts": panel["split"].value_counts().to_dict() if len(panel) else {},
    }
    return panel, attrition, meta


def coverage_breaks_by_month(panel: pd.DataFrame, *, fold: float = 3.0) -> pd.DataFrame:
    """Flag consecutive month headline-count changes exceeding ``fold``."""
    monthly = (
        panel.assign(month=panel["session_date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month", sort=True)["article_count"]
        .sum()
        .rename("article_count")
        .reset_index()
    )
    monthly["prev"] = monthly["article_count"].shift(1)
    monthly["fold_change"] = monthly["article_count"] / monthly["prev"].replace(0, np.nan)
    monthly["break_gt_fold"] = monthly["fold_change"].ge(fold) | (1.0 / monthly["fold_change"]).ge(fold)
    return monthly


def write_panel_outputs(
    panel: pd.DataFrame,
    attrition: pd.DataFrame,
    meta: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "fnspid_firm_day_panel.parquet"
    attrition_path = output_dir / "attrition.csv"
    meta_path = output_dir / "manifest.json"
    coverage_path = output_dir / "monthly_coverage.csv"
    sample_path = output_dir / "panel_sample_head.csv"

    panel.to_parquet(panel_path, index=False)
    attrition.to_csv(attrition_path, index=False)
    monthly = coverage_breaks_by_month(panel)
    monthly.to_csv(coverage_path, index=False)
    panel.head(50).to_csv(sample_path, index=False)

    meta = {
        **meta,
        "outputs": {
            "panel": str(panel_path),
            "attrition": str(attrition_path),
            "monthly_coverage": str(coverage_path),
            "sample_head": str(sample_path),
        },
        "monthly_breaks_gt3x": int(monthly["break_gt_fold"].fillna(False).sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    return {
        "panel": panel_path,
        "attrition": attrition_path,
        "manifest": meta_path,
        "monthly_coverage": coverage_path,
        "sample_head": sample_path,
    }


def main() -> int:
    paths = PanelPaths()
    panel, attrition, meta = build_fnspid_firm_day_panel(paths)
    written = write_panel_outputs(panel, attrition, meta, paths.output_dir)
    print(json.dumps({"n_rows": meta["n_rows"], "n_symbols": meta["n_symbols"], "outputs": {k: str(v) for k, v in written.items()}}, indent=2))
    print(attrition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
