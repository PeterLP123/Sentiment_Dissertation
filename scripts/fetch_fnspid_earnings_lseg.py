#!/usr/bin/env python3
"""Fetch a scheduled-earnings calendar for the FNSPID coherent cohort via LSEG.

Uses the Workspace desktop session (``lseg.data``). Prefers LSEG over yfinance.
Writes licensed artifacts under ``final_experiments/data/earnings/`` (gitignored).

Resume-safe: symbol→RIC map and per-batch event CSVs are checkpointed.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COHORT = REPO_ROOT / "reports" / "loop_vader_scale_20260719" / "cohort_symbols.csv"
DEFAULT_OUT = REPO_ROOT / "final_experiments" / "data" / "earnings"

def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _classify_session_timing(event_date: Any, event_time: Any) -> dict[str, Any]:
    """Map LSEG event date/time to BMO / AMC / unknown using America/New_York.

    Convention (frozen for this gather):
    - missing time → timing_flag = unknown, session_rule = next_session
    - local time < 09:30 → BMO → that XNYS session
    - local time >= 16:00 → AMC → next XNYS session
    - otherwise → during_market → that XNYS session (rare for RES)
    """
    out: dict[str, Any] = {
        "report_date": None,
        "event_time_raw": None,
        "event_time_ny": None,
        "timing_flag": "unknown",
        "session_rule": "next_session",
    }
    if pd.isna(event_date):
        return out
    date = pd.Timestamp(event_date).tz_localize(None).normalize()
    out["report_date"] = date.date().isoformat()
    if pd.isna(event_time) or str(event_time).strip() == "":
        return out

    # Event Start Time often arrives as datetime.time or "HH:MM:SS".
    try:
        if hasattr(event_time, "hour"):
            tod = pd.Timestamp(event_time).strftime("%H:%M:%S")
        else:
            tod = str(event_time).strip()
            if " " in tod:
                tod = tod.split()[-1]
        out["event_time_raw"] = tod
        # LSEG Event Start Time for US names is typically UTC wall-clock.
        utc_ts = pd.Timestamp(f"{out['report_date']} {tod}", tz="UTC")
        ny = utc_ts.tz_convert("America/New_York")
        out["event_time_ny"] = ny.isoformat()
        # If UTC date rolled into previous/next NY calendar day, use NY date as report_date.
        out["report_date"] = ny.date().isoformat()
        minutes = ny.hour * 60 + ny.minute
        if minutes < 9 * 60 + 30:
            out["timing_flag"] = "BMO"
            out["session_rule"] = "same_session"
        elif minutes >= 16 * 60:
            out["timing_flag"] = "AMC"
            out["session_rule"] = "next_session"
        else:
            out["timing_flag"] = "during_market"
            out["session_rule"] = "same_session"
    except Exception:
        out["timing_flag"] = "unknown"
        out["session_rule"] = "next_session"
    return out


def _parse_fiscal_period(title: Any) -> str | None:
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return None
    text = str(title)
    m = re.search(r"\b(Q[1-4])\s+(20\d{2}|19\d{2})\b", text, flags=re.I)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    m = re.search(r"\b(FY|H[12]|1H|2H)\s*(20\d{2}|19\d{2})\b", text, flags=re.I)
    if m:
        return f"{m.group(1).upper()} {m.group(2)}"
    return None


_LIGHT_SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|nv|sa|ag)\b",
    flags=re.I,
)


def _normalize_name(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = _LIGHT_SUFFIX.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _issuer_phrase(document_title: Any, symbol: str) -> str:
    if isinstance(document_title, str) and document_title.strip():
        # Drop parenthetical qualifiers ("Tetra Tech Inc (Delaware)") — they
        # never appear in event titles and caused full-history false rejects.
        head = re.sub(r"\([^)]*\)", "", document_title.split(",")[0])
        return _normalize_name(head)
    return _normalize_name(symbol)


def _title_matches_issuer(title: Any, phrase: str, symbol: str) -> bool:
    """Keep RES rows whose title refers to the mapped issuer.

    LSEG often returns predecessor / same-ticker / related-entity earnings under
    a surviving RIC. Require the DocumentTitle issuer phrase (normalized) to
    appear in the event title. Tickers of length ≥ 3 may also match on symbol.
    """
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return False
    text = str(title)
    if len(symbol) >= 3 and re.search(rf"\b{re.escape(symbol.lower())}\b", text.lower()):
        return True
    if phrase and phrase in _normalize_name(text):
        return True
    return False


def _is_quarterly_release(title: Any) -> bool:
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return False
    return bool(re.search(r"\b(Q[1-4]|FY|H[12]|1H|2H)\b", str(title), flags=re.I))


def resolve_rics(symbols: list[str], batch_size: int = 50) -> pd.DataFrame:
    from lseg.data.content import symbol_conversion

    rows: list[dict[str, Any]] = []
    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i : i + batch_size]
        definition = symbol_conversion.Definition(
            symbols=chunk,
            from_symbol_type=symbol_conversion.SymbolTypes.TICKER_SYMBOL,
            to_symbol_types=[symbol_conversion.SymbolTypes.RIC],
            preferred_country_code=symbol_conversion.CountryCode.USA,
        )
        raw = definition.get_data().data.raw or {}
        matches = raw.get("Matches") or {}
        unmatched = set(raw.get("Unmatched") or []) | set(raw.get("Failed") or [])
        for symbol in chunk:
            hit = matches.get(symbol)
            if not hit:
                rows.append(
                    {
                        "symbol": symbol,
                        "ric": None,
                        "document_title": None,
                        "resolve_status": "unmatched" if symbol in unmatched else "missing",
                        "suspect_non_equity": False,
                    }
                )
                continue
            doc_title = hit.get("DocumentTitle")
            suspect = bool(
                isinstance(doc_title, str)
                and re.search(r"future|commodity|\boption\b", doc_title, flags=re.I)
            )
            rows.append(
                {
                    "symbol": symbol,
                    "ric": hit.get("RIC"),
                    "document_title": doc_title,
                    "resolve_status": "matched",
                    # Tickers can collide with futures/options roots (e.g. BK →
                    # a WTI-Brent future); flag for review, do not silently use.
                    "suspect_non_equity": suspect,
                }
            )
        time.sleep(0.2)
    return pd.DataFrame(rows)


def fetch_earnings_batch(
    rics: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    import lseg.data as ld

    frame = ld.get_data(
        universe=rics,
        fields=[
            "TR.EventStartDate",
            "TR.EventStartTime",
            "TR.EventType",
            "TR.EventTitle",
            "TR.EventStatus",
            "TR.EventEventID",
        ],
        parameters={"SDate": start, "EDate": end, "EventType": "RES"},
    )
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2011-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(args.cohort)
    symbols = sorted(cohort["symbol"].astype(str).str.strip().unique().tolist())
    print(f"cohort symbols: {len(symbols)} from {args.cohort}")

    import lseg.data as ld

    ld.open_session()
    print(f"LSEG session open at {_utc_now()}")

    ric_path = out_dir / "symbol_ric_map.csv"
    if ric_path.exists():
        ric_map = pd.read_csv(ric_path)
        print(f"loaded existing RIC map: {len(ric_map)} rows")
    else:
        print("resolving ticker → RIC via LSEG symbol conversion…")
        ric_map = resolve_rics(symbols)
        ric_map.to_csv(ric_path, index=False)
        print(
            f"RIC map written: matched={int((ric_map.resolve_status=='matched').sum())} "
            f"unmatched={int((ric_map.resolve_status!='matched').sum())}"
        )

    if "suspect_non_equity" not in ric_map.columns:
        # Map built by an earlier version of this script; derive the flag.
        ric_map["suspect_non_equity"] = ric_map["document_title"].astype(str).str.contains(
            r"future|commodity|\boption\b", case=False, regex=True
        )
    suspect_n = int(ric_map["suspect_non_equity"].fillna(False).sum())
    if suspect_n:
        flagged = ric_map.loc[
            ric_map["suspect_non_equity"].fillna(False), "symbol"
        ].astype(str).tolist()
        print(f"WARNING: {suspect_n} tickers resolved to non-equity instruments: {flagged}")

    matched = ric_map.loc[ric_map["ric"].notna() & (ric_map["ric"].astype(str).str.len() > 0)].copy()
    ric_to_symbol = dict(zip(matched["ric"].astype(str), matched["symbol"].astype(str), strict=False))
    # Also allow Instrument returning the bare symbol.
    for _, row in matched.iterrows():
        ric_to_symbol[str(row["symbol"])] = str(row["symbol"])

    rics = matched["ric"].astype(str).tolist()
    batches = [rics[i : i + args.batch_size] for i in range(0, len(rics), args.batch_size)]
    print(f"earnings fetch: {len(rics)} RICs in {len(batches)} batches; window {args.start}→{args.end}")

    batch_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    for idx, batch in enumerate(batches):
        ckpt = ckpt_dir / f"batch_{idx:03d}.csv"
        if ckpt.exists():
            part = pd.read_csv(ckpt)
            batch_frames.append(part)
            print(f"batch {idx+1}/{len(batches)}: resume {len(part)} rows from checkpoint")
            continue
        try:
            raw = fetch_earnings_batch(batch, args.start, args.end)
            if raw.empty:
                part = pd.DataFrame(
                    columns=[
                        "Instrument",
                        "Event Start Date",
                        "Event Start Time",
                        "Company Event Type",
                        "Event Title",
                        "Event Status",
                        "Event Id",
                    ]
                )
            else:
                part = raw.copy()
            part.to_csv(ckpt, index=False)
            batch_frames.append(part)
            print(f"batch {idx+1}/{len(batches)}: {len(part)} raw rows")
        except Exception as exc:
            failures.append({"batch_index": idx, "rics": batch, "error": str(exc)})
            print(f"batch {idx+1}/{len(batches)} FAILED: {exc}")
        time.sleep(args.sleep_seconds)

    raw_all = pd.concat(batch_frames, ignore_index=True) if batch_frames else pd.DataFrame()
    raw_path = out_dir / "earnings_events_raw.csv"
    raw_all.to_csv(raw_path, index=False)

    doc_by_symbol = {
        str(row.symbol): None if pd.isna(row.document_title) else str(row.document_title)
        for row in ric_map.itertuples()
    }

    # Normalize + issuer-title filter (drops predecessor / same-ticker contamination).
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for _, row in raw_all.iterrows():
        instrument = row.get("Instrument")
        if pd.isna(instrument):
            continue
        instrument = str(instrument)
        symbol = ric_to_symbol.get(instrument)
        if symbol is None:
            base = instrument.split(".")[0]
            symbol = ric_to_symbol.get(base)
        if symbol is None:
            continue
        timing = _classify_session_timing(row.get("Event Start Date"), row.get("Event Start Time"))
        if timing["report_date"] is None:
            continue
        title = row.get("Event Title")
        status = row.get("Event Status")
        # LSEG docs: EventStatus True ≈ estimated, False ≈ confirmed (for CA-style flags).
        if pd.isna(status) or str(status).strip() == "":
            status_label = "unknown"
        elif str(status).lower() in {"true", "1"} or status is True:
            status_label = "estimated"
        else:
            status_label = "confirmed"
        title_str = None if pd.isna(title) else str(title)
        record = {
            "symbol": symbol,
            "ric": instrument,
            "report_date": timing["report_date"],
            "event_time_raw": timing["event_time_raw"],
            "event_time_ny": timing["event_time_ny"],
            "timing_flag": timing["timing_flag"],
            "session_rule": timing["session_rule"],
            "event_type": row.get("Company Event Type"),
            "event_title": title_str,
            "fiscal_period": _parse_fiscal_period(title_str),
            "is_quarterly_release": _is_quarterly_release(title_str),
            "event_status": status_label,
            "event_id": None if pd.isna(row.get("Event Id")) else str(row.get("Event Id")),
        }
        phrase = _issuer_phrase(doc_by_symbol.get(symbol), symbol)
        if _title_matches_issuer(title_str, phrase, symbol):
            kept.append(record)
        else:
            rejected.append(record)

    calendar = pd.DataFrame.from_records(kept)
    if not calendar.empty:
        calendar = calendar.drop_duplicates(subset=["symbol", "report_date", "event_id"], keep="first")
        calendar = calendar.sort_values(["symbol", "report_date", "event_id"]).reset_index(drop=True)
    rejected_df = pd.DataFrame.from_records(rejected)
    quarterly = (
        calendar.loc[calendar["is_quarterly_release"]].copy().reset_index(drop=True)
        if not calendar.empty
        else calendar
    )

    cal_path = out_dir / "fnspid_earnings_calendar_2011_2023.csv"
    q_path = out_dir / "fnspid_earnings_calendar_2011_2023_quarterly.csv"
    rej_path = out_dir / "earnings_events_title_rejected.csv"
    calendar.to_csv(cal_path, index=False)
    quarterly.to_csv(q_path, index=False)
    rejected_df.to_csv(rej_path, index=False)

    covered = set(calendar["symbol"].unique()) if not calendar.empty else set()
    covered_q = set(quarterly["symbol"].unique()) if not quarterly.empty else set()
    matched_symbols = set(matched["symbol"].astype(str))
    summary = {
        "built_at": _utc_now(),
        "source": "LSEG Workspace desktop session via lseg.data",
        "event_type_parameter": "RES",
        "fields": [
            "TR.EventStartDate",
            "TR.EventStartTime",
            "TR.EventType",
            "TR.EventTitle",
            "TR.EventStatus",
            "TR.EventEventID",
        ],
        "window": {"start": args.start, "end": args.end},
        "cohort_path": str(args.cohort.relative_to(REPO_ROOT)),
        "cohort_symbols": len(symbols),
        "rics_resolved": int((ric_map["resolve_status"] == "matched").sum()),
        "rics_unresolved": int((ric_map["resolve_status"] != "matched").sum()),
        "rics_suspect_non_equity": suspect_n,
        "earnings_rows": int(len(calendar)),
        "earnings_rows_quarterly": int(len(quarterly)),
        "symbols_with_at_least_one_event": len(covered),
        "symbols_with_at_least_one_quarterly_event": len(covered_q),
        "matched_symbols_with_zero_events": sorted(matched_symbols - covered),
        "timing_flag_counts": calendar["timing_flag"].value_counts(dropna=False).to_dict()
        if not calendar.empty
        else {},
        "timing_flag_counts_quarterly": quarterly["timing_flag"].value_counts(dropna=False).to_dict()
        if not quarterly.empty
        else {},
        "event_status_counts": calendar["event_status"].value_counts(dropna=False).to_dict()
        if not calendar.empty
        else {},
        "title_filter": {
            "rule": (
                "normalized issuer phrase from DocumentTitle (before first comma, "
                "parentheticals stripped) must appear in normalized Event Title; "
                "tickers with len>=3 may match on symbol"
            ),
            "rows_rejected": int(len(rejected_df)),
        },
        "recommended_for_ws6": str(q_path.relative_to(REPO_ROOT)),
        "batch_failures": failures,
        "outputs": {
            "symbol_ric_map": str(ric_path.relative_to(REPO_ROOT)),
            "raw_events": str(raw_path.relative_to(REPO_ROOT)),
            "calendar": str(cal_path.relative_to(REPO_ROOT)),
            "calendar_quarterly": str(q_path.relative_to(REPO_ROOT)),
            "rejected_titles": str(rej_path.relative_to(REPO_ROOT)),
        },
        "session_mapping_convention": {
            "BMO": "timing_flag=BMO → trade that XNYS session open",
            "AMC": "timing_flag=AMC → first XNYS session strictly after report_date",
            "unknown_or_missing_time": "session_rule=next_session (conservative)",
            "during_market": "same_session",
        },
        "sharing": {
            "licensed_lseg_data": True,
            "redistribute": False,
            "note": "Calendar rows are LSEG-derived and stay local/gitignored.",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "cohort_symbols",
                    "rics_resolved",
                    "rics_unresolved",
                    "earnings_rows",
                    "earnings_rows_quarterly",
                    "symbols_with_at_least_one_event",
                    "symbols_with_at_least_one_quarterly_event",
                    "timing_flag_counts_quarterly",
                    "batch_failures",
                )
            },
            indent=2,
        )
    )

    try:
        ld.close_session()
    except Exception:
        pass
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
