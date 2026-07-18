"""Streaming, aggregate-only feasibility audit for the frozen FNSPID corpus."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlsplit

import pandas as pd

from .artifact_io import atomic_write_json, sha256_file
from .runtime_metadata import collect_run_environment


class FnspidAuditError(RuntimeError):
    """Raised when the frozen corpus cannot be audited without substitution."""


@dataclass(frozen=True)
class FnspidAuditConfig:
    coverage_break_ratio: float = 3.0
    minimum_firms: int = 300
    minimum_news_days_per_firm_year: int = 20
    minimum_window_years: int = 3
    minimum_events: int = 50_000
    hash_seed: str = "fnspid-feasibility-v1"
    date_only_policy: str = "exclude"
    full_datetime_policy: str = "publication_date"
    calendar_name: str = "XNYS"


def _hash64(*parts: str, seed: str) -> int:
    digest = hashlib.blake2b(digest_size=8, person=seed.encode("utf-8")[:16])
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little")


@contextmanager
def _open_zstd_csv(path: Path) -> Iterator[TextIO]:
    zstd = shutil.which("zstd")
    if zstd is None:
        raise FnspidAuditError("zstd executable is required to read the frozen archives")
    process = subprocess.Popen([zstd, "-q", "-dc", str(path)], stdout=subprocess.PIPE)
    if process.stdout is None:
        raise FnspidAuditError(f"failed to open decompressor for {path}")
    wrapper = io.TextIOWrapper(process.stdout, encoding="utf-8", errors="replace", newline="")
    try:
        yield wrapper
    finally:
        wrapper.close()
        return_code = process.wait()
        if return_code != 0:
            raise FnspidAuditError(f"zstd failed for {path} with status {return_code}")


def _site(url: str) -> str:
    try:
        hostname = (urlsplit(url).hostname or "").lower()
    except ValueError:
        hostname = ""
    return hostname.removeprefix("www.") or "unknown"


def _timestamp_parts(value: str) -> tuple[str, str, str, bool] | None:
    text = value.strip()
    if len(text) < 10 or text[4:5] != "-" or text[7:8] != "-":
        return None
    day = text[:10]
    if not (day[:4].isdigit() and day[5:7].isdigit() and day[8:10].isdigit()):
        return None
    year = day[:4]
    month = day[:7]
    time_text = text[11:19] if len(text) >= 19 else ""
    full_datetime = len(time_text) == 8 and time_text != "00:00:00"
    return day, month, year, full_datetime


def _normal(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _candidate_windows(
    symbol_year: Counter[tuple[str, str]],
    symbol_year_events: Counter[tuple[str, str]],
    event_year: Counter[str],
    config: FnspidAuditConfig,
) -> pd.DataFrame:
    years = sorted(event_year)
    eligible_by_year: dict[str, set[str]] = {
        year: {
            symbol
            for (symbol, candidate_year), count in symbol_year.items()
            if candidate_year == year and count >= config.minimum_news_days_per_firm_year
        }
        for year in years
    }
    rows: list[dict[str, Any]] = []
    for start_index, start in enumerate(years):
        for end_index in range(start_index + config.minimum_window_years - 1, len(years)):
            selected = years[start_index : end_index + 1]
            if any(int(selected[index + 1]) != int(selected[index]) + 1 for index in range(len(selected) - 1)):
                continue
            firms_each_year = [len(eligible_by_year[year]) for year in selected]
            coherent_firms = set.intersection(*(eligible_by_year[year] for year in selected))
            events = sum(event_year[year] for year in selected)
            coherent_firm_events = sum(symbol_year_events[(symbol, year)] for symbol in coherent_firms for year in selected)
            rows.append(
                {
                    "start_year": start,
                    "end_year": selected[-1],
                    "years": len(selected),
                    "minimum_eligible_firms_in_any_year": min(firms_each_year),
                    "firms_eligible_in_every_year": len(coherent_firms),
                    "total_mappable_deduplicated_events": events,
                    "coherent_firm_mappable_events": coherent_firm_events,
                    "dimension_gate_pass": (len(coherent_firms) >= config.minimum_firms and coherent_firm_events >= config.minimum_events),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "start_year",
                "end_year",
                "years",
                "minimum_eligible_firms_in_any_year",
                "firms_eligible_in_every_year",
                "total_mappable_deduplicated_events",
                "coherent_firm_mappable_events",
                "dimension_gate_pass",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["dimension_gate_pass", "years", "firms_eligible_in_every_year", "coherent_firm_mappable_events"],
        ascending=[False, False, False, False],
        kind="stable",
    )


def audit_fnspid_news(
    archives: tuple[str | Path, ...],
    output_dir: str | Path,
    *,
    config: FnspidAuditConfig | None = None,
    exclude_full_datetime_archives: tuple[str | Path, ...] = (),
) -> Path:
    """Scan headline metadata, deduplicate firm events, and write aggregate-only diagnostics."""

    config = config or FnspidAuditConfig()
    if config.date_only_policy not in {"exclude", "next_trading_session"}:
        raise FnspidAuditError("date_only_policy must be exclude or next_trading_session")
    if config.full_datetime_policy not in {"publication_date", "session_close"}:
        raise FnspidAuditError("full_datetime_policy must be publication_date or session_close")
    destination = Path(output_dir)
    if destination.exists():
        raise FnspidAuditError(f"refusing to overwrite audit output: {destination}")
    archive_paths = tuple(Path(path) for path in archives)
    excluded_timed_archives = {Path(path).resolve() for path in exclude_full_datetime_archives}
    if not archive_paths or not all(path.is_file() for path in archive_paths):
        raise FnspidAuditError("every frozen news archive must exist")
    csv.field_size_limit(sys.maxsize)

    source_rows: Counter[str] = Counter()
    source_symbol_rows: Counter[tuple[str, str]] = Counter()
    source_timestamp: Counter[tuple[str, str]] = Counter()
    source_year_timestamp: Counter[tuple[str, str, str]] = Counter()
    source_month_events: Counter[tuple[str, str]] = Counter()
    source_mappable_events: Counter[str] = Counter()
    symbol_year_days: Counter[tuple[str, str]] = Counter()
    symbol_year_events: Counter[tuple[str, str]] = Counter()
    event_year: Counter[str] = Counter()
    event_day: Counter[str] = Counter()
    schemas: list[dict[str, Any]] = []
    seen_firm_events: set[int] = set()
    seen_story_events: set[int] = set()
    seen_symbol_days: set[int] = set()
    total_rows = 0
    valid_timestamps = 0
    full_datetime_rows = 0
    date_only_or_midnight_rows = 0
    exact_firm_duplicates = 0
    cross_symbol_story_associations = 0
    date_only_rows_assigned_next_session = 0
    excluded_full_datetime_rows = 0

    session_cache: dict[str, str] = {}
    calendar: Any | None = None
    if config.date_only_policy == "next_trading_session" or config.full_datetime_policy == "session_close":
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The 'generic' unit for NumPy timedelta is deprecated",
                    category=DeprecationWarning,
                )
                import exchange_calendars as xcals

                calendar = xcals.get_calendar(config.calendar_name, start="1900-01-01", end="2030-12-31")
        except ImportError as exc:  # pragma: no cover - dependency is part of the project environment
            raise FnspidAuditError("exchange_calendars is required for next-session assignment") from exc

    def assigned_day(raw_timestamp: str, publication_day: str, *, full_datetime: bool) -> str | None:
        nonlocal date_only_rows_assigned_next_session
        if full_datetime:
            if config.full_datetime_policy == "publication_date":
                return publication_day
            if calendar is None:  # pragma: no cover - guarded by configuration above
                raise FnspidAuditError("calendar was not initialized")
            timestamp = pd.Timestamp(raw_timestamp)
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            return calendar.minute_to_session(timestamp.floor("min"), direction="next").date().isoformat()
        if config.date_only_policy == "exclude":
            return None
        cached = session_cache.get(publication_day)
        if cached is None:
            if calendar is None:  # pragma: no cover - guarded by configuration above
                raise FnspidAuditError("calendar was not initialized")
            publication = pd.Timestamp(publication_day)
            session = calendar.date_to_session(publication, direction="next")
            if session.date().isoformat() == publication_day:
                session = calendar.next_session(session)
            cached = session.date().isoformat()
            session_cache[publication_day] = cached
        date_only_rows_assigned_next_session += 1
        return cached

    for archive in archive_paths:
        exclude_full_datetime = archive.resolve() in excluded_timed_archives
        with _open_zstd_csv(archive) as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            required = {"Date", "Article_title", "Stock_symbol", "Url"}
            missing = required - set(fields)
            if missing:
                raise FnspidAuditError(f"{archive} lacks fields: {sorted(missing)}")
            schemas.append({"archive": str(archive), "columns": list(fields)})
            for row in reader:
                total_rows += 1
                if total_rows % 1_000_000 == 0:
                    print(f"audited {total_rows:,} rows", file=sys.stderr, flush=True)
                url = str(row.get("Url") or "").strip()
                site = _site(url)
                source_rows[site] += 1
                symbol = str(row.get("Stock_symbol") or "").strip().upper()
                if symbol:
                    source_symbol_rows[(site, "nonempty_symbol")] += 1
                raw_timestamp = str(row.get("Date") or "")
                timestamp = _timestamp_parts(raw_timestamp)
                if timestamp is None:
                    source_timestamp[(site, "invalid")] += 1
                    continue
                day, month, year, full_datetime = timestamp
                valid_timestamps += 1
                timestamp_class = "full_datetime" if full_datetime else "date_only_or_exact_midnight"
                source_timestamp[(site, timestamp_class)] += 1
                source_year_timestamp[(site, year, timestamp_class)] += 1
                if not full_datetime:
                    date_only_or_midnight_rows += 1
                else:
                    full_datetime_rows += 1
                    if exclude_full_datetime:
                        excluded_full_datetime_rows += 1
                        continue
                if not symbol:
                    continue
                event_session_day = assigned_day(raw_timestamp, day, full_datetime=full_datetime)
                if event_session_day is None:
                    continue
                if full_datetime:
                    source_symbol_rows[(site, "full_datetime_with_symbol")] += 1
                event_year_value = event_session_day[:4]
                event_month = event_session_day[:7]
                title = _normal(row.get("Article_title"))
                normalized_url = _normal(url).rstrip("/")
                identity = normalized_url or title
                firm_key = _hash64(symbol, event_session_day, identity, seed=config.hash_seed)
                if firm_key in seen_firm_events:
                    exact_firm_duplicates += 1
                    continue
                seen_firm_events.add(firm_key)
                story_key = _hash64(event_session_day, identity, seed=config.hash_seed)
                if story_key in seen_story_events:
                    cross_symbol_story_associations += 1
                else:
                    seen_story_events.add(story_key)
                source_month_events[(site, event_month)] += 1
                source_mappable_events[site] += 1
                event_year[event_year_value] += 1
                event_day[event_session_day] += 1
                symbol_year_events[(symbol, event_year_value)] += 1
                symbol_day_key = _hash64(symbol, event_session_day, seed=config.hash_seed)
                if symbol_day_key not in seen_symbol_days:
                    seen_symbol_days.add(symbol_day_key)
                    symbol_year_days[(symbol, event_year_value)] += 1

    source_timestamp_rows: list[dict[str, Any]] = []
    for site in sorted(source_rows):
        total = source_rows[site]
        full = source_timestamp[(site, "full_datetime")]
        date_only = source_timestamp[(site, "date_only_or_exact_midnight")]
        invalid = source_timestamp[(site, "invalid")]
        source_timestamp_rows.append(
            {
                "source_site": site,
                "rows": total,
                "rows_with_nonempty_symbol": source_symbol_rows[(site, "nonempty_symbol")],
                "full_datetime_rows": full,
                "full_datetime_rows_with_nonempty_symbol": source_symbol_rows[(site, "full_datetime_with_symbol")],
                "date_only_or_exact_midnight_rows": date_only,
                "invalid_timestamp_rows": invalid,
                "full_datetime_fraction": full / total if total else math.nan,
                "mappable_deduplicated_firm_events": source_mappable_events[site],
            }
        )
    source_year_rows: list[dict[str, Any]] = [
        {
            "source_site": site,
            "year": year,
            "full_datetime_rows": source_year_timestamp[(site, year, "full_datetime")],
            "date_only_or_exact_midnight_rows": source_year_timestamp[(site, year, "date_only_or_exact_midnight")],
        }
        for site, year in sorted({(site, year) for site, year, _ in source_year_timestamp})
    ]
    for row in source_year_rows:
        denominator = row["full_datetime_rows"] + row["date_only_or_exact_midnight_rows"]
        row["full_datetime_fraction"] = row["full_datetime_rows"] / denominator if denominator else math.nan

    months_by_site: dict[str, list[str]] = {}
    for site, month in source_month_events:
        months_by_site.setdefault(site, []).append(month)
    break_rows: list[dict[str, Any]] = []
    for site, raw_months in sorted(months_by_site.items()):
        months = pd.period_range(min(raw_months), max(raw_months), freq="M")
        previous_count: int | None = None
        previous_month: str | None = None
        for month_period in months:
            month = str(month_period)
            count = source_month_events[(site, month)]
            if previous_count is not None:
                if previous_count == 0 and count == 0:
                    fold = 1.0
                elif previous_count == 0 or count == 0:
                    fold = math.inf
                else:
                    fold = max(count / previous_count, previous_count / count)
                if fold > config.coverage_break_ratio:
                    break_rows.append(
                        {
                            "source_site": site,
                            "previous_month": previous_month,
                            "month": month,
                            "previous_count": previous_count,
                            "count": count,
                            "max_fold_change": fold,
                        }
                    )
            previous_count = count
            previous_month = month

    coverage = pd.DataFrame(
        [{"symbol": symbol, "year": year, "mappable_news_days": count} for (symbol, year), count in sorted(symbol_year_days.items())]
    )
    candidates = _candidate_windows(symbol_year_days, symbol_year_events, event_year, config)
    destination.mkdir(parents=True)
    outputs = {
        "source_timestamp_audit.csv": pd.DataFrame(source_timestamp_rows),
        "source_year_timestamp_audit.csv": pd.DataFrame(source_year_rows),
        "source_month_mappable_events.csv": pd.DataFrame(
            [
                {"source_site": site, "month": month, "mappable_deduplicated_events": count}
                for (site, month), count in sorted(source_month_events.items())
            ]
        ),
        "coverage_breaks.csv": pd.DataFrame(break_rows),
        "symbol_year_coverage.csv": coverage,
        "daily_event_counts.csv": pd.DataFrame(
            [{"date": day, "mappable_deduplicated_events": count} for day, count in sorted(event_day.items())]
        ),
        "candidate_windows.csv": candidates,
    }
    for filename, frame in outputs.items():
        frame.to_csv(destination / filename, index=False, lineterminator="\n")
    summary = {
        "schema_version": 1,
        "status": "completed",
        "timestamp_rule": {
            "date_only": (
                "excluded from primary mapping"
                if config.date_only_policy == "exclude"
                else "assigned strictly to the next XNYS trading session"
            ),
            "full_datetime": (
                "publication calendar date"
                if config.full_datetime_policy == "publication_date"
                else "XNYS session containing the UTC timestamp, or the next session when outside trading hours"
            ),
        },
        "counts": {
            "rows": total_rows,
            "valid_timestamp_rows": valid_timestamps,
            "full_datetime_rows": full_datetime_rows,
            "date_only_or_exact_midnight_rows": date_only_or_midnight_rows,
            "date_only_rows_assigned_next_session": date_only_rows_assigned_next_session,
            "excluded_full_datetime_rows": excluded_full_datetime_rows,
            "mappable_deduplicated_firm_events": len(seen_firm_events),
            "exact_firm_event_duplicates": exact_firm_duplicates,
            "unique_story_events": len(seen_story_events),
            "cross_symbol_story_associations": cross_symbol_story_associations,
            "symbols": len({symbol for symbol, _ in symbol_year_days}),
            "coverage_breaks": len(break_rows),
        },
        "schemas": schemas,
        "config": config.__dict__,
        "inputs": {str(path): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in archive_paths},
        "excluded_full_datetime_archives": sorted(str(path) for path in excluded_timed_archives),
        "files": {
            filename: {
                "sha256": sha256_file(destination / filename),
                "size_bytes": (destination / filename).stat().st_size,
            }
            for filename in outputs
        },
        "environment": collect_run_environment(),
    }
    atomic_write_json(destination / "audit_manifest.json", summary)
    return destination / "audit_manifest.json"
