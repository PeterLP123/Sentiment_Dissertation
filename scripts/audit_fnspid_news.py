#!/usr/bin/env python3
"""Run the frozen FNSPID structure, timestamp, duplicate, and coverage audit."""

from __future__ import annotations

import argparse

from sentiment_benchmark.fnspid_feasibility import FnspidAuditConfig, audit_fnspid_news


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--date-only-policy",
        choices=("exclude", "next_trading_session"),
        default="exclude",
        help="How to handle rows without a reliable intraday time.",
    )
    parser.add_argument(
        "--full-datetime-policy",
        choices=("publication_date", "session_close"),
        default="publication_date",
        help="Whether precise UTC timestamps use their date or the XNYS close rule.",
    )
    args = parser.parse_args()
    config = FnspidAuditConfig(
        date_only_policy=args.date_only_policy,
        full_datetime_policy=args.full_datetime_policy,
    )
    manifest = audit_fnspid_news(tuple(args.archive), args.output_dir, config=config)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
