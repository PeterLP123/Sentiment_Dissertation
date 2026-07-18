#!/usr/bin/env python3
"""Run the frozen FNSPID structure, timestamp, duplicate, and coverage audit."""

from __future__ import annotations

import argparse

from sentiment_benchmark.fnspid_feasibility import audit_fnspid_news


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    manifest = audit_fnspid_news(tuple(args.archive), args.output_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
