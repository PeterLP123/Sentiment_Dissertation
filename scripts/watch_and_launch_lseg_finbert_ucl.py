#!/usr/bin/env python3
"""Poll the UCL GPU pool and launch a staged incremental FinBERT run once."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS_SCRIPT = REPO_ROOT / "scripts/ucl_gpu_status.py"
DEFAULT_PROJECT_ROOT = "/cs/student/project_msc/2025/cf/pprender"


def _append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at_utc": datetime.now(UTC).isoformat(), **event}, sort_keys=True) + "\n")
        handle.flush()


def _free_hosts(args: argparse.Namespace) -> tuple[list[str], dict[str, int]]:
    command = [
        sys.executable,
        str(STATUS_SCRIPT),
        "--user",
        args.user,
        "--jump-host",
        args.jump_host,
        "--workers",
        str(args.workers),
        "--connect-timeout",
        str(args.connect_timeout),
        "--json",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(f"GPU status command failed: {completed.stderr[-500:]}")
    payload = json.loads(completed.stdout)
    reports = payload["reports"]
    counts: dict[str, int] = {}
    for report in reports:
        status = str(report["status"])
        counts[status] = counts.get(status, 0) + 1
    return sorted(str(report["host"]) for report in reports if report["status"] == "FREE"), counts


def _launch(args: argparse.Namespace, host: str) -> subprocess.CompletedProcess[str]:
    project = args.project_root.rstrip("/")
    collection = f"{project}/data/raw/lseg_us_sector_33_backward_incremental/{args.snapshot_id}"
    run_root = f"{project}/runs/labels/lseg_us_sector_33_backward_20240101_20251026/incremental/{args.snapshot_id}"
    output = f"{run_root}/headline_scores_finbert4556_vader_baseline_v1_cacheonly.csv"
    seed = f"{run_root}/headline_scores_finbert4556_vader_baseline_v1_resume_seed.csv"
    log = f"{project}/artifacts/logs/lseg33-backward-finbert-{args.snapshot_id}.log"
    record = f"{run_root}/ucl_run_record.yaml"
    launcher = f"{run_root}/ucl_run_lseg_backward_incremental_finbert.sh"
    remote_script = " ".join(
        [
            "set -e;",
            f"cd {shlex.quote(run_root)};",
            "exec env",
            f"SNAPSHOT_ID={shlex.quote(args.snapshot_id)}",
            f"COLLECTION_ROOT={shlex.quote(collection)}",
            f"RUN_ROOT={shlex.quote(run_root)}",
            f"OUTPUT_PATH={shlex.quote(output)}",
            f"SEED_OUTPUT_PATH={shlex.quote(seed)}",
            f"LABEL_LOG_PATH={shlex.quote(log)}",
            f"RUN_RECORD_PATH={shlex.quote(record)}",
            f"LABEL_TMUX_SESSION={shlex.quote(args.remote_session)}",
            shlex.quote(launcher),
        ]
    )
    target = f"{args.user}@{host}.cs.ucl.ac.uk"
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={args.connect_timeout}",
        "-J",
        args.jump_host,
        target,
        f"bash -lc {shlex.quote(remote_script)}",
    ]
    return subprocess.run(command, check=False, capture_output=True, text=True, timeout=120)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--user", default="pprender")
    parser.add_argument("--jump-host", default="ucl-knuckles")
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--remote-session", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--connect-timeout", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds < 30 or args.workers < 1 or args.connect_timeout < 1:
        parser.error("poll-seconds must be at least 30; workers and connect-timeout must be positive")
    return args


def main() -> int:
    args = parse_args()
    _append_event(args.log, {"event": "watcher_started", "snapshot_id": args.snapshot_id})
    while True:
        try:
            free, counts = _free_hosts(args)
            _append_event(args.log, {"event": "pool_checked", "status_counts": counts, "free_hosts": free})
            for host in free:
                result = _launch(args, host)
                _append_event(
                    args.log,
                    {
                        "event": "launch_attempt",
                        "host": host,
                        "returncode": result.returncode,
                        "stdout_tail": result.stdout[-1_000:],
                        "stderr_tail": result.stderr[-1_000:],
                    },
                )
                if result.returncode == 0:
                    _append_event(args.log, {"event": "launched", "host": host, "remote_session": args.remote_session})
                    return 0
        except Exception as exc:
            _append_event(args.log, {"event": "watch_error", "error": f"{type(exc).__name__}: {exc}"})
        if args.once:
            return 1
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
