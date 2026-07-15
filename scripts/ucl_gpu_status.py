#!/usr/bin/env python3
"""Check the availability of the UCL CS public lab GPUs over SSH.

The host inventory is a dated snapshot of https://tsg.cs.ucl.ac.uk/gpus/.
SSH is always run in batch mode, so this script never prompts for or stores a
password. Configure public-key authentication before using it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

INVENTORY_SOURCE = "https://tsg.cs.ucl.ac.uk/gpus/"
INVENTORY_CHECKED = "2026-07-15"
DOMAIN = ".cs.ucl.ac.uk"

LAB_HOSTS: dict[str, tuple[str, ...]] = {
    "lab105": (
        "aylesbury-l",
        "barnacle-l",
        "brent-l",
        "bufflehead-l",
        "cackling-l",
        "canada-l",
        "crested-l",
        "eider-l",
        "gadwall-l",
        "goosander-l",
        "gressingham-l",
        "harlequin-l",
        "mallard-l",
        "mandarin-l",
        "pintail-l",
        "pocher-l",
        "ruddy-l",
        "scaup-l",
        "scoter-l",
        "shelduck-l",
        "shoveler-l",
        "smew-l",
        "wigeon-l",
    ),
    "lab121": (
        "albacore-l",
        "barbel-l",
        "chub-l",
        "cripps-l",
        "dory-l",
        "elver-l",
        "flounder-l",
        "goldeye-l",
        "hake-l",
        "inanga-l",
        "javelin-l",
        "koi-l",
        "lamprey-l",
        "mackerel-l",
        "mullet-l",
        "nase-l",
        "opah-l",
        "pike-l",
        "plaice-l",
        "quillback-l",
        "roach-l",
        "rudd-l",
        "shark-l",
        "skate-l",
        "tench-l",
        "tope-l",
        "uaru-l",
        "vimba-l",
        "whitebait-l",
        "yellowtail-l",
        "zander-l",
    ),
}

GPU_LIST_MARKER = "__UCL_GPU_LIST__"
PROCESS_LIST_MARKER = "__UCL_GPU_PROCESSES__"
END_MARKER = "__UCL_GPU_END__"
REMOTE_COMMAND = f"""set -eu
printf '{GPU_LIST_MARKER}\\n'
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits
printf '{PROCESS_LIST_MARKER}\\n'
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
printf '{END_MARKER}\\n'
"""


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_total_mib: int | None
    memory_used_mib: int | None
    utilization_percent: int | None
    compute_processes: int
    compute_memory_mib: int | None


@dataclass(frozen=True)
class HostReport:
    host: str
    lab: str
    status: str
    gpus: tuple[GPU, ...] = ()
    detail: str = ""


def _optional_int(value: str) -> int | None:
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return int(normalized)
    except ValueError:
        return None


def _csv_rows(block: str) -> list[list[str]]:
    return [
        [field.strip() for field in row]
        for row in csv.reader(block.splitlines())
        if row and any(field.strip() for field in row) and "No running processes found" not in row[0]
    ]


def parse_nvidia_smi(output: str) -> tuple[GPU, ...]:
    """Parse the two CSV queries emitted by ``REMOTE_COMMAND``."""
    if GPU_LIST_MARKER not in output or PROCESS_LIST_MARKER not in output or END_MARKER not in output:
        raise ValueError("nvidia-smi output markers are missing")

    gpu_block = output.split(GPU_LIST_MARKER, 1)[1].split(PROCESS_LIST_MARKER, 1)[0]
    process_block = output.split(PROCESS_LIST_MARKER, 1)[1].split(END_MARKER, 1)[0]
    if "No devices were found" in gpu_block or "Unable to determine the device handle" in gpu_block:
        raise ValueError("nvidia-smi found no usable GPU (driver or device error)")
    processes: dict[str, list[int | None]] = {}
    for row in _csv_rows(process_block):
        if len(row) < 3:
            continue
        processes.setdefault(row[0], []).append(_optional_int(row[2]))

    gpus: list[GPU] = []
    for row in _csv_rows(gpu_block):
        if len(row) != 6:
            raise ValueError(f"unexpected GPU row with {len(row)} fields")
        process_memory = processes.get(row[1], [])
        known_process_memory = [value for value in process_memory if value is not None]
        gpus.append(
            GPU(
                index=int(row[0]),
                uuid=row[1],
                name=row[2],
                memory_total_mib=_optional_int(row[3]),
                memory_used_mib=_optional_int(row[4]),
                utilization_percent=_optional_int(row[5]),
                compute_processes=len(process_memory),
                compute_memory_mib=sum(known_process_memory) if known_process_memory else None,
            )
        )
    if not gpus:
        raise ValueError("nvidia-smi reported no GPUs")
    return tuple(gpus)


def classify_ssh_failure(stderr: str) -> tuple[str, str]:
    """Map common SSH failures to stable, user-facing states."""
    message = " ".join(stderr.strip().split())
    lowered = message.lower()
    if "permission denied" in lowered or "no supported authentication methods" in lowered:
        return "AUTH", "SSH key authentication failed"
    if "host key verification failed" in lowered or "remote host identification has changed" in lowered:
        return "HOST_KEY", "SSH host-key verification failed"
    if "administratively prohibited" in lowered or "stdio forwarding failed" in lowered:
        return "OFFLINE", "SSH gateway refused forwarding to this host"
    offline_fragments = (
        "could not resolve hostname",
        "connection refused",
        "connection timed out",
        "operation timed out",
        "no route to host",
        "network is unreachable",
        "connection closed",
        "connection reset",
    )
    if any(fragment in lowered for fragment in offline_fragments):
        return "OFFLINE", "host is unreachable or not running Linux"
    if "nvidia-smi" in lowered:
        return "ERROR", "connected, but nvidia-smi failed"
    return "ERROR", message[-160:] or "SSH command failed"


def ssh_command(host: str, user: str, jump_host: str, connect_timeout: int) -> list[str]:
    target = f"{user}@{host}{DOMAIN}" if not host.endswith(DOMAIN) else f"{user}@{host}"
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={connect_timeout}",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "LogLevel=ERROR",
        "-J",
        jump_host,
        target,
        REMOTE_COMMAND,
    ]


def check_host(host: str, lab: str, user: str, jump_host: str, connect_timeout: int) -> HostReport:
    command = ssh_command(host, user, jump_host, connect_timeout)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=max(10, connect_timeout * 3),
        )
    except subprocess.TimeoutExpired:
        return HostReport(host=host, lab=lab, status="OFFLINE", detail="SSH check timed out")

    if completed.returncode != 0:
        status, detail = classify_ssh_failure(completed.stderr)
        return HostReport(host=host, lab=lab, status=status, detail=detail)
    try:
        gpus = parse_nvidia_smi(completed.stdout)
    except ValueError as exc:
        return HostReport(host=host, lab=lab, status="ERROR", detail=str(exc))
    status = "TAKEN" if any(gpu.compute_processes for gpu in gpus) else "FREE"
    return HostReport(host=host, lab=lab, status=status, gpus=gpus)


def infer_user(explicit_user: str | None, jump_host: str) -> str:
    if explicit_user:
        return explicit_user
    environment_user = os.environ.get("UCL_CS_USER")
    if environment_user:
        return environment_user
    alias = jump_host.split("@")[-1]
    completed = subprocess.run(
        ["ssh", "-G", alias],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    settings = dict(
        line.split(maxsplit=1)
        for line in completed.stdout.splitlines()
        if line.startswith(("hostname ", "user "))
    )
    if completed.returncode == 0 and settings.get("hostname") != alias and settings.get("user"):
        return settings["user"]
    raise ValueError("set --user, UCL_CS_USER, or a configured SSH jump-host alias with the UCL username")


def selected_hosts(labs: list[str], explicit_hosts: list[str]) -> list[tuple[str, str]]:
    if explicit_hosts:
        return [(host.removesuffix(DOMAIN), "custom") for host in explicit_hosts]
    return [(host, lab) for lab in labs for host in LAB_HOSTS[lab]]


def _gpu_summary(report: HostReport) -> str:
    if not report.gpus:
        return report.detail
    names = ", ".join(dict.fromkeys(gpu.name for gpu in report.gpus))
    process_count = sum(gpu.compute_processes for gpu in report.gpus)
    memory_used = sum(gpu.memory_used_mib or 0 for gpu in report.gpus)
    memory_total = sum(gpu.memory_total_mib or 0 for gpu in report.gpus)
    usage = f"{memory_used}/{memory_total} MiB" if memory_total else "memory unknown"
    process_text = f", {process_count} compute process{'es' if process_count != 1 else ''}" if process_count else ""
    return f"{names}; {usage}{process_text}"


def print_table(reports: list[HostReport]) -> None:
    widths = (16, 7, 8)
    print(f"{'HOST':<{widths[0]}} {'LAB':<{widths[1]}} {'STATUS':<{widths[2]}} DETAILS")
    print(f"{'-' * widths[0]} {'-' * widths[1]} {'-' * widths[2]} {'-' * 40}")
    for report in reports:
        print(
            f"{report.host:<{widths[0]}} {report.lab:<{widths[1]}} "
            f"{report.status:<{widths[2]}} {_gpu_summary(report)}"
        )
    statuses = ("FREE", "TAKEN", "OFFLINE", "AUTH", "HOST_KEY", "ERROR")
    counts = {status: sum(report.status == status for report in reports) for status in statuses}
    summary = ", ".join(f"{status.lower()}={count}" for status, count in counts.items() if count)
    print(f"\nChecked {len(reports)} hosts: {summary}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", help="UCL CS username (or set UCL_CS_USER; otherwise inferred from the jump alias)")
    parser.add_argument("--jump-host", default="ucl-knuckles", help="SSH gateway or configured alias (default: ucl-knuckles)")
    parser.add_argument("--lab", action="append", choices=sorted(LAB_HOSTS), help="lab to check; repeatable (default: both)")
    parser.add_argument("--host", action="append", default=[], help="check only this host; repeatable")
    parser.add_argument("--workers", type=int, default=12, help="parallel SSH checks (default: 12)")
    parser.add_argument("--connect-timeout", type=int, default=5, help="SSH connection timeout in seconds (default: 5)")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.connect_timeout < 1:
        raise SystemExit("--workers and --connect-timeout must be positive integers")
    try:
        user = infer_user(args.user, args.jump_host)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    hosts = selected_hosts(args.lab or list(LAB_HOSTS), args.host)
    reports: list[HostReport] = []
    with ThreadPoolExecutor(max_workers=min(args.workers, len(hosts))) as executor:
        futures = {
            executor.submit(check_host, host, lab, user, args.jump_host, args.connect_timeout): (host, lab)
            for host, lab in hosts
        }
        for future in as_completed(futures):
            reports.append(future.result())
    reports.sort(key=lambda report: (report.lab, report.host))

    if args.json:
        payload: dict[str, Any] = {
            "inventory": {"source": INVENTORY_SOURCE, "checked": INVENTORY_CHECKED},
            "reports": [asdict(report) for report in reports],
        }
        json.dump(payload, sys.stdout, indent=2)
        print()
    else:
        print_table(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
