from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from . import __version__


def _clean_text(value: object, max_length: int = 256) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    if not text:
        return None
    return text[:max_length]


def _normalize_machine_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", value.strip())
    normalized = normalized.strip("-._:")
    return normalized[:64] or _hash_machine_source(value)


def _hash_machine_source(source: str) -> str:
    digest = hashlib.sha256(f"sentiment-benchmark-machine-v1:{source}".encode()).hexdigest()
    return f"machine-{digest[:16]}"


def _windows_machine_guid() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
    except OSError:
        return None
    return _clean_text(value)


def _linux_machine_id() -> str | None:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    return None


def machine_identity() -> dict[str, str | None]:
    configured_id = _clean_text(os.getenv("SENTIMENT_BENCH_MACHINE_ID"), max_length=128)
    label = _clean_text(os.getenv("SENTIMENT_BENCH_MACHINE_LABEL") or platform.node() or socket.gethostname())
    if configured_id:
        return {
            "id": _normalize_machine_id(configured_id),
            "label": label,
            "id_source": "env:SENTIMENT_BENCH_MACHINE_ID",
        }

    stable_sources = [
        ("windows_machine_guid_sha256", _windows_machine_guid()),
        ("linux_machine_id_sha256", _linux_machine_id()),
        ("hostname_sha256", platform.node() or socket.gethostname()),
    ]
    for source_name, source_value in stable_sources:
        if source_value:
            return {
                "id": _hash_machine_source(source_value),
                "label": label,
                "id_source": source_name,
            }

    return {"id": None, "label": label, "id_source": None}


def _git_value(args: list[str], repo_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _clean_text(result.stdout)


def git_metadata(repo_path: str | Path = ".") -> dict[str, Any]:
    root = Path(repo_path)
    inside = _git_value(["rev-parse", "--is-inside-work-tree"], root)
    if inside != "true":
        return {"available": False}

    status = _git_value(["status", "--short"], root)
    return {
        "available": True,
        "commit": _git_value(["rev-parse", "HEAD"], root),
        "branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(status),
    }


def collect_run_environment(repo_path: str | Path = ".") -> dict[str, Any]:
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    offset_minutes = int(offset.total_seconds() // 60) if offset is not None else None
    return {
        "machine": machine_identity(),
        "platform": {
            "system": _clean_text(platform.system()),
            "release": _clean_text(platform.release()),
            "version": _clean_text(platform.version()),
            "machine": _clean_text(platform.machine()),
            "processor": _clean_text(platform.processor()),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "package_version": __version__,
        "git": git_metadata(repo_path),
        "database": {
            "backend": (os.getenv("SENTIMENT_BENCH_DB_BACKEND", "sqlite").strip().lower() or "sqlite"),
        },
        "timezone": {
            "name": now.tzname(),
            "utc_offset_minutes": offset_minutes,
        },
    }
