#!/usr/bin/env python3
"""Validate the public, licence-safe project checkout with the standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "submission" / "source_manifest.json"
MAX_PUBLIC_FILE_SIZE = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".arrow", ".db", ".feather", ".jsonl", ".parquet", ".pickle", ".pkl", ".sqlite", ".sqlite3"}
CACHE_PARTS = {".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "checkpoints"}
RAW_TEXT_COLUMNS = {"article", "body", "headline", "headline_text", "main_body", "prompt", "raw_response", "text"}
CSV_RAW_TEXT_EXACT = {"Data/derived/labeled/financial_sentiment_v2.csv"}
CSV_RAW_TEXT_PREFIXES = ("Data/source/fiqa_2018/", "tests/fixtures/")
MARKDOWN_FILES = (
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/README.md",
    "docs/research_archive.md",
    "docs/reproducing_the_submission.md",
    "docs/data_and_rights.md",
    "submission/README.md",
)
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def git_visible_files() -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and "\\" not in value and ".." not in path.parts and str(path) == value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(visible: set[str]) -> tuple[list[str], int]:
    errors: list[str] = []
    try:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read submission manifest: {exc}"], 0
    if not isinstance(manifest, dict):
        return ["submission manifest must be a JSON object"], 0
    expected_metadata = {
        "schema_version": 1,
        "transfer_date": "2026-09-01",
        "integration_audit_date": "2026-09-07",
        "experiment_rerun": False,
        "package_root": "submission/news-sentiment-beyond-mean",
    }
    for key, expected in expected_metadata.items():
        if manifest.get(key) != expected:
            errors.append(f"manifest field {key!r} must be {expected!r}")
    source = manifest.get("source", {})
    if source.get("repository") != "https://github.com/PeterLP123/news-sentiment-beyond-mean":
        errors.append("manifest source repository is incorrect")
    if source.get("commit") != "59577da111f69f1d7678b922d3d753208bfba1ef":
        errors.append("manifest source commit is incorrect")
    amendments = manifest.get("amendments", [])
    if not any(item.get("path") == "manuscript/main.tex" for item in amendments):
        errors.append("manifest does not record the candidate-number amendment")
    entries = manifest.get("files", [])
    if len(entries) != 387:
        errors.append(f"manifest must contain 387 submitted files, found {len(entries)}")
    package_root = manifest.get("package_root", "")
    listed: set[str] = set()
    for entry in entries:
        relative = entry.get("path", "")
        if not safe_relative_path(relative):
            errors.append(f"unsafe manifest path: {relative!r}")
            continue
        if relative in listed:
            errors.append(f"duplicate manifest path: {relative}")
            continue
        listed.add(relative)
        project_relative = f"{package_root}/{relative}"
        path = ROOT / project_relative
        if project_relative not in visible:
            errors.append(f"submitted file is not Git-visible: {project_relative}")
            continue
        if path.is_symlink():
            errors.append(f"submitted file must not be a symlink: {project_relative}")
            continue
        if not path.is_file():
            errors.append(f"submitted file is missing: {project_relative}")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry.get("size"):
            errors.append(f"submitted file size changed: {relative} (expected {entry.get('size')}, found {actual_size})")
        actual_hash = sha256(path)
        if actual_hash != entry.get("sha256"):
            errors.append(f"submitted file hash changed: {relative} (expected {entry.get('sha256')}, found {actual_hash})")
    visible_package = {path.removeprefix(f"{package_root}/") for path in visible if path.startswith(f"{package_root}/")}
    extras = sorted(visible_package - listed)
    if extras:
        errors.append("unmanifested files in submitted package: " + ", ".join(extras))
    return errors, len(entries)


def allowed_raw_csv(path: str) -> bool:
    return path in CSV_RAW_TEXT_EXACT or path.startswith(CSV_RAW_TEXT_PREFIXES)


def validate_public_files(visible: set[str]) -> list[str]:
    errors: list[str] = []
    for relative in sorted(visible):
        path = ROOT / relative
        if path.is_symlink():
            errors.append(f"Git-visible file must not be a symlink: {relative}")
            continue
        if not path.is_file():
            errors.append(f"Git-visible path is missing or not a file: {relative}")
            continue
        size = path.stat().st_size
        if size > MAX_PUBLIC_FILE_SIZE:
            errors.append(f"public file exceeds 10 MiB: {relative} ({size} bytes)")
        parts = PurePosixPath(relative).parts
        name = PurePosixPath(relative).name
        if any(part in CACHE_PARTS for part in parts):
            errors.append(f"cache or local environment is Git-visible: {relative}")
        if name.startswith(".env") and name != ".env.example":
            errors.append(f"environment file is Git-visible: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not relative.startswith("tests/fixtures/"):
            errors.append(f"forbidden row-level or binary artifact is Git-visible: {relative}")
        if path.suffix.lower() == ".csv" and not allowed_raw_csv(relative):
            try:
                with path.open(newline="", encoding="utf-8-sig") as handle:
                    header = next(csv.reader(handle), [])
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                errors.append(f"cannot inspect CSV header {relative}: {exc}")
                continue
            exposed = RAW_TEXT_COLUMNS.intersection(column.strip().lower() for column in header)
            if exposed:
                errors.append(f"possible raw-text columns {sorted(exposed)} in {relative}")
    return errors


def markdown_targets(source: str) -> list[str]:
    targets: list[str] = []
    in_fence = False
    for line in source.splitlines():
        if line.lstrip().startswith((chr(96) * 3, "~~~")):
            in_fence = not in_fence
            continue
        if not in_fence:
            targets.extend(match.group(1).strip() for match in LINK_PATTERN.finditer(line))
    return targets


def validate_markdown_links(visible: set[str]) -> tuple[list[str], int]:
    errors: list[str] = []
    checked = 0
    for relative in MARKDOWN_FILES:
        if relative not in visible:
            errors.append(f"required documentation file is not Git-visible: {relative}")
            continue
        source_path = ROOT / relative
        for raw_target in markdown_targets(source_path.read_text(encoding="utf-8")):
            target = raw_target
            if target.startswith("<") and ">" in target:
                target = target[1 : target.index(">")]
            else:
                target = target.split(maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("#"):
                continue
            decoded = unquote(parsed.path)
            if not decoded:
                continue
            checked += 1
            candidate = (ROOT / decoded.lstrip("/") if decoded.startswith("/") else source_path.parent / decoded).resolve()
            try:
                project_relative = candidate.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append(f"{relative}: link escapes repository: {raw_target}")
                continue
            if candidate.is_dir():
                prefix = project_relative.rstrip("/") + "/"
                if not any(path.startswith(prefix) for path in visible):
                    errors.append(f"{relative}: linked directory is empty in Git: {raw_target}")
            elif project_relative not in visible:
                errors.append(f"{relative}: local link is missing from Git: {raw_target}")
    return errors, checked


def main() -> int:
    try:
        visible = git_visible_files()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: cannot enumerate Git-visible files: {exc}", file=sys.stderr)
        return 1
    manifest_errors, manifest_count = validate_manifest(visible)
    public_errors = validate_public_files(visible)
    link_errors, link_count = validate_markdown_links(visible)
    errors = [*manifest_errors, *public_errors, *link_errors]
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"Project validation: FAIL ({len(errors)} issue(s))", file=sys.stderr)
        return 1
    print(
        "Project validation: PASS "
        f"({len(visible)} Git-visible files, {manifest_count} submitted files, {link_count} local documentation links)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
