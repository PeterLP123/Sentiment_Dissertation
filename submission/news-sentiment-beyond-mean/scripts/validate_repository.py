"""Fail fast when the clean repository drifts outside its research boundary."""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".sqlite",
    ".sqlite3",
    ".db",
    ".parquet",
    ".jsonl",
    ".arrow",
    ".feather",
    ".pkl",
    ".pickle",
}
RAW_TEXT_COLUMNS = {
    "headline",
    "headline_text",
    "body",
    "main_body",
    "article",
    "text",
    "raw_response",
    "prompt",
}
SECRET_PATTERNS = {
    "OpenRouter key": re.compile(r"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    "generic API assignment": re.compile(
        r"(?i)(api[_-]?key|token)\s*[=:]\s*['\"][A-Za-z0-9_-]{24,}['\"]"
    ),
}


def repository_files() -> list[Path]:
    ignored_roots = {
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "generated",
    }
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in ignored_roots for part in path.relative_to(ROOT).parts)
    )


def validate_no_large_or_row_level_artifacts(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden row-level or binary data artifact: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB clean-repository limit: {relative}")
        if path.suffix.lower() == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as handle:
                header = next(csv.reader(handle), [])
            exposed = RAW_TEXT_COLUMNS.intersection(
                column.strip().lower() for column in header
            )
            if exposed:
                errors.append(f"possible raw-text columns {sorted(exposed)} in {relative}")
    return errors


def validate_json(files: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in files:
        if path.suffix.lower() not in {".json", ".ipynb"}:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON in {path.relative_to(ROOT)}: {exc}")
    return errors


def validate_secrets(files: list[Path]) -> list[str]:
    errors: list[str] = []
    text_suffixes = {
        ".md",
        ".py",
        ".toml",
        ".json",
        ".ipynb",
        ".tex",
        ".bib",
        ".cff",
        "",
    }
    for path in files:
        if path.suffix.lower() not in text_suffixes or path.stat().st_size > 5 * 1024 * 1024:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"possible {label} in {path.relative_to(ROOT)}")
    return errors


def validate_manuscript_figures() -> list[str]:
    manuscript_root = ROOT / "manuscript"
    main = manuscript_root / "main.tex"
    if not main.exists():
        return ["missing manuscript/main.tex"]
    errors: list[str] = []
    extensions = ("", ".pdf", ".png", ".jpg", ".jpeg", ".svg")
    for tex_path in sorted(manuscript_root.rglob("*.tex")):
        source = tex_path.read_text(encoding="utf-8")
        for name in re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source
        ):
            stems = [
                manuscript_root / name,
                tex_path.parent / name,
                manuscript_root / "figures" / name,
                manuscript_root / "artifacts" / name,
            ]
            if not any(
                Path(f"{stem}{extension}").exists()
                for stem in stems
                for extension in extensions
            ):
                errors.append(
                    f"missing manuscript graphic referenced by "
                    f"{tex_path.relative_to(ROOT)}: {name}"
                )
    return errors


def validate_manuscript_sources() -> list[str]:
    manuscript_root = ROOT / "manuscript"
    main = manuscript_root / "main.tex"
    bibliography = manuscript_root / "references.bib"
    errors: list[str] = []
    if not main.exists() or not bibliography.exists():
        return ["missing manuscript/main.tex or manuscript/references.bib"]

    tex_paths = sorted(manuscript_root.rglob("*.tex"))
    joined = "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)

    for tex_path in tex_paths:
        source = tex_path.read_text(encoding="utf-8")
        for command, name in re.findall(r"\\(include|input)\{([^}]+)\}", source):
            candidates = [
                manuscript_root / f"{name}.tex",
                tex_path.parent / f"{name}.tex",
                manuscript_root / name,
                tex_path.parent / name,
            ]
            if not any(candidate.exists() for candidate in candidates):
                errors.append(
                    f"missing manuscript {command} referenced by "
                    f"{tex_path.relative_to(ROOT)}: {name}"
                )

    marker_patterns = {
        "TODO marker": r"\bTODO\b",
        "TBC marker": r"\bTBC\b",
        "FIXME marker": r"\bFIXME\b",
        "draft-note command": r"\\draftnote\b",
        "placeholder marker": r"\bplaceholder\b",
        "unresolved submission check": r"submission should be checked",
    }
    for label, pattern in marker_patterns.items():
        if re.search(pattern, joined, flags=re.IGNORECASE):
            errors.append(f"{label} remains in manuscript sources")

    citation_pattern = re.compile(
        r"\\(?:cite|citep|citet|citealp|citeauthor|Citet)"
        r"(?:\[[^]]*\]){0,2}\{([^}]+)\}"
    )
    cited = {
        key.strip()
        for group in citation_pattern.findall(joined)
        for key in group.split(",")
        if key.strip()
    }
    bib_source = bibliography.read_text(encoding="utf-8")
    available = {
        match.strip()
        for match in re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib_source)
    }
    missing = sorted(cited - available)
    unused = sorted(available - cited)
    if missing:
        errors.append(f"missing bibliography keys: {missing}")
    if unused:
        errors.append(f"uncited bibliography entries: {unused}")

    reference_pattern = re.compile(
        r"\\(?:ref|autoref|cref|Cref)\{([^}]+)\}"
    )
    referenced_labels = {
        label.strip()
        for group in reference_pattern.findall(joined)
        for label in group.split(",")
        if label.strip()
    }
    float_labels = set(re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", joined))
    unreferenced_floats = sorted(float_labels - referenced_labels)
    if unreferenced_floats:
        errors.append(f"unreferenced manuscript floats: {unreferenced_floats}")

    required_fragments = {
        "research question": "is a higher negative-story-share rank associated",
        "UCL authorship declaration": "substantially the result of my own work",
        "AI disclosure": "OpenAI Codex",
        "project-summary appendix": "\\include{appendices/project_summary}",
    }
    for label, fragment in required_fragments.items():
        if fragment not in joined:
            errors.append(f"missing required manuscript element: {label}")
    return errors


def validate_notebook_boundary() -> list[str]:
    notebook_dir = ROOT / "experiments" / "notebooks"
    notebooks = sorted(notebook_dir.glob("*.ipynb"))
    errors: list[str] = []
    if len(notebooks) != 22:
        errors.append(f"expected 22 curated notebooks, found {len(notebooks)}")
    for path in notebooks:
        source = path.read_text(encoding="utf-8")
        if "final_experiments.lib" in source:
            errors.append(f"legacy helper import remains in {path.relative_to(ROOT)}")
    return errors


def main() -> None:
    files = repository_files()
    errors = [
        *validate_no_large_or_row_level_artifacts(files),
        *validate_json(files),
        *validate_secrets(files),
        *validate_manuscript_figures(),
        *validate_manuscript_sources(),
        *validate_notebook_boundary(),
    ]
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(
        f"Repository boundary valid: {len(files)} files checked; "
        "no raw-data artifact or obvious secret found."
    )


if __name__ == "__main__":
    main()
