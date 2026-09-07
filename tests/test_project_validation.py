from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import check_project


def _manifest(entries: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source": {
            "repository": "https://github.com/PeterLP123/news-sentiment-beyond-mean",
            "commit": "59577da111f69f1d7678b922d3d753208bfba1ef",
        },
        "transfer_date": "2026-09-01",
        "integration_audit_date": "2026-09-07",
        "experiment_rerun": False,
        "amendments": [{"path": "manuscript/main.tex"}],
        "package_root": "submission/news-sentiment-beyond-mean",
        "files": entries,
    }


def _configure_manifest(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    manifest: object,
) -> Path:
    path = root / "submission" / "source_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(check_project, "ROOT", root)
    monkeypatch.setattr(check_project, "MANIFEST_PATH", path)
    return path


@pytest.mark.parametrize("mode", ["missing", "changed"])
def test_manifest_detects_missing_or_changed_snapshot_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    relative = "manuscript/main.tex"
    project_relative = f"submission/news-sentiment-beyond-mean/{relative}"
    submitted = tmp_path / project_relative
    submitted.parent.mkdir(parents=True)
    original = b"submitted bytes"
    if mode == "changed":
        submitted.write_bytes(b"changed bytes")
    entry = {
        "path": relative,
        "sha256": hashlib.sha256(original).hexdigest(),
        "size": len(original),
    }
    _configure_manifest(monkeypatch, tmp_path, _manifest([entry]))

    errors, _ = check_project.validate_manifest({project_relative})

    expected = "missing" if mode == "missing" else "changed"
    assert any(expected in error and relative in error for error in errors)


def test_markdown_link_to_ignored_file_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    readme = tmp_path / "README.md"
    ignored = tmp_path / "private.md"
    readme.write_text("[private](private.md)\n", encoding="utf-8")
    ignored.write_text("local only\n", encoding="utf-8")
    monkeypatch.setattr(check_project, "ROOT", tmp_path)
    monkeypatch.setattr(check_project, "MARKDOWN_FILES", ("README.md",))

    errors, checked = check_project.validate_markdown_links({"README.md"})

    assert checked == 1
    assert errors == ["README.md: local link is missing from Git: private.md"]


def test_licensed_csv_is_rejected_but_public_fiqa_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    licensed = tmp_path / "exports" / "news.csv"
    fiqa = tmp_path / "Data" / "source" / "fiqa_2018" / "task1_headline_ABSA_train.csv"
    licensed.parent.mkdir(parents=True)
    fiqa.parent.mkdir(parents=True)
    licensed.write_text("headline,label\nsecret,negative\n", encoding="utf-8")
    fiqa.write_text("headline,label\npublic,positive\n", encoding="utf-8")
    monkeypatch.setattr(check_project, "ROOT", tmp_path)

    errors = check_project.validate_public_files({"exports/news.csv", "Data/source/fiqa_2018/task1_headline_ABSA_train.csv"})

    assert errors == ["possible raw-text columns ['headline'] in exports/news.csv"]


def test_git_visible_symlink_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "published.txt"
    target.write_text("private\n", encoding="utf-8")
    link.symlink_to(target)
    monkeypatch.setattr(check_project, "ROOT", tmp_path)

    errors = check_project.validate_public_files({"published.txt"})

    assert errors == ["Git-visible file must not be a symlink: published.txt"]


def test_manifest_rejects_unsafe_path_and_non_object_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_manifest(monkeypatch, tmp_path, _manifest([{"path": "../outside"}]))
    errors, _ = check_project.validate_manifest(set())
    assert "unsafe manifest path: '../outside'" in errors

    check_project.MANIFEST_PATH.write_text("[]", encoding="utf-8")
    errors, count = check_project.validate_manifest(set())
    assert errors == ["submission manifest must be a JSON object"]
    assert count == 0
