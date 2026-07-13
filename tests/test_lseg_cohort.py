import json
from pathlib import Path

import pytest

from sentiment_benchmark.artifact_io import sha256_file
from sentiment_benchmark.lseg_cohort import build_lseg_analysis_cohort, load_verified_lseg_cohort
from sentiment_benchmark.lseg_source import LsegNewsError


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    rows = []
    for day in range(1, 11):
        for symbol, name in (("AAA", "Alpha Corp"), ("BBB", "Beta Corp")):
            for index in range(2):
                family = f"urn:{symbol}:{day}:{index}"
                for revision in (0, 1):
                    text = f"{name} announced results. {name} expects growth."
                    rows.append({
                        "article_id": family,
                        "revision_id": f"{family}:{revision}",
                        "story_id": f"{family}:{revision}",
                        "story_family_id": family,
                        "headline": f"{name} update",
                        "clean_text": text,
                        "clean_text_sha256": f"hash-{family}-{revision}",
                        "version_created": f"2026-01-{day:02d}T{10 + revision}:00:00+00:00",
                        "matched_symbols": [symbol],
                        "scoring_eligible": True,
                    })
    articles = root / "articles.jsonl"
    articles.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "config": {"companies": [
            {"symbol": "AAA", "name": "Alpha Corp", "aliases": ["Alpha Corp"]},
            {"symbol": "BBB", "name": "Beta Corp", "aliases": ["Beta Corp"]},
        ]},
        "files": {"articles_jsonl": {"path": "articles.jsonl", "sha256": sha256_file(articles)}},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "analysis.toml"
    path.write_text(
        "[cohort]\n"
        f'output_dir = "{(tmp_path / "out").as_posix()}"\n'
        'timezone = "America/New_York"\n'
        "development_fraction = 0.7\n"
        "development_size = 8\n"
        "holdout_size = 4\n"
        "l3_size = 3\n"
        "seed = 42\n",
        encoding="utf-8",
    )
    return path


def test_build_cohort_is_deterministic_and_uses_earliest_revision(tmp_path: Path) -> None:
    result = build_lseg_analysis_cohort(_corpus(tmp_path), _config(tmp_path))
    manifest, rows = load_verified_lseg_cohort(result.manifest_path)
    assert len(rows) == 12
    assert manifest["counts"]["development"] == 8
    assert manifest["counts"]["holdout"] == 4
    assert all(str(row["version_created"])[11:13] == "10" for row in rows)
    assert {row["chronological_split"] for row in rows} == {"development", "holdout"}
    assert len(result.l3_path.read_text().splitlines()) == 3


def test_build_cohort_refuses_existing_output(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    config = _config(tmp_path)
    build_lseg_analysis_cohort(corpus, config)
    with pytest.raises(LsegNewsError, match="refusing to overwrite"):
        build_lseg_analysis_cohort(corpus, config)
