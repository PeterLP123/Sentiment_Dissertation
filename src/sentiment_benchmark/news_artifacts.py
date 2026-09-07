from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NewsOutputPaths:
    output_dir: Path
    articles_jsonl: Path
    articles_csv: Path
    manifest_json: Path


def publish_news_corpus(
    output_root: str | Path,
    directory_name: str,
    *,
    jsonl_rows: Iterable[Mapping[str, Any]],
    csv_fieldnames: Sequence[str],
    csv_rows: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> NewsOutputPaths:
    """Publish a complete three-file news corpus under a collision-safe name."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{directory_name}.", suffix=".tmp", dir=root))
    try:
        with (staging_dir / "articles.jsonl").open("w", encoding="utf-8", newline="\n") as file:
            for row in jsonl_rows:
                file.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        with (staging_dir / "articles.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        (staging_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
            newline="\n",
        )

        suffix = 1
        while True:
            output_dir = root / (directory_name if suffix == 1 else f"{directory_name}_{suffix}")
            lock_dir = root / f".{output_dir.name}.publish-lock"
            try:
                lock_dir.mkdir()
            except FileExistsError:
                # A concurrent publisher (or an interrupted one) reserved this name.
                suffix += 1
                continue
            try:
                if os.path.lexists(output_dir):
                    suffix += 1
                    continue
                os.rename(staging_dir, output_dir)
            finally:
                lock_dir.rmdir()
            break
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return NewsOutputPaths(
        output_dir=output_dir,
        articles_jsonl=output_dir / "articles.jsonl",
        articles_csv=output_dir / "articles.csv",
        manifest_json=output_dir / "manifest.json",
    )
