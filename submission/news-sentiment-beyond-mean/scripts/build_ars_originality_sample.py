"""Build a deterministic final-integrity originality sample for the manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = "ars-originality-final-v1"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for",
    "from", "has", "have", "in", "is", "it", "its", "not", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "with",
}


def prose_text(raw: str) -> str:
    text = " ".join(line.split("%", 1)[0].strip() for line in raw.splitlines())
    text = re.sub(r"\\(?:cite\w*|[Cc]ref)\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z*]+(?:\[[^]]*\])?\{([^{}]*)\}", r" \1 ", text)
    text = re.sub(r"\\[A-Za-z*]+", " ", text)
    text = re.sub(r"[$~{}_^&]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)


def characteristic_fragment(tokens: list[str], width: int = 9) -> str:
    if len(tokens) <= width:
        return " ".join(tokens)
    best: tuple[int, int, list[str]] | None = None
    for index in range(len(tokens) - width + 1):
        window = tokens[index:index + width]
        score = sum(len(token) for token in window if token.lower() not in STOPWORDS)
        candidate = (score, -index, window)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return " ".join(best[2])


def paragraphs(path: Path) -> list[dict[str, object]]:
    rows = []
    for ordinal, raw in enumerate(re.split(r"\n\s*\n", path.read_text(encoding="utf-8")), start=1):
        text = prose_text(raw)
        tokens = words(text)
        if len(tokens) < 20 or "\\caption" in raw:
            continue
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "paragraph_ordinal": ordinal,
                "word_count": len(tokens),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "fragment": characteristic_fragment(tokens),
                "text": text,
            }
        )
    return rows


def sample_key(row: dict[str, object]) -> str:
    material = f"{SEED}|{row['path']}|{row['paragraph_ordinal']}|{row['text_sha256']}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    chapter_paths = sorted((ROOT / "manuscript" / "chapters").glob("0[1-6]_*.tex"))
    frame = [row for path in chapter_paths for row in paragraphs(path)]
    sample_size = math.ceil(len(frame) * 0.50)
    sampled = sorted(frame, key=sample_key)[:sample_size]
    for row in sampled:
        row["selection_reason"] = ["deterministic_50pct_sample"]

    modified_paths = [
        ROOT / "manuscript" / "chapters" / "05_results.tex",
        ROOT / "manuscript" / "appendices" / "project_summary.tex",
    ]
    modified = []
    for path in modified_paths:
        for row in paragraphs(path):
            if "measured baseline pipeline association" in str(row["text"]):
                modified.append(row)
    selected_by_hash = {str(row["text_sha256"]): row for row in sampled}
    for row in modified:
        existing = selected_by_hash.get(str(row["text_sha256"]))
        if existing is not None:
            existing["selection_reason"].append("modified_round2")
        else:
            row["selection_reason"] = ["modified_round2"]
            sampled.append(row)
            selected_by_hash[str(row["text_sha256"])] = row

    output = {
        "schema_version": "ars-originality-sample/1.0",
        "seed": SEED,
        "chapter_frame_paragraphs": len(frame),
        "chapter_sample_paragraphs": sample_size,
        "chapter_sample_rate": sample_size / len(frame),
        "modified_paragraphs_required": len(modified),
        "total_search_fragments": len(sampled),
        "rows": sampled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"originality sample: {sample_size}/{len(frame)} chapter paragraphs; "
        f"{len(modified)} modified; {len(sampled)} searches"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
