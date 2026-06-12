"""Build a provenance-clean labeled sentiment dataset from the original sources.

Rebuilds the labeled benchmark directly from Financial PhraseBank v1.0
(Malo et al. 2014) and FiQA 2018 Task 1, replacing the Kaggle merge in
``Data/data.csv``. The provenance check of 2026-06-12 (M1-5) found that the
Kaggle file is PhraseBank Sentences_66Agree.txt plus FiQA, with every one of
the 514 negative PhraseBank sentences duplicated under a wrong ``neutral``
label — so its "conflicting duplicates" are corruption, not human
disagreement, and its strictest-disagreement sentences (the 50-65% agreement
tier) are missing entirely.

This script produces ``Data/derived/labeled/financial_sentiment_v2.csv`` with:

- ``Sentence`` / ``Sentiment`` — compatible with ``sentiment_benchmark.dataset``.
- ``source`` — ``financial_phrasebank`` or ``fiqa_2018``.
- ``pb_agreement_tier`` — 100/75/66/50, the strongest annotator-agreement file
  the sentence appears in (graded human-disagreement ground truth).
- ``fiqa_score`` / ``fiqa_n_annotations`` / ``fiqa_sign_conflict`` /
  ``fiqa_format`` — sentence-level aggregation of FiQA's continuous target
  scores. FiQA has no annotated neutral class: labels are score signs, so
  neutral-sensitive analyses should filter to ``source == financial_phrasebank``.

Sources are downloaded into ``Data/source/`` on first run and reused after.

Usage: python scripts/build_labeled_dataset.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "Data" / "source"
OUTPUT_DIR = REPO_ROOT / "Data" / "derived" / "labeled"
OUTPUT_CSV = OUTPUT_DIR / "financial_sentiment_v2.csv"
OUTPUT_README = OUTPUT_DIR / "README.md"

PHRASEBANK_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/"
    "FinancialPhraseBank-v1.0.zip"
)
FIQA_URLS = {
    split: f"https://huggingface.co/datasets/pauri32/fiqa-2018/resolve/main/{split}.csv"
    for split in ("train", "validation", "test")
}

# Strongest agreement file each sentence appears in wins (files are nested).
TIER_FILES = (
    ("Sentences_50Agree.txt", 50),
    ("Sentences_66Agree.txt", 66),
    ("Sentences_75Agree.txt", 75),
    ("Sentences_AllAgree.txt", 100),
)


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response:
        destination.write_bytes(response.read())


def fetch_sources() -> tuple[Path, dict[str, Path]]:
    phrasebank_zip = SOURCE_DIR / "financial_phrasebank" / "FinancialPhraseBank-v1.0.zip"
    _download(PHRASEBANK_URL, phrasebank_zip)
    fiqa_paths: dict[str, Path] = {}
    for split, url in FIQA_URLS.items():
        path = SOURCE_DIR / "fiqa_2018" / f"{split}.csv"
        _download(url, path)
        fiqa_paths[split] = path
    return phrasebank_zip, fiqa_paths


def load_phrasebank(zip_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    """One row per sentence with its label and strongest agreement tier.

    The source files contain a handful of exact-duplicate lines and two
    sentences that appear with different labels inside the same file; the
    duplicates are collapsed and the genuinely conflicting sentences dropped
    (returned for the build report) so every output sentence is unique.
    """
    tier_by_pair: dict[tuple[str, str], int] = {}
    labels_by_sentence: dict[str, set[str]] = defaultdict(set)
    with zipfile.ZipFile(zip_path) as archive:
        for file_name, tier in TIER_FILES:
            member = f"FinancialPhraseBank-v1.0/{file_name}"
            with archive.open(member) as raw:
                text = io.TextIOWrapper(raw, encoding="iso-8859-1")
                for line in text:
                    line = line.strip()
                    if not line:
                        continue
                    sentence, _, label = line.rpartition("@")
                    sentence, label = sentence.strip(), label.strip()
                    pair = (sentence, label)
                    tier_by_pair[pair] = max(tier_by_pair.get(pair, 0), tier)
                    labels_by_sentence[sentence].add(label)

    conflicting = sorted(s for s, labels in labels_by_sentence.items() if len(labels) > 1)
    rows = [
        {
            "Sentence": sentence,
            "Sentiment": label,
            "source": "financial_phrasebank",
            "pb_agreement_tier": tier,
            "fiqa_score": "",
            "fiqa_n_annotations": "",
            "fiqa_sign_conflict": "",
            "fiqa_format": "",
        }
        for (sentence, label), tier in sorted(tier_by_pair.items())
        if sentence not in set(conflicting)
    ]
    return rows, conflicting


def load_fiqa(paths: dict[str, Path]) -> list[dict[str, object]]:
    """One row per unique sentence with the mean of its target-level scores.

    FiQA annotates continuous sentiment per target entity; multi-entity
    sentences have several scores. The sentence-level label is the sign of the
    mean score (an exact zero would be neutral; none occurs), and
    ``fiqa_sign_conflict`` flags sentences whose target scores disagree in
    sign — FiQA's own form of within-sentence ambiguity.
    """
    scores: dict[str, list[float]] = defaultdict(list)
    formats: dict[str, str] = {}
    for path in paths.values():
        with path.open(newline="", encoding="utf-8") as file:
            for record in csv.DictReader(file):
                sentence = (record.get("sentence") or "").strip()
                if not sentence:
                    continue
                scores[sentence].append(float(record["sentiment_score"]))
                formats.setdefault(sentence, (record.get("format") or "").strip())

    rows: list[dict[str, object]] = []
    for sentence in sorted(scores):
        values = scores[sentence]
        mean_score = sum(values) / len(values)
        if mean_score > 0:
            label = "positive"
        elif mean_score < 0:
            label = "negative"
        else:
            label = "neutral"
        signs = {value > 0 for value in values if value != 0}
        rows.append(
            {
                "Sentence": sentence,
                "Sentiment": label,
                "source": "fiqa_2018",
                "pb_agreement_tier": "",
                "fiqa_score": f"{mean_score:.6f}",
                "fiqa_n_annotations": len(values),
                "fiqa_sign_conflict": int(len(signs) > 1),
                "fiqa_format": formats[sentence],
            }
        )
    return rows


def main() -> int:
    phrasebank_zip, fiqa_paths = fetch_sources()
    pb_rows, pb_conflicts = load_phrasebank(phrasebank_zip)
    fiqa_rows = load_fiqa(fiqa_paths)

    overlap = {row["Sentence"] for row in pb_rows} & {row["Sentence"] for row in fiqa_rows}
    if overlap:
        raise SystemExit(f"Unexpected sentence overlap between sources: {len(overlap)} sentences")

    rows = pb_rows + fiqa_rows
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Sentence",
        "Sentiment",
        "source",
        "pb_agreement_tier",
        "fiqa_score",
        "fiqa_n_annotations",
        "fiqa_sign_conflict",
        "fiqa_format",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sha256 = hashlib.sha256(OUTPUT_CSV.read_bytes()).hexdigest()
    label_counts: dict[str, int] = defaultdict(int)
    tier_counts: dict[object, int] = defaultdict(int)
    for row in rows:
        label_counts[str(row["Sentiment"])] += 1
        if row["source"] == "financial_phrasebank":
            tier_counts[row["pb_agreement_tier"]] += 1

    report = f"""# financial_sentiment_v2

Provenance-clean rebuild of the labeled sentiment benchmark, built by
`scripts/build_labeled_dataset.py` directly from the original sources. It
replaces `Data/data.csv` (the Kaggle merge), whose 514 "conflicting
duplicates" were found on 2026-06-12 to be corrupt rows — every negative
PhraseBank sentence duplicated under a wrong `neutral` label — and which
silently dropped the 50-65% agreement tier (the most human-contested
sentences).

- Rows: {len(rows)} ({len(pb_rows)} PhraseBank + {len(fiqa_rows)} FiQA; every sentence unique)
- Labels: {dict(sorted(label_counts.items()))}
- PhraseBank agreement tiers (strongest file containing the sentence): {dict(sorted(tier_counts.items()))}
- PhraseBank sentences dropped for in-source label conflicts: {len(pb_conflicts)}
- SHA-256: `{sha256}`

## Sources

- Financial PhraseBank v1.0 (Malo et al. 2014), CC BY-NC-SA 3.0 —
  `Data/source/financial_phrasebank/FinancialPhraseBank-v1.0.zip`
- FiQA 2018 Task 1 (Maia et al. 2018) via `pauri32/fiqa-2018` on Hugging Face —
  `Data/source/fiqa_2018/{{train,validation,test}}.csv`

## Column notes

- `pb_agreement_tier`: 100/75/66/50 — minimum annotator-agreement file the
  sentence appears in (16 annotators, 5-8 annotations per sentence). This is
  the graded human-disagreement ground truth for ambiguity-proxy validation.
- FiQA rows are sentence-level aggregates of per-target continuous scores;
  the label is the sign of the mean score. FiQA has **no annotated neutral
  class** — restrict neutral-sensitive analyses to `source ==
  financial_phrasebank`. `fiqa_sign_conflict = 1` marks sentences whose
  target-level scores disagree in sign.

## Caveats

- Both sources are public and predate all candidate-model training cutoffs;
  the contamination/look-ahead caveat in `docs/research_protocol.md` applies.
- PhraseBank license is CC BY-NC-SA 3.0 (non-commercial; fine for the
  dissertation with attribution).
"""
    OUTPUT_README.write_text(report, encoding="utf-8")

    print(f"Wrote {OUTPUT_CSV} ({len(rows)} rows)")
    print(f"  PhraseBank: {len(pb_rows)} rows; dropped in-source conflicts: {len(pb_conflicts)}")
    print(f"  FiQA: {len(fiqa_rows)} rows")
    print(f"  Labels: {dict(sorted(label_counts.items()))}")
    print(f"  Tiers: {dict(sorted(tier_counts.items()))}")
    print(f"  SHA-256: {sha256}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
