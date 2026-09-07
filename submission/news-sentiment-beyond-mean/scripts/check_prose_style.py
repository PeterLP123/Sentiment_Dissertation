"""Reject stock AI-writing phrases in the dissertation prose.

This is deliberately narrow. It catches phrases that should never survive the
edit, while the manual review remains responsible for rhythm, precision and
repetitive structure.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "manuscript"

TROPES = {
    "delve": r"\bdelv(?:e|es|ed|ing)\b",
    "landscape": r"\blandscape\b",
    "pivotal": r"\bpivotal\b",
    "multifaceted": r"\bmultifaceted\b",
    "myriad": r"\bmyriad\b",
    "realm": r"\brealm\b",
    "nuanced": r"\bnuanced\b",
    "transformative": r"\btransformative\b",
    "game changer": r"\bgame[- ]changer\b",
    "seamlessly": r"\bseamlessly\b",
    "serves as a testament": r"\bserves? as a testament\b",
    "important to note": r"\bit is (?:important|worth|interesting) to note\b",
    "worth noting": r"\bit is worth noting\b",
    "should be noted": r"\bit should be noted\b",
    "this underscores": r"\bthis underscores\b",
    "underscores importance": r"\bunderscores the (?:importance|need)\b",
    "rapidly evolving": r"\b(?:rapidly|ever)[ -]evolving\b",
    "tapestry": r"\btapestry\b",
    "navigate complexities": r"\bnavigat(?:e|es|ed|ing) the complexit(?:y|ies)\b",
    "at its core": r"\bat (?:its|the) core\b",
    "at the heart of": r"\bat the heart of\b",
    "key takeaway": r"\bkey takeaway\b",
    "sheds light": r"\bsheds? light on\b",
    "plays a crucial role": r"\bplays? a (?:crucial|vital|pivotal) role\b",
    "cannot be overstated": r"\bcannot be overstated\b",
    "unlock": r"\bunlock(?:s|ed|ing)?\b",
    "stock transition": r"\b(?:moreover|furthermore|additionally|more importantly)\b",
    "valuable insights": r"\bvaluable insights?\b",
    "deep dive": r"\bdeep dive\b",
    "groundbreaking": r"\bground[- ]breaking\b",
    "cutting edge": r"\bcutting[- ]edge\b",
    "holistic": r"\bholistic\b",
    "not only but also": r"\bnot only\b.{0,140}\bbut also\b",
    "not merely but also": r"\bnot merely\b.{0,140}\bbut also\b",
    "in conclusion": r"\bin conclusion\b",
    "in summary": r"\bin summary\b",
}


def strip_comments(source: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in source.splitlines())


def manuscript_sources() -> list[Path]:
    return sorted(
        path
        for path in MANUSCRIPT.rglob("*.tex")
        if "generated" not in path.relative_to(MANUSCRIPT).parts
    )


def main() -> None:
    errors: list[str] = []
    for path in manuscript_sources():
        source = strip_comments(path.read_text(encoding="utf-8"))
        prose_source = re.sub(r"\\(?:begin|end)\{[^}]+\}", "", source)
        normalised = re.sub(r"\s+", " ", prose_source)
        for label, raw_pattern in TROPES.items():
            pattern = re.compile(raw_pattern, flags=re.IGNORECASE)
            for match in pattern.finditer(normalised):
                line = prose_source.count(
                    "\n", 0, min(match.start(), len(prose_source))
                ) + 1
                errors.append(
                    f"{path.relative_to(ROOT)}:{line}: stock phrase '{label}'"
                )
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Prose style check passed: {len(manuscript_sources())} TeX files checked.")


if __name__ == "__main__":
    main()
