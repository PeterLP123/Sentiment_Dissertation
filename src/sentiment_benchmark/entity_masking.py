"""Entity masking for the contamination / name-prior ablation arm.

Replaces a company's identifying mentions (name, ticker, curated aliases) with
neutral placeholders so an LLM scores the *text's* sentiment rather than its
memorised prior about the named company. Comparing masked vs unmasked scores
isolates the name-prior contribution — a "Beyond the Score" ablation, and a
second contamination control alongside the knowledge-cutoff stratification.

Pure text utility (dependency-free leaf): operates on strings, not articles, so
it has no import cycle with ``trading_strategy``. Matching is word-boundaried to
avoid the generic-alias over-masking trap (e.g. alias "Apple" must not touch
"pineapple"); see the entity-relevance lessons in the news pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COMPANY_PLACEHOLDER = "[COMPANY]"
TICKER_PLACEHOLDER = "[TICKER]"


@dataclass(frozen=True)
class MaskResult:
    masked_text: str
    n_masked: int
    counts: dict[str, int]  # placeholder -> number of replacements


def mask_entities(
    text: str,
    *,
    name: str,
    ticker: str = "",
    aliases: tuple[str, ...] = (),
) -> MaskResult:
    """Mask company identifiers in ``text``.

    The ticker becomes ``[TICKER]``; the company name and aliases become
    ``[COMPANY]``. Terms are applied longest-first so multi-word names are masked
    before their constituents, and each match is word-boundaried and
    case-insensitive so substrings (e.g. "Apple" inside "pineapple") are left
    intact. Returns the masked text plus replacement counts.
    """
    counts: dict[str, int] = {TICKER_PLACEHOLDER: 0, COMPANY_PLACEHOLDER: 0}
    masked = text

    # Ticker first (distinctive, its own placeholder), then name/aliases by length.
    spec: list[tuple[str, str]] = []
    if ticker.strip():
        spec.append((ticker.strip(), TICKER_PLACEHOLDER))
    company_terms = {term.strip() for term in (name, *aliases) if term.strip()}
    company_terms.discard(ticker.strip())
    for term in sorted(company_terms, key=len, reverse=True):
        spec.append((term, COMPANY_PLACEHOLDER))

    for term, placeholder in spec:
        # Lookarounds (not \b) so terms ending in punctuation like "Apple Inc."
        # still match, while substrings ("pineapple", "Applebee's") do not.
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        masked, n = pattern.subn(placeholder, masked)
        counts[placeholder] += n

    n_masked = counts[TICKER_PLACEHOLDER] + counts[COMPANY_PLACEHOLDER]
    return MaskResult(masked_text=masked, n_masked=n_masked, counts=counts)
