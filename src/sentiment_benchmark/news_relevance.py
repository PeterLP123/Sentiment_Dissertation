"""Entity-relevance screening for the per-ticker Tavily news panel.

Week-1 calibration of configs/tavily_ticker_query_matrix.toml (2026-06-12)
showed Tavily search returns sector-adjacent off-topic articles for many
names, and that a title/snippet substring heuristic over-counts passing
mentions. This module therefore scores the extracted article TEXT: a record
is on-topic for its query family only when a company alias appears in the
title or the lead of the body AND the body mentions the company at least
``min_body_mentions`` times.

Aliases are derived from the matrix ``family`` field ("TICKER — Company
Name") with curated overrides for names where the bare company name or
ticker is too generic ('shell' matches any oil-sector article and the person
name Jeff Shell; 'BA' collides with British Airways; 'AAL' with American
Airlines). For those, exact anchored forms ("Shell plc") replace the bare
substring; tickers are always matched case-sensitively at word boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

DEFAULT_LEAD_WINDOW_CHARS = 2000
DEFAULT_MIN_BODY_MENTIONS = 2

RELEVANCE_ON_TOPIC = "on_topic"
RELEVANCE_OFF_TOPIC = "off_topic"
# Text quality was not "ok", so there is no extracted body to score.
RELEVANCE_NO_TEXT = "no_text"
# No alias list could be derived (family is not "TICKER — Company Name",
# e.g. thematic-matrix families), so the record was never scored.
RELEVANCE_UNSCREENED = "unscreened"

_FAMILY_SEPARATOR = "—"

# Only the ticker may be too short to use as a default alias; single-letter
# tickers (e.g. "C — Citigroup") match far too much as word-boundary tokens.
_MIN_DEFAULT_TICKER_CHARS = 2


@dataclass(frozen=True)
class FamilyAliases:
    family: str
    # Case-insensitive phrases matched at word boundaries ("Shell plc").
    names: tuple[str, ...]
    # Case-sensitive tokens matched at word boundaries (tickers, "Meta").
    exact: tuple[str, ...]


@dataclass(frozen=True)
class RelevanceResult:
    relevance: str
    # Body mentions after merging overlapping alias matches, so "Goldman
    # Sachs" does not also count as a separate "Goldman" mention.
    mention_count: int
    title_match: bool
    lead_match: bool
    matched_aliases: tuple[str, ...]


_UNSCREENED_RESULT = RelevanceResult(
    relevance=RELEVANCE_UNSCREENED,
    mention_count=0,
    title_match=False,
    lead_match=False,
    matched_aliases=(),
)

# Keyed by the ticker half of the family field. Families absent here fall
# back to (company name, ticker). Each entry is (names, exact); an empty
# exact tuple drops the ticker because it collides with something common.
_ALIAS_OVERRIDES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # --- US arm ---
    "GOOGL": (("Alphabet", "Google"), ("GOOGL", "GOOG")),
    # "meta" is a common word; match the company form case-sensitively.
    "META": (("Meta Platforms",), ("META", "Meta")),
    "JPM": (("JPMorgan", "JP Morgan", "J.P. Morgan"), ("JPM",)),
    "GS": (("Goldman Sachs", "Goldman"), ("GS",)),
    "BAC": (("Bank of America", "BofA"), ("BAC",)),
    # "MS" is an abbreviation magnet (Microsoft, multiple sclerosis, ...).
    "MS": (("Morgan Stanley",), ()),
    # Single-letter ticker is unusable; add the common short names instead.
    "C": (("Citigroup", "Citibank", "Citi"), ()),
    "XOM": (("Exxon Mobil", "ExxonMobil", "Exxon"), ("XOM",)),
    "JNJ": (("Johnson & Johnson", "Johnson and Johnson"), ("JNJ",)),
    "LLY": (("Eli Lilly", "Lilly"), ("LLY",)),
    # "BA" collides with British Airways and BAE in the same news pool.
    "BA": (("Boeing",), ()),
    # --- UK arm ---
    # Bare "shell" matches any oil-sector article and the person Jeff Shell.
    "SHEL": (("Shell plc", "Royal Dutch Shell"), ("SHEL",)),
    # Bare case-insensitive "bp" matches basis-point shorthand; keep only the
    # anchored names plus the case-sensitive word-boundary ticker.
    "BP": (("BP plc", "BP p.l.c."), ("BP",)),
    "GSK": (("GlaxoSmithKline",), ("GSK",)),
    # "RR" is generic shorthand; rely on the company name.
    "RR": (("Rolls-Royce",), ()),
    # TSCO is also Tractor Supply (NASDAQ).
    "TSCO": (("Tesco",), ()),
    # BATS is also the Cboe BZX exchange.
    "BATS": (("British American Tobacco",), ()),
    "BA.": (("BAE Systems", "BAE"), ()),
    # AAL is also American Airlines (NASDAQ).
    "AAL": (("Anglo American",), ()),
    # Bare "Prudential" matches Prudential Financial (NYSE: PRU, unrelated).
    "PRU": (("Prudential plc",), ()),
}


def aliases_for_family(family: str) -> FamilyAliases | None:
    """Alias list for a "TICKER — Company Name" family, or None.

    Thematic-matrix families ("Bank earnings") do not name a single company,
    so they return None and their records stay unscreened.
    """
    ticker, separator, name = family.partition(_FAMILY_SEPARATOR)
    if not separator:
        return None
    ticker = ticker.strip()
    name = name.strip()
    if not ticker or not name or " " in ticker:
        return None
    override = _ALIAS_OVERRIDES.get(ticker)
    if override is not None:
        names, exact = override
    else:
        names = (name,)
        exact = (ticker,) if len(ticker) >= _MIN_DEFAULT_TICKER_CHARS else ()
    return FamilyAliases(family=family, names=names, exact=exact)


@lru_cache(maxsize=2048)
def _alias_pattern(alias: str, case_sensitive: bool) -> re.Pattern[str]:
    # Lookarounds instead of \b so phrases ending in punctuation ("BP p.l.c.")
    # still anchor against following word characters; \s+ tolerates rewrapped
    # whitespace in extracted text.
    escaped = re.escape(alias).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", 0 if case_sensitive else re.IGNORECASE)


def _match_spans(text: str, aliases: FamilyAliases) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    candidates = [(alias, False) for alias in aliases.names] + [(alias, True) for alias in aliases.exact]
    for alias, case_sensitive in candidates:
        for match in _alias_pattern(alias, case_sensitive).finditer(text):
            spans.append((match.start(), match.end(), alias))
    return sorted(spans)


def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end, _ in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _ordered_aliases(spans: list[tuple[int, int, str]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(alias for _, _, alias in spans))


def score_entity_relevance(
    title: str,
    article_text: str,
    aliases: FamilyAliases,
    *,
    lead_window_chars: int = DEFAULT_LEAD_WINDOW_CHARS,
    min_body_mentions: int = DEFAULT_MIN_BODY_MENTIONS,
) -> RelevanceResult:
    title_spans = _match_spans(title or "", aliases)
    title_match = bool(title_spans)
    body = article_text or ""
    if not body.strip():
        return RelevanceResult(
            relevance=RELEVANCE_NO_TEXT,
            mention_count=0,
            title_match=title_match,
            lead_match=False,
            matched_aliases=_ordered_aliases(title_spans),
        )
    body_spans = _match_spans(body, aliases)
    mentions = _merge_spans(body_spans)
    lead_match = any(start < lead_window_chars for start, _ in mentions)
    on_topic = (title_match or lead_match) and len(mentions) >= min_body_mentions
    return RelevanceResult(
        relevance=RELEVANCE_ON_TOPIC if on_topic else RELEVANCE_OFF_TOPIC,
        mention_count=len(mentions),
        title_match=title_match,
        lead_match=lead_match,
        matched_aliases=_ordered_aliases(title_spans + body_spans),
    )


def score_relevance_for_families(
    title: str,
    article_text: str,
    families: list[str] | tuple[str, ...],
    *,
    lead_window_chars: int = DEFAULT_LEAD_WINDOW_CHARS,
    min_body_mentions: int = DEFAULT_MIN_BODY_MENTIONS,
) -> RelevanceResult:
    """Score a record fetched under one or more query families.

    A URL pulled in by several ticker queries is on-topic when it concerns
    any of those companies; the best-scoring family wins so the reported
    mention count and aliases describe the company the article is about.
    """
    alias_sets = [aliases for aliases in map(aliases_for_family, families) if aliases is not None]
    if not alias_sets:
        return _UNSCREENED_RESULT
    results = [
        score_entity_relevance(
            title,
            article_text,
            aliases,
            lead_window_chars=lead_window_chars,
            min_body_mentions=min_body_mentions,
        )
        for aliases in alias_sets
    ]
    return max(results, key=_result_rank)


def _result_rank(result: RelevanceResult) -> tuple[int, int]:
    return (1 if result.relevance == RELEVANCE_ON_TOPIC else 0, result.mention_count)
