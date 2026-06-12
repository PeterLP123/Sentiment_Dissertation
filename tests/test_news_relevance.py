from sentiment_benchmark.news_relevance import (
    RELEVANCE_NO_TEXT,
    RELEVANCE_OFF_TOPIC,
    RELEVANCE_ON_TOPIC,
    RELEVANCE_UNSCREENED,
    aliases_for_family,
    score_entity_relevance,
    score_relevance_for_families,
)

APPLE = aliases_for_family("AAPL — Apple")
SHELL = aliases_for_family("SHEL — Shell")
BP = aliases_for_family("BP — BP")
assert APPLE is not None and SHELL is not None and BP is not None

# Roughly 2500 chars, so an alias appearing after it sits past the default lead window.
DEEP_FILLER = "Market commentary filler sentence about macro conditions. " * 43


def test_aliases_for_family_derives_defaults_from_ticker_and_name() -> None:
    assert APPLE.names == ("Apple",)
    assert APPLE.exact == ("AAPL",)


def test_aliases_for_family_returns_none_for_thematic_families() -> None:
    assert aliases_for_family("Bank earnings") is None
    assert aliases_for_family("") is None


def test_aliases_for_family_curated_overrides_drop_generic_forms() -> None:
    assert "Shell" not in SHELL.names  # bare 'shell' matches any oil-sector article
    assert "Shell plc" in SHELL.names

    citigroup = aliases_for_family("C — Citigroup")
    assert citigroup is not None
    assert citigroup.exact == ()  # single-letter ticker is unusable
    assert "Citi" in citigroup.names

    anglo = aliases_for_family("AAL — Anglo American")
    assert anglo is not None
    assert anglo.exact == ()  # AAL is also American Airlines


def test_on_topic_article_passes_title_and_mention_checks() -> None:
    result = score_entity_relevance(
        "Shell plc lifts dividend after record quarter",
        "Shell plc reported higher quarterly profits on Thursday. The board said Shell plc will raise its dividend by 4%.",
        SHELL,
    )
    assert result.relevance == RELEVANCE_ON_TOPIC
    assert result.title_match
    assert result.lead_match
    assert result.mention_count == 2
    assert "Shell plc" in result.matched_aliases


def test_generic_shell_mentions_do_not_count() -> None:
    result = score_entity_relevance(
        "Jeff Shell named CEO of new studio venture",
        "Jeff Shell, the former NBCUniversal executive, will run the venture. "
        "Filings show a shell company structure registered in Delaware. Oil majors were broadly higher.",
        SHELL,
    )
    assert result.relevance == RELEVANCE_OFF_TOPIC
    assert result.mention_count == 0
    assert result.matched_aliases == ()


def test_bp_ticker_matches_word_boundary_case_sensitively() -> None:
    on_topic = score_entity_relevance(
        "BP raises buyback as profits beat",
        "BP said cash flow improved. Analysts expect BP to extend the programme into next year.",
        BP,
    )
    assert on_topic.relevance == RELEVANCE_ON_TOPIC
    assert on_topic.mention_count == 2

    off_topic = score_entity_relevance(
        "Gilt yields fall after inflation data",
        "Yields dropped 10 bps on the day. Traders priced another 25 bp of cuts. The BPS index was unchanged.",
        BP,
    )
    assert off_topic.relevance == RELEVANCE_OFF_TOPIC
    assert off_topic.mention_count == 0


def test_american_airlines_is_off_topic_for_anglo_american() -> None:
    anglo = aliases_for_family("AAL — Anglo American")
    assert anglo is not None
    result = score_entity_relevance(
        "American Airlines stock soars on travel demand",
        "American Airlines (AAL) shares jumped after the carrier raised guidance. AAL now expects record summer bookings.",
        anglo,
    )
    assert result.relevance == RELEVANCE_OFF_TOPIC
    assert result.mention_count == 0


def test_mentions_past_lead_window_need_a_title_match() -> None:
    body = DEEP_FILLER + "Apple results impressed analysts. Apple raised guidance."

    buried = score_entity_relevance("Quarterly results roundup", body, APPLE)
    assert buried.relevance == RELEVANCE_OFF_TOPIC
    assert buried.mention_count == 2
    assert not buried.lead_match

    titled = score_entity_relevance("Apple earnings preview", body, APPLE)
    assert titled.relevance == RELEVANCE_ON_TOPIC
    assert titled.title_match

    widened = score_entity_relevance("Quarterly results roundup", body, APPLE, lead_window_chars=5000)
    assert widened.relevance == RELEVANCE_ON_TOPIC


def test_single_passing_mention_fails_count_threshold() -> None:
    result = score_entity_relevance(
        "Tech stocks slide as rates rise",
        "Apple slipped alongside the rest of the sector. Chipmakers and software names also fell on the day.",
        APPLE,
    )
    assert result.relevance == RELEVANCE_OFF_TOPIC
    assert result.mention_count == 1

    relaxed = score_entity_relevance(
        "Tech stocks slide as rates rise",
        "Apple slipped alongside the rest of the sector. Chipmakers and software names also fell on the day.",
        APPLE,
        min_body_mentions=1,
    )
    assert relaxed.relevance == RELEVANCE_ON_TOPIC


def test_overlapping_aliases_count_as_one_mention() -> None:
    goldman = aliases_for_family("GS — Goldman Sachs")
    assert goldman is not None
    result = score_entity_relevance(
        "Goldman Sachs beats estimates",
        "Goldman Sachs posted stronger trading revenue. Goldman Sachs also lifted its dividend.",
        goldman,
    )
    assert result.relevance == RELEVANCE_ON_TOPIC
    assert result.mention_count == 2  # "Goldman" inside "Goldman Sachs" is not double-counted


def test_missing_text_yields_no_text() -> None:
    result = score_entity_relevance("Apple earnings preview", "", APPLE)
    assert result.relevance == RELEVANCE_NO_TEXT
    assert result.title_match
    assert result.mention_count == 0


def test_score_relevance_for_families_picks_best_family_and_skips_thematic() -> None:
    assert score_relevance_for_families("Title", "Body", ["Bank earnings"]).relevance == RELEVANCE_UNSCREENED

    result = score_relevance_for_families(
        "Microsoft lifts guidance",
        "Microsoft reported strong cloud growth. Microsoft shares rose in late trading.",
        ["AAPL — Apple", "MSFT — Microsoft"],
    )
    assert result.relevance == RELEVANCE_ON_TOPIC
    assert "Microsoft" in result.matched_aliases
