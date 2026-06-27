import asyncio
import json
from pathlib import Path

import pytest

import sentiment_benchmark.lseg_source as lseg_source
from sentiment_benchmark.lseg_source import (
    LsegCollectionConfig,
    LsegCompanyConfig,
    LsegHeadlinePage,
    LsegNewsClient,
    LsegNewsError,
    LsegStoryResponse,
    collection_windows,
    fetch_lseg_news,
    load_lseg_collection_config,
)


def _config(tmp_path: Path, *, collection_id: str = "test_lseg") -> LsegCollectionConfig:
    return LsegCollectionConfig(
        collection_id=collection_id,
        start="2026-06-01T00:00:00Z",
        end="2026-06-03T00:00:00Z",
        page_size=2,
        max_pages=3,
        story_concurrency=2,
        retries=0,
        requests_per_second=1_000_000,
        min_text_chars=10,
        raw_output_root=tmp_path / "raw",
        derived_output_root=tmp_path / "derived",
        companies=(
            LsegCompanyConfig("AAPL", "Apple Inc.", "AAPL.O", "R:AAPL.O", ("Apple", "AAPL")),
            LsegCompanyConfig("MSFT", "Microsoft Corp.", "MSFT.O", "R:MSFT.O", ("Microsoft", "MSFT")),
        ),
    )


class FakeBackend:
    def __init__(self) -> None:
        self.page_calls: list[tuple[str, str | None]] = []
        self.story_calls: list[str] = []

    async def headline_page(self, *, query, start, end, count, cursor):
        self.page_calls.append((query, cursor))
        if query == "R:AAPL.O" and cursor is None:
            return LsegHeadlinePage(
                [
                    {
                        "storyId": "urn:test:apple:1",
                        "headline": "Apple reports growth",
                        "firstCreated": "2026-06-01T09:00:00Z",
                        "versionCreated": "2026-06-01T09:05:00Z",
                        "sourceCode": "NS:RTRS",
                        "language": "en",
                    }
                ],
                "next-aapl",
                {"meta": {"next": "next-aapl"}},
            )
        if query == "R:AAPL.O" and cursor == "next-aapl":
            return LsegHeadlinePage(
                [
                    {
                        "storyId": "urn:test:shared:1",
                        "headline": "Apple and Microsoft sign agreement",
                        "versionCreated": "2026-06-01T10:00:00Z",
                    }
                ],
                None,
            )
        if query == "R:MSFT.O":
            return LsegHeadlinePage(
                [
                    {
                        "storyId": "urn:test:shared:1",
                        "headline": "Apple and Microsoft sign a major agreement",
                        "versionCreated": "2026-06-01T10:00:00Z",
                    },
                    {
                        "storyId": "urn:test:web:1",
                        "headline": "Microsoft external report",
                        "versionCreated": "2026-06-01T11:00:00Z",
                    },
                ],
                None,
            )
        raise AssertionError((query, cursor, start, end, count))

    async def story(self, story_id):
        self.story_calls.append(story_id)
        if story_id == "urn:test:web:1":
            return LsegStoryResponse(story_id, "story_unavailable", web_url="https://publisher.test/story")
        return LsegStoryResponse(story_id, "success", f"<p>{story_id} body text with enough content.</p>", "html")


def test_load_lseg_config_requires_utc_and_desktop(tmp_path: Path) -> None:
    path = tmp_path / "lseg.toml"
    path.write_text(
        """
[collection]
id = "x"
start = "2026-06-01T00:00:00Z"
end = "2026-06-02T00:00:00Z"
[[companies]]
symbol = "AAPL"
name = "Apple"
ric = "AAPL.O"
aliases = ["Apple"]
"""
    )

    config = load_lseg_collection_config(path)

    assert config.companies[0].news_query == "R:AAPL.O and Language:LEN"
    assert config.session == "desktop"
    assert config.start.endswith("Z")


def test_load_lseg_config_accepts_window_days(tmp_path: Path) -> None:
    path = tmp_path / "lseg.toml"
    path.write_text(
        """
[collection]
id = "x"
start = "2026-06-01T00:00:00Z"
end = "2026-06-03T00:00:00Z"
window_days = 1
requests_per_second = 3.0
[[companies]]
symbol = "AAPL"
name = "Apple"
ric = "AAPL.O"
aliases = ["Apple"]
"""
    )

    config = load_lseg_collection_config(path)

    assert config.window_days == 1
    assert config.requests_per_second == 3.0
    assert collection_windows(config) == [
        lseg_source.LsegCollectionWindow(1, "2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z"),
        lseg_source.LsegCollectionWindow(2, "2026-06-02T00:00:00Z", "2026-06-03T00:00:00Z"),
    ]


def test_load_lseg_config_rejects_request_rate_above_workspace_limit(tmp_path: Path) -> None:
    path = tmp_path / "lseg.toml"
    path.write_text(
        """
[collection]
id = "x"
start = "2026-06-01T00:00:00Z"
end = "2026-06-02T00:00:00Z"
requests_per_second = 5.1
[[companies]]
symbol = "AAPL"
name = "Apple"
ric = "AAPL.O"
"""
    )

    with pytest.raises(LsegNewsError, match="requests_per_second must be between 0.1 and 5"):
        load_lseg_collection_config(path)


def test_story_content_reads_lseg_sdk_story_content() -> None:
    class Content:
        html = "<p>Full story body.</p>"
        text = "Full story body."
        web_url = "https://publisher.test/story"

    class Story:
        content = Content()

    class Data:
        story = Story()

    body, body_format, web_url = lseg_source._story_content(Data())

    assert body == "<p>Full story body.</p>"
    assert body_format == "html"
    assert web_url == "https://publisher.test/story"


def test_fetch_paginates_deduplicates_and_resumes_without_calls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    backend = FakeBackend()
    result = asyncio.run(fetch_lseg_news(config, LsegNewsClient(backend)))

    assert result.headline_count == 3
    assert result.story_count == 3
    assert result.failed_story_count == 1
    assert (result.raw_dir / "headline_pages" / "001-AAPL-page-0001.json").exists()
    assert backend.story_calls.count("urn:test:shared:1") == 1
    headlines = [json.loads(line) for line in (result.raw_dir / "headlines.jsonl").read_text().splitlines()]
    shared = next(row for row in headlines if row["story_id"] == "urn:test:shared:1")
    assert shared["matched_symbols"] == ["AAPL", "MSFT"]
    assert shared["headline"] == "Apple and Microsoft sign a major agreement"

    class NoCalls:
        async def headline_page(self, **kwargs):
            raise AssertionError(kwargs)

        async def story(self, story_id):
            raise AssertionError(story_id)

    resumed = asyncio.run(fetch_lseg_news(config, LsegNewsClient(NoCalls())))
    assert resumed.resumed is True


def test_fetch_windowed_collection_checkpoint_names_and_counts(tmp_path: Path) -> None:
    class WindowBackend:
        def __init__(self) -> None:
            self.page_calls: list[tuple[str, str, str, str | None]] = []
            self.story_calls: list[str] = []

        async def headline_page(self, *, query, start, end, count, cursor):
            self.page_calls.append((query, start, end, cursor))
            day = start[:10]
            return LsegHeadlinePage(
                [
                    {
                        "storyId": f"urn:test:{query}:{day}",
                        "headline": f"{query} headline {day}",
                        "versionCreated": start,
                        "language": "en",
                    }
                ],
                None,
            )

        async def story(self, story_id):
            self.story_calls.append(story_id)
            return LsegStoryResponse(story_id, "success", f"<p>{story_id} body text with enough content.</p>", "html")

    config = LsegCollectionConfig(**{**_config(tmp_path).__dict__, "window_days": 1, "companies": (_config(tmp_path).companies[0],)})
    progress_updates = []
    result = asyncio.run(
        fetch_lseg_news(
            config,
            LsegNewsClient(WindowBackend()),
            progress_callback=progress_updates.append,
        )
    )

    assert result.headline_count == 2
    assert result.story_count == 2
    assert (result.raw_dir / "headline_pages" / "001-AAPL-window-0001-20260601T000000z-20260602T000000z-page-0001.json").exists()
    assert (result.raw_dir / "headline_pages" / "001-AAPL-window-0002-20260602T000000z-20260603T000000z-page-0001.json").exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["counts"]["headline_pages"] == 2
    assert manifest["queries"]["AAPL"]["windows"] == 2
    headline_updates = [update for update in progress_updates if update.phase == "headlines"]
    story_updates = [update for update in progress_updates if update.phase == "stories"]
    assert headline_updates[0].completed == 0
    assert headline_updates[-1].completed == 2
    assert headline_updates[-1].total == 2
    assert headline_updates[-1].raw_headline_rows == 2
    assert headline_updates[-1].unique_headlines == 2
    assert story_updates[0].completed == 0
    assert story_updates[-1].completed == 2
    assert story_updates[-1].total == 2
    assert story_updates[-1].requests_started == 4
    assert story_updates[-1].requests_per_second == 1_000_000


def test_progress_starts_from_completed_window_checkpoints(tmp_path: Path) -> None:
    config = LsegCollectionConfig(
        **{
            **_config(tmp_path).__dict__,
            "window_days": 1,
            "companies": (_config(tmp_path).companies[0],),
        }
    )
    backend = FakeBackend()
    first_updates = []
    asyncio.run(fetch_lseg_news(config, LsegNewsClient(backend), progress_callback=first_updates.append))

    manifest = json.loads(config.raw_dir.joinpath("manifest.json").read_text())
    manifest["status"] = "in_progress"
    manifest["config"]["collection"].pop("requests_per_second")
    manifest["config_sha256"] = "legacy-config-without-request-pacing"
    config.raw_dir.joinpath("manifest.json").write_text(json.dumps(manifest))
    config.raw_dir.joinpath("headlines.jsonl").unlink()
    config.raw_dir.joinpath("stories.jsonl").unlink()
    second_updates = []

    class NoNewCalls:
        async def headline_page(self, **kwargs):
            raise AssertionError(kwargs)

        async def story(self, story_id):
            raise AssertionError(story_id)

    asyncio.run(fetch_lseg_news(config, LsegNewsClient(NoNewCalls()), progress_callback=second_updates.append))

    first_headline_update = next(update for update in second_updates if update.phase == "headlines")
    first_story_update = next(update for update in second_updates if update.phase == "stories")
    assert first_headline_update.completed == 2
    assert first_headline_update.checkpointed == 2
    assert first_story_update.completed == 2
    assert first_story_update.checkpointed == 2


def test_completed_collection_rejects_changed_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path)
    asyncio.run(fetch_lseg_news(config, LsegNewsClient(FakeBackend())))
    changed = LsegCollectionConfig(**{**config.__dict__, "end": "2026-06-04T00:00:00Z"})

    with pytest.raises(LsegNewsError, match="different configuration"):
        asyncio.run(fetch_lseg_news(changed, LsegNewsClient(FakeBackend())))


def test_completed_collection_reports_no_requests_for_noop_resume(tmp_path: Path) -> None:
    config = _config(tmp_path)
    asyncio.run(fetch_lseg_news(config, LsegNewsClient(FakeBackend())))

    result = asyncio.run(fetch_lseg_news(config, LsegNewsClient(FakeBackend())))

    assert result.resumed is True
    assert result.request_count == 0
    assert result.retry_count == 0


def test_in_progress_collection_allows_only_page_limit_increase(tmp_path: Path) -> None:
    class TwoPageBackend:
        def __init__(self) -> None:
            self.cursors = []

        async def headline_page(self, *, cursor, **kwargs):
            self.cursors.append(cursor)
            if cursor is None:
                return LsegHeadlinePage(
                    [{"storyId": "urn:test:page:1", "headline": "First"}],
                    "next-page",
                )
            return LsegHeadlinePage(
                [{"storyId": "urn:test:page:2", "headline": "Second"}],
                None,
            )

        async def story(self, story_id):
            return LsegStoryResponse(story_id, "success", f"<p>{story_id}</p>", "html")

    base = _config(tmp_path)
    limited = LsegCollectionConfig(
        **{
            **base.__dict__,
            "end": "2026-06-02T00:00:00Z",
            "max_pages": 1,
            "window_days": 1,
            "companies": (base.companies[0],),
        }
    )
    backend = TwoPageBackend()
    with pytest.raises(LsegNewsError, match="exceeded collection.max_pages=1"):
        asyncio.run(fetch_lseg_news(limited, LsegNewsClient(backend)))

    expanded = LsegCollectionConfig(**{**limited.__dict__, "max_pages": 2})
    result = asyncio.run(fetch_lseg_news(expanded, LsegNewsClient(backend)))

    assert result.headline_count == 2
    assert backend.cursors == [None, "next-page"]


def test_repeated_cursor_is_rejected(tmp_path: Path) -> None:
    class Repeating:
        async def headline_page(self, *, cursor, **kwargs):
            story_number = 1 if cursor is None else 2
            return LsegHeadlinePage([{"storyId": f"urn:test:new:{story_number}"}], "same")

        async def story(self, story_id):
            raise AssertionError(story_id)

    config = LsegCollectionConfig(**{**_config(tmp_path).__dict__, "companies": (_config(tmp_path).companies[0],)})
    with pytest.raises(LsegNewsError, match="repeated cursor with 1 new stories"):
        asyncio.run(fetch_lseg_news(config, LsegNewsClient(Repeating())))


def test_duplicate_only_repeated_cursor_is_audited_terminal_page(tmp_path: Path) -> None:
    class DuplicateTerminal:
        async def headline_page(self, *, cursor, **kwargs):
            return LsegHeadlinePage([{"storyId": "urn:test:duplicate:1", "headline": "Same story"}], "same")

        async def story(self, story_id):
            return LsegStoryResponse(story_id, "success", "<p>Story body.</p>", "html")

    config = LsegCollectionConfig(**{**_config(tmp_path).__dict__, "companies": (_config(tmp_path).companies[0],)})
    result = asyncio.run(fetch_lseg_news(config, LsegNewsClient(DuplicateTerminal())))

    assert result.headline_count == 1
    assert result.pagination_anomaly_count == 1
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["counts"]["pagination_anomalies"] == 1
    assert manifest["pagination_anomalies"][0]["reason"] == "repeated_cursor_duplicate_page"
    second_page = result.raw_dir / "headline_pages" / "001-AAPL-page-0002.json"
    assert json.loads(second_page.read_text())["pagination_terminal_reason"] == "repeated_cursor_duplicate_page"


def test_request_pacer_smooths_concurrent_request_starts() -> None:
    now = 0.0
    sleeps = []

    async def fake_sleep(delay):
        nonlocal now
        sleeps.append(delay)
        now += delay

    pacer = lseg_source._RequestPacer(2.0, clock=lambda: now, sleeper=fake_sleep)

    async def run_requests():
        await asyncio.gather(pacer.acquire(), pacer.acquire(), pacer.acquire())

    asyncio.run(run_requests())

    metrics = pacer.snapshot()
    assert sleeps == [0.5, 0.5]
    assert metrics.requests_started == 3
    assert metrics.paced_waits == 2
    assert metrics.paced_wait_seconds == 1.0
    assert metrics.requests_per_second == 2.0


def test_retry_uses_exponential_backoff(monkeypatch) -> None:
    attempts = 0
    delays: list[float] = []
    retry_events = []

    async def operation():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("transient")
        return "ok"

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(lseg_source.asyncio, "sleep", fake_sleep)

    assert (
        asyncio.run(
            lseg_source._retry(
                operation,
                3,
                on_retry=lambda delay, exc: retry_events.append((delay, type(exc).__name__)),
            )
        )
        == "ok"
    )
    assert delays == [1, 2]
    assert retry_events == [(1, "TimeoutError"), (2, "TimeoutError")]
