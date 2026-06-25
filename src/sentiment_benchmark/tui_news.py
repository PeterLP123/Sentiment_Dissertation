"""Tavily news-collection feature logic for the TUI."""

from __future__ import annotations

import os
from contextlib import suppress

from textual.widgets import Button, Checkbox, Input, RichLog, Select, Static

from .news_source import TavilyNewsClient, make_news_fetch_config, write_news_corpus
from .tui_base import AppMixin


class NewsMixin(AppMixin):
    def _set_news_log(self, message: str) -> None:
        self.news_lines.append(message)
        self.news_lines = self.news_lines[-500:]
        try:
            log = self.query_one("#news-log", RichLog)
        except Exception:
            return
        should_follow = bool(log.is_vertical_scroll_end)
        log.write(message, scroll_end=should_follow)

    def _make_news_client(self) -> TavilyNewsClient:
        return TavilyNewsClient()

    def _refresh_news_summary(self) -> None:
        try:
            summary = self.query_one("#news-summary", Static)
        except Exception:
            return
        api_state = "present" if os.getenv("TAVILY_API_KEY") else "missing"
        summary.update(
            "\n".join(
                [
                    f"Tavily API key: {api_state}",
                    f"Output directory: {self.news_output_dir}",
                    "Generated corpora are unlabeled source material.",
                ]
            )
        )

    def _set_news_busy(self, busy: bool) -> None:
        self._news_in_progress = busy
        for button_id in ("#news-check", "#news-fetch"):
            with suppress(Exception):
                self.query_one(button_id, Button).disabled = busy

    def _news_config_from_ui(self, *, check_only: bool = False):
        try:
            query = self.query_one("#news-query", Input).value.strip()
            max_results = 1 if check_only else int(self.query_one("#news-max-results", Input).value.strip())
            topic_value = self.query_one("#news-topic", Select).value
            time_range_value = self.query_one("#news-time-range", Select).value
            if topic_value is Select.BLANK or time_range_value is Select.BLANK:
                raise ValueError("Choose a Tavily topic and time range")
            extract = False if check_only else bool(self.query_one("#news-extract", Checkbox).value)
            return make_news_fetch_config(
                query=query,
                max_results=max_results,
                topic=str(topic_value),
                time_range=str(time_range_value),
                extract=extract,
            )
        except Exception as exc:
            raise ValueError(f"News setup error: {exc}") from exc

    async def _check_news(self) -> None:
        if self._news_in_progress:
            self._notify_error("A Tavily request is already in progress.", title="Busy")
            return
        try:
            config = self._news_config_from_ui(check_only=True)
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot check")
            return
        self._set_news_busy(True)
        self._set_news_log(f"Checking Tavily with query: {config.query}")
        try:
            async with self._make_news_client() as client:
                result = await client.fetch(config)
            credits = result.search_usage.get("credits") if isinstance(result.search_usage, dict) else None
            credit_text = f", credits={credits}" if credits is not None else ""
            self._set_news_log(
                f"Tavily check OK: {len(result.records)} result(s), request_id={result.search_request_id or '-'}{credit_text}"
            )
            self._notify_info("Tavily news API check succeeded.", title="Tavily")
        except Exception as exc:
            self._notify_error(f"Tavily check failed: {exc}", title="Tavily failed")
            self._set_news_log(f"Tavily check failed: {exc}")
        finally:
            self._set_news_busy(False)

    async def _fetch_news(self) -> None:
        if self._news_in_progress:
            self._notify_error("A Tavily request is already in progress.", title="Busy")
            return
        try:
            config = self._news_config_from_ui()
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot fetch")
            return
        self._set_news_busy(True)
        self._set_news_log(f"Fetching Tavily news: {config.query}")
        try:
            async with self._make_news_client() as client:
                result = await client.fetch(config)
            paths = write_news_corpus(result, self.news_output_dir)
            failed = sum(1 for record in result.records if record.extraction_status == "failed")
            self._set_news_log(
                f"Saved {len(result.records)} article record(s), failed extractions={failed}: {paths.output_dir}"
            )
            self._set_news_log(f"JSONL: {paths.articles_jsonl}")
            self._set_news_log(f"CSV: {paths.articles_csv}")
            self._set_news_log(f"Manifest: {paths.manifest_json}")
            self._notify_info(f"Saved Tavily article corpus to {paths.output_dir}", title="News saved")
        except Exception as exc:
            self._notify_error(f"Tavily fetch failed: {exc}", title="Fetch failed")
            self._set_news_log(f"Tavily fetch failed: {exc}")
        finally:
            self._set_news_busy(False)
