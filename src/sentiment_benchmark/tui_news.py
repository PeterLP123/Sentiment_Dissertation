"""News-collection feature logic for the TUI."""

from __future__ import annotations

import asyncio
import importlib.util
import os
from contextlib import suppress
from pathlib import Path

from textual.widgets import Button, Checkbox, Input, RichLog, Select, Static

from .lseg_catalog import build_lseg_catalog, read_lseg_catalog
from .lseg_corpus import build_lseg_corpus
from .lseg_source import LsegNewsClient, LsegNewsError, check_lseg_news, fetch_lseg_news, load_lseg_collection_config
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

    def _make_lseg_news_client(self) -> LsegNewsClient:
        return LsegNewsClient()

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

    def _refresh_lseg_summary(self) -> None:
        try:
            summary = self.query_one("#lseg-summary", Static)
        except Exception:
            return
        dependency_state = "installed" if importlib.util.find_spec("lseg") is not None else "missing"
        lines = [f"LSEG SDK: {dependency_state}", f"Config: {self.lseg_config_path}"]
        try:
            config = load_lseg_collection_config(self.lseg_config_path)
            lines.extend(
                [
                    f"Collection: {config.collection_id}",
                    f"Date range: {config.start} to {config.end}",
                    f"Companies: {len(config.companies)} ({', '.join(company.symbol for company in config.companies)})",
                    f"Window days: {config.window_days or 'whole interval'}",
                    f"Raw: {config.raw_dir}",
                    f"Clean: {config.derived_dir}",
                ]
            )
            try:
                catalog = read_lseg_catalog(config.derived_output_root / "catalog.json")
                corpora = catalog.get("corpora") if isinstance(catalog.get("corpora"), list) else []
                match = next(
                    (row for row in corpora if isinstance(row, dict) and row.get("collection_id") == config.collection_id),
                    None,
                )
                if isinstance(match, dict):
                    lines.append(f"Catalog: {match.get('articles', 0)} articles, {match.get('eligible', 0)} eligible")
                else:
                    lines.append("Catalog: no entry for this config")
            except LsegNewsError:
                lines.append("Catalog: not built")
        except Exception as exc:
            lines.append(f"Config status: {exc}")
        summary.update("\n".join(lines))

    def _set_news_busy(self, busy: bool) -> None:
        self._news_in_progress = busy
        for button_id in ("#news-check", "#news-fetch"):
            with suppress(Exception):
                self.query_one(button_id, Button).disabled = busy

    def _set_lseg_busy(self, busy: bool) -> None:
        self._lseg_in_progress = busy
        for button_id in ("#lseg-check", "#lseg-fetch", "#lseg-build", "#lseg-catalog"):
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

    def _lseg_config_from_ui(self):
        try:
            value = self.query_one("#lseg-config-path", Input).value.strip() or "configs/lseg_us_mega_cap_1y.toml"
            self.lseg_config_path = Path(value)
            return load_lseg_collection_config(self.lseg_config_path)
        except Exception as exc:
            raise ValueError(f"LSEG setup error: {exc}") from exc

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

    async def _check_lseg(self) -> None:
        if self._lseg_in_progress:
            self._notify_error("An LSEG request is already in progress.", title="Busy")
            return
        try:
            config = self._lseg_config_from_ui()
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot check")
            return
        self._set_lseg_busy(True)
        self._set_news_log(f"Checking LSEG collection {config.collection_id}")
        try:
            async with self._make_lseg_news_client() as client:
                result = await check_lseg_news(config, client)
            self._set_news_log(
                f"LSEG check OK: query={result['query']}, headlines={result['headline_count']}, story_status={result['story_status']}"
            )
            self._notify_info("LSEG Workspace check succeeded.", title="LSEG")
        except Exception as exc:
            self._notify_error(f"LSEG check failed: {exc}", title="LSEG failed")
            self._set_news_log(f"LSEG check failed: {exc}")
        finally:
            self._set_lseg_busy(False)
            self._refresh_lseg_summary()

    async def _fetch_lseg(self) -> None:
        if self._lseg_in_progress:
            self._notify_error("An LSEG request is already in progress.", title="Busy")
            return
        try:
            config = self._lseg_config_from_ui()
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot fetch")
            return
        self._set_lseg_busy(True)
        self._set_news_log(f"Fetching LSEG collection {config.collection_id}")
        try:
            async with self._make_lseg_news_client() as client:
                result = await fetch_lseg_news(config, client)
            self._set_news_log(
                f"LSEG raw saved: headlines={result.headline_count}, stories={result.story_count}, "
                f"failed={result.failed_story_count}, resumed={result.resumed}"
            )
            self._set_news_log(f"Raw manifest: {result.manifest_path}")
            self._notify_info(f"Saved LSEG raw collection to {result.raw_dir}", title="LSEG saved")
        except Exception as exc:
            self._notify_error(f"LSEG fetch failed: {exc}", title="Fetch failed")
            self._set_news_log(f"LSEG fetch failed: {exc}")
        finally:
            self._set_lseg_busy(False)
            self._refresh_lseg_summary()

    async def _build_lseg(self) -> None:
        if self._lseg_in_progress:
            self._notify_error("An LSEG request is already in progress.", title="Busy")
            return
        try:
            config = self._lseg_config_from_ui()
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot build")
            return
        self._set_lseg_busy(True)
        self._set_news_log(f"Building LSEG clean corpus {config.collection_id}")
        try:
            result = await asyncio.to_thread(build_lseg_corpus, config.raw_dir)
            catalog = await asyncio.to_thread(build_lseg_catalog, config.derived_output_root)
            self._set_news_log(
                f"LSEG clean corpus built: articles={result.article_count}, eligible={result.eligible_count}, resumed={result.resumed}"
            )
            self._set_news_log(f"Clean manifest: {result.manifest_path}")
            self._set_news_log(f"Catalog: {catalog.catalog_json}")
            self._notify_info(f"Built LSEG clean corpus in {result.derived_dir}", title="LSEG clean")
        except Exception as exc:
            self._notify_error(f"LSEG build failed: {exc}", title="Build failed")
            self._set_news_log(f"LSEG build failed: {exc}")
        finally:
            self._set_lseg_busy(False)
            self._refresh_lseg_summary()

    async def _refresh_lseg_catalog_action(self) -> None:
        if self._lseg_in_progress:
            self._notify_error("An LSEG request is already in progress.", title="Busy")
            return
        try:
            config = self._lseg_config_from_ui()
        except ValueError as exc:
            self._notify_error(str(exc), title="Cannot catalog")
            return
        self._set_lseg_busy(True)
        try:
            result = await asyncio.to_thread(build_lseg_catalog, config.derived_output_root)
            self._set_news_log(
                f"LSEG catalog refreshed: corpora={result.corpus_count}, eligible={result.eligible_count}, json={result.catalog_json}"
            )
            self._notify_info("LSEG catalog refreshed.", title="LSEG catalog")
        except Exception as exc:
            self._notify_error(f"LSEG catalog failed: {exc}", title="Catalog failed")
            self._set_news_log(f"LSEG catalog failed: {exc}")
        finally:
            self._set_lseg_busy(False)
            self._refresh_lseg_summary()

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
            self._set_news_log(f"Saved {len(result.records)} article record(s), failed extractions={failed}: {paths.output_dir}")
            self._set_news_log(f"JSONL: {paths.articles_jsonl}")
            self._set_news_log(f"CSV: {paths.articles_csv}")
            self._set_news_log(f"Manifest: {paths.manifest_json}")
            self._notify_info(f"Saved Tavily article corpus to {paths.output_dir}", title="News saved")
        except Exception as exc:
            self._notify_error(f"Tavily fetch failed: {exc}", title="Fetch failed")
            self._set_news_log(f"Tavily fetch failed: {exc}")
        finally:
            self._set_news_busy(False)
