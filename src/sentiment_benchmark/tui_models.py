"""Model selection, fetching, and Ollama-cloud feature logic for the TUI."""

from __future__ import annotations

from contextlib import suppress

from rich.text import Text
from textual.widgets import DataTable, Input, Static

from .models import ModelConfig
from .ollama_cloud import _to_cloud_tag, cloud_catalog_overridden, fetch_ollama_cloud_models, ollama_cloud_catalog
from .providers import make_llm_client
from .tui_base import AppMixin

_SELECTED_MARK = "[x]"
_UNSELECTED_MARK = "[ ]"


class ModelsMixin(AppMixin):
    def _render_selected_table(self) -> None:
        try:
            table = self.query_one("#selected-table", DataTable)
        except Exception:
            return
        table.clear()
        for model_id in self.selected_models:
            name = self._model_names.get(model_id) or "(manual)"
            table.add_row(model_id, name, key=model_id)
        self._refresh_selected_summary()
        self._refresh_status_bar()

    def _refresh_selected_summary(self) -> None:
        try:
            summary = self.query_one("#selected-summary", Static)
        except Exception:
            return
        count = len(self.selected_models)
        if count == 0:
            summary.update("No models selected yet.")
        else:
            cloud = self._selected_cloud_count()
            cloud_note = f" ({cloud} cloud)" if cloud else ""
            summary.update(f"{count} model(s) selected{cloud_note}.")

    def _render_model_table(self) -> None:
        try:
            table = self.query_one("#model-table", DataTable)
        except Exception:
            return
        table.clear()
        try:
            query = self.query_one("#model-search", Input).value.strip().lower()
        except Exception:
            query = ""
        selected = set(self.selected_models)
        for model in self._all_models:
            if query:
                haystack = f"{model.model_id} {model.name or ''}".lower()
                if query not in haystack:
                    continue
            mark = (
                Text(_SELECTED_MARK, style="bold green")
                if model.model_id in selected
                else Text(_UNSELECTED_MARK, style="grey58")
            )
            cloud_cell = (
                Text("cloud", style="cyan") if self._is_cloud_model(model.model_id, model) else Text("", style="grey58")
            )
            table.add_row(
                mark,
                model.model_id,
                model.name or "",
                str(model.context_length or ""),
                cloud_cell,
                key=model.model_id,
            )
        self._refresh_ollama_thinking_recommendation()

    def _selected_model_configs(self) -> list[ModelConfig]:
        by_id = {model.model_id: model for model in self._all_models}
        return [by_id[model_id] for model_id in self.selected_models if model_id in by_id]

    @staticmethod
    def _is_cloud_model(model_id: str, model: ModelConfig | None = None) -> bool:
        """Heuristic for Ollama Cloud models, which carry a ``-cloud``/``:cloud`` tag.

        These run on Ollama's hosted infrastructure (proxied through a signed-in
        local daemon) rather than local VRAM, so they are flagged for the user.
        """
        lowered = (model_id or "").lower()
        if lowered.endswith("-cloud") or lowered.endswith(":cloud") or "-cloud" in lowered:
            return True
        raw = model.raw_metadata if model is not None else {}
        if isinstance(raw, dict) and (raw.get("remote") or raw.get("cloud")):
            return True
        return False

    def _normalize_selected_ollama_cloud_models(self) -> None:
        """Migrate old cached cloud IDs to the current runnable Ollama tag."""
        if self.provider != "ollama":
            return
        normalized: list[str] = []
        for model_id in self.selected_models:
            next_id = model_id
            name = self._model_names.get(model_id)
            if name and self._is_cloud_model(model_id):
                next_id = _to_cloud_tag(name)
                self._model_names.setdefault(next_id, name)
            if next_id not in normalized:
                normalized.append(next_id)
        self.selected_models = normalized

    def _selected_cloud_count(self) -> int:
        by_id = {model.model_id: model for model in self._all_models}
        return sum(1 for model_id in self.selected_models if self._is_cloud_model(model_id, by_id.get(model_id)))

    @staticmethod
    def _model_recommends_disabled_thinking(model_id: str, model: ModelConfig | None = None) -> bool:
        lowered = model_id.lower()
        if "gemma4" in lowered:
            return True
        raw = model.raw_metadata if model is not None else {}
        capabilities = raw.get("capabilities") if isinstance(raw, dict) else None
        if isinstance(capabilities, list) and any(str(item).lower() == "thinking" for item in capabilities):
            return True
        details = raw.get("details") if isinstance(raw, dict) else None
        family = details.get("family") if isinstance(details, dict) else None
        return isinstance(family, str) and "gemma4" in family.lower()

    def _refresh_ollama_thinking_recommendation(self) -> None:
        try:
            target = self.query_one("#ollama-thinking-recommendation", Static)
        except Exception:
            return
        by_id = {model.model_id: model for model in self._all_models}
        recommended_models = [
            model_id
            for model_id in self.selected_models
            if self._model_recommends_disabled_thinking(model_id, by_id.get(model_id))
        ]
        if self.provider != "ollama":
            target.update("Used only for Ollama runs; OpenRouter runs ignore this setting.")
            return
        if recommended_models:
            state = "ON" if self.disable_ollama_thinking else "OFF"
            target.update(
                Text(
                    "Recommendation: keep Disable Ollama thinking ON for "
                    f"{', '.join(recommended_models[:3])}"
                    f"{'...' if len(recommended_models) > 3 else ''}. Current setting: {state}.",
                    style="bold yellow",
                )
            )
            return
        target.update("Recommended for Gemma 4 and other thinking-capable Ollama models.")

    def _toggle_model(self, model_id: str, name: str | None = None) -> None:
        if not model_id:
            return
        if model_id in self.selected_models:
            self.selected_models.remove(model_id)
        else:
            self.selected_models.append(model_id)
            if name:
                self._model_names[model_id] = name
        self._render_model_table()
        self._render_selected_table()
        self._refresh_run_estimate()
        self._save_session()

    def _select_model(self, model_id: str, name: str | None = None) -> None:
        if not model_id or model_id in self.selected_models:
            return
        self.selected_models.append(model_id)
        if name:
            self._model_names[model_id] = name
        self._render_model_table()
        self._render_selected_table()
        self._refresh_run_estimate()
        self._save_session()

    def _deselect_model(self, model_id: str) -> None:
        if model_id not in self.selected_models:
            return
        self.selected_models.remove(model_id)
        self._render_model_table()
        self._render_selected_table()
        self._refresh_run_estimate()
        self._save_session()

    async def _fetch_models(self) -> None:
        with self._busy("fetch-models", label="Fetching…", spinner_widget="model-table"):
            try:
                async with make_llm_client(self.provider, base_url=self.base_url, ollama_host=self.ollama_host) as client:
                    models = await client.list_models()
            except Exception as exc:
                self._notify_error(f"Could not fetch models: {exc}", title="Fetch failed")
                return
            self._all_models = list(models[:200])
            for model in self._all_models:
                if model.name:
                    self._model_names[model.model_id] = model.name
            self._render_model_table()
            self._render_selected_table()
            self._set_monitor(
                f"Fetched {len(models)} {self._provider_title()} models. Type in the search box to filter, press Enter on a row to toggle."
            )
        if self.provider == "ollama":
            await self._fetch_loaded_models()

    async def _show_cloud_catalog(self) -> None:
        """List Ollama Cloud models so they can be browsed and selected even when
        none are pulled locally. Fetches the live catalogue from ollama.com and
        falls back to the env override / cache / built-in list when offline."""
        if self.provider != "ollama":
            self._notify_error(
                "Ollama Cloud models run through the Ollama provider. Switch Provider to Ollama first.",
                title="Cloud catalog",
            )
            return
        with self._busy("show-cloud-catalog", label="Loading…", spinner_widget="model-table"):
            if cloud_catalog_overridden():
                catalog = ollama_cloud_catalog()
                source = "OLLAMA_CLOUD_MODELS override"
            else:
                try:
                    catalog = await fetch_ollama_cloud_models()
                    source = "ollama.com (live)"
                except Exception as exc:
                    catalog = ollama_cloud_catalog()
                    source = f"offline fallback — fetch failed: {exc}"

            existing = {model.model_id for model in self._all_models}
            added = 0
            for model in catalog:
                if model.model_id not in existing:
                    self._all_models.append(model)
                    existing.add(model.model_id)
                    added += 1
                if model.name:
                    self._model_names.setdefault(model.model_id, model.name)
            # Filter the table to the cloud entries so they are visible immediately.
            with suppress(Exception):
                self.query_one("#model-search", Input).value = "cloud"
            self._render_model_table()
            self._render_selected_table()
        self._set_monitor(
            f"Loaded {len(catalog)} Ollama Cloud model(s) from {source} ({added} new). These route through a "
            "signed-in daemon (run 'ollama signin'); no local pull needed. Press Enter on a row to select. "
            "Set OLLAMA_CLOUD_MODELS to pin a custom list."
        )

    def _set_ollama_loaded(self, text: str) -> None:
        with suppress(Exception):
            self.query_one("#ollama-loaded", Static).update(text)

    def _reset_ollama_loaded_hint(self) -> None:
        self._set_ollama_loaded(
            "Press 'Loaded in Ollama' to query /api/ps for models held in VRAM."
            if self.provider == "ollama"
            else "Ollama only — switch the provider to Ollama to see models loaded in VRAM."
        )

    @staticmethod
    def _format_vram(size_bytes: int | None) -> str:
        if not isinstance(size_bytes, (int, float)) or size_bytes <= 0:
            return "-"
        return f"{size_bytes / 1e9:.1f} GB"

    async def _fetch_loaded_models(self) -> None:
        if self.provider != "ollama":
            self._set_ollama_loaded("Ollama only — switch the provider to Ollama to see models loaded in VRAM.")
            return
        with self._busy("fetch-loaded", label="Querying…", spinner_widget="ollama-loaded"):
            try:
                async with make_llm_client("ollama", base_url=self.base_url, ollama_host=self.ollama_host) as client:
                    loaded = await client.list_loaded_models()
            except Exception as exc:
                self._set_ollama_loaded(f"Could not query Ollama /api/ps: {exc}")
                return
            if not loaded:
                self._set_ollama_loaded("No models are currently loaded in Ollama.")
                return
            lines = []
            for entry in loaded:
                vram = self._format_vram(entry.get("size_vram") or entry.get("size"))
                lines.append(f"{entry['model']} | VRAM {vram}")
            self._set_ollama_loaded("\n".join(lines))
