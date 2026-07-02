"""Dashboard and resource/GPU monitor feature logic for the TUI."""

from __future__ import annotations

import asyncio
import os
import subprocess
from contextlib import suppress

from textual.widgets import Input, Static

from .dataset import compute_stats, load_dataset
from .storage import BenchmarkStore
from .tui_base import AppMixin


class MonitorMixin(AppMixin):
    def _refresh_dashboard(self) -> None:
        rows = load_dataset(self.dataset_path)
        stats = compute_stats(rows)
        if self.provider == "openrouter":
            credential_line = f"OpenRouter API key: {'present' if os.getenv('OPENROUTER_API_KEY') else 'missing'}"
        elif self.provider == "cerebras":
            credential_line = f"Cerebras API key: {'present' if os.getenv('CEREBRAS_API_KEY') else 'missing'}"
        else:
            credential_line = "Ollama API key: not required"
        endpoint = self._active_endpoint()
        storage_label = BenchmarkStore(self.db_path).storage_label()
        self.query_one("#dashboard", Static).update(
            "\n".join(
                [
                    f"Dataset: {self.dataset_path}",
                    f"Rows: {stats.row_count}",
                    f"Labels: {stats.label_counts}",
                    f"Conflicting duplicate rows excluded from primary metrics: {stats.conflicting_duplicate_rows}",
                    f"Primary scoring rows: {stats.primary_row_count}",
                    f"Result DB: {storage_label}",
                    f"Provider: {self._provider_title()}",
                    f"Endpoint: {endpoint}",
                    credential_line,
                ]
            )
        )
        self._refresh_status_bar()

    @staticmethod
    def _format_gpu_value(value: str, suffix: str = "") -> str:
        value = value.strip()
        if not value or "not supported" in value.lower():
            return "-"
        try:
            number = float(value)
        except ValueError:
            return value
        if number.is_integer():
            return f"{int(number)}{suffix}"
        return f"{number:.1f}{suffix}"

    def _gpu_monitor_lines(self) -> list[str]:
        command = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                check=True,
                text=True,
                timeout=2,
            )
        except FileNotFoundError:
            return ["GPU: nvidia-smi not found."]
        except subprocess.TimeoutExpired:
            return ["GPU: nvidia-smi timed out."]
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or exc.stdout or str(exc)).strip()
            return [f"GPU: nvidia-smi failed: {error}"]

        lines: list[str] = []
        for index, raw_line in enumerate(result.stdout.splitlines()):
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 7:
                continue
            name, util, mem_used, mem_total, power_draw, power_limit, temp = parts[:7]
            used = self._format_gpu_value(mem_used)
            total = self._format_gpu_value(mem_total)
            memory_percent = "-"
            with suppress(ValueError, ZeroDivisionError):
                memory_percent = f"{(float(mem_used) / float(mem_total)) * 100:.0f}%"
            lines.append(
                "GPU "
                f"{index}: {name} | util {self._format_gpu_value(util, '%')} | "
                f"VRAM {used}/{total} MB ({memory_percent}) | "
                f"power {self._format_gpu_value(power_draw)}/{self._format_gpu_value(power_limit)} W | "
                f"temp {self._format_gpu_value(temp)} C"
            )
        return lines or ["GPU: no NVIDIA devices reported."]

    def _refresh_resource_monitor(self) -> None:
        try:
            concurrency = self.query_one("#concurrency", Input).value.strip() or "?"
        except Exception:
            concurrency = "?"
        try:
            max_tokens = self.query_one("#max-tokens", Input).value.strip() or "?"
        except Exception:
            max_tokens = "?"

        provider_lines = []
        if self.provider == "cerebras":
            provider_lines.append("Cerebras request pace: calibrated from live API-key quota headers")
        elif self.provider == "ollama":
            provider_lines.extend(
                [
                    f"Ollama NUM_PARALLEL env visible to TUI: {os.getenv('OLLAMA_NUM_PARALLEL') or 'default'}",
                    f"Ollama thinking: {'disabled' if self.disable_ollama_thinking else 'provider default'}",
                ]
            )
        lines = [
            f"Provider: {self._provider_title()} | Endpoint: {self._active_endpoint()}",
            f"Selected models: {len(self.selected_models)} | Fetched provider models: {len(self._all_models)}",
            f"Run settings: TUI concurrency {concurrency} | max completion tokens {max_tokens}",
            *provider_lines,
            *self._gpu_lines,
        ]
        content = "\n".join(lines)
        for widget_id in ("#dashboard-resource-monitor", "#run-resource-monitor"):
            with suppress(Exception):
                self.query_one(widget_id, Static).update(content)

    async def _refresh_gpu_lines(self) -> None:
        """Refresh cached GPU stats in a worker thread, then re-render the monitor."""
        try:
            self._gpu_lines = await asyncio.to_thread(self._gpu_monitor_lines)
        except Exception:  # pragma: no cover - defensive
            self._gpu_lines = ["GPU: monitor unavailable."]
        self._refresh_resource_monitor()
