from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .constants import DEFAULT_REASONING_MAX_COMPLETION_TOKENS

OutputMode = Literal["label_only", "explanation"]
RunMode = Literal["pilot", "full"]
ParseStatus = Literal["valid", "invalid", "error"]
ResponseStatus = Literal["success", "api_error", "transport_error", "malformed_response", "skipped"]


@dataclass(frozen=True)
class DatasetRow:
    row_number: int
    sentence: str
    hidden_label: str
    is_duplicate: bool
    has_conflicting_duplicate: bool
    duplicate_group_size: int

    def blind(self) -> BlindExample:
        return BlindExample(row_number=self.row_number, sentence=self.sentence)


@dataclass(frozen=True)
class BlindExample:
    row_number: int
    sentence: str


@dataclass(frozen=True)
class PromptConfig:
    prompt_id: str
    system_prompt: str
    user_template: str
    output_mode: OutputMode
    prompt_hash: str


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    name: str | None = None
    context_length: int | None = None
    pricing: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    models: list[str]
    prompt: PromptConfig
    mode: RunMode
    dataset_path: str
    db_path: str
    base_url: str
    sample_per_class: int = 30
    seed: int = 42
    temperature: float = 0.0
    max_completion_tokens: int = 64
    concurrency: int = 1
    retries: int = 3
    reasoning_max_completion_tokens: int = DEFAULT_REASONING_MAX_COMPLETION_TOKENS
    model_max_completion_tokens: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedResponse:
    normalized_label: str | None
    parse_status: ParseStatus
    explanation: str | None = None


@dataclass(frozen=True)
class LLMResponseRecord:
    row_number: int
    model_id: str
    prompt_hash: str
    raw_content: str | None
    normalized_label: str | None
    parse_status: ParseStatus
    status: ResponseStatus
    explanation: str | None = None
    raw_response_json: dict[str, Any] | None = None
    latency_ms: float | None = None
    error: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    generation_id: str | None = None


@dataclass(frozen=True)
class EvaluationResult:
    model_id: str
    scope: str
    row_count: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: dict[str, dict[str, int]]
    invalid_output_count: int
    api_error_count: int
    mean_latency_ms: float | None
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
