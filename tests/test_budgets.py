from sentiment_benchmark.budgets import is_reasoning_model, resolve_max_completion_tokens
from sentiment_benchmark.constants import DEFAULT_REASONING_MAX_COMPLETION_TOKENS


def test_is_reasoning_model_detects_known_families() -> None:
    assert is_reasoning_model("openai/gpt-5.5")
    assert is_reasoning_model("openai/o3-mini")
    assert is_reasoning_model("google/gemini-3.5-flash")
    assert is_reasoning_model("minimax-m3:cloud")
    assert is_reasoning_model("deepseek-v4-pro:cloud")
    assert not is_reasoning_model("openai/gpt-4o-mini")
    assert not is_reasoning_model("anthropic/claude-3.5-sonnet")


def test_reasoning_model_gets_bumped_budget() -> None:
    assert resolve_max_completion_tokens("openai/gpt-5.5", default=64) == DEFAULT_REASONING_MAX_COMPLETION_TOKENS


def test_reasoning_bump_never_lowers_a_larger_default() -> None:
    assert resolve_max_completion_tokens("openai/gpt-5.5", default=4096) == 4096


def test_non_reasoning_model_uses_default() -> None:
    assert resolve_max_completion_tokens("openai/gpt-4o-mini", default=64) == 64


def test_explicit_override_wins_over_reasoning_bump() -> None:
    overrides = {"openai/gpt-5.5": 100}
    assert resolve_max_completion_tokens("openai/gpt-5.5", default=64, overrides=overrides) == 100
