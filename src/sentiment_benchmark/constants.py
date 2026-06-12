from pathlib import Path

ALLOWED_LABELS = ("positive", "negative", "neutral")
VALID_LABELS = frozenset(ALLOWED_LABELS)
PREDICTION_SENTINELS = ("__invalid__", "__error__")
CONFUSION_PREDICTION_LABELS = (*ALLOWED_LABELS, *PREDICTION_SENTINELS)


def is_valid_label(label: str) -> bool:
    return label in VALID_LABELS
DEFAULT_DATASET_PATH = Path("Data/derived/labeled/financial_sentiment_v2.csv")
DEFAULT_DB_PATH = Path("results/sentiment_benchmark.sqlite")
DEFAULT_PROMPTS_PATH = Path("configs/default_prompts.toml")
DEFAULT_PROVIDER = "openrouter"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_APP_TITLE = "Sentiment Dissertation Benchmark"
DEFAULT_SEED = 42
DEFAULT_PILOT_PER_CLASS = 30
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_COMPLETION_TOKENS = 64
# Soft-label responses are a JSON object with three float fields; the default
# 64-token budget truncates that for some tokenizers, producing invalid parses
# on paid calls. Runs with a soft_label prompt get at least this budget.
SOFT_LABEL_MIN_COMPLETION_TOKENS = 128
DEFAULT_CONCURRENCY = 1
DEFAULT_RETRIES = 3
# Reasoning-capable models spend completion tokens on hidden reasoning before
# emitting the label, so a small budget leaves no room for the answer. This
# larger budget is applied automatically to models matching REASONING_MODEL_MARKERS.
DEFAULT_REASONING_MAX_COMPLETION_TOKENS = 2048
REASONING_MODEL_MARKERS = (
    "gpt-5",
    "o1",
    "o3",
    "o4-mini",
    "reasoning",
    "thinking",
    "deepseek-r",
    "qwq",
    "gemini-2.5",
    "gemini-3",
    "minimax-m3",
    "deepseek-v4",
)
