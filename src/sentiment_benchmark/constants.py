from pathlib import Path

ALLOWED_LABELS = ("positive", "negative", "neutral")
DEFAULT_DATASET_PATH = Path("Data/data.csv")
DEFAULT_DB_PATH = Path("results/sentiment_benchmark.sqlite")
DEFAULT_PROMPTS_PATH = Path("configs/default_prompts.toml")
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_APP_TITLE = "Sentiment Dissertation Benchmark"
DEFAULT_SEED = 42
DEFAULT_PILOT_PER_CLASS = 30
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_COMPLETION_TOKENS = 8
DEFAULT_CONCURRENCY = 1
DEFAULT_RETRIES = 3

