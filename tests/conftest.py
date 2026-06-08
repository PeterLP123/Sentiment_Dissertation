import pytest


@pytest.fixture(autouse=True)
def _local_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SENTIMENT_BENCH_DB_BACKEND", "sqlite")
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_ID", "test-machine")
    monkeypatch.setenv("SENTIMENT_BENCH_MACHINE_LABEL", "test-host")
    monkeypatch.setenv("SENTIMENT_BENCH_PROVIDER", "openrouter")
    monkeypatch.setenv("SENTIMENT_BENCH_AUTO_FETCH_MODELS", "0")
