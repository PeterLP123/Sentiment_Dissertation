import os

from sentiment_benchmark.env import load_env_file


def test_load_env_file_sets_missing_values(tmp_path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENROUTER_API_KEY='test-key'\nEXISTING=from-file\n", encoding="utf-8")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING", "already-set")
    load_env_file(env_file)
    assert os.environ["OPENROUTER_API_KEY"] == "test-key"
    assert os.environ["EXISTING"] == "already-set"

