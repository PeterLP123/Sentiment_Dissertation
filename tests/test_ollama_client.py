import asyncio
from types import SimpleNamespace

from sentiment_benchmark.models import BlindExample
from sentiment_benchmark.ollama_client import OllamaClient
from sentiment_benchmark.prompts import make_prompt
from sentiment_benchmark.providers import endpoint_for_provider, normalize_provider


def run(coro):
    return asyncio.run(coro)


class FakeOllamaAsyncClient:
    def __init__(self) -> None:
        self.chat_calls: list[dict] = []
        self.closed = False

    async def list(self):
        return {
            "models": [
                {
                    "model": "gemma3:latest",
                    "name": "Gemma 3",
                    "details": {"context_length": 8192},
                }
            ]
        }

    async def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return {
            "model": kwargs["model"],
            "message": {"role": "assistant", "content": "positive"},
            "prompt_eval_count": 12,
            "eval_count": 2,
        }

    async def aclose(self) -> None:
        self.closed = True


def test_ollama_lists_models_from_injected_client() -> None:
    async def scenario() -> None:
        fake = FakeOllamaAsyncClient()
        client = OllamaClient(host="http://desktop-pc:11434", client=fake)

        models = await client.list_models()
        await client.close()

        assert len(models) == 1
        assert models[0].model_id == "gemma3:latest"
        assert models[0].name == "Gemma 3"
        assert models[0].context_length == 8192
        assert fake.closed is False

    run(scenario())


def test_ollama_classify_uses_chat_options_and_existing_parser() -> None:
    async def scenario() -> None:
        fake = FakeOllamaAsyncClient()
        client = OllamaClient(host="http://desktop-pc:11434", client=fake)
        prompt = make_prompt("test", "Return a label.", "Sentence:\n{sentence}\n\nSentiment label:", "label_only")

        record = await client.classify(
            "gemma3:latest",
            prompt,
            BlindExample(7, "Profits rose sharply."),
            temperature=0.2,
            max_completion_tokens=32,
        )

        assert record.status == "success"
        assert record.normalized_label == "positive"
        assert record.prompt_tokens == 12
        assert record.completion_tokens == 2
        assert record.total_tokens == 14
        call = fake.chat_calls[0]
        assert call["model"] == "gemma3:latest"
        assert call["stream"] is False
        assert call["options"] == {"temperature": 0.2, "num_predict": 32}
        assert call["messages"][0]["role"] == "system"

    run(scenario())


def test_ollama_classify_can_disable_thinking() -> None:
    async def scenario() -> None:
        fake = FakeOllamaAsyncClient()
        client = OllamaClient(client=fake)
        prompt = make_prompt("test", "Return a label.", "{sentence}", "label_only")

        record = await client.classify(
            "gemma4:12b",
            prompt,
            BlindExample(1, "Profits rose sharply."),
            ollama_think=False,
        )

        assert record.status == "success"
        assert fake.chat_calls[0]["think"] is False

    run(scenario())


def test_ollama_object_response_shape_is_supported() -> None:
    async def scenario() -> None:
        class ObjectResponseClient(FakeOllamaAsyncClient):
            async def chat(self, **kwargs):
                return SimpleNamespace(
                    message=SimpleNamespace(content="neutral"),
                    prompt_eval_count=3,
                    eval_count=1,
                )

        prompt = make_prompt("test", "Return a label.", "{sentence}", "label_only")
        record = await OllamaClient(client=ObjectResponseClient()).classify("gemma3", prompt, BlindExample(1, "Flat."))

        assert record.status == "success"
        assert record.normalized_label == "neutral"
        assert record.total_tokens == 4

    run(scenario())


def test_ollama_response_error_becomes_api_error() -> None:
    async def scenario() -> None:
        class FakeResponseError(Exception):
            status_code = 404
            error = "model not found"

        class FailingClient(FakeOllamaAsyncClient):
            async def chat(self, **kwargs):
                raise FakeResponseError("missing")

        prompt = make_prompt("test", "Return a label.", "{sentence}", "label_only")
        record = await OllamaClient(client=FailingClient()).classify("missing-model", prompt, BlindExample(1, "Flat."))

        assert record.status == "api_error"
        assert "Ollama HTTP 404: model not found" in (record.error or "")

    run(scenario())


def test_provider_helpers_resolve_ollama_endpoint() -> None:
    assert normalize_provider("OLLAMA") == "ollama"
    assert endpoint_for_provider("ollama", base_url="https://openrouter.test", ollama_host="http://pc:11434/") == "http://pc:11434"
    assert endpoint_for_provider("openrouter", base_url="https://openrouter.test/", ollama_host="http://pc:11434") == "https://openrouter.test"
