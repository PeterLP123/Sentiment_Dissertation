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
                    "digest": "sha256:abc123",
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


def test_ollama_structured_label_keep_alive_and_default_thinking() -> None:
    async def scenario() -> None:
        fake = FakeOllamaAsyncClient()
        client = OllamaClient(
            client=fake,
            keep_alive="30m",
            structured_label_output=True,
            default_think=False,
        )
        prompt = make_prompt("test", "Return a label.", "{sentence}", "label_only")

        record = await client.classify("gemma3:latest", prompt, BlindExample(1, "Profits rose."))
        digest = await client.model_digest("gemma3:latest")

        assert record.normalized_label == "positive"
        assert digest == "sha256:abc123"
        call = fake.chat_calls[0]
        assert call["format"] == {"type": "string", "enum": ["positive", "negative", "neutral"]}
        assert call["keep_alive"] == "30m"
        assert call["think"] is False

    run(scenario())


def test_ollama_structured_label_uses_explicit_enum_contract() -> None:
    async def scenario() -> None:
        class FiveLevelClient(FakeOllamaAsyncClient):
            async def chat(self, **kwargs):
                self.chat_calls.append(kwargs)
                return {"message": {"content": "very_positive"}}

        fake = FiveLevelClient()
        client = OllamaClient(client=fake, structured_label_output=True)
        prompt = make_prompt("strategy", "Return an enum.", "{sentence}", "label_only")
        labels = ("very_negative", "negative", "neutral", "positive", "very_positive")
        record = await client.classify("gemma", prompt, BlindExample(1, "News"), allowed_labels=labels)

        assert record.normalized_label == "very_positive"
        assert fake.chat_calls[0]["format"] == {"type": "string", "enum": list(labels)}

    run(scenario())


def test_ollama_never_unload_keep_alive_is_sent_as_integer() -> None:
    async def scenario() -> None:
        fake = FakeOllamaAsyncClient()
        client = OllamaClient(client=fake, keep_alive="-1")
        prompt = make_prompt("test", "Return a label.", "{sentence}", "label_only")

        await client.classify("gemma4:e4b-it-qat", prompt, BlindExample(1, "Profits rose."))

        assert fake.chat_calls[0]["keep_alive"] == -1

    run(scenario())


def test_ollama_structured_soft_label_uses_required_probability_schema() -> None:
    async def scenario() -> None:
        class SoftClient(FakeOllamaAsyncClient):
            async def chat(self, **kwargs):
                self.chat_calls.append(kwargs)
                return {
                    "model": kwargs["model"],
                    "message": {"role": "assistant", "content": '{"positive":0.6,"negative":0.1,"neutral":0.3}'},
                    "prompt_eval_count": 12,
                    "eval_count": 12,
                }

        fake = SoftClient()
        client = OllamaClient(client=fake, structured_label_output=True)
        prompt = make_prompt("soft", "Return JSON.", "{sentence}", "soft_label")
        record = await client.classify("gemma4:e4b-it-qat", prompt, BlindExample(1, "Profits rose."))

        assert record.status == "success"
        schema = fake.chat_calls[0]["format"]
        assert schema["required"] == ["positive", "negative", "neutral"]
        assert schema["additionalProperties"] is False
        assert schema["properties"]["positive"] == {"type": "number", "minimum": 0, "maximum": 1}

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
    assert normalize_provider("CEREBRAS") == "cerebras"
    assert endpoint_for_provider("ollama", base_url="https://openrouter.test", ollama_host="http://pc:11434/") == "http://pc:11434"
    assert endpoint_for_provider("openrouter", base_url="https://openrouter.test/", ollama_host="http://pc:11434") == "https://openrouter.test"
    assert (
        endpoint_for_provider(
            "cerebras",
            base_url="https://openrouter.test/",
            ollama_host="http://pc:11434",
            cerebras_base_url="https://cerebras.test/v1/",
        )
        == "https://cerebras.test/v1"
    )


def test_ollama_empty_completion_is_malformed() -> None:
    class EmptyResponseClient(FakeOllamaAsyncClient):
        async def chat(self, **kwargs):
            return {
                "model": kwargs["model"],
                "message": {"role": "assistant", "content": ""},
                "prompt_eval_count": 12,
                "eval_count": 0,
            }

    prompt = make_prompt("test", "Return a label.", "{sentence}", "label_only")
    record = run(OllamaClient(client=EmptyResponseClient()).classify("gemma4:12b", prompt, BlindExample(1, "Flat.")))

    # An empty completion must not count as success: resumes would otherwise
    # treat the row as permanently complete.
    assert record.status == "malformed_response"
    assert record.parse_status == "error"
    assert "empty completion" in (record.error or "")
