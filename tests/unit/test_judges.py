import httpx
import pytest
from deepeval.models import AnthropicModel, AzureOpenAIModel, OllamaModel, OpenAIModel

from evalkit.config import Settings
from evalkit.judges import OfflineJudge, get_judge, judge_status
from evalkit.judges.factory import ollama_status
from evalkit.judges.offline import OfflineJudgeCalled


def settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


@pytest.mark.parametrize(
    ("kw", "cls"),
    [
        ({"judge_provider": "ollama"}, OllamaModel),
        ({"judge_provider": "openai", "openai_api_key": "sk-test"}, OpenAIModel),
        (
            {
                "judge_provider": "anthropic",
                "anthropic_api_key": "k",
                "judge_model": "claude-sonnet-5-5",
            },
            AnthropicModel,
        ),
        (
            {
                "judge_provider": "azure",
                "azure_openai_api_key": "k",
                "azure_openai_endpoint": "https://x.openai.azure.com",
                "azure_openai_deployment": "d",
                "judge_model": "gpt-4o",
            },
            AzureOpenAIModel,
        ),
    ],
)
def test_factory_builds_the_configured_provider(kw, cls):
    assert isinstance(get_judge(settings(**kw)), cls)


@pytest.mark.parametrize(
    ("kw", "missing"),
    [
        ({"judge_provider": "openai"}, "OPENAI_API_KEY"),
        ({"judge_provider": "anthropic"}, "ANTHROPIC_API_KEY"),
        ({"judge_provider": "azure", "azure_openai_api_key": "k"}, "AZURE_OPENAI_ENDPOINT"),
    ],
)
def test_status_names_the_missing_credential(kw, missing):
    status = judge_status(settings(**kw))
    assert not status.ok
    assert missing in status.reason


def _fake_get(payload=None, exc=None):
    def get(url, timeout):
        if exc:
            raise exc
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    return get


def test_ollama_status_ok_when_model_pulled(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({"models": [{"name": "llama3.1:8b"}]}))
    assert ollama_status("http://x", "llama3.1:8b").ok


def test_ollama_status_matches_latest_tag(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({"models": [{"name": "qwen2.5:latest"}]}))
    assert ollama_status("http://x", "qwen2.5").ok


def test_ollama_status_tells_you_to_pull(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get({"models": [{"name": "other:1b"}]}))
    status = ollama_status("http://x", "llama3.1:8b")
    assert not status.ok
    assert "ollama pull llama3.1:8b" in status.reason


def test_ollama_status_unreachable(monkeypatch):
    monkeypatch.setattr(httpx, "get", _fake_get(exc=httpx.ConnectError("refused")))
    status = ollama_status("http://x", "m")
    assert not status.ok
    assert "not reachable" in status.reason


def test_offline_judge_fails_loudly_when_called():
    with pytest.raises(OfflineJudgeCalled, match=r"pytest\.mark\.llm"):
        OfflineJudge().generate("score this")
