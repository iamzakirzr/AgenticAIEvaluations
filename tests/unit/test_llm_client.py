import json

import httpx
import pytest

from evalkit.config import Settings
from evalkit.sut.llm_client import OllamaChat, OpenAIChat, _to_openai, build_chat_model


def _capture(monkeypatch, response_json):
    sent = {}

    def post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, json=json, headers=headers)
        return httpx.Response(200, json=response_json, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", post)
    return sent


def test_ollama_parses_tool_calls(monkeypatch):
    sent = _capture(
        monkeypatch,
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}
                ],
            }
        },
    )
    r = OllamaChat("m", "http://o:11434/").chat([{"role": "user", "content": "hi"}], [{"t": 1}])
    assert sent["url"] == "http://o:11434/api/chat"
    assert sent["json"]["options"]["temperature"] == 0
    assert sent["json"]["tools"] == [{"t": 1}]
    call = r.tool_calls[0]
    assert (call.name, call.arguments) == ("calculator", {"expression": "1+1"})


def test_openai_parses_json_string_arguments(monkeypatch):
    sent = _capture(
        monkeypatch,
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {
                                    "name": "get_order",
                                    "arguments": '{"order_id": "INV-1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )
    r = OpenAIChat("gpt", "sk-x", "https://api/v1").chat([{"role": "user", "content": "q"}])
    assert sent["headers"]["Authorization"] == "Bearer sk-x"
    assert r.content == ""
    assert r.tool_calls[0].arguments == {"order_id": "INV-1"}


def test_history_is_converted_to_openai_tool_call_ids():
    history = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "a", "arguments": {"x": 1}}},
                {"function": {"name": "b", "arguments": {}}},
            ],
        },
        {"role": "tool", "content": "ra", "name": "a"},
        {"role": "tool", "content": "rb", "name": "b"},
    ]
    out = _to_openai(history)
    ids = [c["id"] for c in out[1]["tool_calls"]]
    assert [m.get("tool_call_id") for m in out[2:]] == ids
    assert json.loads(out[1]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}


def test_openai_sut_requires_a_key():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_chat_model(Settings(_env_file=None, sut_provider="openai"))
