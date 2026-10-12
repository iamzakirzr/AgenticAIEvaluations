"""A minimal chat-completion client for the bundled apps, plus a scripted stand-in.

Deliberately not LangChain: the bundled apps exist to be *tested*, and the
less framework between the prompt and the HTTP call, the easier it is to see
why a metric moved.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from evalkit.config import Settings, get_settings


@dataclass
class ToolCallRequest:
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    content: str
    tool_calls: list[ToolCallRequest] = field(default_factory=list)


Message = dict[str, Any]


class ChatModel(Protocol):
    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ChatResult: ...


class OllamaChat:
    def __init__(self, model: str, base_url: str, timeout_s: float = 120.0):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.timeout_s = timeout_s

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0},
        }
        if tools:
            payload["tools"] = tools
        resp = httpx.post(self.url, json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        msg = resp.json()["message"]
        calls = [
            ToolCallRequest(c["function"]["name"], c["function"].get("arguments") or {})
            for c in msg.get("tool_calls") or []
        ]
        return ChatResult(msg.get("content") or "", calls)


class OpenAIChat:
    """Any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, LiteLLM, ...)."""

    def __init__(self, model: str, api_key: str, base_url: str, timeout_s: float = 120.0):
        self.model = model
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.timeout_s = timeout_s

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai(messages),
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools
        resp = httpx.post(self.url, json=payload, headers=self.headers, timeout=self.timeout_s)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        calls = [
            ToolCallRequest(c["function"]["name"], json.loads(c["function"]["arguments"] or "{}"))
            for c in msg.get("tool_calls") or []
        ]
        return ChatResult(msg.get("content") or "", calls)


def _to_openai(messages: list[Message]) -> list[Message]:
    """Convert the Ollama-style history the apps keep into OpenAI's wire format.

    OpenAI needs an id on every assistant tool call, JSON-string arguments, and
    a matching `tool_call_id` on each tool result.
    """
    out: list[Message] = []
    pending: list[str] = []
    for i, msg in enumerate(messages):
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            ids = [f"call_{i}_{j}" for j in range(len(msg["tool_calls"]))]
            pending.extend(ids)
            out.append(
                {
                    "role": "assistant",
                    "content": msg.get("content") or None,
                    "tool_calls": [
                        {
                            "id": cid,
                            "type": "function",
                            "function": {
                                "name": c["function"]["name"],
                                "arguments": json.dumps(c["function"]["arguments"]),
                            },
                        }
                        for cid, c in zip(ids, msg["tool_calls"], strict=True)
                    ],
                }
            )
        elif msg["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": msg["content"]})
        else:
            out.append({"role": msg["role"], "content": msg["content"]})
    return out


class ScriptedChat:
    """Deterministic model: a function from the conversation so far to a reply.

    Used by the unit tier to drive the real app code (retrieval, tool loop,
    history handling) without a model, so the framework's own plumbing is
    tested on every pull request.
    """

    def __init__(self, policy: Callable[[list[Message], list[dict] | None], ChatResult]):
        self.policy = policy
        self.calls: list[list[Message]] = []

    def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ChatResult:
        self.calls.append([dict(m) for m in messages])
        return self.policy(messages, tools)


def build_chat_model(settings: Settings | None = None) -> ChatModel:
    s = settings or get_settings()
    if s.sut_provider == "ollama":
        return OllamaChat(s.sut_model, s.ollama_base_url, s.request_timeout_s)
    if s.sut_provider == "openai":
        if not s.openai_api_key:
            raise ValueError("SUT_PROVIDER=openai needs OPENAI_API_KEY")
        return OpenAIChat(s.sut_model, s.openai_api_key, s.openai_base_url, s.request_timeout_s)
    raise ValueError(
        "SUT_PROVIDER=scripted has no generic model; build the app with its scripted policy "
        "(see evalkit.sut.factory)."
    )
