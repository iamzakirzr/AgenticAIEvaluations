"""The contract between the framework and anything it tests.

Tests only ever talk to these Protocols. To evaluate your own application,
write a class with the same method (an HTTP client calling your service is
typical) and return it from the fixture in tests/conftest.py. Nothing else in
the framework changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


@dataclass
class RAGResponse:
    answer: str
    # Each retrieved chunk as the generator saw it, prefixed "[source] ".
    contexts: list[str]
    sources: list[str]
    latency_s: float = 0.0


@dataclass
class ToolInvocation:
    name: str
    args: dict[str, Any]
    output: str


@dataclass
class AgentResponse:
    output: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    # Number of model calls the agent made, including the final answer.
    steps: int = 0
    stopped_reason: Literal["final_answer", "step_limit", "error"] = "final_answer"
    latency_s: float = 0.0
    # Execution trace (agent span with llm/tool child spans) in the shape
    # DeepEval's trace-based metrics read. Optional for third-party agents.
    trace: dict[str, Any] | None = None


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@runtime_checkable
class RAGApp(Protocol):
    def query(self, question: str) -> RAGResponse: ...


@runtime_checkable
class AgentApp(Protocol):
    def run(self, task: str) -> AgentResponse: ...


@runtime_checkable
class ChatApp(Protocol):
    # The bot description that RoleAdherence and red teaming score against.
    role_description: str

    def reply(self, history: list[ChatMessage], message: str) -> str: ...
