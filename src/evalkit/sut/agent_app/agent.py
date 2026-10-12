"""A tool-calling agent: a while-loop around a model that may request tools.

It records its own trajectory as a trace so trajectory metrics (step budget,
redundant calls, DeepEval's StepEfficiency) can score *how* it got to the
answer, not only the answer.
"""

from __future__ import annotations

import re
import time
from typing import Any

from evalkit.sut.agent_app.tools import Tool, default_tools
from evalkit.sut.base import AgentResponse, ToolInvocation
from evalkit.sut.llm_client import ChatModel, ChatResult, Message, ToolCallRequest

SYSTEM_PROMPT = """You are the Nimbus Notes billing assistant.
Use tools for every fact: search_docs for policy and prices, get_order for invoices,
calculator for any arithmetic. Do not guess numbers. Call each tool only when needed,
then give a short final answer that states the result."""

STEP_LIMIT_MESSAGE = "I could not finish this request within the step limit."


class ToolAgent:
    def __init__(self, model: ChatModel, tools: list[Tool] | None = None, max_steps: int = 6):
        self.model = model
        self.tools = {t.name: t for t in (tools or default_tools())}
        self.max_steps = max_steps

    def run(self, task: str) -> AgentResponse:
        start = time.perf_counter()
        messages: list[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]
        schemas = [t.schema() for t in self.tools.values()]
        invocations: list[ToolInvocation] = []
        spans: list[dict[str, Any]] = []

        for step in range(1, self.max_steps + 1):
            result = self.model.chat(messages, schemas)
            spans.append(
                {
                    "name": "model",
                    "type": "llm",
                    "input": messages[-1]["content"],
                    "output": result.content or [c.name for c in result.tool_calls],
                }
            )
            if not result.tool_calls:
                return self._finish(
                    task, result.content.strip(), invocations, spans, step, "final_answer", start
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": result.content,
                    "tool_calls": [_wire(c) for c in result.tool_calls],
                }
            )
            for call in result.tool_calls:
                output = self._invoke(call)
                invocations.append(ToolInvocation(call.name, call.arguments, output))
                spans.append(
                    {"name": call.name, "type": "tool", "input": call.arguments, "output": output}
                )
                messages.append({"role": "tool", "content": output, "name": call.name})

        return self._finish(
            task, STEP_LIMIT_MESSAGE, invocations, spans, self.max_steps, "step_limit", start
        )

    def _invoke(self, call: ToolCallRequest) -> str:
        tool = self.tools.get(call.name)
        if tool is None:
            return f"error: unknown tool {call.name}"
        try:
            return tool.fn(**call.arguments)
        except TypeError as exc:
            return f"error: bad arguments for {call.name}: {exc}"

    @staticmethod
    def _finish(task, output, invocations, spans, steps, reason, start) -> AgentResponse:
        trace = {
            "name": "billing_agent",
            "type": "agent",
            "input": task,
            "output": output,
            "children": spans,
        }
        return AgentResponse(output, invocations, steps, reason, time.perf_counter() - start, trace)


def _wire(call: ToolCallRequest) -> dict[str, Any]:
    return {"function": {"name": call.name, "arguments": call.arguments}}


_INVOICE = re.compile(r"\bINV-\d+\b", re.IGNORECASE)
_ARITHMETIC = re.compile(r"\d[\d.\s]*[-+*/][\d.\s+\-*/()]*\d")


def keyword_policy(messages: list[Message], tools: list[dict] | None = None) -> ChatResult:
    """Scripted 'model' for the unit tier: route by keyword, answer from the tool output."""
    if messages[-1]["role"] == "tool":
        observations = [m["content"] for m in messages if m["role"] == "tool"]
        return ChatResult(f"Result: {observations[-1]}")
    task = messages[-1]["content"]
    if m := _INVOICE.search(task):
        return ChatResult("", [ToolCallRequest("get_order", {"order_id": m.group(0).upper()})])
    if m := _ARITHMETIC.search(task):
        return ChatResult("", [ToolCallRequest("calculator", {"expression": m.group(0).strip()})])
    return ChatResult("", [ToolCallRequest("search_docs", {"query": task})])
