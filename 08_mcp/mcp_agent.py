"""
Building a LangChain agent over MCP tools -- and measuring whether it routes well.

=============================================================================
TWO WAYS TO BUILD AN AGENT IN THIS STACK, AND WHEN TO USE EACH
=============================================================================
Lesson 03 built an agent by hand with a LangGraph `StateGraph`: explicit nodes,
explicit edges, explicit state. Roughly 80 lines.

`langchain.agents.create_agent` is the prebuilt version: one call, and you get
a compiled LangGraph graph with the same loop inside it.

    create_agent(model, tools, system_prompt=..., checkpointer=..., middleware=...)

It returns a `CompiledStateGraph`, so everything lesson 03 taught still applies
-- `.invoke`, `.stream`, `thread_id`, interrupts, checkpointers.

WHICH TO USE:
    create_agent   the loop is standard (model -> tools -> model) and you want
                   middleware, structured output and checkpointing for free
    StateGraph     you need a node the loop does not have: a validation gate, a
                   fan-out, a human approval step in the middle, a custom
                   router

The honest answer for interviews: start with `create_agent`; drop to
`StateGraph` the moment you need a step that is not "call a tool". Reaching for
`StateGraph` first is a common way to write 200 lines that the prebuilt already
does, and reaching for `create_agent` when you need a custom node is how people
end up abusing middleware to fake control flow.

=============================================================================
THE MEASUREMENT THIS FILE IS REALLY FOR
=============================================================================
An agent with three MCP servers has a failure mode that no RAG metric detects:
it asks the WRONG SERVER. The answer that comes back is fluent, cited, and
faithful to the passages it was handed -- because it is faithful; the passages
were just from the wrong place.

So the first thing to measure on a multi-MCP agent is tool-selection accuracy,
against labelled cases, with no judge. `KeywordRouter` below exists to give
that metric a BASELINE to beat: if your LLM router cannot beat a dozen
hand-written keywords, you are paying for latency and variance and getting
nothing. Measuring against a cheap baseline is ordinary engineering practice
that eval work routinely skips.
=============================================================================
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from mcp_client import ToolRegistry, text_of

SYSTEM_PROMPT = """You answer questions using the tools you are given.

Tool choice rules, in order:
1. Questions about retrieval, chunking, embeddings, evaluation concepts or this
   system's own documentation -> the corpus tools (internal knowledge base).
2. Questions about current events, weather, or anything outside the internal
   knowledge base -> the web tools.
3. Requests to CHECK or SCORE an answer that already exists -> the evaluator tools.

If the corpus search returns NO_MATCH, say exactly:
"The provided context does not contain this information."
Do not answer from your own knowledge. Cite passages as [doc_id]."""


def build_mcp_agent(registry: ToolRegistry, model: Any, **kwargs: Any):
    """Wire every tool in the registry into one prebuilt agent.

    Refuses to build on a colliding registry. That refusal is the point: an
    ambiguous tool name is a latent 50/50 bug, and the failure is far cheaper
    at startup than in production, where it presents as "the agent sometimes
    searches the web for internal questions" and reproduces one time in two.
    """
    if registry.collisions:
        raise ValueError(
            f"ambiguous tool names {registry.collisions} -- two MCP servers export the "
            f"same tool. Load with prefix=True, or the model's tool choice is "
            f"resolved by list order rather than by intent."
        )
    return create_agent(model, registry.tools, system_prompt=SYSTEM_PROMPT, **kwargs)


# ---------------------------------------------------------------------------
# A trajectory view over the agent's messages
# ---------------------------------------------------------------------------


@dataclass
class AgentRun:
    """What the agent did, in a shape you can assert on.

    Same argument as lesson 03 and lesson 07: the final string is not the unit
    of evaluation for an agent. WHICH tools it called, in WHAT order, and from
    WHICH server is the behaviour you actually care about.
    """

    messages: list[BaseMessage] = field(default_factory=list)
    registry: ToolRegistry | None = None

    @property
    def tool_calls(self) -> list[str]:
        names: list[str] = []
        for message in self.messages:
            if isinstance(message, AIMessage):
                names.extend(call["name"] for call in (message.tool_calls or []))
        return names

    @property
    def servers_used(self) -> list[str]:
        """The ORIGIN servers, deduplicated in first-use order.

        This is the line that answers "where did this answer come from?" during
        an incident, and it is only available because ToolRegistry kept the
        origin that get_tools() discards.
        """
        if self.registry is None:
            return []
        seen: list[str] = []
        for name in self.tool_calls:
            server = self.registry.origin.get(name, "unknown")
            if server not in seen:
                seen.append(server)
        return seen

    @property
    def tool_outputs(self) -> list[str]:
        return [text_of(m.content) for m in self.messages if isinstance(m, ToolMessage)]

    @property
    def answer(self) -> str:
        for message in reversed(self.messages):
            if isinstance(message, AIMessage) and message.content:
                return text_of(message.content)
        return ""

    @property
    def refused(self) -> bool:
        return "does not contain this information" in self.answer.lower()


async def arun_agent(
    agent: Any, question: str, registry: ToolRegistry | None = None
) -> AgentRun:
    """Invoke the agent and wrap the result in something assertable.

    =========================================================================
    TRAP 5 -- MCP TOOLS ARE ASYNC-ONLY. `agent.invoke()` RAISES.
    =========================================================================
    MEASURED, not recalled. Calling the synchronous `.invoke()` on an agent
    holding MCP tools fails inside the tools node with:

        NotImplementedError: StructuredTool does not support sync invocation.

    `convert_mcp_tool_to_langchain_tool` builds a StructuredTool with `coroutine`
    set and `func` left None, because the underlying MCP client session is
    async. There is no sync path, and there is no warning at bind time -- the
    agent constructs perfectly and dies on the first tool call.

    This matters more than it sounds: the overwhelming majority of LangChain
    examples, including ones you will copy, use `.invoke()`. Swap in MCP tools
    and the agent builds, binds, plans, and then explodes the moment it tries
    to use a tool -- so it fails in the integration test, not the unit test.

    Async all the way down is the answer. `run_agent` below is a convenience
    for scripts and tests; inside a running event loop (a FastAPI handler --
    see lesson 09) you must await `arun_agent` instead.
    """
    state = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    return AgentRun(messages=list(state["messages"]), registry=registry)


def run_agent(agent: Any, question: str, registry: ToolRegistry | None = None) -> AgentRun:
    """Synchronous convenience wrapper. NOT usable from inside an event loop.

    `asyncio.run` refuses to nest, so calling this from an async web handler
    raises "asyncio.run() cannot be called from a running event loop". That
    error is the correct one to get: it tells you to await `arun_agent`.
    """
    return asyncio.run(arun_agent(agent, question, registry))


# ---------------------------------------------------------------------------
# The baseline every LLM router should have to beat
# ---------------------------------------------------------------------------


@dataclass
class KeywordRouter:
    """Route a question to a tool by keyword. Zero cost, zero latency, zero variance.

    Rules are checked in order and the first hit wins, so the ORDER encodes
    precedence. That matters for the trap cases: "today's news about embeddings"
    contains an internal word, so the external rules must be checked first.

    Being explicit about precedence is the advantage a keyword router has over
    an LLM router, and it is why the baseline is often genuinely competitive on
    a narrow domain. Ship the LLM router only after it beats this.
    """

    rules: list[tuple[tuple[str, ...], str]]
    default: str

    def __call__(self, question: str) -> str:
        lowered = question.lower()
        for keywords, tool_name in self.rules:
            if any(keyword in lowered for keyword in keywords):
                return tool_name
        return self.default


def default_keyword_router() -> KeywordRouter:
    """The baseline router for the prefixed three-server registry."""
    return KeywordRouter(
        rules=[
            # Evaluator first: "check ... citation" also contains retrieval words.
            (("check", "valid", "score this", "recall@", "citation"), "evaluator_check_citations"),
            # External signals next, because they beat internal vocabulary.
            (("weather", "today", "news", "current", "latest"), "web_search"),
            (("list the documents", "what documents"), "corpus_list_documents"),
        ],
        default="corpus_search",
    )


__all__ = [
    "SYSTEM_PROMPT",
    "AgentRun",
    "KeywordRouter",
    "arun_agent",
    "build_mcp_agent",
    "default_keyword_router",
    "run_agent",
]
