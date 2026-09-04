"""
The agent: a LangGraph StateGraph, built by hand so the loop is visible.

=============================================================================
WHY BUILD THE GRAPH INSTEAD OF CALLING create_react_agent
=============================================================================
LangGraph ships `create_react_agent(model, tools)`, which produces exactly this
graph in one line. Use it in production. Do not use it to learn, because the
thing you need to understand is the LOOP, and the one-liner hides it.

The graph is three parts:

        START -> [agent] --(has tool calls?)--> [tools] --+
                    ^                                     |
                    +-------------------------------------+
                    |
                    +--(no tool calls)--> END

  agent node    calls the model with the conversation so far
  tools node    executes whatever tools the model asked for, appends results
  conditional   the ONLY thing deciding whether to loop again is: did the last
                message contain tool calls?

That last point is the whole mechanism. There is no planner and no controller.
An "agent" is a while-loop whose exit condition is a model's choice, which is
why agents fail in ways pipelines cannot: they can loop forever, call tools in
a nonsensical order, or stop early.

=============================================================================
WHY THIS RETURNS AN AgentTrace, NOT A STRING
=============================================================================
Same argument as core/trace.py, one level up. A RAG pipeline needs its chunks
preserved; an agent needs its TRAJECTORY preserved -- every tool call, its
arguments, its result, and the order.

Without the trajectory you can only ask "was the final answer right?", which
core/corpus/agent_metrics.md explains is not enough: an agent that reached the
right answer after nine wasted calls and one wrong turn looks identical to one
that went straight there. Tool Correctness, Step Efficiency and Loop Detection
all read the trajectory, not the answer.
=============================================================================
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from tools import ALL_TOOLS, TOOLS_BY_NAME  # noqa: E402

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """What flows between nodes.

    `Annotated[list, add_messages]` is the important part. It is a REDUCER: it
    tells LangGraph how to merge a node's output into the existing state.
    Without it, each node's return value would REPLACE the message list and the
    agent would forget everything on every step.

    add_messages appends, and also de-duplicates by message id -- which is what
    makes resuming from a checkpoint safe.
    """

    messages: Annotated[list[BaseMessage], add_messages]

    # Guard against infinite loops. Not decoration: an agent whose model keeps
    # emitting the same tool call will otherwise run until something else
    # kills it, which in production looks like a timeout rather than a bug.
    steps: int


# ---------------------------------------------------------------------------
# TRAJECTORY
# ---------------------------------------------------------------------------


@dataclass
class ToolInvocation:
    """One tool call, recorded for evaluation."""

    name: str
    args: dict[str, Any]
    result: str = ""

    def __str__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v!r}' for k, v in self.args.items())})"


@dataclass
class AgentTrace:
    """Everything the agent did, in order. The unit of agent evaluation."""

    question: str
    answer: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)
    steps: int = 0
    hit_step_limit: bool = False
    elapsed_ms: float = 0.0

    @property
    def tool_names(self) -> list[str]:
        """Tool names in call order -- what Tool Correctness compares against."""
        return [call.name for call in self.tool_calls]

    @property
    def refused(self) -> bool:
        """True if the agent explicitly used the refuse tool.

        Making refusal a TOOL rather than free text is what turns "did it
        refuse?" from an LLM-judged question into a boolean. Prefer designs
        that make the behaviour you care about deterministically observable.
        """
        return "refuse" in self.tool_names

    def repeated_calls(self) -> list[str]:
        """Tool calls made more than once with identical arguments.

        This is LOOP DETECTION, computed with no LLM. An agent repeating the
        exact same call is not making progress, and per
        core/corpus/agent_metrics.md this failure is invisible to outcome-only
        evaluation because the agent usually times out rather than answering
        wrongly.
        """
        seen: set[str] = set()
        repeats: list[str] = []
        for call in self.tool_calls:
            signature = str(call)
            if signature in seen and signature not in repeats:
                repeats.append(signature)
            seen.add(signature)
        return repeats

    def summary(self) -> str:
        lines = [f"Q: {self.question}", f"A: {self.answer[:200]}"]
        lines.append(f"steps: {self.steps}" + ("  (HIT LIMIT)" if self.hit_step_limit else ""))
        lines.append("trajectory:")
        for i, call in enumerate(self.tool_calls, start=1):
            lines.append(f"  {i}. {call}  -> {call.result[:70]!r}")
        if self.repeated_calls():
            lines.append(f"  LOOP DETECTED: {self.repeated_calls()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# THE GRAPH
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a research assistant for a knowledge base about \
retrieval-augmented generation and LLM evaluation.

- Use search_knowledge_base to find facts before answering.
- Answer only from what the tools return, never from your own knowledge.
- If searching finds nothing relevant, call refuse rather than guessing.
- Cite passages by their number, like [1].
- Do not call the same tool twice with the same arguments."""

MAX_STEPS = 6


class ResearchAgent:
    """A tool-using agent over the knowledge base."""

    def __init__(self, llm: BaseChatModel, tools=None, max_steps: int = MAX_STEPS) -> None:
        self.tools = tools if tools is not None else ALL_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        # bind_tools attaches the tools' JSON schemas to every model request.
        self.llm = llm.bind_tools(self.tools)
        self.max_steps = max_steps
        self.graph = self._build()

    # ---- nodes ------------------------------------------------------------

    def _agent_node(self, state: AgentState) -> dict:
        """Ask the model what to do next, given everything so far."""
        response = self.llm.invoke(state["messages"])
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def _tools_node(self, state: AgentState) -> dict:
        """Execute every tool the model just asked for."""
        last = state["messages"][-1]
        outputs: list[BaseMessage] = []

        for call in getattr(last, "tool_calls", []) or []:
            tool = self.tools_by_name.get(call["name"])
            if tool is None:
                # An unknown tool name is a REAL agent failure (models
                # hallucinate tool names). Feed the error back as a
                # ToolMessage so the agent can recover, rather than crashing
                # the graph -- and so the failure appears in the trajectory.
                result = f"ERROR: unknown tool {call['name']!r}"
            else:
                try:
                    result = str(tool.invoke(call["args"]))
                except Exception as exc:
                    result = f"ERROR: {type(exc).__name__}: {exc}"

            outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))

        return {"messages": outputs}

    # ---- the conditional edge --------------------------------------------

    def _should_continue(self, state: AgentState) -> str:
        """The entire control logic of the agent, in five lines.

        Loop if the model asked for tools AND we have budget left. Otherwise
        stop. There is nothing else -- no planner, no supervisor.
        """
        if state.get("steps", 0) >= self.max_steps:
            return "end"  # circuit breaker
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)

        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent",
            self._should_continue,
            {"tools": "tools", "end": END},
        )
        # After running tools we ALWAYS return to the agent so it can read the
        # results. This edge is what makes it a loop rather than a pipeline.
        graph.add_edge("tools", "agent")

        return graph.compile()

    # ---- running ----------------------------------------------------------

    def run(self, question: str) -> AgentTrace:
        """Answer a question and return the full trajectory."""
        started = time.perf_counter()

        initial: AgentState = {
            "messages": [
                ("system", SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
            "steps": 0,
        }
        # recursion_limit is LangGraph's own safety net, separate from our
        # steps counter. Ours produces a clean trace; this one raises. Belt
        # and braces, because an agent that never terminates is the failure
        # mode that takes production down.
        final = self.graph.invoke(initial, config={"recursion_limit": self.max_steps * 3})

        return self._to_trace(question, final, (time.perf_counter() - started) * 1000)

    def _to_trace(self, question: str, state: dict, elapsed_ms: float) -> AgentTrace:
        """Reconstruct the trajectory from the message history.

        Tool calls live on AIMessages and their results arrive later as
        ToolMessages, matched by tool_call_id. Stitching them back together is
        what turns a flat message list into an evaluable trajectory.
        """
        messages = state["messages"]

        results_by_id = {
            m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)
        }

        calls: list[ToolInvocation] = []
        for message in messages:
            for call in getattr(message, "tool_calls", []) or []:
                calls.append(
                    ToolInvocation(
                        name=call["name"],
                        args=call.get("args", {}),
                        result=str(results_by_id.get(call["id"], "")),
                    )
                )

        # The answer is the last AIMessage that has text and no tool calls.
        answer = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                answer = message.content
                break

        steps = state.get("steps", 0)
        return AgentTrace(
            question=question,
            answer=answer,
            tool_calls=calls,
            messages=messages,
            steps=steps,
            hit_step_limit=steps >= self.max_steps,
            elapsed_ms=elapsed_ms,
        )


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_offline_agent(script: list[AIMessage], max_steps: int = MAX_STEPS) -> ResearchAgent:
    """A fully deterministic agent driven by a scripted model.

    Real tools, real graph, real trajectory recording -- only the model's
    decisions are pre-written. Enough to test routing, step limits, loop
    detection, error recovery and trace reconstruction with no LLM.
    """
    from scripted_model import ScriptedToolCallingModel

    return ResearchAgent(ScriptedToolCallingModel(script=script), max_steps=max_steps)


def build_ollama_agent(max_steps: int = MAX_STEPS) -> ResearchAgent:
    """The real agent, with a local model choosing its own actions."""
    from core.providers import get_chat_model

    return ResearchAgent(get_chat_model(), max_steps=max_steps)


def bind_default_retriever() -> None:
    """Wire the search tool to a lesson-02 pipeline over the shared corpus."""
    sys.path.insert(0, str(_ROOT / "02_langchain"))
    from pipeline import RagPipeline
    from tools import bind_retriever

    from core.providers import LexicalEmbeddings

    bind_retriever(RagPipeline(embeddings=LexicalEmbeddings(dim=2048)).ingest())
