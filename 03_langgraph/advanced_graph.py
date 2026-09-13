"""
ADVANCED LANGGRAPH: parallelism, reducers, subgraphs and streaming.

=============================================================================
WHAT THIS ADDS OVER THE LESSON-03 AGENT
=============================================================================
Lesson 03 built one loop: agent -> tools -> agent. That is the shape most
agents have, and it is sequential.

Four further capabilities come up constantly in real systems and in interviews:

  1. PARALLEL FAN-OUT / FAN-IN   run independent steps at the same time and
                                 merge the results. The single biggest latency
                                 win available in a RAG pipeline.
  2. REDUCERS                    how concurrent writes to the same state key
                                 are merged. Get this wrong and parallel
                                 branches silently overwrite each other.
  3. SUBGRAPHS                   a whole graph used as one node. Composition
                                 and reuse -- and independently testable.
  4. STREAMING                   emit progress as it happens instead of after.
                                 Perceived-latency win; also the only sane way
                                 to debug a slow agent.

=============================================================================
THE ONE THAT BITES: CONCURRENT WRITES NEED A REDUCER
=============================================================================
In LangGraph, nodes that run in the same superstep both return updates to the
state. If two of them write the same key and that key has NO reducer, you get

    InvalidUpdateError: At key 'x': Can receive only one value per step.

This is a GOOD error -- it is LangGraph refusing to silently pick a winner. The
fix is to declare how the merge should work:

    results: Annotated[list[str], operator.add]     # concatenate
    messages: Annotated[list, add_messages]         # append + dedupe by id

Anyone who has debugged a race condition will recognise the pattern
immediately: the framework is making you state your merge strategy up front
instead of discovering it in production.
=============================================================================
"""

from __future__ import annotations

import operator
import sys
import time
from pathlib import Path
from typing import Annotated, Any, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.graph import END, START, StateGraph

# ===========================================================================
# 1. PARALLEL FAN-OUT / FAN-IN
# ===========================================================================


class ResearchState(TypedDict):
    """State for a fan-out research graph.

    `findings` carries the `operator.add` reducer, which is what makes
    concurrent writes legal: each branch returns a one-element list and
    LangGraph concatenates them. Without the annotation the parallel branches
    raise InvalidUpdateError rather than silently clobbering one another.
    """

    question: str
    findings: Annotated[list[str], operator.add]
    timings: Annotated[list[float], operator.add]
    answer: str


def build_parallel_research_graph(searchers: dict[str, Any], synthesise) -> Any:
    """Fan out to several independent searchers, then fan in to synthesise.

        START -> plan -> ┬─ searcher_a ─┐
                         ├─ searcher_b ─┼─> synthesise -> END
                         └─ searcher_c ─┘

    WHY THIS IS THE BIGGEST LATENCY WIN IN A RAG PIPELINE: querying three
    indexes sequentially costs the SUM of their latencies; in parallel it costs
    the MAX. With three 400ms searches that is 1200ms versus 400ms, for no
    change in quality.

    LangGraph runs nodes in "supersteps": every node reachable at the same time
    executes together, and their state updates are merged by the reducers
    before the next superstep begins.
    """
    graph = StateGraph(ResearchState)

    def plan(state: ResearchState) -> dict:
        return {}  # a real planner would decide WHICH searchers to run

    graph.add_node("plan", plan)

    for name, search_fn in searchers.items():
        def make_node(fn=search_fn, label=name):
            def node(state: ResearchState) -> dict:
                started = time.perf_counter()
                finding = fn(state["question"])
                # Return a LIST of one. The reducer concatenates; returning a
                # bare string would replace, which is the bug this shape avoids.
                return {
                    "findings": [f"[{label}] {finding}"],
                    "timings": [(time.perf_counter() - started) * 1000],
                }

            return node

        graph.add_node(name, make_node())
        graph.add_edge("plan", name)   # fan out
        graph.add_edge(name, "join")   # fan in

    def join(state: ResearchState) -> dict:
        return {"answer": synthesise(state["question"], state["findings"])}

    graph.add_node("join", join)
    graph.add_edge(START, "plan")
    graph.add_edge("join", END)

    return graph.compile()


# ===========================================================================
# 2. REDUCERS
# ===========================================================================


class CounterState(TypedDict):
    """Demonstrates three merge strategies side by side."""

    # Concatenate. The usual choice for accumulating results.
    log: Annotated[list[str], operator.add]
    # Sum. Useful for token counts and costs across parallel branches.
    tokens: Annotated[int, operator.add]
    # NO reducer: last-write-wins, and concurrent writes are an ERROR.
    status: str


def build_reducer_demo(branches: int = 3, write_status: bool = False) -> Any:
    """A graph whose branches all write the same keys.

    With `write_status=True` the branches also write the un-reduced `status`
    key, which raises InvalidUpdateError -- deliberately, so the failure is
    visible in a test rather than discovered in production.
    """
    graph = StateGraph(CounterState)

    def start(state: CounterState) -> dict:
        return {}

    graph.add_node("start", start)

    for i in range(branches):
        def make(index=i):
            def node(state: CounterState) -> dict:
                update: dict[str, Any] = {"log": [f"branch-{index}"], "tokens": 10}
                if write_status:
                    update["status"] = f"done-{index}"
                return update

            return node

        graph.add_node(f"branch-{i}", make())
        graph.add_edge("start", f"branch-{i}")
        graph.add_edge(f"branch-{i}", END)

    graph.add_edge(START, "start")
    return graph.compile()


# ===========================================================================
# 3. SUBGRAPHS
# ===========================================================================


class RetrievalState(TypedDict):
    question: str
    chunks: Annotated[list[str], operator.add]


class PipelineState(TypedDict):
    question: str
    chunks: Annotated[list[str], operator.add]
    answer: str


def build_retrieval_subgraph(retrieve) -> Any:
    """A self-contained retrieval graph, usable as a single node elsewhere.

    WHY SUBGRAPHS EARN THEIR KEEP:

      - TESTABLE IN ISOLATION. You can assert on retrieval behaviour without
        constructing the whole agent, which is the same argument as extracting
        a function.
      - REUSABLE. The same retrieval subgraph serves the chatbot, the batch
        job and the evaluation harness.
      - BOUNDED. The parent only sees the keys they share, so a subgraph cannot
        accidentally depend on parent state that happens to be lying around.

    The catch: the parent and child must agree on shared state keys, INCLUDING
    their reducers. A child appending to `chunks` while the parent replaces it
    produces results that depend on execution order.
    """
    graph = StateGraph(RetrievalState)

    def expand(state: RetrievalState) -> dict:
        return {}

    def search(state: RetrievalState) -> dict:
        return {"chunks": list(retrieve(state["question"]))}

    graph.add_node("expand", expand)
    graph.add_node("search", search)
    graph.add_edge(START, "expand")
    graph.add_edge("expand", "search")
    graph.add_edge("search", END)
    return graph.compile()


def build_pipeline_with_subgraph(retrieve, generate) -> Any:
    """Parent graph that uses the retrieval subgraph as one node."""
    subgraph = build_retrieval_subgraph(retrieve)

    graph = StateGraph(PipelineState)
    # A compiled graph is a Runnable, so it drops straight in as a node.
    graph.add_node("retrieval", subgraph)

    def answer(state: PipelineState) -> dict:
        return {"answer": generate(state["question"], state["chunks"])}

    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", END)
    return graph.compile()


# ===========================================================================
# 4. STREAMING
# ===========================================================================


def stream_updates(graph, initial_state: dict, config: dict | None = None):
    """Yield ``(node_name, update)`` as each node finishes.

    `stream_mode="updates"` emits only what each node CHANGED, which is what
    you want for progress reporting. The alternatives:

        "values"    the whole state after each step -- verbose, but the right
                    choice when debugging a state-shape problem
        "updates"   just the delta per node        <- used here
        "messages"  token-by-token LLM output, for typing-indicator UX
        "custom"    whatever your nodes emit via a writer

    WHY THIS MATTERS BEYOND UX: an agent that takes 40 seconds is nearly
    impossible to debug from its final output alone. Streaming updates tells
    you WHICH node consumed the time and what it produced, live -- which is the
    same reason lesson 06 puts a span around every tool call.
    """
    for chunk in graph.stream(initial_state, config=config, stream_mode="updates"):
        # Each chunk is {node_name: update}; a superstep with parallel branches
        # yields one entry per branch.
        yield from chunk.items()


def collect_stream(graph, initial_state: dict, config: dict | None = None) -> list[str]:
    """Node names in completion order. Handy for asserting execution order."""
    return [name for name, _ in stream_updates(graph, initial_state, config)]
