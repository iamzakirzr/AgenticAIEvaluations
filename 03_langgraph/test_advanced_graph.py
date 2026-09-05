"""
Advanced LangGraph tests. FAST TIER -- no model needed.

Run:  pytest 03_langgraph/test_advanced_graph.py -v

Parallelism, reducers, subgraphs and streaming are all structural properties of
the graph, so all of them are testable with plain functions standing in for
model calls.
"""

from __future__ import annotations

import time

import pytest
from advanced_graph import (
    build_parallel_research_graph,
    build_pipeline_with_subgraph,
    build_reducer_demo,
    build_retrieval_subgraph,
    collect_stream,
    stream_updates,
)


def slow_searcher(name: str, delay: float = 0.15):
    def search(question: str) -> str:
        time.sleep(delay)
        return f"{name} found something about {question[:20]}"

    return search


def synthesise(question: str, findings: list[str]) -> str:
    return f"combined {len(findings)} findings"


# ===========================================================================
# PARALLELISM
# ===========================================================================


def test_parallel_branches_all_contribute():
    graph = build_parallel_research_graph(
        {"web": slow_searcher("web", 0), "docs": slow_searcher("docs", 0)},
        synthesise,
    )
    out = graph.invoke(
        {"question": "what is chunk overlap?", "findings": [], "timings": [], "answer": ""}
    )

    assert len(out["findings"]) == 2
    assert out["answer"] == "combined 2 findings"


def test_parallel_branches_actually_run_concurrently():
    """The measurement, not the claim.

    Three 150ms searches cost the MAX (~150ms) in parallel and the SUM (~450ms)
    sequentially. This is the single biggest latency win available in a RAG
    pipeline, and it is worth proving rather than assuming.
    """
    graph = build_parallel_research_graph(
        {
            "web": slow_searcher("web", 0.15),
            "docs": slow_searcher("docs", 0.15),
            "db": slow_searcher("db", 0.15),
        },
        synthesise,
    )

    started = time.perf_counter()
    graph.invoke(
        {"question": "q", "findings": [], "timings": [], "answer": ""}
    )
    elapsed = (time.perf_counter() - started) * 1000

    assert elapsed < 350, (
        f"took {elapsed:.0f}ms for 3x150ms searches -- they ran sequentially "
        f"(~450ms expected if so)"
    )


def test_fan_in_waits_for_every_branch():
    """The join must not run early. A synthesiser that fires before all
    branches report produces a partial answer that looks complete."""
    graph = build_parallel_research_graph(
        {f"s{i}": slow_searcher(f"s{i}", 0.02 * i) for i in range(4)},
        synthesise,
    )
    out = graph.invoke({"question": "q", "findings": [], "timings": [], "answer": ""})
    assert out["answer"] == "combined 4 findings"


def test_a_single_branch_still_works():
    graph = build_parallel_research_graph({"only": slow_searcher("only", 0)}, synthesise)
    out = graph.invoke({"question": "q", "findings": [], "timings": [], "answer": ""})
    assert len(out["findings"]) == 1


# ===========================================================================
# REDUCERS
# ===========================================================================


def test_a_reducer_merges_concurrent_writes():
    """`operator.add` concatenates, so three branches produce three entries."""
    out = build_reducer_demo(branches=3).invoke({"log": [], "tokens": 0, "status": ""})

    assert sorted(out["log"]) == ["branch-0", "branch-1", "branch-2"]
    assert out["tokens"] == 30, "the integer reducer did not sum across branches"


def test_concurrent_writes_to_an_unreduced_key_are_rejected():
    """THE error worth meeting in a test rather than in production.

    Two nodes in the same superstep writing the same key with NO reducer raises
    InvalidUpdateError. That is LangGraph REFUSING to silently pick a winner --
    a good error, and the same discipline as declaring a merge strategy for a
    shared resource instead of discovering a race condition later.
    """
    graph = build_reducer_demo(branches=3, write_status=True)

    with pytest.raises(Exception) as excinfo:
        graph.invoke({"log": [], "tokens": 0, "status": ""})

    message = str(excinfo.value)
    assert "status" in message or "one value per step" in message.lower(), (
        f"expected an InvalidUpdateError about concurrent writes, got: {message}"
    )


def test_a_single_branch_may_write_an_unreduced_key():
    """One writer is fine. The constraint is about CONCURRENT writes only."""
    out = build_reducer_demo(branches=1, write_status=True).invoke(
        {"log": [], "tokens": 0, "status": ""}
    )
    assert out["status"] == "done-0"


# ===========================================================================
# SUBGRAPHS
# ===========================================================================


def test_a_subgraph_runs_standalone():
    """The main argument for subgraphs: testable in isolation.

    Same reasoning as extracting a function -- you can assert on retrieval
    without constructing the whole pipeline around it.
    """
    subgraph = build_retrieval_subgraph(lambda q: ["chunk about " + q])
    out = subgraph.invoke({"question": "chunking", "chunks": []})
    assert out["chunks"] == ["chunk about chunking"]


def test_a_subgraph_composes_into_a_parent():
    pipeline = build_pipeline_with_subgraph(
        retrieve=lambda q: ["chunk one", "chunk two"],
        generate=lambda q, chunks: f"answer from {len(chunks)} chunks",
    )
    out = pipeline.invoke({"question": "q", "chunks": [], "answer": ""})

    assert out["chunks"] == ["chunk one", "chunk two"]
    assert out["answer"] == "answer from 2 chunks"


def test_the_subgraph_can_be_swapped_without_touching_the_parent():
    """Reuse, demonstrated: same parent, two retrieval strategies."""
    a = build_pipeline_with_subgraph(lambda q: ["a"], lambda q, c: f"{len(c)}")
    b = build_pipeline_with_subgraph(lambda q: ["a", "b", "c"], lambda q, c: f"{len(c)}")

    assert a.invoke({"question": "q", "chunks": [], "answer": ""})["answer"] == "1"
    assert b.invoke({"question": "q", "chunks": [], "answer": ""})["answer"] == "3"


# ===========================================================================
# STREAMING
# ===========================================================================


def test_streaming_emits_each_node_as_it_completes():
    """An agent that takes 40 seconds is nearly impossible to debug from its
    final output. Streaming tells you which node consumed the time, live."""
    pipeline = build_pipeline_with_subgraph(
        retrieve=lambda q: ["chunk"], generate=lambda q, c: "done"
    )
    names = collect_stream(pipeline, {"question": "q", "chunks": [], "answer": ""})

    assert "retrieval" in names
    assert "answer" in names
    assert names.index("retrieval") < names.index("answer"), "wrong execution order"


def test_stream_updates_carry_the_node_delta():
    pipeline = build_pipeline_with_subgraph(
        retrieve=lambda q: ["chunk one"], generate=lambda q, c: "final answer"
    )
    updates = dict(
        stream_updates(pipeline, {"question": "q", "chunks": [], "answer": ""})
    )

    assert updates["answer"]["answer"] == "final answer"


def test_streaming_a_parallel_graph_reports_every_branch():
    graph = build_parallel_research_graph(
        {"web": slow_searcher("web", 0), "docs": slow_searcher("docs", 0)}, synthesise
    )
    names = collect_stream(
        graph, {"question": "q", "findings": [], "timings": [], "answer": ""}
    )

    assert {"web", "docs", "join"} <= set(names)
    assert names[-1] == "join", "the fan-in should complete last"
