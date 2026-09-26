"""
Tests for lesson 08 -- one agent, several MCP servers.

=============================================================================
THESE TESTS SPAWN REAL MCP SERVERS
=============================================================================
No mocks. Each one starts an actual Python subprocess that speaks the MCP
protocol over stdin/stdout, and talks to it. That costs about a second per
connection, which is why the loaded registry is built ONCE and cached rather
than per test.

It is worth the second. Every trap this lesson teaches -- name collisions, the
TaskGroup failure mode, per-call process spawning, content-block returns -- is
invisible to a mock, because a mock is written from the same wrong mental model
that produced the bug.

Still no model, no GPU, no network and no API key, so it stays in the fast tier.

Run:  pytest 08_mcp -v
"""

from __future__ import annotations

import asyncio
import sys
from functools import lru_cache
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE), str(_ROOT / "03_langgraph")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp_agent import (
    AgentRun,
    arun_agent,
    build_mcp_agent,
    default_keyword_router,
    run_agent,
)
from mcp_client import (
    DEFAULT_SERVERS,
    TOOL_CHOICE_CASES,
    ToolRegistry,
    contract_diff,
    evaluate_tool_selection,
    load_tools_resiliently,
    stdio_server,
    text_of,
    tool_contract,
)
from measure_threshold import top_scores
from scripted_model import ScriptedToolCallingModel


@lru_cache(maxsize=2)
def _registry(prefix: bool) -> ToolRegistry:
    """Load all three servers once per prefix mode, then reuse.

    lru_cache rather than a pytest fixture so it survives across test classes
    and stays usable from a plain script. Three subprocess handshakes is about
    a second; paying it per test would add half a minute to the fast tier and
    the fast tier's whole value is that nobody is tempted to switch it off.
    """
    return ToolRegistry.from_loads(asyncio.run(load_tools_resiliently(prefix=prefix)))


# ---------------------------------------------------------------------------
# The protocol actually works
# ---------------------------------------------------------------------------


def test_all_three_servers_start_and_expose_tools():
    registry = _registry(True)
    assert registry.degraded == []
    assert registry.by_server() == {
        "corpus": ["corpus_list_documents", "corpus_search", "corpus_stats"],
        "evaluator": [
            "evaluator_check_citations",
            "evaluator_retrieval_mrr",
            "evaluator_retrieval_recall",
        ],
        "web": ["web_fetch_url", "web_search"],
    }


def test_a_real_tool_call_crosses_a_process_boundary():
    """Real retrieval, in another process, over JSON-RPC, with no model."""
    tool = _registry(True).get("corpus_search")
    result = text_of(asyncio.run(tool.ainvoke({"query": "chunk overlap boundary", "k": 2})))
    assert "chunking#" in result
    assert "score" in result


def test_the_server_flags_low_confidence_rather_than_pretending_to_refuse():
    """Nonsense still returns passages -- flagged, not suppressed.

    The tempting design is NO_MATCH below a threshold. We measured whether a
    threshold exists (see the next test) and it does not, so suppressing would
    be a refusal the evidence cannot support.
    """
    tool = _registry(True).get("corpus_search")
    result = text_of(asyncio.run(tool.ainvoke({"query": "zzzz qqqq xxxx", "k": 2})))
    assert result.startswith("LOW_CONFIDENCE")


def test_no_similarity_threshold_separates_real_questions_from_gibberish():
    """MEASURED: 42 golden questions vs 200 random strings.

        real       min 0.208   median 0.313
        gibberish  median 0.190   p95 0.290   MAX 0.433

    The gibberish maximum EXCEEDS the real minimum, so the distributions
    overlap and no cut-off separates them. At 0.25 you reject 16.7% of genuine
    questions and still accept 10% of gibberish.

    This test exists to stop a future contributor "fixing" the LOW_CONFIDENCE
    hint into a hard gate. If a better embedder ever makes the distributions
    separate, this fails -- and that failure is the signal that a real
    threshold has become defensible.
    """
    real, noise = top_scores(samples=200)
    assert max(noise) > min(real), "distributions now separate -- revisit the design"

    rejected = sum(1 for value in real if value < 0.25) / len(real)
    accepted = sum(1 for value in noise if value >= 0.25) / len(noise)
    assert rejected > 0.10  # a threshold that cheap already refuses real users
    assert accepted > 0.05  # and still lets nonsense through


# ---------------------------------------------------------------------------
# TRAP 4 -- the return shape
# ---------------------------------------------------------------------------


def test_an_mcp_tool_returns_content_blocks_not_a_value():
    """MEASURED, not recalled: a native tool returning 5 gives 5; over MCP you
    get [{'type': 'text', 'text': '5', 'id': 'lc_...'}].

    Every assertion written against the native tool breaks on the swap, and it
    breaks as a confusing type error far from the tool call.
    """
    tool = _registry(True).get("evaluator_check_citations")
    raw = asyncio.run(tool.ainvoke({"answer": "grounded [1] and [9]", "passage_count": 3}))

    assert isinstance(raw, list)
    assert raw[0]["type"] == "text"
    assert text_of(raw) == "INVALID_CITATIONS: [9]"


def test_text_of_is_total_over_the_shapes_you_actually_meet():
    assert text_of("plain") == "plain"
    assert text_of([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a\nb"
    assert text_of([]) == ""


# ---------------------------------------------------------------------------
# TRAP 2 -- colliding names
# ---------------------------------------------------------------------------


def test_colliding_tool_names_are_silently_ambiguous():
    """Two servers, one name, no error. This is the bug this lesson is for.

    `corpus` and `web` both export `search`. Without prefixing the registry
    holds two tools called `search`, the model emits {"name": "search"}, and
    which server runs is decided by list order.
    """
    registry = _registry(False)
    assert registry.names.count("search") == 2
    assert registry.collisions == ["search"]
    # And the origin map -- a plain dict keyed by name -- has already lost one
    # of them. That is what makes the failure unattributable after the fact.
    assert len([n for n in registry.names if n == "search"]) > len(
        {n for n in registry.names if n == "search"}
    )


def test_prefixing_resolves_the_collision():
    registry = _registry(True)
    assert registry.collisions == []
    assert "corpus_search" in registry.names
    assert "web_search" in registry.names


def test_building_an_agent_on_a_colliding_registry_is_refused():
    """Fail at startup, loudly, rather than 50/50 in production."""
    with pytest.raises(ValueError, match="ambiguous tool names"):
        build_mcp_agent(_registry(False), model=ScriptedToolCallingModel(script=[]))


# ---------------------------------------------------------------------------
# TRAP 1 -- one dead server must not kill the others
# ---------------------------------------------------------------------------


def test_one_dead_server_does_not_take_the_others_down():
    """The whole reason load_tools_resiliently exists.

    The naive single-client version raises an ExceptionGroup here and returns
    NO tools at all -- see the next test, which pins that behaviour so the
    contrast is not just a claim in a comment.
    """
    servers = {
        "corpus": DEFAULT_SERVERS["corpus"],
        "ghost": stdio_server("does_not_exist.py"),
    }
    loads = asyncio.run(load_tools_resiliently(servers))
    registry = ToolRegistry.from_loads(loads)

    assert registry.degraded == ["ghost"]
    assert "corpus_search" in registry.names  # the healthy server still works

    failed = next(load for load in loads if load.name == "ghost")
    # Unwrapped: the raw ExceptionGroup str is "unhandled errors in a TaskGroup
    # (1 sub-exception)", which tells an operator nothing at all.
    assert "TaskGroup" not in failed.error
    assert failed.error


def test_the_naive_single_client_really_does_lose_everything():
    """Pinning the behaviour the fix exists for. If this ever stops raising,
    the fix is no longer needed and this lesson should be rewritten."""
    servers = {
        "corpus": DEFAULT_SERVERS["corpus"],
        "ghost": stdio_server("does_not_exist.py"),
    }
    client = MultiServerMCPClient(servers, tool_name_prefix=True)
    # BaseException on purpose, and it is the finding: asyncio raises an
    # ExceptionGroup, which is NOT an Exception subclass, so the `except
    # Exception` most people write around get_tools() does not catch this at
    # all. Narrowing this assertion would hide exactly what the test is for.
    with pytest.raises(BaseException):  # noqa: B017
        asyncio.run(client.get_tools())


# ---------------------------------------------------------------------------
# TRAP 3 -- sessions and server-side state
# ---------------------------------------------------------------------------


def test_a_sessionless_tool_loses_server_state_between_calls():
    """MEASURED: two calls, two processes, two different pids, counter reset.

    Anything the server holds in memory -- a cursor, a login, a cache, a
    transaction -- silently vanishes between tool calls. An agent built on a
    stateful MCP server without a session is subtly, intermittently wrong, and
    nothing in any log says so.
    """
    connection = stdio_server("stateful_probe_mcp.py")

    async def sessionless() -> list[str]:
        client = MultiServerMCPClient({"probe": connection})
        tool = (await client.get_tools())[0]
        return [text_of(await tool.ainvoke({})) for _ in range(2)]

    async def with_session() -> list[str]:
        client = MultiServerMCPClient({"probe": connection})
        async with client.session("probe") as session:
            tool = (await load_mcp_tools(session))[0]
            return [text_of(await tool.ainvoke({})) for _ in range(2)]

    a, b = asyncio.run(sessionless())
    assert a.split()[1] == "count=1"
    assert b.split()[1] == "count=1"  # state LOST
    assert a.split()[0] != b.split()[0]  # different pid: a whole new process

    c, d = asyncio.run(with_session())
    assert c.split()[1] == "count=1"
    assert d.split()[1] == "count=2"  # state KEPT
    assert c.split()[0] == d.split()[0]  # same pid


# ---------------------------------------------------------------------------
# Contract snapshotting -- your behaviour lives in someone else's repo
# ---------------------------------------------------------------------------


def test_tool_contract_captures_what_the_model_routes_on():
    contract = tool_contract(_registry(True).tools)["corpus_search"]
    assert contract["required"] == ["query"]
    assert contract["parameters"] == {"k": "integer", "query": "string"}
    # The description is in the contract because the description IS the prompt.
    assert "internal knowledge base" in contract["description"]


def test_contract_diff_ranks_a_new_required_argument_as_breaking():
    before = tool_contract(_registry(True).tools)
    after = {k: dict(v) for k, v in before.items()}
    after["corpus_search"]["required"] = ["query", "k"]
    after.pop("web_fetch_url")
    after["web_newtool"] = {"description": "x", "required": [], "parameters": {}}

    findings = contract_diff(before, after)
    assert "REMOVED tool web_fetch_url" in findings
    assert "ADDED tool web_newtool" in findings
    assert any(f.startswith("BREAKING corpus_search") for f in findings)


def test_a_reworded_description_is_reported_as_a_behaviour_change():
    """Not cosmetic: the description is what the model routes on, so rewording
    it changes tool selection with no schema diff and no code diff."""
    before = tool_contract(_registry(True).tools)
    after = {k: dict(v) for k, v in before.items()}
    after["corpus_search"]["description"] = "Search stuff."
    assert any("REWORDED corpus_search" in f for f in contract_diff(before, after))


# ---------------------------------------------------------------------------
# The metric: tool-selection accuracy
# ---------------------------------------------------------------------------


def test_the_keyword_baseline_is_measured_not_assumed():
    """MEASURED on the 8 labelled cases: 75%, missing two paraphrases.

    Reported as a number rather than a claim, because that number is the bar an
    LLM router has to clear to be worth its latency and variance. Ship the LLM
    router only when it beats 75% on this set.
    """
    report = evaluate_tool_selection(default_keyword_router(), _registry(True))
    assert report.total == 8
    assert report.accuracy == pytest.approx(0.75)
    misrouted = {question for question, _, _ in report.mistakes}
    assert "is it raining in London right now?" in misrouted


def test_the_routing_set_has_discriminating_power():
    """A guard against the set drifting back to saturated.

    The first version of TOOL_CHOICE_CASES had five items and the cheapest
    possible baseline scored 5/5. A set your baseline aces cannot rank two
    routers, which is the same failure recall@k has on this corpus. If someone
    later deletes the hard cases, this fails.
    """
    report = evaluate_tool_selection(default_keyword_router(), _registry(True))
    assert 0.0 < report.accuracy < 1.0, "baseline is saturated -- add harder cases"


def test_every_routing_case_names_a_server_that_exists():
    """Dataset integrity. A typo'd expected_server makes the metric unpassable
    and looks exactly like a routing bug."""
    servers = set(_registry(True).origin.values())
    for case in TOOL_CHOICE_CASES:
        assert case.expected_server in servers, case


# ---------------------------------------------------------------------------
# The agent itself
# ---------------------------------------------------------------------------


def test_the_agent_calls_a_real_mcp_tool_and_records_its_origin():
    """A scripted model drives the loop; the TOOL CALL is real, over MCP."""
    registry = _registry(True)
    model = ScriptedToolCallingModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "corpus_search",
                        "args": {"query": "chunk overlap", "k": 2},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Overlap protects boundary facts [chunking#4]."),
        ]
    )
    run = asyncio.run(
        arun_agent(build_mcp_agent(registry, model), "what is chunk overlap?", registry)
    )

    assert run.tool_calls == ["corpus_search"]
    assert run.servers_used == ["corpus"]  # attributable after the fact
    assert "chunking#" in run.tool_outputs[0]
    assert not run.refused


def test_the_agent_can_use_two_servers_in_one_trajectory():
    """Retrieve from corpus, then check its own citations with the evaluator.

    The self-check is a CONTROL SIGNAL, not a measurement -- see
    servers/evaluator_mcp.py. It changes the agent's behaviour; it is not
    evidence about the agent's quality.
    """
    registry = _registry(True)
    model = ScriptedToolCallingModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "corpus_search", "args": {"query": "chunking"}, "id": "c1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "evaluator_check_citations",
                        "args": {"answer": "see [1]", "passage_count": 3},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content="Chunking splits documents [1]."),
        ]
    )
    run = asyncio.run(arun_agent(build_mcp_agent(registry, model), "explain chunking", registry))

    assert run.tool_calls == ["corpus_search", "evaluator_check_citations"]
    assert run.servers_used == ["corpus", "evaluator"]
    assert run.tool_outputs[1] == "OK"


def test_the_agent_degrades_rather_than_crashing_when_a_server_is_down():
    """With `web` unavailable the agent must still start on the remaining tools.

    An agent that refuses to boot because one optional server is down converts
    a degraded experience into an outage. Record the degradation, keep serving.
    """
    loads = asyncio.run(
        load_tools_resiliently(
            {"corpus": DEFAULT_SERVERS["corpus"], "web": stdio_server("does_not_exist.py")}
        )
    )
    registry = ToolRegistry.from_loads(loads)
    model = ScriptedToolCallingModel(script=[AIMessage(content="Answered without the web.")])

    run = asyncio.run(arun_agent(build_mcp_agent(registry, model), "anything", registry))

    assert registry.degraded == ["web"]
    assert run.answer == "Answered without the web."
    assert model.bound_tools and all(not n.startswith("web_") for n in model.bound_tools)


def test_agent_run_reads_an_empty_trajectory_without_crashing():
    """Defensive: the properties are read during incidents, on odd states."""
    empty = AgentRun()
    assert empty.tool_calls == []
    assert empty.servers_used == []
    assert empty.answer == ""
    assert not empty.refused


def test_sync_invoke_on_an_mcp_agent_raises():
    """TRAP 5, pinned. The majority of LangChain examples use `.invoke()`.

    Swap in MCP tools and the agent builds, binds and plans fine, then dies on
    the first tool call -- so it passes every unit test and fails in
    integration. Async all the way down, or you meet this in staging.
    """
    registry = _registry(True)
    model = ScriptedToolCallingModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[{"name": "corpus_search", "args": {"query": "x"}, "id": "c1"}],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = build_mcp_agent(registry, model)
    with pytest.raises(NotImplementedError, match="sync invocation"):
        agent.invoke({"messages": [HumanMessage(content="x")]})


def test_run_agent_sync_wrapper_works_outside_an_event_loop():
    """The convenience wrapper is real, not decorative -- but see its docstring
    for why a FastAPI handler must await `arun_agent` instead."""
    registry = _registry(True)
    model = ScriptedToolCallingModel(script=[AIMessage(content="plain answer")])
    assert run_agent(build_mcp_agent(registry, model), "hi", registry).answer == "plain answer"
