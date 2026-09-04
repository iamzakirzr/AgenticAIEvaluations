"""
Lesson 03 tests -- FAST TIER unless marked.

Run:  pytest 03_langgraph/ -v
      pytest -m ollama 03_langgraph/ -v

=============================================================================
THE IDEA THIS FILE EXISTS TO PROVE
=============================================================================
Agent failures are TRAJECTORY failures, and trajectories are testable without
a language model.

Every pathology below -- infinite loops, repeated calls, hallucinated tool
names, bad arguments, stopping early -- is reproduced here with a scripted
model and asserted deterministically. No GPU, no flakiness, milliseconds.

That matters because these are the failures that actually take agents down in
production, and an LLM-judged test is the worst possible way to catch them:
slow, expensive, and non-deterministic about a thing that is perfectly
deterministic.
=============================================================================
"""

from __future__ import annotations

import pytest
from agent import ResearchAgent, build_offline_agent, bind_default_retriever
from langchain_core.messages import AIMessage, ToolMessage
from scripted_model import ScriptedToolCallingModel, final_answer, tool_call
from tools import ALL_TOOLS, list_documents, reciprocal_rank, refuse, search_knowledge_base


@pytest.fixture(scope="module", autouse=True)
def _retriever():
    """Point the search tool at a real indexed corpus, once per module."""
    bind_default_retriever()


# ===========================================================================
# THE TOOLS THEMSELVES
# ===========================================================================


def test_tool_schema_comes_from_the_docstring_and_annotations():
    """What the model actually sees is a schema derived from your code.

    The docstring is not documentation -- it is the prompt that decides whether
    the model picks this tool. Asserting it is non-empty stops someone deleting
    it during a tidy-up and silently degrading tool selection.
    """
    assert search_knowledge_base.name == "search_knowledge_base"
    assert search_knowledge_base.description, "no description -- the model is choosing blind"
    assert "query" in search_knowledge_base.args


def test_search_tool_returns_numbered_passages_with_provenance():
    result = search_knowledge_base.invoke({"query": "chunk overlap"})
    assert "[1]" in result
    assert ".md" in result, "no source document -- citations become unverifiable"


def test_reciprocal_rank_tool_computes_correctly():
    assert "0.2500" in reciprocal_rank.invoke({"position": 4})
    assert "1.0000" in reciprocal_rank.invoke({"position": 1})


def test_tool_returns_a_readable_error_instead_of_raising():
    """A raised exception kills the graph. A returned error lets the agent recover.

    Whether the agent ACTUALLY recovers is itself a behaviour worth measuring --
    see test_agent_can_recover_from_a_bad_argument below.
    """
    result = reciprocal_rank.invoke({"position": 0})
    assert result.startswith("ERROR")
    assert "Retry" in result


def test_refuse_is_a_tool_so_refusal_is_observable():
    """Making refusal a tool call turns 'did it refuse?' into a boolean.

    If refusal were free text you would need a judge to detect it. Design for
    deterministic observability whenever you can.
    """
    assert refuse.invoke({"reason": "not in the corpus"}).startswith("REFUSED")


# ===========================================================================
# THE SCRIPTED MODEL
# ===========================================================================


def test_bind_tools_records_what_was_bound():
    """bind_tools attaches schemas to requests; it does not mutate the model."""
    model = ScriptedToolCallingModel(script=[final_answer("hi")])
    bound = model.bind_tools(ALL_TOOLS)
    assert set(bound.bound_tools) == {t.name for t in ALL_TOOLS}


def test_every_emitted_message_gets_a_unique_id():
    """A real LangGraph gotcha, pinned so nobody 'simplifies' it away.

    `add_messages` de-duplicates BY MESSAGE ID. If a model returns the same
    AIMessage object twice, the second is not appended -- it REPLACES the first
    in its original position. The newest message is then no longer last, the
    conditional edge reads a stale ToolMessage, and the agent loop exits after
    two steps looking completely healthy.

    The symptom is an agent that mysteriously stops early with no error. This
    test cost a debugging session to find; it exists so it costs zero next time.
    """
    model = ScriptedToolCallingModel(script=[tool_call("list_documents", call_id="fixed")])

    first = model.invoke("a")
    second = model.invoke("b")

    assert first.id != second.id, "identical ids would be silently swallowed by add_messages"
    assert first.tool_calls[0]["id"] != second.tool_calls[0]["id"], (
        "reused tool_call_id would stitch the wrong result onto the wrong call"
    )


def test_scripted_model_repeats_its_last_message_when_exhausted():
    """Clamping rather than wrapping is deliberate.

    Repeating the last message models an agent stuck in a loop -- the exact
    pathology the step limit exists to catch. Wrapping back to the start would
    let a stuck agent accidentally 'recover' and hide the bug.
    """
    model = ScriptedToolCallingModel(script=[final_answer("only one")])
    assert model.invoke("a").content == "only one"
    assert model.invoke("b").content == "only one"


# ===========================================================================
# GRAPH ROUTING -- the entire control logic of an agent
# ===========================================================================


def test_agent_ends_immediately_when_the_model_asks_for_no_tools():
    """No tool calls -> no loop. This is the whole exit condition."""
    agent = build_offline_agent([final_answer("I already know this.")])
    trace = agent.run("hello")

    assert trace.answer == "I already know this."
    assert trace.tool_calls == []
    assert trace.steps == 1


def test_agent_loops_once_then_answers():
    """The canonical trajectory: search, read the result, answer."""
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", query="chunk overlap"),
            final_answer("Overlap protects boundary facts [1]."),
        ]
    )
    trace = agent.run("what is chunk overlap for?")

    assert trace.tool_names == ["search_knowledge_base"]
    assert trace.steps == 2  # agent -> tools -> agent
    assert "boundary facts" in trace.answer
    assert not trace.hit_step_limit


def test_agent_can_chain_several_different_tools():
    agent = build_offline_agent(
        [
            tool_call("list_documents", call_id="c1"),
            tool_call("search_knowledge_base", call_id="c2", query="reciprocal rank"),
            tool_call("reciprocal_rank", call_id="c3", position=4),
            final_answer("The reciprocal rank at position 4 is 0.25 [1]."),
        ]
    )
    trace = agent.run("what is the reciprocal rank at position 4?")

    assert trace.tool_names == ["list_documents", "search_knowledge_base", "reciprocal_rank"]
    assert "0.25" in trace.answer


# ===========================================================================
# THE FAILURE MODES THAT ACTUALLY MATTER
# ===========================================================================


def test_step_limit_stops_an_agent_that_would_loop_forever():
    """THE most important test in this file.

    The scripted model emits the same tool call forever. Without the circuit
    breaker this runs until something external kills it -- which in production
    presents as a timeout or a runaway bill, not as an obvious bug.

    core/corpus/agent_metrics.md: loop failures are invisible to outcome-only
    evaluation, because the agent never returns a wrong answer. It just never
    returns.
    """
    agent = build_offline_agent(
        [tool_call("list_documents")],  # exhausted immediately, so it repeats
        max_steps=4,
    )
    trace = agent.run("this will never terminate on its own")

    assert trace.hit_step_limit is True
    assert trace.steps == 4, "the circuit breaker did not fire at the configured limit"


def test_loop_detection_finds_repeated_identical_calls():
    """Loop detection with no LLM: same tool, same args, more than once."""
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="chunking"),
            tool_call("search_knowledge_base", call_id="c2", query="chunking"),
            final_answer("done"),
        ]
    )
    trace = agent.run("q")

    repeats = trace.repeated_calls()
    assert repeats, "an agent repeating an identical call was not flagged"
    assert "search_knowledge_base" in repeats[0]


def test_different_arguments_are_not_flagged_as_a_loop():
    """Searching twice for DIFFERENT things is legitimate refinement, not a loop.

    A loop detector that fires on this would be worse than useless -- it would
    penalise the multi-query expansion described in core/corpus/retrieval.md.
    """
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="chunking"),
            tool_call("search_knowledge_base", call_id="c2", query="reranking"),
            final_answer("done"),
        ]
    )
    assert build_trace_repeats(agent) == []


def build_trace_repeats(agent) -> list[str]:
    return agent.run("q").repeated_calls()


def test_agent_survives_a_hallucinated_tool_name():
    """Models invent tool names. The graph must not crash when they do.

    The error is fed back as a ToolMessage so the agent can correct itself, and
    so the failure shows up in the trajectory instead of as a stack trace.
    """
    agent = build_offline_agent(
        [
            tool_call("search_the_internet", query="anything"),  # does not exist
            final_answer("Sorry, I could not do that."),
        ]
    )
    trace = agent.run("q")

    assert trace.tool_names == ["search_the_internet"]
    assert "unknown tool" in trace.tool_calls[0].result
    assert trace.answer, "the agent should still produce an answer, not crash"


def test_agent_can_recover_from_a_bad_argument():
    """ARGUMENT CORRECTNESS is a distinct failure from tool selection.

    Here the agent picks the right tool and passes an invalid position, reads
    the error, and retries correctly. Being able to tell "wrong tool" from
    "right tool, wrong arguments" is why core/corpus/agent_metrics.md lists
    them as separate metrics.
    """
    agent = build_offline_agent(
        [
            tool_call("reciprocal_rank", call_id="c1", position=0),  # invalid
            tool_call("reciprocal_rank", call_id="c2", position=4),  # corrected
            final_answer("It is 0.25."),
        ]
    )
    trace = agent.run("reciprocal rank at position 4?")

    assert trace.tool_calls[0].result.startswith("ERROR")
    assert "0.2500" in trace.tool_calls[1].result
    assert trace.tool_names == ["reciprocal_rank", "reciprocal_rank"]


def test_refusal_is_detectable_from_the_trajectory():
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="capital of France"),
            tool_call("refuse", call_id="c2", reason="the corpus does not cover geography"),
            final_answer("The provided context does not contain this information."),
        ]
    )
    trace = agent.run("What is the capital of France?")

    assert trace.refused is True
    assert "does not contain" in trace.answer


# ===========================================================================
# TRAJECTORY RECONSTRUCTION
# ===========================================================================


def test_trace_stitches_tool_calls_back_to_their_results():
    """Calls live on AIMessages, results arrive later as ToolMessages.

    Matching them by tool_call_id is what turns a flat message list into an
    evaluable trajectory. Get this wrong and every agent metric reads garbage.
    """
    agent = build_offline_agent(
        [
            tool_call("reciprocal_rank", call_id="abc", position=2),
            final_answer("0.5"),
        ]
    )
    trace = agent.run("q")

    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call.name == "reciprocal_rank"
    assert call.args == {"position": 2}
    assert "0.5000" in call.result, "the result was not matched back to its call"


def test_trace_preserves_the_full_message_history():
    agent = build_offline_agent(
        [tool_call("list_documents"), final_answer("done")]
    )
    trace = agent.run("q")

    assert any(isinstance(m, AIMessage) for m in trace.messages)
    assert any(isinstance(m, ToolMessage) for m in trace.messages)


def test_summary_surfaces_a_loop_for_a_human_reader():
    agent = build_offline_agent(
        [
            tool_call("list_documents", call_id="c1"),
            tool_call("list_documents", call_id="c2"),
            final_answer("done"),
        ]
    )
    assert "LOOP DETECTED" in agent.run("q").summary()


# ===========================================================================
# OLLAMA TIER -- a real model choosing its own actions
# ===========================================================================


@pytest.mark.ollama
def test_real_agent_chooses_to_search_for_a_factual_question():
    """Weak but real evidence that tool selection works with a local model.

    Small models are noticeably worse at tool choice than frontier models. If
    this fails on llama3.1:8b that is a finding worth writing down, not
    necessarily a bug in the agent -- and it is exactly the kind of result the
    calibration work in lesson 04 exists to quantify.
    """
    from agent import build_ollama_agent

    from core.providers import ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    trace = build_ollama_agent().run("How many dimensions does nomic-embed-text produce?")
    print("\n" + trace.summary())

    assert "search_knowledge_base" in trace.tool_names, (
        f"agent answered without searching. Trajectory: {trace.tool_names}"
    )
    assert not trace.hit_step_limit, "agent failed to terminate within the step budget"


@pytest.mark.ollama
def test_real_agent_terminates():
    """Non-termination is the agent failure that hurts most in production."""
    from agent import build_ollama_agent

    from core.providers import ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    trace = build_ollama_agent().run("What documents are in the knowledge base?")
    print("\n" + trace.summary())
    assert not trace.hit_step_limit
    assert trace.answer, "agent hit no limit but produced no final answer"
