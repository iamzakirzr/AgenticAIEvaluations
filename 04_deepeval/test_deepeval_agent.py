"""
DeepEval on AGENTS -- and the surprising amount that needs no LLM.

Run:  pytest 04_deepeval/test_deepeval_agent.py -v         # fast tier
      pytest -m judge 04_deepeval/test_deepeval_agent.py   # judged tier

=============================================================================
THE FINDING THIS FILE IS BUILT AROUND
=============================================================================
`ToolCorrectnessMetric` performs a purely deterministic comparison of the tools
an agent called against the tools it should have called. No model involved.

BUT: in DeepEval 4.2.1 it still constructs a default OpenAI model in its
__init__ and raises `DeepEvalError: OpenAI API key is not configured` if
OPENAI_API_KEY is unset -- even though it never calls it.

That is a genuine trap. It makes a deterministic metric look like it requires a
paid API. The fix is to pass any DeepEvalBaseLLM; we pass `ExplodingJudge`,
which raises if it is ever actually invoked. That turns "I believe this metric
is deterministic" into a test that fails the day it stops being true.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for path in (str(_HERE), str(_HERE.parent / "03_langgraph"), str(_HERE.parent / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent import bind_default_retriever, build_offline_agent
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval_adapters import agent_trace_to_test_case
from ollama_judge import ExplodingJudge
from scripted_model import final_answer, tool_call


@pytest.fixture(scope="module", autouse=True)
def _retriever():
    bind_default_retriever()


def tool_metric(**kwargs) -> ToolCorrectnessMetric:
    """ToolCorrectnessMetric that proves it never touches a model.

    ExplodingJudge satisfies the constructor without an API key AND asserts the
    metric stays deterministic.
    """
    return ToolCorrectnessMetric(model=ExplodingJudge(), **kwargs)


# ===========================================================================
# TOOL CORRECTNESS -- deterministic, free, CI-safe
# ===========================================================================


def test_tool_correctness_is_genuinely_non_llm():
    """If this ever fails, the metric started calling a model and must leave CI."""
    case = LLMTestCase(
        input="q",
        actual_output="a",
        tools_called=[ToolCall(name="search_knowledge_base")],
        expected_tools=[ToolCall(name="search_knowledge_base")],
    )
    metric = tool_metric()
    metric.measure(case)  # ExplodingJudge raises AssertionError if consulted
    assert metric.score == 1.0


def test_perfect_trajectory_scores_one():
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", query="chunk overlap"),
            final_answer("Overlap protects boundary facts [1]."),
        ]
    )
    case = agent_trace_to_test_case(agent.run("what is chunk overlap?"),
                                    expected_tools=["search_knowledge_base"])
    metric = tool_metric()
    metric.measure(case)
    assert metric.score == 1.0


def test_calling_the_wrong_tool_scores_zero():
    agent = build_offline_agent(
        [tool_call("list_documents"), final_answer("Here are the documents.")]
    )
    case = agent_trace_to_test_case(agent.run("what is chunk overlap?"),
                                    expected_tools=["search_knowledge_base"])
    metric = tool_metric()
    metric.measure(case)

    assert metric.score == 0.0
    assert "missing" in (metric.reason or "").lower()


def test_partial_credit_when_some_expected_tools_were_used():
    """Scores between 0 and 1 are what make this useful as a gate.

    A binary metric can only tell you pass/fail. A graded one lets you set a
    threshold and see trajectories drift before they break completely.
    """
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="rank"),
            final_answer("done"),
        ]
    )
    case = agent_trace_to_test_case(
        agent.run("q"), expected_tools=["search_knowledge_base", "reciprocal_rank"]
    )
    metric = tool_metric()
    metric.measure(case)

    assert 0.0 < metric.score < 1.0, f"expected partial credit, got {metric.score}"


def test_refusal_trajectory_is_scored_against_expected_refusal():
    """Refusal as a tool call means the CORRECT behaviour on an unanswerable
    question is itself a checkable trajectory, not a judged string."""
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="capital of France"),
            tool_call("refuse", call_id="c2", reason="not covered by the corpus"),
            final_answer("The provided context does not contain this information."),
        ]
    )
    trace = agent.run("What is the capital of France?")
    case = agent_trace_to_test_case(trace, expected_tools=["search_knowledge_base", "refuse"])

    metric = tool_metric()
    metric.measure(case)
    assert metric.score == 1.0
    assert trace.refused is True


# ===========================================================================
# THE ADAPTER
# ===========================================================================


def test_adapter_carries_tool_arguments_not_just_names():
    """Argument correctness is a distinct failure from tool selection.

    Dropping the arguments in the adapter would make it invisible.
    """
    agent = build_offline_agent(
        [tool_call("reciprocal_rank", position=4), final_answer("0.25")]
    )
    case = agent_trace_to_test_case(agent.run("q"))

    assert case.tools_called[0].name == "reciprocal_rank"
    assert case.tools_called[0].input_parameters == {"position": 4}


def test_adapter_exposes_trajectory_health_in_metadata():
    """Loop and step-limit information must survive into the test case.

    These are the agent failures that outcome metrics cannot see, so losing
    them at the adapter boundary would defeat the point.
    """
    agent = build_offline_agent([tool_call("list_documents")], max_steps=4)
    case = agent_trace_to_test_case(agent.run("q"))

    meta = case.metadata
    assert meta["hit_step_limit"] is True
    assert meta["repeated_calls"], "a looping agent showed no repeated calls"


def test_adapter_passes_search_results_as_retrieval_context():
    """An agent that searches is still doing retrieval, so RAG metrics apply."""
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", query="chunk overlap"),
            final_answer("Overlap protects boundary facts [1]."),
        ]
    )
    case = agent_trace_to_test_case(agent.run("q"))
    assert case.retrieval_context, "search results were dropped; faithfulness now impossible"


def test_adapter_handles_an_agent_that_produced_no_answer():
    """A looping agent returns no final message. The adapter must not crash.

    DeepEval rejects an empty actual_output, so we substitute an explicit
    placeholder -- and the metadata still records that the limit was hit.
    """
    agent = build_offline_agent([tool_call("list_documents")], max_steps=3)
    case = agent_trace_to_test_case(agent.run("q"))

    assert case.actual_output  # non-empty, so metrics can run
    assert case.metadata["hit_step_limit"] is True


# ===========================================================================
# JUDGED TIER -- metrics that genuinely need a model
# ===========================================================================


@pytest.mark.judge
def test_task_completion_judges_the_whole_trajectory():
    """TaskCompletionMetric infers the user's goal from the full trace.

    Unlike ToolCorrectness it needs no expected-tools label, which makes it
    usable on production traces you never annotated. The trade is that it needs
    a judge, and is therefore slow and noisy.
    """
    from deepeval.metrics import TaskCompletionMetric

    from core.providers import ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    from ollama_judge import OllamaJudge

    judge = OllamaJudge()
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", query="reciprocal rank position 4"),
            final_answer("The reciprocal rank for position 4 is 0.25 [1]."),
        ]
    )
    case = agent_trace_to_test_case(agent.run("What is the reciprocal rank at position 4?"))

    metric = TaskCompletionMetric(model=judge, threshold=0.5)
    metric.measure(case)

    print(f"\nTaskCompletion = {metric.score:.2f}\nreason: {metric.reason}")
    print(judge.stats.report())
    assert metric.score is not None


@pytest.mark.judge
def test_real_agent_trajectory_against_expected_tools():
    """Score a REAL local model's tool choices. Expect this to be imperfect.

    Small models are materially worse at tool selection than frontier models.
    A failure here is a finding about llama3.1:8b, not necessarily a bug in the
    agent -- write it down rather than tuning until it passes.
    """
    from agent import build_ollama_agent

    from core.providers import ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    trace = build_ollama_agent().run("How many dimensions does nomic-embed-text produce?")
    case = agent_trace_to_test_case(trace, expected_tools=["search_knowledge_base"])

    metric = tool_metric()
    metric.measure(case)
    print(f"\n{trace.summary()}\nToolCorrectness = {metric.score}")

    assert metric.score >= 0.5, (
        f"local model chose {trace.tool_names}, expected search_knowledge_base"
    )
