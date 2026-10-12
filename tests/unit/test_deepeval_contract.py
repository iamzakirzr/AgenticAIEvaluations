"""Pin the DeepEval behaviours this framework relies on.

If a DeepEval upgrade breaks any of these, the unit tier fails here, with a
name that says what changed, instead of the nightly evals quietly producing
wrong numbers.
"""

import inspect

from deepeval import assert_test
from deepeval.metrics import StepEfficiencyMetric, ToolCorrectnessMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn

from evalkit.judges import OfflineJudge


def test_step_efficiency_reads_the_private_trace_attribute():
    # evalkit.builders.agent_test_case sets `_trace_dict` directly.
    assert "_trace_dict" in inspect.getsource(StepEfficiencyMetric)
    tc = LLMTestCase(input="q", actual_output="a")
    tc._trace_dict = {"type": "agent"}
    assert tc._trace_dict == {"type": "agent"}


def test_tool_correctness_is_deterministic_without_available_tools():
    tc = LLMTestCase(
        input="q",
        actual_output="a",
        tools_called=[ToolCall(name="calculator")],
        expected_tools=[ToolCall(name="calculator")],
    )
    # OfflineJudge raises if called, so passing proves no LLM was used.
    assert_test(
        tc, [ToolCorrectnessMetric(model=OfflineJudge(), async_mode=False)], run_async=False
    )


def test_tool_correctness_fails_on_a_missing_tool():
    tc = LLMTestCase(
        input="q",
        actual_output="a",
        tools_called=[ToolCall(name="search_docs")],
        expected_tools=[ToolCall(name="calculator")],
    )
    metric = ToolCorrectnessMetric(model=OfflineJudge(), async_mode=False)
    assert metric.measure(tc) == 0.0


def test_conversational_test_case_fields_used_by_builders():
    tc = ConversationalTestCase(
        turns=[Turn(role="user", content="hi")],
        scenario="s",
        expected_outcome="o",
        chatbot_role="r",
        user_description="u",
    )
    assert (tc.scenario, tc.chatbot_role) == ("s", "r")
