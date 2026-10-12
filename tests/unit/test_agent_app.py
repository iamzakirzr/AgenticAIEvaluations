import pytest

from evalkit.sut.agent_app import ToolAgent
from evalkit.sut.agent_app.agent import STEP_LIMIT_MESSAGE
from evalkit.sut.agent_app.tools import calculate, get_order
from evalkit.sut.llm_client import ChatResult, ScriptedChat, ToolCallRequest


@pytest.mark.parametrize(
    ("expr", "expected"),
    [("14 * 5 * 12", "840"), ("14 * 5 * 12 * 0.8", "672"), ("-3 + 2", "-1"), ("7 / 2", "3.5")],
)
def test_calculator(expr, expected):
    assert calculate(expr) == expected


@pytest.mark.parametrize(
    "expr", ["__import__('os').system('id')", "open('/etc/passwd')", "2 ** 1000", "x + 1"]
)
def test_calculator_rejects_anything_but_arithmetic(expr):
    assert calculate(expr).startswith("error:")


def test_calculator_reports_division_by_zero():
    assert "division by zero" in calculate("672 / 0")


def test_get_order_is_case_insensitive_and_reports_unknown_ids():
    assert "status=refunded" in get_order("inv-1003")
    assert get_order("INV-9999").startswith("error: no order")


@pytest.mark.parametrize(
    ("task", "tool", "args"),
    [
        ("What is the status of invoice INV-1003?", "get_order", {"order_id": "INV-1003"}),
        ("What is 14 * 5 * 12?", "calculator", {"expression": "14 * 5 * 12"}),
        (
            "What support do Pro users get?",
            "search_docs",
            {"query": "What support do Pro users get?"},
        ),
    ],
)
def test_agent_calls_the_tool_and_answers_from_its_output(scripted_agent, task, tool, args):
    r = scripted_agent.run(task)
    assert [(c.name, c.args) for c in r.tool_calls] == [(tool, args)]
    assert r.stopped_reason == "final_answer"
    assert r.steps == 2
    assert r.output == f"Result: {r.tool_calls[0].output}"


def test_agent_stops_at_the_step_limit():
    looping = ScriptedChat(
        lambda m, t: ChatResult("", [ToolCallRequest("calculator", {"expression": "1+1"})])
    )
    r = ToolAgent(looping, max_steps=3).run("loop forever")
    assert (r.stopped_reason, r.steps, r.output) == ("step_limit", 3, STEP_LIMIT_MESSAGE)
    assert len(r.tool_calls) == 3


def test_unknown_tool_and_bad_arguments_become_tool_errors_not_crashes():
    calls = iter(
        [
            ChatResult("", [ToolCallRequest("delete_account", {})]),
            ChatResult("", [ToolCallRequest("calculator", {"expr": "1"})]),
            ChatResult("done"),
        ]
    )
    r = ToolAgent(ScriptedChat(lambda m, t: next(calls))).run("x")
    assert r.tool_calls[0].output == "error: unknown tool delete_account"
    assert r.tool_calls[1].output.startswith("error: bad arguments for calculator")
    assert r.output == "done"


def test_trace_records_every_llm_and_tool_span(scripted_agent):
    r = scripted_agent.run("What is 2 + 2?")
    assert r.trace["type"] == "agent"
    assert [s["type"] for s in r.trace["children"]] == ["llm", "tool", "llm"]
    assert r.trace["children"][1]["output"] == "4"
