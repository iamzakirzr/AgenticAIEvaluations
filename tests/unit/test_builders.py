from deepeval.dataset import ConversationalGolden, Golden
from deepeval.test_case import ToolCall, Turn

from evalkit.builders import agent_test_case, rag_test_case, run_conversation
from evalkit.sut.base import AgentResponse, RAGResponse, ToolInvocation


def golden(**kw):
    meta = {"id": "g1", "category": "answerable", "tags": ["t"]}
    return Golden(input="q", expected_output="e", name="g1", additional_metadata=meta, **kw)


def test_rag_case_uses_what_the_app_returned():
    r = RAGResponse("answer [a.md]", ["[a.md] S: text"], ["a.md"], 0.25)
    tc = rag_test_case(golden(context=["truth"]), r)
    assert (tc.actual_output, tc.retrieval_context, tc.context) == (
        "answer [a.md]",
        ["[a.md] S: text"],
        ["truth"],
    )
    assert tc.completion_time == 0.25
    assert tc.metadata["sources"] == ["a.md"]
    assert tc.tags == ["t"]


def test_rag_case_with_no_retrieval_has_none_not_empty_list():
    tc = rag_test_case(golden(), RAGResponse("x", [], []))
    assert tc.retrieval_context is None


def test_agent_case_maps_tool_calls_and_attaches_trace():
    trace = {"name": "a", "type": "agent", "children": []}
    r = AgentResponse(
        "out",
        [ToolInvocation("calculator", {"expression": "1+1"}, "2")],
        2,
        "final_answer",
        0.1,
        trace,
    )
    tc = agent_test_case(golden(expected_tools=[ToolCall(name="calculator")]), r)
    assert tc.tools_called[0].name == "calculator"
    assert tc.tools_called[0].input_parameters == {"expression": "1+1"}
    assert tc.tools_called[0].output == "2"
    assert tc.expected_tools[0].name == "calculator"
    assert tc.metadata["steps"] == 2
    assert tc._trace_dict is trace


def test_conversation_is_played_live_against_the_app(scripted_chat):
    g = ConversationalGolden(
        scenario="s",
        expected_outcome="o",
        name="c1",
        turns=[
            Turn(role="user", content="Hi, my name is Priya."),
            Turn(role="user", content="What is my name?"),
        ],
        additional_metadata={"id": "c1", "category": "conversation"},
    )
    tc = run_conversation(scripted_chat, g)
    assert [t.role for t in tc.turns] == ["user", "assistant", "user", "assistant"]
    assert tc.turns[-1].content == "Your name is Priya."
    assert tc.chatbot_role == scripted_chat.role_description
    assert (tc.scenario, tc.expected_outcome) == ("s", "o")
