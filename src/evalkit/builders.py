"""Turn a SUT response into a DeepEval test case.

This is the seam between "what the app did" and "what DeepEval scores". Keep
it boring: every field DeepEval reads must be filled from what the app
actually produced, never from the golden, or the metric measures the test
data instead of the system.
"""

from __future__ import annotations

from deepeval.dataset import ConversationalGolden, Golden
from deepeval.test_case import ConversationalTestCase, LLMTestCase, ToolCall, Turn

from evalkit.sut.base import AgentResponse, ChatApp, ChatMessage, RAGResponse


def _meta(golden: Golden) -> dict:
    return dict(golden.additional_metadata or {})


def rag_test_case(golden: Golden, response: RAGResponse) -> LLMTestCase:
    return LLMTestCase(
        input=golden.input,
        actual_output=response.answer,
        expected_output=golden.expected_output,
        retrieval_context=response.contexts or None,
        context=golden.context,
        completion_time=response.latency_s,
        name=golden.name,
        tags=_meta(golden).get("tags") or None,
        metadata={**_meta(golden), "sources": response.sources},
    )


def agent_test_case(golden: Golden, response: AgentResponse) -> LLMTestCase:
    case = LLMTestCase(
        input=golden.input,
        actual_output=response.output,
        expected_output=golden.expected_output,
        tools_called=[
            ToolCall(name=c.name, input_parameters=c.args, output=c.output)
            for c in response.tool_calls
        ],
        expected_tools=golden.expected_tools,
        completion_time=response.latency_s,
        name=golden.name,
        tags=_meta(golden).get("tags") or None,
        metadata={
            **_meta(golden),
            "steps": response.steps,
            "stopped_reason": response.stopped_reason,
        },
    )
    if response.trace is not None:
        # DeepEval's trace-based metrics (StepEfficiency) read this private
        # attribute; normally it is set by @observe + evals_iterator. Setting it
        # directly keeps agent evals plain pytest tests. Guarded by
        # tests/unit/test_deepeval_contract.py so an upgrade that renames it
        # fails the unit tier instead of silently scoring an empty trace.
        case._trace_dict = response.trace
    return case


def run_conversation(app: ChatApp, golden: ConversationalGolden) -> ConversationalTestCase:
    """Play the golden's user turns against the app and record the dialogue."""
    history: list[ChatMessage] = []
    for turn in golden.turns:
        answer = app.reply(history, turn.content)
        history += [ChatMessage("user", turn.content), ChatMessage("assistant", answer)]
    return ConversationalTestCase(
        turns=[Turn(role=m.role, content=m.content) for m in history],
        scenario=golden.scenario,
        expected_outcome=golden.expected_outcome,
        user_description=golden.user_description,
        chatbot_role=app.role_description,
        name=golden.name,
        tags=(golden.additional_metadata or {}).get("tags") or None,
        metadata=dict(golden.additional_metadata or {}),
    )
