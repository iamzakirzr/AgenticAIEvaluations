"""
A deterministic chat model that can emit TOOL CALLS.

=============================================================================
WHY THIS FILE HAS TO EXIST
=============================================================================
`FakeListChatModel` (used in lessons 01-02) replays strings. That is enough to
test a RAG pipeline, because a RAG pipeline calls the model exactly once.

An agent is different: it LOOPS. The model emits a tool call, the graph runs
the tool, feeds the result back, and the model decides again. To test that loop
deterministically -- routing, step limits, trajectory recording, loop detection
-- you need a fake model that can produce tool calls on demand.

`FakeMessagesListChatModel` gets close (it replays AIMessage objects, tool
calls included) but does not implement `bind_tools`, which LangGraph calls on
every model before wiring it into a graph. So we implement the smallest model
that does both.

=============================================================================
WHAT bind_tools ACTUALLY DOES (worth knowing)
=============================================================================
`llm.bind_tools(tools)` returns a NEW runnable that will include the tools'
JSON schemas in every request. It does not mutate the model and it does not
teach the model anything -- it just attaches schemas to the request payload.

Our scripted model ignores the schemas entirely, because its replies are
pre-written. Recording them anyway (in `self.bound_tools`) lets tests assert
that the agent bound the tools it was supposed to.
=============================================================================
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedToolCallingModel(BaseChatModel):
    """Replays a fixed list of AIMessages, tool calls and all.

    Example -- an agent that searches once, then answers:

        ScriptedToolCallingModel(script=[
            AIMessage(content="", tool_calls=[
                {"name": "search_knowledge_base",
                 "args": {"query": "chunk overlap"}, "id": "call_1"},
            ]),
            AIMessage(content="Chunk overlap protects boundary facts [1]."),
        ])

    The first invocation returns the tool call, the second returns the final
    answer. If the script runs out, the last message repeats -- which is what
    lets us build a model that loops forever, to test the step limit.
    """

    script: list[AIMessage]
    call_count: int = 0
    bound_tools: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedToolCallingModel":
        """Record which tools were bound, then return self.

        A real implementation converts each tool to a JSON schema and attaches
        it to the request. Ours only needs to not crash, and to remember what
        it was given so tests can assert on it.
        """
        names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        self.bound_tools = names
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        if not self.script:
            raise ValueError("ScriptedToolCallingModel needs a non-empty script")

        # Clamp rather than wrap: repeating the LAST message forever models an
        # agent stuck in a loop, which is exactly the pathology the step limit
        # and loop detection exist to catch. Wrapping to the start would
        # accidentally let a stuck agent 'recover' and hide the bug.
        index = min(self.call_count, len(self.script) - 1)
        template = self.script[index]
        self.call_count += 1

        # -------------------------------------------------------------------
        # EVERY EMITTED MESSAGE NEEDS A FRESH ID. This is not defensive
        # boilerplate -- it is required for correctness, and the reason is
        # genuinely surprising:
        #
        # LangGraph's `add_messages` reducer de-duplicates BY MESSAGE ID. If we
        # returned the same AIMessage object twice, the second one would not be
        # appended; it would REPLACE the first one in its original position.
        # The newest message would therefore no longer be last, `_should_continue`
        # would look at a stale ToolMessage instead of the new AIMessage, and
        # the agent loop would exit after two steps looking perfectly healthy.
        #
        # The symptom is an agent that mysteriously stops early with no error.
        # We hit exactly this while writing the step-limit test.
        #
        # Tool call ids get the same treatment, because a ToolMessage is matched
        # back to its request by tool_call_id -- reusing one would stitch the
        # wrong result onto the wrong call in the trajectory.
        # -------------------------------------------------------------------
        import uuid

        suffix = uuid.uuid4().hex[:8]
        message = template.model_copy(
            update={
                "id": f"scripted-{self.call_count}-{suffix}",
                "tool_calls": [
                    {**call, "id": f"{call['id']}-{self.call_count}"}
                    for call in (template.tool_calls or [])
                ],
            }
        )

        return ChatResult(generations=[ChatGeneration(message=message)])

    def reset(self) -> None:
        self.call_count = 0


def tool_call(name: str, call_id: str = "call_1", **args: Any) -> AIMessage:
    """Shorthand for an AIMessage that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def final_answer(text: str) -> AIMessage:
    """Shorthand for an AIMessage with no tool calls, which ends the loop."""
    return AIMessage(content=text)
