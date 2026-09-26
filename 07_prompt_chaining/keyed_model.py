"""
A deterministic chat model whose reply depends on the PROMPT, not on a counter.

=============================================================================
WHY THE COUNTER-BASED FAKE IS NOT ENOUGH HERE
=============================================================================
`FakeListChatModel` (core.providers.scripted_chat_model) replays responses in
order: first call gets responses[0], second gets responses[1], and so on. That
is perfect for a RAG pipeline, which calls the model exactly once.

It is actively WRONG for testing parallel chains. `RunnableParallel` runs its
branches in a thread pool, so the order in which they reach the model is not
determined. A counter-based fake therefore assigns replies to branches at
random, and a test that asserts "summary == ..." passes or fails depending on
thread scheduling. That is a flaky test you wrote yourself, and it is the exact
category of flake this repository keeps arguing you must not accept.

Keying on the prompt removes the race: the same branch always gets the same
reply no matter when it runs. Deterministic tests for concurrent code.

This is a generally useful trick beyond LangChain -- when faking a dependency
that will be called concurrently, key the fake on the REQUEST, never on call
order.
=============================================================================
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class KeyedChatModel(BaseChatModel):
    """Replies with the value of the first rule whose pattern matches the prompt.

        model = KeyedChatModel(rules=[
            (r"Classify as", "factual"),
            (r"Summarise",   "A one sentence summary."),
        ], default="(no rule matched)")

    Rules are ordered; the first match wins, like a routing table. `default`
    is returned when nothing matches -- and a test that gets the default back
    is telling you the prompt changed, which is usually the thing you wanted
    to know.
    """

    rules: list[tuple[str, str]]
    default: str = "(no rule matched)"
    calls: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "keyed-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> KeyedChatModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(m.content) for m in messages)
        # Appending from multiple threads is safe: list.append is atomic under
        # the GIL. Ordering is NOT guaranteed, so assert on membership and
        # length, never on index.
        self.calls.append(prompt)

        text = self.default
        for pattern, reply in self.rules:
            if re.search(pattern, prompt, flags=re.IGNORECASE):
                text = reply
                break
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


__all__ = ["KeyedChatModel"]
