"""A judge that refuses to be called.

Several DeepEval metrics construct an LLM client in `__init__` even when the
scoring path they take is purely deterministic (ToolCorrectnessMetric with no
`available_tools` is one). Without an explicit model they fall back to OpenAI
and crash on a missing API key. Passing `OfflineJudge()` satisfies the
constructor and turns any accidental LLM call into a loud failure, which is
exactly what the no-model unit tier needs.
"""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM


class OfflineJudgeCalled(RuntimeError):
    pass


class OfflineJudge(DeepEvalBaseLLM):
    def __init__(self) -> None:
        super().__init__("offline")

    def load_model(self) -> None:
        return None

    def generate(self, prompt: str, *args, **kwargs):
        raise OfflineJudgeCalled(
            "A metric tried to call an LLM in the offline tier. Mark the test "
            "@pytest.mark.llm or use a deterministic metric."
        )

    async def a_generate(self, prompt: str, *args, **kwargs):
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "offline"
