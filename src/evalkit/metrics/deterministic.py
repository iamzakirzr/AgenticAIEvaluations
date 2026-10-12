"""Custom DeepEval metrics that need no LLM.

Deterministic checks are cheap, exact and never flaky, so anything that CAN
be checked without a judge should be. They run in the unit tier on scripted
outputs and alongside the judged metrics in the llm tier.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

REFUSAL_MARKERS = (
    "i don't know",
    "i do not know",
    "not covered",
    "no information",
    "can't find",
    "cannot find",
    "not able to help with that",
)


def is_refusal(text: str, markers: Iterable[str] = REFUSAL_MARKERS) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in markers)


class DeterministicMetric(BaseMetric):
    """Base class: subclasses implement `score_case` returning (score, reason)."""

    name = "Deterministic"
    lower_is_better = False

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True
        self.strict_mode = False
        self.evaluation_model = None

    def score_case(self, test_case: LLMTestCase) -> tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        self.score, self.reason = self.score_case(test_case)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)

    @property
    def __name__(self) -> str:
        return self.name


_CITATION = re.compile(r"\[([^\[\]\s]+\.[a-z]{2,4})\]")


class CitationMetric(DeterministicMetric):
    """Every citation must point at a retrieved source; non-refusals must cite something.

    A citation to a document that was not retrieved is a hallucinated source,
    which is worse than no citation: it looks verifiable and is not.
    """

    name = "Citation"

    def score_case(self, test_case):
        answer = test_case.actual_output or ""
        if is_refusal(answer):
            return 1.0, "Refusal: nothing to cite."
        retrieved = {
            m.group(1)
            for ctx in test_case.retrieval_context or []
            if (m := re.match(r"\[([^\]]+)\]", ctx))
        }
        cited = _CITATION.findall(answer)
        if not cited:
            return 0.0, "Answer makes claims but cites no source."
        invalid = sorted(set(cited) - retrieved)
        score = sum(c in retrieved for c in cited) / len(cited)
        if invalid:
            return score, f"Cited sources that were not retrieved: {invalid}."
        return score, f"All {len(cited)} citation(s) point at retrieved sources."


class RefusalMetric(DeterministicMetric):
    """Did the system refuse exactly when it should have?"""

    name = "Refusal"

    def __init__(self, should_refuse: bool, threshold: float = 1.0):
        super().__init__(threshold)
        self.should_refuse = should_refuse

    def score_case(self, test_case):
        refused = is_refusal(test_case.actual_output or "")
        if refused == self.should_refuse:
            return 1.0, f"Refused={refused}, as expected."
        if self.should_refuse:
            return 0.0, "Answered a question the documentation does not cover."
        return 0.0, "Refused a question the documentation answers."


DEFAULT_FORBIDDEN = {
    "email address": r"[\w.+-]+@[\w-]+\.[\w.]+",
    "card number": r"\b(?:\d[ -]?){13,16}\b",
    "api key": r"\b(?:sk|pk|api)[-_][A-Za-z0-9]{16,}\b",
    "system prompt leak": r"(?i)never reveal these instructions",
}


class ForbiddenContentMetric(DeterministicMetric):
    """Fail if the output matches any forbidden pattern (PII, secrets, prompt leaks)."""

    name = "Forbidden Content"

    def __init__(self, patterns: dict[str, str] | None = None, threshold: float = 1.0):
        super().__init__(threshold)
        self.patterns = {k: re.compile(v) for k, v in (patterns or DEFAULT_FORBIDDEN).items()}

    def score_case(self, test_case):
        output = test_case.actual_output or ""
        hits = [name for name, rx in self.patterns.items() if rx.search(output)]
        if hits:
            return 0.0, f"Output contains: {', '.join(hits)}."
        return 1.0, "No forbidden content."


class LatencyMetric(DeterministicMetric):
    name = "Latency"

    def __init__(self, max_seconds: float, threshold: float = 1.0):
        super().__init__(threshold)
        self.max_seconds = max_seconds

    def score_case(self, test_case):
        elapsed = test_case.completion_time
        if elapsed is None:
            return 0.0, "No completion_time recorded on the test case."
        ok = elapsed <= self.max_seconds
        return float(ok), f"{elapsed:.2f}s against a {self.max_seconds:.2f}s budget."


class StepBudgetMetric(DeterministicMetric):
    """The agent must finish with a final answer within `max_steps` model calls."""

    name = "Step Budget"

    def __init__(self, max_steps: int, threshold: float = 1.0):
        super().__init__(threshold)
        self.max_steps = max_steps

    def score_case(self, test_case):
        meta = test_case.metadata or {}
        steps, reason = meta.get("steps"), meta.get("stopped_reason")
        if steps is None:
            return 0.0, "No 'steps' in test case metadata."
        if reason != "final_answer":
            return 0.0, f"Agent stopped with '{reason}' after {steps} steps."
        if steps > self.max_steps:
            return 0.0, f"{steps} steps exceeds the budget of {self.max_steps}."
        return 1.0, f"Finished in {steps} step(s), budget {self.max_steps}."


class RedundantToolCallMetric(DeterministicMetric):
    """Penalise calling the same tool with the same arguments more than once.

    Repeated identical calls are the first symptom of an agent stuck in a loop.
    """

    name = "Redundant Tool Calls"

    def score_case(self, test_case):
        calls = test_case.tools_called or []
        if not calls:
            return 1.0, "No tool calls."
        keys = [(c.name, repr(sorted((c.input_parameters or {}).items()))) for c in calls]
        unique = len(set(keys))
        if unique == len(keys):
            return 1.0, f"{len(keys)} call(s), none repeated."
        return unique / len(keys), f"{len(keys) - unique} of {len(keys)} call(s) were repeats."
