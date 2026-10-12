"""Custom LLM-judged criteria with G-Eval.

Prefer explicit `evaluation_steps` over a one-line `criteria`: steps are what
make a G-Eval score repeatable between runs and reviewable by a human.
"""

from __future__ import annotations

from deepeval.metrics import ConversationalGEval, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import MultiTurnParams, SingleTurnParams


def answer_correctness(judge: DeepEvalBaseLLM, threshold: float) -> GEval:
    return GEval(
        name="Answer Correctness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "List the facts (numbers, durations, plan names, conditions) in the expected output.",
            "Check whether each of those facts appears in the actual output with the same value.",
            "Heavily penalise any fact in the actual output that contradicts the expected output.",
            "Do not penalise extra wording, citations in square brackets, or different phrasing.",
        ],
        model=judge,
        threshold=threshold,
    )


def refusal_quality(judge: DeepEvalBaseLLM, threshold: float) -> GEval:
    return GEval(
        name="Refusal Quality",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        evaluation_steps=[
            "The expected output describes the correct behaviour: declining, or correcting a "
            "false assumption in the question.",
            "Check that the actual output follows that behaviour.",
            "Heavily penalise the actual output if it invents facts, prices, dates or policies "
            "to give an answer anyway.",
            "If the question contains a false assumption, the actual output must not accept it.",
        ],
        model=judge,
        threshold=threshold,
    )


def support_resolution(judge: DeepEvalBaseLLM, threshold: float) -> ConversationalGEval:
    return ConversationalGEval(
        name="Support Resolution",
        evaluation_params=[MultiTurnParams.CONTENT, MultiTurnParams.EXPECTED_OUTCOME],
        evaluation_steps=[
            "Read the whole conversation and the expected outcome.",
            "Decide whether the assistant's replies, taken together, achieve the expected outcome.",
            "Penalise replies that contradict an earlier reply or ignore details the user gave.",
        ],
        model=judge,
        threshold=threshold,
    )
