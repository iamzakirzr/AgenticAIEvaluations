"""
Converting our traces into DeepEval's test-case types.

=============================================================================
WHY THIS LIVES HERE AND NOT IN core/
=============================================================================
The repo README argues against a "unified metric abstraction" over DeepEval and
RAGAS, and this file is where that argument becomes concrete.

DeepEval and RAGAS disagree about what a test case IS:

    DeepEval          LLMTestCase(input, actual_output, expected_output,
                                  retrieval_context, context,
                                  tools_called, expected_tools)
                      -> a single object, evaluated one at a time

    RAGAS             SingleTurnSample(user_input, response, reference,
                                       retrieved_contexts, reference_contexts)
                      -> collected into an EvaluationDataset, scored in bulk

Note the traps hiding in the naming:

  * DeepEval has BOTH `context` and `retrieval_context`, and they mean
    different things. `retrieval_context` is what your retriever returned (use
    this). `context` is the IDEAL context -- ground truth used by the
    Hallucination metric. Putting retrieved chunks in `context` silently turns
    hallucination detection into a tautology.

  * RAGAS's `reference` is DeepEval's `expected_output`. Same concept, and no
    two libraries agree on the name.

A shared `Metric` interface would have to flatten these differences and would
lose information from both. Two small adapters, each fluent in one library's
dialect, cost less and lie less. This is the general lesson: adapt at the
edges, do not unify in the middle.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deepeval.test_case import LLMTestCase, ToolCall

from core.golden import GoldenItem
from core.trace import RagTrace


def rag_trace_to_test_case(trace: RagTrace, item: GoldenItem | None = None) -> LLMTestCase:
    """Convert a RagTrace into a DeepEval LLMTestCase.

    Which metrics become computable depends entirely on which fields you fill:

        input             + actual_output      -> AnswerRelevancy, GEval, Bias,
                                                  Toxicity, PIILeakage
        + retrieval_context                    -> Faithfulness,
                                                  ContextualRelevancy
        + expected_output                      -> ContextualPrecision,
                                                  ContextualRecall

    Leaving `expected_output` empty is the most common way people accidentally
    make half the RAG metrics unavailable and then conclude the library is
    broken.
    """
    return LLMTestCase(
        input=trace.question,
        actual_output=trace.answer,
        # What the retriever ACTUALLY returned. Faithfulness is judged against
        # this, so it must not be idealised.
        retrieval_context=trace.contexts,
        # The reference answer, when we have a label for this question.
        expected_output=item.reference_answer if item else trace.reference_answer,
        # NOTE: `context` is deliberately left unset. It means "the ideal
        # ground-truth context" and is used by HallucinationMetric. Filling it
        # with the retrieved chunks would make that metric compare the
        # retrieved context against itself, which always passes.
        metadata={
            "golden_id": item.id if item else None,
            "category": item.category if item else None,
            "retrieved_doc_ids": trace.retrieved_doc_ids,
            "expected_doc_ids": item.reference_doc_ids if item else [],
            "invalid_citations": trace.metadata.get("invalid_citations", []),
        },
    )


def agent_trace_to_test_case(trace, expected_tools: list[str] | None = None) -> LLMTestCase:
    """Convert a lesson-03 AgentTrace into a DeepEval LLMTestCase.

    `tools_called` and `expected_tools` are what ToolCorrectnessMetric compares.
    Both are lists of ToolCall objects, and ORDER MATTERS to the metric unless
    you relax it via evaluation_params.

    The retrieved passages the agent gathered are passed as retrieval_context so
    that RAG metrics still apply to an agentic pipeline -- an agent that
    searches is still doing retrieval, and faithfulness is still meaningful.
    """
    contexts = [
        call.result
        for call in trace.tool_calls
        if call.name == "search_knowledge_base" and call.result
    ]

    return LLMTestCase(
        input=trace.question,
        actual_output=trace.answer or "(no final answer produced)",
        retrieval_context=contexts or None,
        tools_called=[
            ToolCall(name=call.name, input_parameters=call.args) for call in trace.tool_calls
        ],
        expected_tools=(
            [ToolCall(name=name) for name in expected_tools] if expected_tools else None
        ),
        metadata={
            "steps": trace.steps,
            "hit_step_limit": trace.hit_step_limit,
            "repeated_calls": trace.repeated_calls(),
            "refused": trace.refused,
        },
    )
