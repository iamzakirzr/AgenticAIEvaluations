"""
Converting our traces into RAGAS's sample types.

=============================================================================
THE SAME CONCEPTS, DIFFERENT NAMES -- a translation table worth memorising
=============================================================================

    concept                     DeepEval              RAGAS
    -------------------------   -------------------   ----------------------
    the question                input                 user_input
    what the system said        actual_output         response
    the reference answer        expected_output       reference
    what the retriever found    retrieval_context     retrieved_contexts
    ideal/ground-truth context  context               reference_contexts

Five fields, ten names, zero overlap. This table is the entire reason lesson 04
and lesson 05 have SEPARATE adapters instead of one shared abstraction: a
unified interface would have to pick one vocabulary, and would silently mislead
anyone reading it with the other library's docs open.

Note especially the last row. DeepEval's `context` and RAGAS's
`reference_contexts` both mean "the ideal context", and in BOTH libraries the
mistake is the same: filling it with what your retriever actually returned
turns the metrics that use it into tautologies.

=============================================================================
ONE OBJECT PER SAMPLE, OR A DATASET?
=============================================================================
RAGAS 0.4 supports both styles:

  NEW (used here)     metric.ascore(user_input=..., response=..., ...)
                      explicit arguments, type-checked, one sample at a time.

  CLASSIC             EvaluationDataset([SingleTurnSample(...), ...])
                      then evaluate(dataset, metrics=[...]) scores in bulk.

We build BOTH from the same trace, because you will meet both: the dataset form
is what almost every tutorial and the `evaluate()` entry point use, and the
explicit form is what the current metric classes want.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.compat import bootstrap  # noqa: E402

bootstrap()

from ragas import EvaluationDataset, SingleTurnSample  # noqa: E402

from core.golden import GoldenItem  # noqa: E402
from core.trace import RagTrace  # noqa: E402


def rag_trace_to_sample(trace: RagTrace, item: GoldenItem | None = None) -> SingleTurnSample:
    """Convert a RagTrace into a RAGAS SingleTurnSample.

    Which metrics become computable, by field:

        user_input + response                   -> AnswerRelevancy
        + retrieved_contexts                    -> Faithfulness,
                                                   ResponseGroundedness
        + reference                             -> ContextRecall,
                                                   ContextPrecisionWithReference,
                                                   FactualCorrectness,
                                                   NoiseSensitivity
    """
    return SingleTurnSample(
        user_input=trace.question,
        response=trace.answer,
        retrieved_contexts=trace.contexts,
        reference=item.reference_answer if item else trace.reference_answer,
        # reference_contexts is deliberately NOT set to trace.contexts. It means
        # "the ideal context", and setting it to what we actually retrieved
        # would make context-comparison metrics compare a thing with itself.
    )


def traces_to_dataset(
    pairs: list[tuple[RagTrace, GoldenItem | None]],
) -> EvaluationDataset:
    """Build the classic bulk-evaluation dataset from many traces."""
    return EvaluationDataset(samples=[rag_trace_to_sample(t, i) for t, i in pairs])


def sample_to_kwargs(sample: SingleTurnSample) -> dict:
    """Flatten a sample into the keyword arguments the new metric classes take.

    The bridge between the two API styles. Each metric wants a different subset
    -- Faithfulness takes (user_input, response, retrieved_contexts) while
    AnswerRelevancy takes only (user_input, response) -- so callers pick what
    they need from this dict rather than splatting it wholesale.
    """
    return {
        "user_input": sample.user_input,
        "response": sample.response,
        "retrieved_contexts": list(sample.retrieved_contexts or []),
        "reference": sample.reference,
    }
