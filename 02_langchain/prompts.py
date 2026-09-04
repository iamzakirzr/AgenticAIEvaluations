"""
The prompts. Kept in their own file because they are the most-edited,
highest-impact, least-version-controlled part of most RAG systems.

=============================================================================
WHY THE PROMPT IS AN EVALUATION CONCERN, NOT A STYLE CONCERN
=============================================================================
core/corpus/hallucination.md makes a strong claim:

    "Instruct the model explicitly to answer only from the provided context
     and to say it does not know when the context is insufficient. This is the
     single highest-value prompt change in a RAG system, and its effect is
     measurable as a rise in faithfulness on unanswerable questions."

This file contains TWO prompts so you can test that claim rather than believe
it. ``GROUNDED_PROMPT`` has the instruction; ``NAIVE_PROMPT`` does not. Lesson
04 runs both against the unanswerable questions in the golden dataset and
measures the difference in hallucination rate.

That is the shape of every prompt decision worth making: two variants, one
dataset, one number.
=============================================================================
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# THE GROUNDED PROMPT (what the pipeline uses by default)
# ---------------------------------------------------------------------------
# Every line here is doing a specific job. Annotated inline.
# ---------------------------------------------------------------------------

GROUNDED_SYSTEM = """You are a precise technical assistant answering questions about \
retrieval-augmented generation and LLM evaluation.

Rules you must follow:

1. Answer ONLY using the numbered context passages provided below. Do not use \
any knowledge from your training that is not present in the context.

2. If the context does not contain enough information to answer, reply exactly: \
"The provided context does not contain this information." Do not guess, and do \
not offer a partial answer built from general knowledge.

3. If the question contains a claim that the context contradicts, correct the \
claim explicitly rather than accepting it.

4. Cite the passages you used by their number, like [1] or [2][3]. Only cite \
passages you actually relied on.

5. Be concise. Two or three sentences unless the question needs more."""

# Why each rule exists:
#
#  Rule 1  Blocks parametric leakage. Without it, a model asked "what is the
#          capital of France" answers from training data even though the corpus
#          says nothing about France -- which is golden item un-02.
#
#  Rule 2  Gives the model a specific escape hatch. Models hallucinate partly
#          because refusing feels like failing; naming the exact refusal string
#          makes refusal the easy path. It also makes refusal DETECTABLE, which
#          is what lets 04_deepeval score it.
#
#  Rule 3  Targets sycophancy. Golden items ad-01..ad-05 contain false premises
#          ("cosine similarity ranges from 0 to 100"). Without this rule models
#          tend to accept the premise and answer within it.
#
#  Rule 4  Makes citation hallucination checkable. Because we number the
#          passages, a citation like [7] when only 4 were supplied is a
#          deterministic, programmatically detectable error -- no judge needed.
#          See `extract_citations` below.
#
#  Rule 5  Verbosity bias is real: LLM judges rate longer answers higher
#          (core/corpus/llm_as_judge.md). Constraining length keeps the judge
#          honest and reduces the surface area for unsupported claims.

GROUNDED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", GROUNDED_SYSTEM),
        ("human", "Context passages:\n\n{context}\n\nQuestion: {question}"),
    ]
)


# ---------------------------------------------------------------------------
# THE NAIVE PROMPT (the control, for the experiment)
# ---------------------------------------------------------------------------

NAIVE_SYSTEM = """You are a helpful assistant. Use the context below to answer \
the user's question."""

NAIVE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", NAIVE_SYSTEM),
        ("human", "Context:\n\n{context}\n\nQuestion: {question}"),
    ]
)


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------


def format_context(texts: list[str]) -> str:
    """Number the passages so the model can cite them.

    The numbering is not cosmetic. It converts "did the model make up a
    source?" from a judgement call into an integer comparison, which is the
    kind of transformation you should always look for: a deterministic check
    beats an LLM-judged one on cost, speed AND reliability.

    Passages are separated by a blank line and a marker rather than run
    together, because models are noticeably worse at attributing claims to
    sources when passage boundaries are ambiguous.
    """
    if not texts:
        return "(no passages were retrieved)"
    return "\n\n".join(f"[{i}] {text.strip()}" for i, text in enumerate(texts, start=1))


def extract_citations(answer: str) -> set[int]:
    """Pull the [n] citation markers out of an answer.

    Used by the deterministic citation check: any cited number greater than the
    count of supplied passages is a fabricated source, detectable with no LLM.
    """
    import re

    return {int(n) for n in re.findall(r"\[(\d+)\]", answer)}


def invalid_citations(answer: str, n_passages: int) -> set[int]:
    """Citation numbers that refer to passages which were never supplied.

    Non-empty means citation hallucination, caught deterministically.
    """
    return {n for n in extract_citations(answer) if n < 1 or n > n_passages}
