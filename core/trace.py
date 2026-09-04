"""
core.trace -- the data structure every evaluator consumes.

=============================================================================
WHY A TRACE, NOT A STRING
=============================================================================
The single biggest mistake in beginner RAG projects is building a pipeline
whose only output is the answer text:

    answer = rag("What is chunk overlap?")   # -> "Chunk overlap is..."

You cannot evaluate that. Almost every meaningful RAG metric needs to see the
RETRIEVED CONTEXT, not just the answer:

  - Faithfulness            "is the answer supported by the retrieved chunks?"
  - Context Precision       "were the retrieved chunks relevant?"
  - Context Recall          "did we retrieve everything needed?"
  - Contextual Relevancy    "what fraction of retrieved text was useful?"

If your pipeline throws away the chunks, all four are impossible and you are
left with answer-only metrics, which cannot tell you WHY something failed.
That distinction -- "the retriever failed" vs "the generator hallucinated" --
is the whole point of RAG evaluation. A trace preserves it.

So every pipeline in this repo returns a ``RagTrace``. DeepEval and RAGAS each
have their own adapter that converts one into their native format; those
adapters live in the respective lesson directories, deliberately NOT here,
because the two libraries disagree about what a test case is and forcing them
into one shape would lose information from both.
=============================================================================
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievedChunk:
    """One piece of context the retriever returned, with its provenance."""

    # The text handed to the generator. This is what faithfulness is judged
    # against.
    text: str

    # Which source document this came from. Needed for retrieval metrics that
    # compare against known-correct source IDs (recall@k, MRR) WITHOUT an LLM.
    # Those are the metrics that make the fast, free CI tier possible.
    doc_id: str

    # Position of this chunk within its source document (0-based). Useful when
    # debugging "the answer was in chunk 7 but we only retrieved 4".
    chunk_index: int = 0

    # Retriever score. Higher = more similar. Meaning depends on the retriever
    # (cosine similarity here); only the ORDER is comparable across retrievers.
    score: float = 0.0

    def __str__(self) -> str:  # nice output in pytest failure messages
        preview = self.text[:70].replace("\n", " ")
        return f"[{self.doc_id}#{self.chunk_index} score={self.score:.3f}] {preview}..."


@dataclass
class RagTrace:
    """Everything that happened while answering one question.

    Think of this as the flight recorder. If a metric scores badly, this object
    contains enough information to work out why without re-running anything.
    """

    # --- the question asked -------------------------------------------------
    question: str

    # --- what the system produced ------------------------------------------
    answer: str
    retrieved: list[RetrievedChunk] = field(default_factory=list)

    # --- reference data, when we have it (from the golden dataset) ---------
    # Kept on the trace so a single object can be handed to any evaluator.
    reference_answer: str | None = None
    reference_doc_ids: list[str] = field(default_factory=list)

    # --- operational telemetry ---------------------------------------------
    # Quality is not the only thing worth gating on. A change that improves
    # faithfulness by 2% while tripling latency is usually a bad trade, and you
    # can only see that if you record it.
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # --- reproducibility ----------------------------------------------------
    # Which configuration produced this trace. Without these fields, comparing
    # two eval runs is meaningless -- you cannot tell whether the score moved
    # because the code changed or because the model did.
    chat_model: str = ""
    embed_model: str = ""
    chunk_size: int = 0
    top_k: int = 0

    # Free-form space for lesson-specific extras (agent tool calls, etc.).
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- convenience accessors --------------------------------------------

    @property
    def contexts(self) -> list[str]:
        """Just the chunk texts, in rank order.

        Both DeepEval (``retrieval_context``) and RAGAS (``retrieved_contexts``)
        want a plain list of strings, so this is the most-used property here.
        """
        return [c.text for c in self.retrieved]

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """Source document IDs in rank order, de-duplicated but order-preserving.

        A single document can contribute several chunks; for document-level
        recall we care about the set, but order still matters for MRR, so we
        cannot just use set().
        """
        seen: list[str] = []
        for chunk in self.retrieved:
            if chunk.doc_id not in seen:
                seen.append(chunk.doc_id)
        return seen

    @property
    def total_ms(self) -> float:
        return self.retrieval_ms + self.generation_ms

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def summary(self) -> str:
        """One-screen human summary -- print this when a test fails."""
        lines = [
            f"Q: {self.question}",
            f"A: {self.answer[:300]}{'...' if len(self.answer) > 300 else ''}",
            f"retrieved {len(self.retrieved)} chunks from {self.retrieved_doc_ids}",
        ]
        if self.reference_doc_ids:
            lines.append(f"expected docs: {self.reference_doc_ids}")
        lines.append(
            f"timing: retrieval {self.retrieval_ms:.0f}ms + generation "
            f"{self.generation_ms:.0f}ms = {self.total_ms:.0f}ms"
        )
        for chunk in self.retrieved:
            lines.append(f"  {chunk}")
        return "\n".join(lines)
