"""
The agent's tools.

=============================================================================
A TOOL IS A FUNCTION PLUS A CONTRACT THE MODEL CAN READ
=============================================================================
`@tool` turns a Python function into something a model can call. What actually
gets sent to the model is a JSON schema derived from three things:

  1. the function NAME          -> how the model refers to it
  2. the type ANNOTATIONS       -> the argument schema it must satisfy
  3. the DOCSTRING              -> the only description of when to use it

The docstring is not documentation for you. It is a PROMPT. If an agent keeps
picking the wrong tool, rewriting the docstring is usually a bigger lever than
changing the system prompt, and people rarely try it first.

=============================================================================
WHY THESE PARTICULAR TOOLS
=============================================================================
They are chosen to make specific agent failures observable and testable:

  search_knowledge_base   The RAG retriever, now as a tool the agent chooses
                          to call. Turns single-shot RAG into agentic RAG.
  list_documents          Cheap and safe. An agent that calls this repeatedly
                          instead of searching is demonstrating poor planning
                          -- exactly what Step Efficiency catches.
  reciprocal_rank         Pure arithmetic with a strict numeric argument. Lets
                          us test ARGUMENT CORRECTNESS separately from tool
                          selection: calling the right tool with rank=0
                          instead of rank=4 is a distinct failure.
  refuse                  An explicit "I cannot answer from the corpus" action.
                          Making refusal a TOOL CALL rather than free text
                          means refusal becomes deterministically detectable
                          in the trajectory, with no judge needed.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.tools import tool

from core.golden import corpus_doc_ids

# Populated by `bind_retriever()` so the tools can reach the vector store
# without importing lesson 02 at module scope (which would couple the lessons
# and slow every import).
_RETRIEVER = None


def bind_retriever(pipeline) -> None:
    """Point the search tool at an ingested RagPipeline from lesson 02."""
    global _RETRIEVER
    _RETRIEVER = pipeline


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for passages relevant to a query.

    Use this whenever the user asks a factual question about retrieval,
    embeddings, chunking, evaluation metrics, agents or hallucination.
    Returns numbered passages with their source document.
    """
    if _RETRIEVER is None:
        return "ERROR: no retriever is bound. Call bind_retriever() first."

    chunks = _RETRIEVER.retrieve(query)
    if not chunks:
        return "No passages found."

    return "\n\n".join(
        f"[{i}] (from {c.doc_id}.md, score {c.score:.3f})\n{c.text}"
        for i, c in enumerate(chunks, start=1)
    )


@tool
def list_documents() -> str:
    """List the titles of every document available in the knowledge base.

    Use this only to check what topics exist. It does NOT return document
    contents, so it cannot answer a factual question on its own.
    """
    # The second sentence exists because agents genuinely do get stuck calling
    # a cheap listing tool over and over instead of committing to a search.
    # Naming the limitation in the docstring measurably reduces that.
    return ", ".join(sorted(corpus_doc_ids()))


@tool
def reciprocal_rank(position: int) -> str:
    """Compute the reciprocal rank for a 1-based result position.

    Use this for arithmetic about ranking metrics. Position must be a positive
    integer: position 1 gives 1.0, position 2 gives 0.5, position 4 gives 0.25.
    """
    if position < 1:
        # Return an error the AGENT can read and recover from, rather than
        # raising. A raised exception kills the graph; a returned error string
        # gives the model a chance to retry with a corrected argument -- and
        # whether it actually does is a behaviour worth evaluating.
        return f"ERROR: position must be >= 1, got {position}. Retry with a valid position."
    return f"The reciprocal rank for position {position} is {1.0 / position:.4f}."


@tool
def refuse(reason: str) -> str:
    """Decline to answer because the knowledge base does not cover the question.

    Use this when searching has returned nothing relevant. Prefer refusing over
    answering from your own general knowledge.
    """
    return f"REFUSED: {reason}"


ALL_TOOLS = [search_knowledge_base, list_documents, reciprocal_rank, refuse]

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
