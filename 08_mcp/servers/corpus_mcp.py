"""
MCP SERVER 1 of 3 -- the knowledge base, exposed as tools over MCP.

Run it standalone to poke at it:
    python 08_mcp/servers/corpus_mcp.py        # speaks MCP over stdin/stdout

=============================================================================
WHAT AN MCP SERVER ACTUALLY IS
=============================================================================
A process that speaks JSON-RPC over a transport (stdio here) and answers four
kinds of request:

    tools/list      "what can you do?"      -> name, description, JSON schema
    tools/call      "do this"               -> content blocks
    resources/list  "what can you read?"    -> URIs
    prompts/list    "what prompt templates do you have?"

That is the entire protocol surface that matters. The value is not the
protocol -- it is that the CONTRACT is discoverable at runtime, so an agent can
be pointed at a server it was never built against.

The eval consequence, which is the reason this lesson exists: your agent's
tool surface is now DECIDED BY A PROCESS YOU DO NOT CONTROL. A server upgrade
can rename a tool, tighten a schema, or change what a description implies --
and your agent's behaviour changes with no diff in your repository. Lesson:
snapshot-test the tool contract. `test_mcp_integration.py` does.

=============================================================================
WHY THIS SERVER IS DELIBERATELY BORING
=============================================================================
It wraps the same lexical retriever as lesson 01. Nothing new happens inside
it. That is on purpose: the interesting part is entirely in the seam between
agent and server, so the server should not be where your attention goes.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

from core.config import settings
from core.providers import LexicalEmbeddings, cosine_similarity

mcp = FastMCP("corpus")

# The knee of the measured curve, NOT a separating boundary -- see `search`.
LOW_CONFIDENCE_SCORE = 0.25

# Built once at import time, i.e. once per server process. Every tool call on
# the SAME session reuses it. This is exactly the state that silently
# disappears when the client spawns a new process per call -- see
# `test_a_sessionless_tool_loses_server_state`.
_DOCS: list[tuple[str, str]] = []
_EMBEDDER = LexicalEmbeddings()
_MATRIX: list[list[float]] = []


def _ensure_index() -> None:
    global _MATRIX
    if _DOCS:
        return
    corpus_dir = _ROOT / "core" / "corpus"
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # One chunk per paragraph, which is coarse but keeps the server simple.
        for index, para in enumerate(p for p in text.split("\n\n") if len(p.strip()) > 80):
            _DOCS.append((f"{path.stem}#{index}", para.strip()))
    # embed_documents FITS the IDF statistics as a side effect, so it must
    # run before any embed_query call -- see LexicalEmbeddings.corpus_fitted.
    _MATRIX = _EMBEDDER.embed_documents([text for _, text in _DOCS])


@mcp.tool()
def search(query: str, k: int = 3) -> str:
    """Search the internal knowledge base for passages matching a query.

    The docstring IS the tool description the model sees. It is a prompt, not a
    comment: "internal knowledge base" versus "search" is the difference
    between the model choosing this tool and choosing the web one. Lesson 08's
    tool-selection evaluation measures exactly that.
    """
    _ensure_index()
    vector = _EMBEDDER.embed_query(query)
    scored = sorted(
        ((cosine_similarity(vector, row), doc_id, text) for row, (doc_id, text) in zip(_MATRIX, _DOCS)),
        reverse=True,
    )
    top = scored[: max(1, min(k, 10))]
    if not top:
        return "NO_MATCH"

    body = "\n\n".join(f"[{doc_id}] (score {score:.3f}) {text[:400]}" for score, doc_id, text in top)

    # ---------------------------------------------------------------------
    # WHY THIS IS A HINT AND NOT A GATE -- a measured result, not an opinion
    # ---------------------------------------------------------------------
    # The obvious design is "if the top score is below T, return NO_MATCH".
    # We measured whether any T exists, over 42 answerable golden questions and
    # 200 random gibberish queries (run `python 08_mcp/measure_threshold.py`):
    #
    #     real questions   min 0.208   median 0.313
    #     gibberish        median 0.190   p95 0.290   MAX 0.433
    #
    # The gibberish maximum EXCEEDS the real minimum, so the distributions
    # overlap and no threshold separates them. At T=0.25 you reject 16.7% of
    # genuine questions and still accept 10% of gibberish.
    #
    # Hash-based lexical embeddings are the reason: unknown tokens collide into
    # slots that real tokens occupy, so nothing ever scores zero. A neural
    # embedder separates far better -- but you must MEASURE that on your own
    # corpus before trusting it, which is the transferable point.
    #
    # So the score is surfaced as a hint for the model and the operator, and
    # the refusal decision stays where a deterministic check can back it up:
    # the grounding instruction plus citation validation.
    if top[0][0] < LOW_CONFIDENCE_SCORE:
        return f"LOW_CONFIDENCE (top score {top[0][0]:.3f})\n\n{body}"
    return body


@mcp.tool()
def list_documents() -> str:
    """List the document ids available in the internal knowledge base."""
    _ensure_index()
    return ", ".join(sorted({doc_id.split("#")[0] for doc_id, _ in _DOCS}))


@mcp.tool()
def stats() -> str:
    """Report how many passages the internal knowledge base holds."""
    _ensure_index()
    return f"passages={len(_DOCS)} top_k_default={settings.top_k}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
