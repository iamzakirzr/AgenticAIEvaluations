"""
MCP SERVER 3 of 3 -- evaluation utilities, exposed as tools.

=============================================================================
THE IDEA WORTH STEALING FROM THIS FILE
=============================================================================
Evaluation is usually something you run AROUND an agent. MCP lets you hand the
agent its own evaluators, so it can check itself mid-trajectory: retrieve,
score its own citations, and retry before answering.

Be sceptical about how far that goes. A system checking its own work with a
tool it also controls is not independent evidence, and none of these numbers
belong in a quality report -- they are a CONTROL SIGNAL for the agent, not a
measurement of it. The measurement still has to happen outside, with a
reference, in lessons 04 and 05.

That distinction -- control signal versus measurement -- is worth being able
to articulate in an interview. Self-evaluation improves behaviour and proves
nothing.

The tools here are deliberately DETERMINISTIC (regex and set arithmetic, no
model), which is why they can be trusted as a control signal at all.
=============================================================================
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mcp.server.fastmcp import FastMCP

from core.metrics import recall_at_k, reciprocal_rank

mcp = FastMCP("evaluator")


@mcp.tool()
def check_citations(answer: str, passage_count: int) -> str:
    """Check that every [n] citation in an answer refers to a passage that exists.

    Returns OK, or the list of invalid citations. A hallucinated citation is
    the cheapest hallucination signal there is: it costs one regex and no model.
    """
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    if not cited:
        return "NO_CITATIONS"
    invalid = sorted(n for n in cited if n < 1 or n > passage_count)
    return "OK" if not invalid else f"INVALID_CITATIONS: {invalid}"


@mcp.tool()
def retrieval_recall(retrieved_ids: list[str], expected_ids: list[str]) -> str:
    """Compute recall@k for one retrieval, given the ids that should have come back."""
    return f"recall={recall_at_k(retrieved_ids, expected_ids, len(retrieved_ids)):.3f}"


@mcp.tool()
def retrieval_mrr(retrieved_ids: list[str], expected_ids: list[str]) -> str:
    """Compute the reciprocal rank of the first correct document."""
    return f"rr={reciprocal_rank(retrieved_ids, expected_ids):.3f}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
