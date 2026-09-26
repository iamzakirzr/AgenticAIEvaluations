"""
A deliberately STATEFUL MCP server, used to prove trap 3.

It holds a counter in process memory. If the client keeps one session the
counter increments across calls; if the client opens a new connection per call
the counter resets and the pid changes, because it is a different process.

That difference is invisible in any mock, which is why this file exists rather
than a `MagicMock` in the test.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("probe")
_state = {"count": 0}


@mcp.tool()
def bump() -> str:
    """Increment a counter held in this server process's memory."""
    _state["count"] += 1
    return f"pid={os.getpid()} count={_state['count']}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
