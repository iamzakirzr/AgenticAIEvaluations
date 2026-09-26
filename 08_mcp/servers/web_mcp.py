"""
MCP SERVER 2 of 3 -- an offline stand-in for a public web-search server.

=============================================================================
WHY THIS SERVER EXISTS, AND WHY ITS TOOL IS ALSO CALLED `search`
=============================================================================
It is here to COLLIDE.

Point an agent at two MCP servers that each expose a tool called `search` and
you get two tools called `search`. No error. No warning. Nothing in any log.
The model emits `{"name": "search"}` and the runtime picks whichever matched
first, which is list order, which is dict order, which is your config file.

`test_colliding_tool_names_are_silently_ambiguous` pins that behaviour, and
`test_prefixing_resolves_the_collision` shows the one-flag fix. This is the
single most likely way a working multi-MCP agent breaks when someone adds a
third server, and it is invisible in code review.

Responses are canned, so the fast tier needs no network. The point is the
seam, not the search quality.
=============================================================================
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("web")

_CANNED = {
    "weather": "London: 14C, light rain.",
    "news": "No major incidents reported today.",
    "python": "Python 3.13 is the current stable release.",
}


@mcp.tool()
def search(query: str) -> str:
    """Search the public web for current information not in any internal store."""
    for key, value in _CANNED.items():
        if key in query.lower():
            return value
    return "NO_MATCH"


@mcp.tool()
def fetch_url(url: str) -> str:
    """Fetch the text content of a public URL.

    Note what this tool would mean on a PUBLICLY EXPOSED agent: a user-supplied
    string becomes an outbound request from your server. That is server-side
    request forgery with a friendly interface. Lesson 09 covers why an exposed
    agent needs an allowlist here, not a blocklist.
    """
    if not url.startswith(("http://", "https://")):
        return "REFUSED: not an http(s) url"
    return f"(offline stub) content of {url}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
