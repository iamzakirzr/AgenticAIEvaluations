#!/usr/bin/env python
"""
What do my MCP servers actually expose? Ask them.

    python 08_mcp/inspect_servers.py

=============================================================================
WHY THIS IS THE FIRST THING TO RUN AGAINST ANY MCP SERVER
=============================================================================
Your agent's tool surface is defined by a process you may not own. Before you
bind anything to a model, look at what you are actually binding: the names, the
required arguments, and the DESCRIPTIONS -- because the description is the
prompt the model routes on, and a reworded one changes behaviour with no diff
in your repository.

The output doubles as the contract you should snapshot in a test. See
`tool_contract` and `contract_diff` in mcp_client.py.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The MCP servers log every request at INFO to stderr, which drowns the report.
logging.getLogger("mcp").setLevel(logging.WARNING)

from mcp_agent import default_keyword_router
from mcp_client import (
    ToolRegistry,
    evaluate_tool_selection,
    load_tools_resiliently,
    tool_contract,
)


def main() -> int:
    loads = asyncio.run(load_tools_resiliently(prefix=True))
    registry = ToolRegistry.from_loads(loads)

    if registry.degraded:
        print(f"DEGRADED servers (loaded independently, so the rest still work): "
              f"{registry.degraded}\n")
        for load in loads:
            if not load.ok:
                print(f"  {load.name}: {load.error}")
        print()

    contract = tool_contract(registry.tools)
    for server, names in registry.by_server().items():
        print(f"{server}")
        for name in names:
            entry = contract[name]
            args = ", ".join(f"{k}: {v}" for k, v in entry["parameters"].items()) or "-"
            required = f"  required={entry['required']}" if entry["required"] else ""
            print(f"  {name}({args}){required}")
            print(f"      {entry['description'].splitlines()[0]}")
        print()

    collisions = registry.collisions
    print(f"name collisions: {collisions or 'none'}")
    if collisions:
        print("  ^ two servers export the same tool. Tool choice is resolved by")
        print("    list order, not by intent. Load with prefix=True.")

    print()
    report = evaluate_tool_selection(default_keyword_router(), registry)
    print(report.markdown())
    print()
    print("That is the BASELINE. An LLM router has to beat it to be worth its")
    print("latency and its variance -- measure yours before you ship it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
