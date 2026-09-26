"""
Connecting ONE agent to SEVERAL MCP servers -- and the four traps in doing so.

=============================================================================
READ THIS BEFORE THE CODE
=============================================================================
Every trap below was found by running the code in this directory against the
installed langchain-mcp-adapters 0.3.2, not recalled from documentation. Each
has a test in `test_mcp_integration.py` that fails if the behaviour changes.

TRAP 1 -- ONE DEAD SERVER KILLS THEM ALL.
    `MultiServerMCPClient.get_tools()` gathers every server in one asyncio
    TaskGroup. One server that fails to start raises an ExceptionGroup out of
    the whole call, and you get NO tools -- not even from the healthy servers.
    Three MCP servers therefore means three single points of failure.
    Fix: `load_tools_resiliently`, which loads each server independently.

TRAP 2 -- COLLIDING TOOL NAMES ARE SILENT.
    Two servers exposing `search` produce two tools named `search`. No error.
    The model emits {"name": "search"} and the runtime resolves it by list
    order. Fix: `tool_name_prefix=True`, asserted by `ToolRegistry.collisions`.

TRAP 3 -- A SESSIONLESS TOOL CALL SPAWNS A NEW SERVER PROCESS.
    Tools from `get_tools()` open a fresh connection per invocation. Anything
    the server held in memory between calls is gone -- measured: pid 1665 then
    pid 1669, counter reset to 1 both times. Fix: `client.session(name)` for a
    stateful server, which keeps one process across calls.

TRAP 4 -- AN MCP TOOL DOES NOT RETURN WHAT A NATIVE TOOL RETURNS.
    A native LangChain tool returning `5` gives you `5`. The identical tool
    over MCP gives you `[{"type": "text", "text": "5", "id": "lc_..."}]`.
    Swap a native tool for its MCP twin and every assertion you wrote breaks.
    Fix: `text_of`, and never assert on the raw return.

=============================================================================
THE EVAL QUESTION THIS LESSON ANSWERS
=============================================================================
"How do I test an agent whose tools come from somewhere else?"

You cannot unit-test the servers -- they are not yours. What you CAN test, and
what this file gives you, is:

    the CONTRACT   -- snapshot the tool names and schemas; fail when they move
    the SELECTION  -- given a question, does the agent pick the right server?
    the DEGRADATION-- when a server is down, does the agent degrade or explode?

Tool-selection accuracy is the metric that matters most and the one nobody
measures. It needs no judge: you label which server should serve each question
and count. `evaluate_tool_selection` does exactly that.
=============================================================================
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

SERVERS_DIR = _HERE / "servers"


def stdio_server(script: str) -> dict[str, Any]:
    """Connection config for one of this lesson's local servers.

    `sys.executable` rather than "python" on purpose: inside a virtualenv,
    under tox, or in CI, "python" may not be the interpreter that has `mcp`
    installed, and the failure ("No module named mcp") points at the server
    rather than at the launcher that chose the wrong interpreter.
    """
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(SERVERS_DIR / script)],
    }


# The three servers this lesson wires into one agent. `corpus` and `web` BOTH
# expose a tool called `search`; that collision is the lesson, not an oversight.
DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
    "corpus": stdio_server("corpus_mcp.py"),
    "web": stdio_server("web_mcp.py"),
    "evaluator": stdio_server("evaluator_mcp.py"),
}


# ---------------------------------------------------------------------------
# TRAP 4 -- reading a result
# ---------------------------------------------------------------------------


def text_of(result: Any) -> str:
    """Flatten an MCP tool result to plain text.

    Handles the content-block list, a bare string, and the ToolMessage-ish
    shapes, because which one you get depends on the adapter version and on
    whether the tool declared structured output. Writing this once is cheaper
    than discovering the difference in an assertion at 2am.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for block in result:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    content = getattr(result, "content", None)
    return text_of(content) if content is not None else str(result)


# ---------------------------------------------------------------------------
# TRAP 1 -- per-server isolation
# ---------------------------------------------------------------------------


@dataclass
class ServerLoad:
    """The outcome of trying to load one server. Failure is data, not an exception."""

    name: str
    tools: list[BaseTool] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def load_tools_resiliently(
    servers: dict[str, dict[str, Any]] | None = None,
    *,
    prefix: bool = True,
    timeout: float = 30.0,
) -> list[ServerLoad]:
    """Load each server INDEPENDENTLY, so one failure does not take the rest down.

    This is the whole fix for trap 1, and it is six lines. The version most
    people write -- one MultiServerMCPClient over every server -- is shorter
    and has the availability of the least available server in the set.

    Returns one ServerLoad per configured server, healthy or not, so the caller
    can decide: proceed degraded, or refuse to start. Both are defensible;
    silently starting with a missing tool is not.
    """
    servers = servers or DEFAULT_SERVERS
    results: list[ServerLoad] = []

    for name, connection in servers.items():
        client = MultiServerMCPClient({name: connection}, tool_name_prefix=prefix)
        try:
            tools = await asyncio.wait_for(client.get_tools(), timeout=timeout)
            results.append(ServerLoad(name=name, tools=list(tools)))
        except Exception as exc:
            # asyncio TaskGroups raise ExceptionGroup, whose str() is the
            # useless "unhandled errors in a TaskGroup (1 sub-exception)".
            # Unwrap it or the operator learns nothing from the log line.
            results.append(ServerLoad(name=name, error=_describe(exc)))
    return results


def _describe(exc: BaseException) -> str:
    inner = getattr(exc, "exceptions", None)
    if inner:
        return "; ".join(_describe(e) for e in inner)
    return f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# TRAP 2 -- collisions
# ---------------------------------------------------------------------------


@dataclass
class ToolRegistry:
    """Every tool the agent can see, with its origin server recorded.

    The origin is the bit `get_tools()` throws away, and it is the bit you need
    to answer "which server served that answer?" during an incident.
    """

    tools: list[BaseTool] = field(default_factory=list)
    origin: dict[str, str] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)

    @classmethod
    def from_loads(cls, loads: list[ServerLoad]) -> ToolRegistry:
        registry = cls()
        for load in loads:
            if not load.ok:
                registry.degraded.append(load.name)
                continue
            for tool in load.tools:
                registry.tools.append(tool)
                registry.origin[tool.name] = load.name
        return registry

    @property
    def names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    @property
    def collisions(self) -> list[str]:
        """Tool names exposed more than once. MUST be empty before you bind.

        Assert on this at startup. A collision is a 50/50 coin flip on every
        call of that tool, and it will not show up in any test that happens to
        run with the servers in the lucky order.
        """
        return sorted(name for name, count in Counter(self.names).items() if count > 1)

    def by_server(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for name, server in self.origin.items():
            grouped[server].append(name)
        return {k: sorted(v) for k, v in sorted(grouped.items())}

    def get(self, name: str) -> BaseTool:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"no tool named {name!r}; have {self.names}")


# ---------------------------------------------------------------------------
# CONTRACT SNAPSHOTTING -- your agent's behaviour lives in someone else's repo
# ---------------------------------------------------------------------------


def tool_contract(tools: list[BaseTool]) -> dict[str, dict[str, Any]]:
    """A stable, comparable description of the tool surface.

    Snapshot this in a test. When an MCP server you depend on renames a tool,
    drops a parameter or makes an optional argument required, the diff appears
    in YOUR test run rather than in production behaviour that nobody can
    explain because no code changed.

    Descriptions are included deliberately: the description is the prompt the
    model routes on, so a reworded description is a behaviour change even when
    the schema is byte-identical.
    """
    contract: dict[str, dict[str, Any]] = {}
    for tool in tools:
        schema = tool.args_schema or {}
        if not isinstance(schema, dict):  # pydantic model rather than raw schema
            schema = getattr(schema, "model_json_schema", dict)()
        properties = schema.get("properties", {}) or {}
        contract[tool.name] = {
            "description": (tool.description or "").strip(),
            "required": sorted(schema.get("required", []) or []),
            "parameters": {
                key: value.get("type", "any") for key, value in sorted(properties.items())
            },
        }
    return contract


def contract_diff(
    old: dict[str, dict[str, Any]], new: dict[str, dict[str, Any]]
) -> list[str]:
    """Human-readable differences between two tool contracts.

    Ordered by how likely each change is to break an agent silently:
    a removed tool fails loudly; a NEW REQUIRED PARAMETER fails on the first
    call the model makes with the old argument shape, which may be days later.
    """
    findings: list[str] = []
    for name in sorted(set(old) - set(new)):
        findings.append(f"REMOVED tool {name}")
    for name in sorted(set(new) - set(old)):
        findings.append(f"ADDED tool {name}")
    for name in sorted(set(old) & set(new)):
        before, after = old[name], new[name]
        added_required = set(after["required"]) - set(before["required"])
        if added_required:
            findings.append(f"BREAKING {name}: newly required {sorted(added_required)}")
        if before["parameters"] != after["parameters"]:
            findings.append(f"CHANGED {name} parameters: {before['parameters']} -> {after['parameters']}")
        if before["description"] != after["description"]:
            # Not cosmetic. The description is what the model routes on.
            findings.append(f"REWORDED {name} description (routing may change)")
    return findings


# ---------------------------------------------------------------------------
# THE METRIC: TOOL-SELECTION ACCURACY
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolChoiceCase:
    """One labelled routing question: which SERVER should answer this?"""

    question: str
    expected_server: str
    why: str = ""


# A deliberately small, deliberately adversarial routing set. Items 4 and 5 are
# the ones that matter: both contain a word that pulls towards the wrong server.
TOOL_CHOICE_CASES: tuple[ToolChoiceCase, ...] = (
    ToolChoiceCase("what does chunk overlap protect against?", "corpus", "internal concept"),
    ToolChoiceCase("list the documents you have", "corpus", "explicitly internal"),
    ToolChoiceCase("what is the weather in London today?", "web", "current, external"),
    ToolChoiceCase(
        "what is today's news about embeddings?",
        "web",
        "TRAP: 'embeddings' is an internal word but 'today's news' is external",
    ),
    ToolChoiceCase(
        "check whether the citation [4] in my answer is valid",
        "evaluator",
        "TRAP: 'answer' and 'citation' sound retrieval-ish but this is a checker",
    ),
    # --- The three below exist because the first five were SATURATED. --------
    # The keyword baseline scored 5/5 on them, which means they cannot tell two
    # routers apart -- the same saturation problem lesson 01 found in recall@k,
    # recurring in a completely different metric. A test set on which your
    # cheapest baseline is perfect has no discriminating power, and adding hard
    # cases is the only fix. These three are PARAPHRASES that carry no keyword.
    ToolChoiceCase("is it raining in London right now?", "web", "no 'weather' keyword"),
    ToolChoiceCase(
        "did I cite anything that does not exist?",
        "evaluator",
        "'cite' is not the substring 'citation' -- keyword routing misses it",
    ),
    ToolChoiceCase(
        "how does splitting text affect what gets found?",
        "corpus",
        "paraphrase of chunking, no shared vocabulary",
    ),
)


@dataclass
class SelectionReport:
    total: int
    correct: int
    mistakes: list[tuple[str, str, str]]  # question, expected, chosen

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def markdown(self) -> str:
        lines = [
            f"Tool-selection accuracy: {self.accuracy:.0%} ({self.correct}/{self.total})",
        ]
        for question, expected, chosen in self.mistakes:
            lines.append(f"  MISROUTED {question!r}: expected {expected}, chose {chosen}")
        return "\n".join(lines)


def evaluate_tool_selection(
    chooser: Any,
    registry: ToolRegistry,
    cases: tuple[ToolChoiceCase, ...] = TOOL_CHOICE_CASES,
) -> SelectionReport:
    """Score a router against labelled cases. NO JUDGE REQUIRED.

    `chooser` is any callable question -> tool name. That can be your agent's
    model, a keyword router, or an embedding classifier; the metric does not
    care, which is what makes it useful for comparing them.

    Why this metric earns its place: an agent that retrieves from the wrong
    server produces a fluent, well-cited, confidently wrong answer. Faithfulness
    scores it highly -- it IS faithful to the passages it was given. The error
    is upstream of everything a RAG metric can see.
    """
    correct = 0
    mistakes: list[tuple[str, str, str]] = []
    for case in cases:
        tool_name = chooser(case.question)
        chosen = registry.origin.get(tool_name, "unknown")
        if chosen == case.expected_server:
            correct += 1
        else:
            mistakes.append((case.question, case.expected_server, chosen))
    return SelectionReport(total=len(cases), correct=correct, mistakes=mistakes)


__all__ = [
    "DEFAULT_SERVERS",
    "SERVERS_DIR",
    "TOOL_CHOICE_CASES",
    "SelectionReport",
    "ServerLoad",
    "ToolChoiceCase",
    "ToolRegistry",
    "contract_diff",
    "evaluate_tool_selection",
    "load_tools_resiliently",
    "stdio_server",
    "text_of",
    "tool_contract",
]
