"""
PRODUCTION SCENARIOS for the agent: memory, approval gates, and budgets.

=============================================================================
WHAT THE LESSON-03 AGENT CANNOT DO YET
=============================================================================

  1. IT HAS NO MEMORY. Every `run()` starts from nothing, so a follow-up like
     "what about the second one?" is unanswerable. Production chat needs
     conversation state, and LangGraph provides it through CHECKPOINTERS.

  2. IT CANNOT BE STOPPED MID-RUN FOR APPROVAL. Some tool calls should not
     happen without a human saying yes -- deleting records, sending email,
     spending money. LangGraph provides `interrupt()` for exactly this, and
     it works by SUSPENDING the graph and persisting its state, not by
     blocking a thread.

  3. IT HAS A STEP LIMIT BUT NOT A COST LIMIT. Six steps of a cheap tool and
     six steps of an expensive one cost very different amounts.

=============================================================================
HOW CHECKPOINTING ACTUALLY WORKS (the part that surprises people)
=============================================================================
Compile the graph with a checkpointer, then pass a `thread_id`:

    graph = builder.compile(checkpointer=InMemorySaver())
    graph.invoke(state, config={"configurable": {"thread_id": "user-42"}})

The thread_id IS the conversation. Same id -> the graph resumes with all
previous messages. Different id -> a fresh conversation. There is no session
object to manage; it is a key into the checkpointer's store.

The official docs are explicit that `InMemorySaver` is for development only --
it loses everything on restart. `SqliteSaver` and `PostgresSaver` are the
persistent options. That distinction matters more than it looks: an agent that
silently forgets every conversation on deploy is a bug users report as "it
keeps asking me things I already told it".

=============================================================================
HOW interrupt() ACTUALLY WORKS (the part that surprises people MORE)
=============================================================================
`interrupt(payload)` does not block. It raises a special signal that suspends
the graph and PERSISTS its state via the checkpointer. `invoke()` returns with
an `__interrupt__` key describing what approval is needed. Later -- possibly in
a different process, hours afterwards -- you resume:

    graph.invoke(Command(resume="approved"), config=same_thread_config)

and the original `interrupt()` call RETURNS that value, continuing from exactly
where it stopped.

Two consequences that catch people out:

  a) INTERRUPT REQUIRES A CHECKPOINTER. Without one there is nowhere to persist
     the suspended state, and it fails.

  b) THE NODE RE-RUNS FROM THE TOP ON RESUME. Everything before the
     `interrupt()` call executes a second time. So a node that interrupts must
     not have side effects before the interrupt -- that is why the approval
     check below happens BEFORE any tool is executed, never after.
=============================================================================
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from tools import ALL_TOOLS

from core.resilience import Budget, BudgetExceeded

# Tools that must never run without explicit human approval. In a real system
# this list is the output of a threat-modelling exercise, not an afterthought:
# anything that spends money, mutates state, or contacts a third party.
DANGEROUS_TOOLS: set[str] = {"refuse"}  # 'refuse' stands in for a real side effect


class ProductionAgentState(TypedDict):
    """Adds budget accounting to the lesson-03 state."""

    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    tool_calls_made: int
    approvals: list[dict[str, Any]]


@dataclass
class RunOutcome:
    """What happened, including the two non-happy endings."""

    answer: str = ""
    steps: int = 0
    tool_names: list[str] = field(default_factory=list)
    interrupted: bool = False
    interrupt_payload: dict[str, Any] | None = None
    budget_exceeded: bool = False
    hit_step_limit: bool = False
    elapsed_ms: float = 0.0

    @property
    def completed(self) -> bool:
        return not (self.interrupted or self.budget_exceeded or self.hit_step_limit)


class ProductionAgent:
    """A LangGraph agent with memory, an approval gate and a spend ceiling."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools=None,
        max_steps: int = 6,
        budget: Budget | None = None,
        dangerous_tools: set[str] | None = None,
        checkpointer=None,
        require_approval: bool = True,
    ) -> None:
        self.tools = tools if tools is not None else ALL_TOOLS
        self.tools_by_name = {t.name: t for t in self.tools}
        self.llm = llm.bind_tools(self.tools)
        self.max_steps = max_steps
        self.budget = budget
        self.dangerous_tools = (
            dangerous_tools if dangerous_tools is not None else DANGEROUS_TOOLS
        )
        self.require_approval = require_approval

        # InMemorySaver is DEVELOPMENT ONLY -- it loses everything on restart.
        # Use SqliteSaver or PostgresSaver in production; the interface is
        # identical, which is the point of the abstraction.
        self.checkpointer = checkpointer if checkpointer is not None else InMemorySaver()
        self.graph = self._build()

    # ---- nodes ------------------------------------------------------------

    def _agent_node(self, state: ProductionAgentState) -> dict:
        # Budget is checked BEFORE the call, not after. Checking afterwards has
        # already spent the money you were trying not to spend.
        if self.budget is not None:
            self.budget.check()

        response = self.llm.invoke(state["messages"])

        if self.budget is not None:
            self.budget.record(
                input_tokens=sum(len(str(m.content)) for m in state["messages"]) // 4,
                output_tokens=len(str(response.content)) // 4,
            )
        return {"messages": [response], "steps": state.get("steps", 0) + 1}

    def _tools_node(self, state: ProductionAgentState) -> dict:
        """Execute tools, pausing for approval before any dangerous one.

        CRITICAL ORDERING: the approval check happens BEFORE the tool runs, and
        before any other side effect in this node. On resume LangGraph re-runs
        the node from the top, so anything done before `interrupt()` happens
        TWICE. Putting the interrupt first is what makes that harmless.
        """
        last = state["messages"][-1]
        calls = list(getattr(last, "tool_calls", []) or [])

        approvals = list(state.get("approvals", []))
        outputs: list[BaseMessage] = []

        for call in calls:
            name = call["name"]

            if self.require_approval and name in self.dangerous_tools:
                # Suspends the graph and persists state. Returns the resume
                # value when the run is continued later.
                decision = interrupt(
                    {
                        "reason": "human approval required",
                        "tool": name,
                        "args": call.get("args", {}),
                    }
                )
                approved = _is_approval(decision)
                approvals.append({"tool": name, "decision": str(decision), "approved": approved})

                if not approved:
                    outputs.append(
                        ToolMessage(
                            content=f"DENIED by human reviewer: {name} was not executed.",
                            tool_call_id=call["id"],
                        )
                    )
                    continue

            tool = self.tools_by_name.get(name)
            if tool is None:
                result = f"ERROR: unknown tool {name!r}"
            else:
                try:
                    result = str(tool.invoke(call["args"]))
                except Exception as exc:
                    result = f"ERROR: {type(exc).__name__}: {exc}"

            outputs.append(ToolMessage(content=result, tool_call_id=call["id"]))

        return {
            "messages": outputs,
            "tool_calls_made": state.get("tool_calls_made", 0) + len(calls),
            "approvals": approvals,
        }

    def _should_continue(self, state: ProductionAgentState) -> str:
        if state.get("steps", 0) >= self.max_steps:
            return "end"
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    def _build(self):
        graph = StateGraph(ProductionAgentState)
        graph.add_node("agent", self._agent_node)
        graph.add_node("tools", self._tools_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges(
            "agent", self._should_continue, {"tools": "tools", "end": END}
        )
        graph.add_edge("tools", "agent")
        # The checkpointer is what makes BOTH memory and interrupt() possible.
        return graph.compile(checkpointer=self.checkpointer)

    # ---- running ----------------------------------------------------------

    def config_for(self, thread_id: str) -> dict:
        """The thread_id IS the conversation identity.

        Docs note: keep it under 255 characters. A per-user or per-session id
        is the usual choice.
        """
        return {"configurable": {"thread_id": thread_id}}

    def run(self, question: str, thread_id: str = "default") -> RunOutcome:
        """Start (or continue) a conversation on ``thread_id``."""
        return self._execute(
            {
                "messages": [HumanMessage(content=question)],
                "steps": 0,
                "tool_calls_made": 0,
                "approvals": [],
            },
            thread_id,
        )

    def resume(self, decision: str, thread_id: str = "default") -> RunOutcome:
        """Continue a run that stopped for approval.

        `Command(resume=...)` makes the original `interrupt()` call return
        ``decision``, and execution continues from that exact point -- possibly
        in a different process, hours later.
        """
        return self._execute(Command(resume=decision), thread_id)

    def _execute(self, payload, thread_id: str) -> RunOutcome:
        started = time.perf_counter()
        config = self.config_for(thread_id)
        outcome = RunOutcome()

        try:
            final = self.graph.invoke(
                payload, config={**config, "recursion_limit": self.max_steps * 3}
            )
        except BudgetExceeded:
            # Not an error to swallow: the run really did stop early, and the
            # caller must know the answer is incomplete rather than final.
            outcome.budget_exceeded = True
            outcome.elapsed_ms = (time.perf_counter() - started) * 1000
            state = self.graph.get_state(config)
            # Guard BOTH reads. When the budget trips on the very first agent
            # node the graph may have no persisted state yet, and `state.values`
            # is None -- so an unguarded .get() turns a clean budget stop into
            # an AttributeError that hides the real reason the run ended.
            values = state.values or {}
            outcome.steps = values.get("steps", 0)
            outcome.tool_names = _tool_names(values.get("messages", []))
            return outcome

        outcome.elapsed_ms = (time.perf_counter() - started) * 1000
        outcome.steps = final.get("steps", 0)
        outcome.hit_step_limit = outcome.steps >= self.max_steps
        outcome.tool_names = _tool_names(final.get("messages", []))

        # A suspended run surfaces as an `__interrupt__` entry in the result.
        interrupts = final.get("__interrupt__")
        if interrupts:
            outcome.interrupted = True
            first = interrupts[0]
            outcome.interrupt_payload = getattr(first, "value", first)
            return outcome

        for message in reversed(final.get("messages", [])):
            if isinstance(message, AIMessage) and message.content and not message.tool_calls:
                outcome.answer = str(message.content)
                break
        return outcome

    # ---- inspection --------------------------------------------------------

    def history(self, thread_id: str = "default") -> list[BaseMessage]:
        """Every message on this thread -- what gives the agent its memory."""
        state = self.graph.get_state(self.config_for(thread_id))
        return list((state.values or {}).get("messages", []))

    def approvals(self, thread_id: str = "default") -> list[dict[str, Any]]:
        """The audit trail: who approved what.

        In any regulated setting this is the artifact an auditor asks for, and
        it must be durable -- which means a real checkpointer, not InMemorySaver.
        """
        state = self.graph.get_state(self.config_for(thread_id))
        return list((state.values or {}).get("approvals", []))


# ---------------------------------------------------------------------------


def _is_approval(decision: Any) -> bool:
    """Only an explicit, recognised yes counts.

    FAILING CLOSED IS THE WHOLE POINT. An unrecognised value -- None, an empty
    string, a typo, a timeout sentinel -- must mean "not approved". A guard that
    fails open is worse than no guard, because it creates the belief that
    someone is checking.
    """
    if decision is True:
        return True
    if not isinstance(decision, str):
        return False
    return decision.strip().lower() in {"approve", "approved", "yes", "y", "ok"}


def _tool_names(messages: list[BaseMessage]) -> list[str]:
    names: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", []) or []:
            names.append(call["name"])
    return names
