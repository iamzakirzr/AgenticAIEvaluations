"""
Lesson 03 production scenarios. FAST TIER -- no model, no network.

Run:  pytest 03_langgraph/test_production_agent.py -v

Memory, approval gates and spend ceilings are all testable deterministically
with a scripted model. None of this needs a GPU, and all of it is the kind of
behaviour that only shows up in production if you did not test it.
"""

from __future__ import annotations

import pytest
from agent import bind_default_retriever
from langgraph.checkpoint.memory import InMemorySaver
from production_agent import ProductionAgent, _is_approval
from scripted_model import ScriptedToolCallingModel, final_answer, tool_call

from core.resilience import Budget


@pytest.fixture(scope="module", autouse=True)
def _retriever():
    bind_default_retriever()


def agent(script, **kwargs) -> ProductionAgent:
    return ProductionAgent(ScriptedToolCallingModel(script=script), **kwargs)


# ===========================================================================
# CHECKPOINTING AND MEMORY
# ===========================================================================


def test_a_thread_remembers_previous_turns():
    """The thread_id IS the conversation. Same id -> the graph resumes."""
    a = agent([final_answer("First answer."), final_answer("Second answer.")])

    a.run("What is chunk overlap?", thread_id="user-42")
    a.run("And what about reranking?", thread_id="user-42")

    history = a.history("user-42")
    contents = [str(m.content) for m in history]

    assert "What is chunk overlap?" in contents
    assert "And what about reranking?" in contents
    assert len(history) >= 4, "the second turn did not see the first"


def test_different_threads_are_isolated():
    """Two users must not see each other's conversations.

    Getting this wrong is a data-leak bug, not a UX bug.
    """
    a = agent([final_answer("answer")])

    a.run("Alice's private question about her account", thread_id="alice")
    a.run("Bob's unrelated question", thread_id="bob")

    alice = " ".join(str(m.content) for m in a.history("alice"))
    bob = " ".join(str(m.content) for m in a.history("bob"))

    assert "Alice's private question" in alice
    assert "Alice's private question" not in bob, "conversation state leaked across threads"


def test_an_unknown_thread_starts_empty():
    a = agent([final_answer("x")])
    assert a.history("never-used") == []


def test_checkpointer_is_injectable_so_production_can_swap_it():
    """InMemorySaver is DEVELOPMENT ONLY -- it loses everything on restart.

    The official docs say so explicitly. SqliteSaver/PostgresSaver have the
    same interface, which is what makes the swap a one-line change. An agent
    that silently forgets every conversation on deploy is reported by users as
    "it keeps asking me things I already told it".
    """
    saver = InMemorySaver()
    a = agent([final_answer("x")], checkpointer=saver)
    a.run("hello", thread_id="t1")

    assert a.checkpointer is saver
    assert a.history("t1"), "nothing was persisted to the injected checkpointer"


# ===========================================================================
# HUMAN-IN-THE-LOOP APPROVAL
# ===========================================================================


def test_a_dangerous_tool_suspends_the_run_for_approval():
    """interrupt() does NOT block a thread.

    It suspends the graph, persists its state through the checkpointer, and
    returns control with an `__interrupt__` payload. The run can be resumed
    later, in another process.
    """
    a = agent(
        [
            tool_call("refuse", call_id="c1", reason="needs a human decision"),
            final_answer("done"),
        ],
        dangerous_tools={"refuse"},
    )

    outcome = a.run("do something risky", thread_id="approval-1")

    assert outcome.interrupted
    assert not outcome.completed
    assert outcome.interrupt_payload["tool"] == "refuse"
    assert "args" in outcome.interrupt_payload


def test_approving_resumes_from_exactly_where_it_stopped():
    a = agent(
        [
            tool_call("refuse", call_id="c1", reason="needs a human decision"),
            final_answer("Completed after approval."),
        ],
        dangerous_tools={"refuse"},
    )

    a.run("do something risky", thread_id="approval-2")
    outcome = a.resume("approved", thread_id="approval-2")

    assert not outcome.interrupted
    assert outcome.answer == "Completed after approval."

    trail = a.approvals("approval-2")
    assert trail and trail[0]["approved"] is True


def test_denying_records_the_decision_and_does_not_run_the_tool():
    a = agent(
        [
            tool_call("refuse", call_id="c1", reason="risky"),
            final_answer("Understood, I did not do that."),
        ],
        dangerous_tools={"refuse"},
    )

    a.run("do something risky", thread_id="approval-3")
    outcome = a.resume("denied", thread_id="approval-3")

    trail = a.approvals("approval-3")
    assert trail[0]["approved"] is False

    denied = [
        str(m.content) for m in a.history("approval-3") if "DENIED" in str(m.content)
    ]
    assert denied, "the denial was not fed back to the agent"
    assert outcome.answer


def test_approval_fails_closed_on_anything_unrecognised():
    """THE most important property of an approval gate.

    An unrecognised value -- None, empty string, a typo, a timeout sentinel --
    must mean NOT approved. A guard that fails open is worse than no guard,
    because it creates the belief that someone is checking.
    """
    for yes in ("approve", "approved", "YES", "y", " ok ", True):
        assert _is_approval(yes) is True, f"{yes!r} should be an approval"

    for no in ("", None, "maybe", "nope", "APPROVE_LATER", 0, 1, [], {"approved": True}):
        assert _is_approval(no) is False, f"{no!r} must NOT count as approval"


def test_safe_tools_are_not_gated():
    """An approval prompt on every action trains people to click yes."""
    a = agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="chunk overlap"),
            final_answer("Overlap protects boundary facts [1]."),
        ],
        dangerous_tools={"refuse"},
    )

    outcome = a.run("what is chunk overlap?", thread_id="safe-1")

    assert not outcome.interrupted
    assert outcome.completed
    assert outcome.tool_names == ["search_knowledge_base"]


def test_approval_can_be_disabled_for_batch_or_offline_use():
    """Evaluation sweeps have no human to ask."""
    a = agent(
        [tool_call("refuse", call_id="c1", reason="x"), final_answer("done")],
        dangerous_tools={"refuse"},
        require_approval=False,
    )

    outcome = a.run("risky", thread_id="batch-1")
    assert not outcome.interrupted
    assert outcome.answer == "done"


def test_the_approval_trail_is_an_audit_artifact():
    """In a regulated setting this is what an auditor asks for.

    Which also means it must be durable -- a real checkpointer, not InMemorySaver.
    """
    a = agent(
        [tool_call("refuse", call_id="c1", reason="x"), final_answer("done")],
        dangerous_tools={"refuse"},
    )
    a.run("risky", thread_id="audit-1")
    a.resume("approved", thread_id="audit-1")

    trail = a.approvals("audit-1")
    assert len(trail) == 1
    assert trail[0]["tool"] == "refuse"
    assert trail[0]["decision"] == "approved"


# ===========================================================================
# BUDGETS
# ===========================================================================


def test_a_cost_budget_stops_a_looping_agent_before_the_step_limit():
    """Six steps of a cheap tool and six of an expensive one cost differently.

    A step limit bounds iterations; only a budget bounds spend.
    """
    a = agent(
        [tool_call("list_documents", call_id="c1")],  # loops forever
        max_steps=20,
        budget=Budget(max_calls=3),
    )

    outcome = a.run("this would loop", thread_id="budget-1")

    assert outcome.budget_exceeded
    assert not outcome.completed
    assert outcome.steps < 20, "the budget did not stop the run early"


def test_a_generous_budget_does_not_interfere():
    a = agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="chunk overlap"),
            final_answer("Overlap protects boundary facts [1]."),
        ],
        budget=Budget(max_calls=100, max_tokens=10**6),
    )

    outcome = a.run("what is chunk overlap?", thread_id="budget-2")
    assert outcome.completed
    assert outcome.answer


def test_budget_records_usage_as_the_run_proceeds():
    budget = Budget(max_calls=50)
    a = agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="chunk overlap"),
            final_answer("answer"),
        ],
        budget=budget,
    )
    a.run("what is chunk overlap?", thread_id="budget-3")

    assert budget.calls >= 2  # one agent step per loop iteration
    assert budget.tokens > 0


def test_step_limit_still_applies_without_a_budget():
    a = agent([tool_call("list_documents", call_id="c1")], max_steps=4)
    outcome = a.run("loops", thread_id="steps-1")

    assert outcome.hit_step_limit
    assert not outcome.completed
    assert outcome.steps == 4


# ===========================================================================
# OUTCOME REPORTING
# ===========================================================================


def test_the_three_non_happy_endings_are_distinguishable():
    """A caller must be able to tell 'finished' from 'stopped early', and WHY.

    Returning the same shape for all of them is how a truncated answer gets
    presented to a user as a final one.
    """
    completed = agent([final_answer("done")]).run("q", thread_id="o1")
    assert completed.completed

    limited = agent([tool_call("list_documents", call_id="c")], max_steps=2).run(
        "q", thread_id="o2"
    )
    assert limited.hit_step_limit and not limited.completed

    broke = agent(
        [tool_call("list_documents", call_id="c")], max_steps=20, budget=Budget(max_calls=2)
    ).run("q", thread_id="o3")
    assert broke.budget_exceeded and not broke.completed

    paused = agent(
        [tool_call("refuse", call_id="c", reason="x")], dangerous_tools={"refuse"}
    ).run("q", thread_id="o4")
    assert paused.interrupted and not paused.completed
