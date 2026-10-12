"""The whole pipeline -- golden -> app -> test case -> metrics -> outcomes -- with no model.

Scripted policies are deliberately limited, which makes them useful: they show
the deterministic metrics catching real failure modes (a single-tool policy on a
multi-tool task), not only passing.
"""

import pytest

from evalkit.builders import agent_test_case, rag_test_case, run_conversation
from evalkit.config import get_thresholds
from evalkit.datasets import case_id, load_conversations, load_goldens
from evalkit.metrics import suites
from evalkit.runner import run_metrics

T = get_thresholds()


def by_metric(outcomes):
    return {o.metric: o for o in outcomes}


def test_every_rag_golden_runs_and_every_answer_cites_correctly(goldens_dir, scripted_rag):
    for g in load_goldens(goldens_dir / "rag.json"):
        category = g.additional_metadata["category"]
        tc = rag_test_case(g, scripted_rag.query(g.input))
        outcomes = by_metric(
            run_metrics("rag", case_id(g), tc, suites.rag_deterministic(category, T))
        )
        assert all(o.error is None for o in outcomes.values()), case_id(g)
        assert outcomes["Citation"].success, (case_id(g), tc.actual_output)


def test_scripted_rag_refuses_exactly_the_unanswerable_questions(goldens_dir, scripted_rag):
    for g in load_goldens(goldens_dir / "rag.json"):
        category = g.additional_metadata["category"]
        if category == "false_premise":
            continue
        tc = rag_test_case(g, scripted_rag.query(g.input))
        refusal = by_metric(
            run_metrics("rag", case_id(g), tc, suites.rag_deterministic(category, T))
        )["Refusal"]
        assert refusal.success, (case_id(g), tc.actual_output)


@pytest.fixture
def agent_outcomes(goldens_dir, scripted_agent):
    results = {}
    for g in load_goldens(goldens_dir / "agent.json"):
        tc = agent_test_case(g, scripted_agent.run(g.input))
        results[case_id(g)] = (
            g.additional_metadata["category"],
            by_metric(run_metrics("agent", case_id(g), tc, suites.agent_deterministic(T))),
        )
    return results


def test_single_tool_tasks_pass_deterministic_agent_metrics(agent_outcomes):
    for cid, (category, outcomes) in agent_outcomes.items():
        if category in {"tool_use", "error_handling"}:
            failed = [o.metric for o in outcomes.values() if not o.success]
            assert not failed, (cid, failed)


def test_tool_correctness_catches_the_missing_second_tool(agent_outcomes):
    for cid, (category, outcomes) in agent_outcomes.items():
        if category == "multi_tool":
            assert not outcomes["Tool Correctness"].success, cid


def test_conversations_play_out_against_the_bot(goldens_dir, scripted_chat):
    cases = {
        case_id(g): run_conversation(scripted_chat, g)
        for g in load_conversations(goldens_dir / "conversations.json")
    }
    for cid, tc in cases.items():
        assert len(tc.turns) % 2 == 0 and tc.turns[-1].role == "assistant", cid
    assert "Priya" in cases["chat-remembers-name"].turns[-1].content
    assert "Team" in cases["chat-plan-context"].turns[-1].content
