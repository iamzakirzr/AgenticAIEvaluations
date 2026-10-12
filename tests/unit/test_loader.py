import json

import pytest

from evalkit.datasets import GoldenError, case_id, load_conversations, load_goldens, to_dataset
from evalkit.datasets.loader import CATEGORIES


def test_shipped_goldens_load(goldens_dir):
    rag = load_goldens(goldens_dir / "rag.json")
    agent = load_goldens(goldens_dir / "agent.json")
    chats = load_conversations(goldens_dir / "conversations.json")
    safety = load_goldens(goldens_dir / "safety.json")
    assert (len(rag), len(agent), len(chats), len(safety)) == (12, 7, 4, 6)
    assert len(to_dataset(rag).goldens) == 12


def test_rag_goldens_are_not_happy_path_only(goldens_dir):
    """A golden set without failure-mode cases reports great scores and catches nothing."""
    categories = [g.additional_metadata["category"] for g in load_goldens(goldens_dir / "rag.json")]
    for required in ("answerable", "multi_hop", "unanswerable", "false_premise"):
        assert categories.count(required) >= 2, required


def test_every_agent_golden_names_expected_tools(goldens_dir):
    for g in load_goldens(goldens_dir / "agent.json"):
        assert g.expected_tools, case_id(g)


def test_metadata_and_tool_params_are_carried(goldens_dir):
    g = next(
        g for g in load_goldens(goldens_dir / "agent.json") if case_id(g) == "agent-order-status"
    )
    assert g.additional_metadata["category"] == "tool_use"
    assert g.expected_tools[0].input_parameters == {"order_id": "INV-1003"}


def _write(tmp_path, rows):
    path = tmp_path / "g.json"
    path.write_text(json.dumps(rows) if not isinstance(rows, str) else rows)
    return path


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ("{not json", "invalid JSON"),
        ([], "non-empty JSON array"),
        ([{"category": "answerable", "input": "q"}], "missing 'id'"),
        ([{"id": "a", "category": "vibes", "input": "q"}], "unknown category"),
        ([{"id": "a", "category": "answerable"}], "missing 'input'"),
        (
            [
                {"id": "a", "category": "answerable", "input": "q"},
                {"id": "a", "category": "answerable", "input": "q"},
            ],
            "duplicate id",
        ),
    ],
)
def test_bad_golden_files_fail_with_a_precise_message(tmp_path, rows, message):
    with pytest.raises(GoldenError, match=message):
        load_goldens(_write(tmp_path, rows))


def test_conversation_needs_user_turns(tmp_path):
    rows = [
        {
            "id": "c",
            "category": "conversation",
            "scenario": "s",
            "expected_outcome": "o",
            "user_turns": [],
        }
    ]
    with pytest.raises(GoldenError, match="user_turns"):
        load_conversations(_write(tmp_path, rows))


def test_categories_are_documented():
    assert {"answerable", "unanswerable", "false_premise", "tool_use", "safety"} <= CATEGORIES
