"""Load golden test data from JSON into DeepEval Golden objects.

The JSON is the test plan. Validation here is strict on purpose: a typo in a
golden file should fail collection, not silently produce a test that checks
nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval.dataset import ConversationalGolden, EvaluationDataset, Golden
from deepeval.test_case import ToolCall, Turn

CATEGORIES = {
    "answerable",  # the answer is in the corpus
    "multi_hop",  # needs facts from more than one chunk
    "unanswerable",  # not in the corpus: correct behaviour is to refuse
    "false_premise",  # the question assumes something untrue
    "tool_use",  # agent: one tool
    "multi_tool",  # agent: several tools in sequence
    "error_handling",  # agent: a tool returns an error
    "conversation",
    "safety",
}


class GoldenError(ValueError):
    pass


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        rows = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise GoldenError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(rows, list) or not rows:
        raise GoldenError(f"{path}: expected a non-empty JSON array")
    seen: set[str] = set()
    for i, row in enumerate(rows):
        where = f"{path.name}[{i}]"
        if not isinstance(row, dict):
            raise GoldenError(f"{where}: expected an object")
        for key in ("id", "category"):
            if not row.get(key):
                raise GoldenError(f"{where}: missing '{key}'")
        if row["id"] in seen:
            raise GoldenError(f"{where}: duplicate id '{row['id']}'")
        seen.add(row["id"])
        if row["category"] not in CATEGORIES:
            raise GoldenError(f"{where}: unknown category '{row['category']}'")
    return rows


def load_goldens(path: Path) -> list[Golden]:
    goldens = []
    for i, row in enumerate(_read(path)):
        if not row.get("input"):
            raise GoldenError(f"{path.name}[{i}]: missing 'input'")
        expected_tools = [
            ToolCall(name=t["name"], input_parameters=t.get("input_parameters"))
            for t in row.get("expected_tools", [])
        ]
        goldens.append(
            Golden(
                input=row["input"],
                expected_output=row.get("expected_output"),
                context=row.get("context"),
                expected_tools=expected_tools or None,
                name=row["id"],
                additional_metadata={
                    "id": row["id"],
                    "category": row["category"],
                    "tags": row.get("tags", []),
                    **row.get("metadata", {}),
                },
            )
        )
    return goldens


def load_conversations(path: Path) -> list[ConversationalGolden]:
    goldens = []
    for i, row in enumerate(_read(path)):
        user_turns = row.get("user_turns")
        if not user_turns or not all(isinstance(t, str) and t for t in user_turns):
            raise GoldenError(f"{path.name}[{i}]: 'user_turns' must be non-empty strings")
        if not row.get("scenario") or not row.get("expected_outcome"):
            raise GoldenError(f"{path.name}[{i}]: needs 'scenario' and 'expected_outcome'")
        goldens.append(
            ConversationalGolden(
                scenario=row["scenario"],
                expected_outcome=row["expected_outcome"],
                user_description=row.get("user_description"),
                turns=[Turn(role="user", content=t) for t in user_turns],
                name=row["id"],
                additional_metadata={
                    "id": row["id"],
                    "category": row["category"],
                    "tags": row.get("tags", []),
                },
            )
        )
    return goldens


def case_id(golden: Golden | ConversationalGolden) -> str:
    return golden.additional_metadata["id"]


def to_dataset(goldens: list[Golden] | list[ConversationalGolden]) -> EvaluationDataset:
    return EvaluationDataset(goldens=goldens)
