"""
core.golden -- the labelled evaluation dataset and its schema.

=============================================================================
WHY THE DATASET IS THE MOST IMPORTANT FILE IN AN EVAL REPO
=============================================================================
Metrics are library code -- you install them. The dataset is the part that is
actually yours, and it determines what your evaluation can and cannot detect.

A dataset of twenty questions you know your system answers well will report
excellent scores and catch nothing. A useful dataset is adversarial toward your
own system. This one has four categories, and three of them exist specifically
to make the system fail:

  single_hop     The answer is in one chunk of one document. Baseline
                 competence. If these fail, retrieval is broken.

  multi_hop      The answer requires combining facts from two or more
                 documents. These break naive top-k retrieval, because no
                 single chunk contains the answer, and they are where context
                 RECALL diverges from context PRECISION.

  unanswerable   The corpus genuinely does not contain the answer. The correct
                 behaviour is refusal. THIS IS THE MOST IMPORTANT CATEGORY.
                 A system with no unanswerable questions in its test set has
                 never been measured on its worst failure mode: confident
                 fabrication. A dataset without them will happily give a
                 hallucinating system a perfect score.

  adversarial    Questions containing a false premise, a leading assumption,
                 or wording that lexically matches the wrong document. These
                 catch sycophancy (agreeing with a wrong premise) and
                 retrieval that is fooled by surface word overlap.

=============================================================================
ON reference_doc_ids -- WHY THIS FIELD BUYS YOU FREE CI
=============================================================================
Recording WHICH documents should be retrieved (not just what the answer is)
lets you compute recall@k, precision@k and MRR with no LLM at all. Those
metrics are deterministic, run in milliseconds, and cost nothing -- which is
what makes it possible to gate every pull request on retrieval quality while
running the expensive judged metrics only nightly.

Most tutorial datasets omit this field. It is the cheapest thing you can add
that makes a dataset genuinely useful.
=============================================================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from core.config import CORPUS_DIR, GOLDEN_PATH

Category = Literal["single_hop", "multi_hop", "unanswerable", "adversarial"]

VALID_CATEGORIES: set[str] = {"single_hop", "multi_hop", "unanswerable", "adversarial"}

# Sentinel used in reference answers for questions the corpus cannot answer.
# Tests look for a refusal rather than string-matching this exact text.
REFUSAL = "The provided context does not contain this information."


@dataclass(frozen=True)
class GoldenItem:
    """One labelled evaluation example."""

    # Stable identifier. Never renumber these -- eval reports reference them,
    # and a renumbering silently invalidates every stored baseline.
    id: str

    question: str

    # The answer a perfect system would give. Used by context recall, answer
    # correctness, and semantic similarity. For unanswerable items this is a
    # refusal.
    reference_answer: str

    # Document IDs (corpus filename stems) that must be retrieved to answer.
    # EMPTY for unanswerable items -- by definition no document contains it.
    reference_doc_ids: list[str] = field(default_factory=list)

    category: str = "single_hop"

    # 1 = easy lookup, 3 = requires synthesis or resisting a false premise.
    difficulty: int = 1

    # Optional note explaining what this item is designed to catch. Shown in
    # failure output so you know why the question exists.
    rationale: str = ""

    @property
    def is_answerable(self) -> bool:
        return self.category != "unanswerable"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "GoldenItem":
        return cls(
            id=raw["id"],
            question=raw["question"],
            reference_answer=raw["reference_answer"],
            reference_doc_ids=list(raw.get("reference_doc_ids", [])),
            category=raw.get("category", "single_hop"),
            difficulty=int(raw.get("difficulty", 1)),
            rationale=raw.get("rationale", ""),
        )


def load_golden(path: Path | None = None) -> list[GoldenItem]:
    """Read the golden dataset from JSONL.

    JSONL (one JSON object per line) rather than a single JSON array, because
    it produces readable git diffs: adding a question changes exactly one line.
    With a JSON array, reformatting can rewrite the whole file and make review
    impossible.
    """
    target = path or GOLDEN_PATH
    items: list[GoldenItem] = []
    with target.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue  # allow blank lines and // comments between sections
            try:
                items.append(GoldenItem.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"{target}:{lineno} is not a valid golden item: {exc}") from exc
    return items


def by_category(category: str, path: Path | None = None) -> list[GoldenItem]:
    """All items in one category -- used to run a focused experiment."""
    return [item for item in load_golden(path) if item.category == category]


def corpus_doc_ids(corpus_dir: Path | None = None) -> set[str]:
    """Every document ID the corpus provides.

    A document's ID is its filename without the .md extension, which is what
    the ingestion pipeline records on each chunk.
    """
    target = corpus_dir or CORPUS_DIR
    return {p.stem for p in target.glob("*.md")}


def iter_corpus(corpus_dir: Path | None = None) -> Iterator[tuple[str, str]]:
    """Yield ``(doc_id, text)`` for every document, sorted for reproducibility.

    Sorting matters: an unsorted glob returns filesystem order, which differs
    between machines. That would make chunk indices differ between your laptop
    and CI, and chunk indices appear in eval reports.
    """
    target = corpus_dir or CORPUS_DIR
    for path in sorted(target.glob("*.md")):
        yield path.stem, path.read_text(encoding="utf-8")
