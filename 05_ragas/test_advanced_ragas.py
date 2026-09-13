"""
Advanced RAGAS tests. FAST TIER except the generation itself.

Run:  pytest 05_ragas/test_advanced_ragas.py -v
      pytest -m judge 05_ragas/test_advanced_ragas.py -v
"""

from __future__ import annotations

import json

import pytest
from advanced_ragas import ReviewQueue, corpus_as_langchain_documents

from core.golden import GoldenItem


def test_corpus_converts_to_langchain_documents():
    docs = corpus_as_langchain_documents()
    assert len(docs) >= 6
    assert all(d.metadata.get("doc_id") for d in docs)
    assert all(d.page_content for d in docs)


def test_document_limit_is_respected():
    assert len(corpus_as_langchain_documents(limit=3)) == 3


def test_generated_answers_never_become_ground_truth():
    """THE rule. A generated question may enter the dataset; a generated ANSWER
    may not become ground truth.

    The generator wrote it by reading the same corpus the system retrieves
    from, so accepting it makes the dataset circular -- certifying current
    behaviour as correct and hiding every existing bug permanently.
    """

    class FakeTestset:
        def to_pandas(self):
            import types

            rows = [
                {"user_input": "What is chunk overlap?", "reference": "A model wrote this."}
            ]
            return types.SimpleNamespace(to_dict=lambda orient: rows)

    queue = ReviewQueue.from_testset(FakeTestset())

    assert len(queue.items) == 1
    assert queue.items[0].reference_answer == "TODO: a human must confirm this"
    assert "A model wrote this" not in queue.items[0].reference_answer


def test_coverage_gaps_names_the_categories_a_generator_will_not_produce():
    """Always run this before congratulating yourself on 200 generated questions.

    A generator reading your corpus produces answerable, in-vocabulary lookups.
    The categories that catch real bugs -- unanswerable and adversarial -- will
    be missing, and their absence is invisible unless you check.
    """
    queue = ReviewQueue(
        items=[
            GoldenItem(id="g1", question="q", reference_answer="a", category="single_hop"),
            GoldenItem(id="g2", question="q", reference_answer="a", category="multi_hop"),
        ]
    )
    gaps = queue.coverage_gaps()

    assert gaps == ["adversarial", "unanswerable"]


def test_a_complete_queue_reports_no_gaps():
    queue = ReviewQueue(
        items=[
            GoldenItem(id=f"g{i}", question="q", reference_answer="a", category=c)
            for i, c in enumerate(
                ["single_hop", "multi_hop", "unanswerable", "adversarial"]
            )
        ]
    )
    assert queue.coverage_gaps() == []


def test_review_queue_writes_reviewable_jsonl(tmp_path):
    queue = ReviewQueue(
        items=[GoldenItem(id="gen-001", question="What is HNSW?", reference_answer="TODO")]
    )
    path = queue.to_jsonl(tmp_path / "generated.jsonl")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert rows[0]["id"] == "gen-001"
    assert rows[0]["question"] == "What is HNSW?"


def test_empty_queue_writes_an_empty_file(tmp_path):
    assert ReviewQueue(items=[]).to_jsonl(tmp_path / "none.jsonl").read_text() == ""


@pytest.mark.judge
def test_generate_a_real_testset_from_the_corpus():
    """Requires a live model. Slow -- the generator builds a knowledge graph
    over the corpus before it produces anything."""
    from advanced_ragas import generate_testset
    from ragas_setup import ragas_ready

    ready, reason = ragas_ready()
    if not ready:
        pytest.skip(reason)

    testset = generate_testset(size=4, docs=corpus_as_langchain_documents(limit=3))
    queue = ReviewQueue.from_testset(testset)

    print(f"\ngenerated {len(queue.items)} questions")
    for item in queue.items:
        print(f"  {item.id}: {item.question}")
    print(f"coverage gaps: {queue.coverage_gaps()}")

    assert queue.items
    assert all(i.reference_answer.startswith("TODO") for i in queue.items)
    assert queue.coverage_gaps(), (
        "the generator produced every category, which would be surprising -- "
        "verify it really invented unanswerable and adversarial questions"
    )
