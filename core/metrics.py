"""
core.metrics -- retrieval metrics that require NO language model.

=============================================================================
WHY THESE MATTER MORE THAN THEY LOOK
=============================================================================
Every metric in this file is computed from two things: the ranked list of
document IDs the retriever returned, and the list of document IDs we know are
correct. No LLM. No embeddings. No network.

That has three consequences that make these the backbone of a real eval setup:

  1. DETERMINISTIC. The same input always yields the same number. You can
     assert on exact values in a unit test, which you can never do with a
     judged metric.

  2. FREE AND INSTANT. Thousands of queries score in milliseconds. This is
     what makes it viable to gate every pull request on retrieval quality.

  3. DIAGNOSTIC. They isolate the retriever. If recall@k is 0.4, the generator
     is irrelevant -- the answer was never in the context. Judged metrics
     cannot tell you this cleanly because they blend retrieval and generation.

Beginners skip straight to faithfulness and answer relevancy because those
sound more sophisticated. That is backwards. If you fix retrieval first, many
generation problems disappear on their own.

=============================================================================
A NOTE ON GRANULARITY
=============================================================================
These operate on DOCUMENT ids, not chunk ids. Document-level is the right
granularity for a golden dataset because labelling "chunk 7 of doc 3 is
relevant" is brittle -- it breaks the moment you change chunk size, which is
the exact experiment you most want to run. Labelling "the answer is in
retrieval.md" survives re-chunking.
=============================================================================
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Fraction of the relevant documents that appear in the top k results.

    Answers: "did we even fetch the answer?"

    Returns 1.0 when there are no relevant documents to find. That convention
    matters: for an UNANSWERABLE question the relevant set is empty, and
    "retrieved everything that was needed" is vacuously true. Returning 0.0
    there would punish correct behaviour.

    >>> recall_at_k(["a", "b", "c"], ["a", "d"])   # found 1 of 2
    0.5
    >>> recall_at_k(["a", "b"], [])                # nothing to find
    1.0
    """
    if not relevant:
        return 1.0
    top = list(retrieved)[: k if k is not None else len(retrieved)]
    found = len(set(top) & set(relevant))
    return found / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Fraction of the top k results that are relevant.

    Answers: "how much of what we fetched was worth fetching?"

    Note the asymmetry with recall on empty relevant sets. Here, retrieving
    documents for an unanswerable question means everything retrieved is
    irrelevant, so precision is genuinely 0.0. That is not a bug -- the
    retriever really did return only noise. What saves the system is refusing
    to answer despite that noise, which is a GENERATION behaviour measured
    elsewhere.

    >>> precision_at_k(["a", "b", "c", "d"], ["a", "c"])
    0.5
    """
    top = list(retrieved)[: k if k is not None else len(retrieved)]
    if not top:
        return 0.0
    if not relevant:
        return 0.0
    hits = sum(1 for doc in top if doc in set(relevant))
    return hits / len(top)


def hit_rate(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """1.0 if at least one relevant document is in the top k, else 0.0.

    Recall@k collapsed to a yes/no. Useful as a coarse gate: "did retrieval
    completely whiff on this question?" Averaged over a dataset it is the
    fraction of questions where the system had any chance at all.
    """
    if not relevant:
        return 1.0
    top = set(list(retrieved)[: k if k is not None else len(retrieved)])
    return 1.0 if top & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1 / (rank of the first relevant document), or 0.0 if none appear.

    Ranks are 1-based: position 1 -> 1.0, position 2 -> 0.5, position 4 -> 0.25.

    WHY RANK POSITION MATTERS AND IS NOT PEDANTRY: generators attend most
    strongly to the start of their context ("lost in the middle"). A system
    that always puts the right chunk 5th has the same recall@5 as one that puts
    it 1st, but produces worse answers. Recall cannot see that difference; MRR
    can.

    >>> reciprocal_rank(["x", "y", "a"], ["a"])
    0.3333333333333333
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    for index, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(rankings: Sequence[tuple[Sequence[str], Sequence[str]]]) -> float:
    """Average reciprocal rank over many (retrieved, relevant) pairs."""
    if not rankings:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in rankings) / len(rankings)


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Normalised Discounted Cumulative Gain with binary relevance.

    DCG sums the gain of each result discounted by log2 of its position, so a
    hit at rank 1 is worth 1/log2(2) = 1.0 and a hit at rank 4 is worth
    1/log2(5) = 0.43. Normalising by the ideal ordering (IDCG) puts the score
    on a 0-1 scale so it is comparable across queries with different numbers of
    relevant documents.

    nDCG is the standard information-retrieval metric and handles GRADED
    relevance too, though we use binary here because our golden dataset labels
    documents as relevant or not, with no in-between.
    """
    top = list(retrieved)[: k if k is not None else len(retrieved)]
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0

    dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(top, start=1) if doc in relevant_set)
    # Ideal DCG: every relevant document packed into the highest positions.
    ideal_hits = min(len(relevant_set), len(top))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class RetrievalReport:
    """Aggregate retrieval quality across a whole dataset.

    Carries the per-question breakdown alongside the means, because -- as
    core/corpus/rag_metrics.md explains -- a mean hides bimodal behaviour, and
    bimodal behaviour needs a completely different fix from uniform mediocrity.
    """

    n: int
    recall: float
    precision: float
    hit_rate: float
    mrr: float
    ndcg: float
    per_item: dict[str, dict[str, float]]

    def failures(self, threshold: float = 1.0) -> list[str]:
        """IDs of questions whose recall fell below ``threshold``.

        This is what you actually read when a CI gate fails: not the average,
        but which specific questions broke.
        """
        return [
            item_id
            for item_id, scores in self.per_item.items()
            if scores["recall"] < threshold
        ]

    def format_table(self) -> str:
        lines = [
            f"Retrieval over {self.n} questions",
            f"  recall@k    {self.recall:.3f}",
            f"  precision@k {self.precision:.3f}",
            f"  hit rate    {self.hit_rate:.3f}",
            f"  MRR         {self.mrr:.3f}",
            f"  nDCG@k      {self.ndcg:.3f}",
        ]
        return "\n".join(lines)


def evaluate_retrieval(
    results: dict[str, tuple[Sequence[str], Sequence[str]]],
    k: int | None = None,
) -> RetrievalReport:
    """Score a whole dataset at once.

    Args:
        results: ``{item_id: (retrieved_doc_ids, relevant_doc_ids)}``
        k: cutoff; None means use everything retrieved.

    Returns:
        A RetrievalReport with both aggregate means and the per-question detail.
    """
    per_item: dict[str, dict[str, float]] = {}
    for item_id, (retrieved, relevant) in results.items():
        per_item[item_id] = {
            "recall": recall_at_k(retrieved, relevant, k),
            "precision": precision_at_k(retrieved, relevant, k),
            "hit_rate": hit_rate(retrieved, relevant, k),
            "rr": reciprocal_rank(retrieved, relevant),
            "ndcg": ndcg_at_k(retrieved, relevant, k),
        }

    n = len(per_item)
    if n == 0:
        return RetrievalReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})

    def mean(key: str) -> float:
        return sum(scores[key] for scores in per_item.values()) / n

    return RetrievalReport(
        n=n,
        recall=mean("recall"),
        precision=mean("precision"),
        hit_rate=mean("hit_rate"),
        mrr=mean("rr"),
        ndcg=mean("ndcg"),
        per_item=per_item,
    )
