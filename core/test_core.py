"""
Tests for the shared spine. ALL FAST TIER -- no model, no network.

Run:  pytest core/ -v

Read these before the lesson tests. They establish the two things every later
lesson depends on: that the dataset is internally consistent, and that the
retrieval metrics compute what they claim to.
"""

from __future__ import annotations

import pytest

from core import compat
from core.golden import REFUSAL, VALID_CATEGORIES, corpus_doc_ids, iter_corpus
from core.metrics import (
    evaluate_retrieval,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from core.providers import LexicalEmbeddings, cosine_similarity, tokenize

# ===========================================================================
# COMPATIBILITY SHIM
# ===========================================================================


def test_shim_reports_whether_it_is_still_needed():
    """Tripwire for the ragas/langchain-community breakage.

    core/compat.py monkeypatches a missing module so `import ragas` works.
    Monkeypatches must not outlive their cause. When a future ragas release
    drops the stale `langchain_community.chat_models.vertexai` import, or the
    module comes back, `is_shim_needed()` flips to False -- and this test tells
    you to go delete the shim rather than carrying it forever.
    """
    still_needed = compat.is_shim_needed()
    if not still_needed:
        pytest.fail(
            "core/compat.py's ragas shim is no longer necessary: "
            "langchain_community.chat_models.vertexai now imports cleanly. "
            "Delete install_ragas_langchain_shim() and this test."
        )
    assert still_needed is True


def test_is_shim_needed_ignores_our_own_stub():
    """Regression test for a bug this repo actually shipped and then caught.

    The naive implementation of is_shim_needed() just tries to import the
    module. But once the shim is installed, OUR STUB is in sys.modules, so the
    import succeeds -- and the function reports "shim not needed" precisely
    because the shim is working.

    The symptom was a test that passed or failed depending on whether anything
    had imported ragas earlier in the same pytest session. Order-dependent
    tests are the worst kind, because they look flaky rather than wrong.

    The fix is a marker attribute on the stub, asserted here.
    """
    compat.bootstrap()  # guarantee the stub is installed
    assert compat.is_shim_needed() is True, (
        "is_shim_needed() was fooled by our own stub -- the marker check broke"
    )


def test_bootstrap_is_idempotent():
    """Calling bootstrap() repeatedly must be harmless.

    Every lesson in 05_ragas calls it at import time, so it will run many times
    in one pytest session.
    """
    compat.bootstrap()
    compat.bootstrap()
    import sys

    stub = sys.modules["langchain_community.chat_models.vertexai"]
    # The placeholder must refuse to be constructed, loudly.
    with pytest.raises(NotImplementedError, match="not available in this project"):
        stub.ChatVertexAI()


# ===========================================================================
# GOLDEN DATASET INTEGRITY
# ===========================================================================
# These are the cheapest high-value tests in the repo. A dataset with a typo in
# a doc_id silently makes recall@k unreachable for that question, and you will
# spend an afternoon debugging the retriever instead of the label.
# ===========================================================================


def test_golden_dataset_loads(golden):
    assert len(golden) >= 30, "dataset is too small to detect anything useful"


def test_every_item_has_a_unique_id(golden):
    ids = [item.id for item in golden]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate golden ids: {duplicates}"


def test_every_reference_doc_id_exists_in_the_corpus(golden):
    """Catch typos in labels.

    If a golden item points at "retreival" instead of "retrieval", recall@k for
    that question is permanently 0 no matter how good the retriever is. Without
    this test that looks like a retrieval bug.
    """
    known = corpus_doc_ids()
    for item in golden:
        for doc_id in item.reference_doc_ids:
            assert doc_id in known, (
                f"golden item {item.id} references unknown document {doc_id!r}. "
                f"Known documents: {sorted(known)}"
            )


def test_categories_are_valid(golden):
    for item in golden:
        assert item.category in VALID_CATEGORIES, f"{item.id} has bad category {item.category!r}"


def test_unanswerable_items_have_no_reference_docs(golden):
    """An unanswerable question cannot have a correct source document.

    If it did, it would be answerable. This test enforces the definition, which
    matters because recall@k treats an empty relevant set as vacuously satisfied.
    """
    for item in golden:
        if item.category == "unanswerable":
            assert item.reference_doc_ids == [], (
                f"{item.id} is marked unanswerable but cites {item.reference_doc_ids}"
            )
            assert item.reference_answer == REFUSAL


def test_answerable_items_cite_at_least_one_document(golden):
    for item in golden:
        if item.is_answerable:
            assert item.reference_doc_ids, f"{item.id} is answerable but cites no source document"


def test_dataset_has_all_four_categories(golden):
    """The categories that make a dataset adversarial must actually be present.

    A dataset of only single_hop questions cannot detect hallucination on
    unanswerable input or sycophancy on a false premise -- the two failures that
    matter most in production.
    """
    present = {item.category for item in golden}
    assert present == VALID_CATEGORIES, f"missing categories: {VALID_CATEGORIES - present}"


def test_enough_unanswerable_coverage(golden):
    """At least 10% of the dataset should be unanswerable.

    Arbitrary but deliberate: too few and the mean refusal rate is dominated by
    noise from a handful of items.
    """
    unanswerable = [i for i in golden if not i.is_answerable]
    ratio = len(unanswerable) / len(golden)
    assert ratio >= 0.10, f"only {ratio:.0%} of the dataset is unanswerable; aim for >=10%"


def test_multi_hop_items_span_multiple_documents_or_are_synthesis(golden):
    """Multi-hop items must genuinely need more than a lookup.

    Some legitimately cite one document but require joining two separate
    sections of it, so we assert difficulty instead of blindly requiring two
    doc_ids -- and record why in the item's rationale.
    """
    for item in golden:
        if item.category == "multi_hop":
            assert item.difficulty >= 2, f"{item.id} is multi_hop but marked trivially easy"
            assert item.rationale, f"{item.id} must explain what it is designed to catch"


def test_corpus_is_non_trivial():
    docs = dict(iter_corpus())
    assert len(docs) >= 6, "corpus too small for retrieval to be interesting"
    for doc_id, text in docs.items():
        assert len(text) > 800, f"{doc_id}.md is too short to require chunking"


# ===========================================================================
# RETRIEVAL METRICS
# ===========================================================================
# Exact-value assertions. This is only possible because these metrics involve
# no model -- and it is exactly why they, not faithfulness, belong in CI.
# ===========================================================================


def test_recall_at_k_basic():
    # Retrieved 3, of the 2 relevant documents we found 1.
    assert recall_at_k(["a", "b", "c"], ["a", "d"]) == 0.5
    assert recall_at_k(["a", "b", "c"], ["a", "b"]) == 1.0
    assert recall_at_k(["x", "y"], ["a"]) == 0.0


def test_recall_respects_the_k_cutoff():
    """Relevant doc sits at position 3; with k=2 we must not count it."""
    assert recall_at_k(["x", "y", "a"], ["a"], k=2) == 0.0
    assert recall_at_k(["x", "y", "a"], ["a"], k=3) == 1.0


def test_recall_is_one_when_nothing_is_relevant():
    """The unanswerable-question convention, asserted explicitly.

    An empty relevant set means there was nothing to find, so retrieval cannot
    have missed anything. Returning 0.0 here would penalise the system for
    correctly having no source to fetch.
    """
    assert recall_at_k(["a", "b"], []) == 1.0


def test_precision_at_k_basic():
    assert precision_at_k(["a", "b", "c", "d"], ["a", "c"]) == 0.5
    assert precision_at_k([], ["a"]) == 0.0


def test_precision_is_zero_for_unanswerable_questions():
    """Deliberately asymmetric with recall -- and correct.

    For an unanswerable question every retrieved document IS noise, so
    precision is genuinely 0. The system is rescued not by retrieval but by
    refusing to answer, which is a generation behaviour measured elsewhere.
    """
    assert precision_at_k(["a", "b"], []) == 0.0


def test_reciprocal_rank_positions():
    """The worked example from the corpus: position 4 gives 0.25."""
    assert reciprocal_rank(["a"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y", "z", "a"], ["a"]) == 0.25
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_mrr_averages_across_queries():
    rankings = [
        (["a", "x"], ["a"]),        # rr = 1.0
        (["x", "y", "b"], ["b"]),   # rr = 1/3
    ]
    assert mean_reciprocal_rank(rankings) == pytest.approx((1.0 + 1 / 3) / 2)


def test_mrr_distinguishes_orderings_that_recall_cannot():
    """The reason MRR earns its place alongside recall.

    Both orderings retrieve the relevant document within the top 3, so recall@3
    is 1.0 for each. Only MRR notices that one puts it first and the other puts
    it last -- which matters because of the 'lost in the middle' effect.
    """
    good = (["a", "x", "y"], ["a"])
    bad = (["x", "y", "a"], ["a"])

    assert recall_at_k(*good, k=3) == recall_at_k(*bad, k=3) == 1.0
    assert reciprocal_rank(*good) > reciprocal_rank(*bad)


def test_hit_rate_is_binary():
    assert hit_rate(["x", "a"], ["a"]) == 1.0
    assert hit_rate(["x", "y"], ["a"]) == 0.0


def test_ndcg_rewards_higher_placement():
    perfect = ndcg_at_k(["a", "b", "x"], ["a", "b"])
    worse = ndcg_at_k(["x", "a", "b"], ["a", "b"])
    assert perfect == pytest.approx(1.0)
    assert worse < perfect


def test_evaluate_retrieval_reports_per_item_failures():
    """The aggregate report must name the questions that failed, not just a mean."""
    report = evaluate_retrieval(
        {
            "q1": (["a"], ["a"]),          # perfect
            "q2": (["x"], ["b"]),          # complete miss
        }
    )
    assert report.n == 2
    assert report.recall == pytest.approx(0.5)
    assert report.failures() == ["q2"]
    assert "MRR" in report.format_table()


# ===========================================================================
# LEXICAL EMBEDDINGS
# ===========================================================================


def test_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("The CAT sat on the mat!") == ["the", "cat", "sat", "on", "the", "mat"]


def test_embeddings_are_deterministic_across_instances():
    """Non-negotiable for reproducible evaluation.

    Python randomises str hashing per process (PYTHONHASHSEED), so using the
    builtin hash() would make vectors differ between runs -- and therefore make
    two eval runs incomparable. core.providers uses FNV-1a instead. This test
    is the guard on that decision.
    """
    docs = ["chunk overlap protects facts at boundaries", "cosine similarity measures direction"]

    a = LexicalEmbeddings(dim=256)
    b = LexicalEmbeddings(dim=256)
    assert a.embed_documents(docs) == b.embed_documents(docs)


def test_embeddings_are_l2_normalised(lexical_embeddings):
    import numpy as np

    vectors = lexical_embeddings.embed_documents(["hello world", "goodbye world"])
    for vec in vectors:
        assert np.linalg.norm(vec) == pytest.approx(1.0, abs=1e-9)


def test_lexical_embeddings_carry_real_semantic_signal(lexical_embeddings):
    """The whole reason we do not use random fake embeddings in the fast tier.

    A retrieval test against random vectors proves nothing. These vectors must
    actually rank a topically-related document above an unrelated one.
    """
    docs = [
        "Chunk overlap means consecutive chunks share text at their boundary.",
        "HNSW is a graph based approximate nearest neighbour index.",
    ]
    lexical_embeddings.embed_documents(docs)
    doc_vectors = [lexical_embeddings.embed_query(d) for d in docs]
    query = lexical_embeddings.embed_query("what does chunk overlap do?")

    assert cosine_similarity(query, doc_vectors[0]) > cosine_similarity(query, doc_vectors[1])


def test_lexical_embeddings_fail_on_synonyms(lexical_embeddings):
    """The documented limitation, asserted so you SEE it rather than read it.

    "car" and "automobile" mean the same thing but hash to different slots, so
    a purely lexical model scores them as unrelated. This failure is the entire
    reason neural embedding models exist -- lesson 01 reruns this exact
    comparison against nomic-embed-text, where it passes.
    """
    docs = ["I drove my car to work.", "Cryptography relies on prime numbers."]
    lexical_embeddings.embed_documents(docs)
    doc_vectors = [lexical_embeddings.embed_query(d) for d in docs]
    query = lexical_embeddings.embed_query("I drove my automobile to work.")

    sim_to_paraphrase = cosine_similarity(query, doc_vectors[0])
    # "drove/my/to/work" still overlap, so it is not zero -- but the *synonym*
    # itself contributes nothing, which is the point.
    assert sim_to_paraphrase < 0.99, "lexical matching should not achieve near-identity on a synonym swap"


def test_cosine_similarity_edge_cases():
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0  # no division by zero
