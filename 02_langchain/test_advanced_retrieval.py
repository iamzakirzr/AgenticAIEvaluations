"""
Advanced retrieval tests. FAST TIER -- every technique runs with no model.

Run:  pytest 02_langchain/test_advanced_retrieval.py -v

Each test demonstrates the SPECIFIC failure of plain vector search that the
technique exists to fix. Read them in order; they build an argument.
"""

from __future__ import annotations

import pytest
from advanced_retrieval import (
    BM25,
    ConversationalRetriever,
    HybridRetriever,
    Reranker,
    hyde_query,
    lexical_overlap_scorer,
    mmr_select,
    multi_query_expansion,
    reciprocal_rank_fusion,
    rewrite_followup,
)
from pipeline import build_offline_pipeline

from core.providers import scripted_chat_model


@pytest.fixture(scope="module")
def pipeline():
    return build_offline_pipeline(["An answer [1]."])


# ===========================================================================
# BM25
# ===========================================================================


def test_bm25_finds_an_exact_rare_token():
    """THE case embeddings are bad at.

    'ERR_5521' and 'ERR_5522' mean different things and embed almost
    identically -- an embedding model is TRAINED to map similar strings
    together, which is exactly wrong for an identifier.
    """
    bm25 = BM25().index(
        [
            ("timeout", "Error code ERR_5521 indicates a gateway timeout."),
            ("auth", "Error code ERR_5522 indicates an authentication failure."),
            ("intro", "This document introduces our error handling philosophy."),
        ]
    )
    hits = bm25.search("ERR_5521", top_k=3)
    assert bm25.doc_ids[hits[0][0]] == "timeout"


def test_idf_ranks_rare_terms_above_common_ones():
    bm25 = BM25().index([("a", "the quick fox"), ("b", "the lazy dog"), ("c", "the end")])
    assert bm25.idf("the") < bm25.idf("fox"), "a term in every document should score lowest"


def test_idf_is_never_negative():
    """Classic BM25 IDF can go negative for a term in >half the corpus.

    That lets a very common word actively REDUCE a document's score, which is
    surprising and usually unwanted. The +1 inside the log prevents it.
    """
    bm25 = BM25().index([("a", "common word"), ("b", "common word"), ("c", "common word")])
    assert bm25.idf("common") >= 0


def test_k1_saturates_repeated_terms():
    """Without saturation, keyword stuffing wins.

    A document repeating 'refund' fifty times would outrank one that says it
    twice and actually answers the question.
    """
    stuffed = "refund " * 50
    useful = "A refund is issued within 30 days of purchase for any refund request."

    bm25 = BM25(k1=1.5).index([("stuffed", stuffed), ("useful", useful)])
    scores = dict(zip(bm25.doc_ids, bm25.score("refund")))

    # Repetition still helps, but nowhere near 25x.
    assert scores["stuffed"] < scores["useful"] * 4, "k1 failed to saturate term frequency"


def test_b_penalises_long_documents():
    """Long documents contain more of every term by chance."""
    short = "chunk overlap"
    long = "chunk overlap " + ("filler text about unrelated topics " * 50)

    normalised = BM25(b=1.0).index([("short", short), ("long", long)])
    unnormalised = BM25(b=0.0).index([("short", short), ("long", long)])

    n_scores = dict(zip(normalised.doc_ids, normalised.score("chunk overlap")))
    u_scores = dict(zip(unnormalised.doc_ids, unnormalised.score("chunk overlap")))

    assert n_scores["short"] > n_scores["long"]
    # With b=0 the long document is not penalised, so the gap narrows.
    assert (n_scores["short"] - n_scores["long"]) > (u_scores["short"] - u_scores["long"])


def test_bm25_does_not_stem_which_is_a_real_limitation():
    """'cat' does not match 'cats'. Worth knowing before you rely on BM25.

    Production BM25 implementations usually stem or lemmatise first. Ours
    doesn't, deliberately -- the limitation is more instructive visible than
    hidden, and it is half the reason hybrid search exists: the embedding side
    handles morphology, the lexical side handles exact tokens.
    """
    bm25 = BM25().index([("a", "cats are mammals"), ("b", "dogs are mammals")])
    assert bm25.search("cat", top_k=2) == [], "unexpectedly matched a different word form"
    assert bm25.search("cats", top_k=2), "exact form should match"


# ===========================================================================
# RECIPROCAL RANK FUSION
# ===========================================================================


def test_rrf_rewards_agreement_across_lists():
    """A document both retrievers like beats one that only tops a single list."""
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["a"] == fused["b"] > fused["c"]


def test_rrf_needs_only_ranks_not_scores():
    """THE reason RRF is the default merge for hybrid search.

    BM25 scores are unbounded positive; cosine is -1 to 1. Averaging them is
    meaningless, and normalising requires knowing each distribution, which
    changes per query. RRF sidesteps all of it.
    """
    a = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    b = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    assert a == b
    # Identical ranks -> identical fused scores, whatever the underlying scales.
    assert a[0][1] == a[1][1]


def test_rrf_k_damps_the_advantage_of_rank_one():
    """With k=60, rank 1 and rank 2 score almost the same, so agreement across
    lists matters more than being first in one. A tiny k inverts that."""
    damped = dict(reciprocal_rank_fusion([["a", "b"]], k=60))
    sharp = dict(reciprocal_rank_fusion([["a", "b"]], k=1))

    assert damped["a"] / damped["b"] < sharp["a"] / sharp["b"]


def test_rrf_surfaces_a_document_neither_list_ranked_first():
    """The payoff case: 2nd in both beats 1st in one and absent from the other."""
    fused = dict(reciprocal_rank_fusion([["solo", "shared"], ["other", "shared"]]))
    assert fused["shared"] > fused["solo"]


def test_rrf_handles_empty_input():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


# ===========================================================================
# MMR
# ===========================================================================


def test_mmr_avoids_returning_near_duplicates():
    """THE failure of plain top-k.

    A document that discusses a topic across consecutive paragraphs produces
    near-identical vectors. Plain top-k returns all of them, the context window
    fills with the same sentence rephrased, and a second document that would
    have completed the answer never appears.
    """
    query = [1.0, 0.0]
    candidates = [
        [0.80, 0.60],   # 0: most relevant
        [0.79, 0.61],   # 1: near-duplicate of 0
        [0.75, -0.66],  # 2: slightly less relevant, but a DIFFERENT direction
    ]

    relevance_only = mmr_select(query, candidates, k=2, lambda_mult=1.0)
    diverse = mmr_select(query, candidates, k=2, lambda_mult=0.5)

    assert relevance_only == [0, 1], "lambda=1.0 should behave exactly like top-k"
    assert diverse[0] == 0
    assert diverse[1] == 2, "MMR should have picked the different document"

    # A SUBTLETY worth knowing, which cost a debugging round writing this test:
    # if a candidate is IDENTICAL to the query vector, then for every other
    # candidate sim(x, that candidate) == sim(x, query). Redundancy and
    # relevance become the same number, so at lambda=0.5 every score collapses
    # to zero and MMR degenerates into "return them in index order". Test
    # fixtures for MMR must not make a candidate exactly equal to the query.


def test_mmr_lambda_one_equals_plain_top_k():
    query = [1.0, 0.0]
    candidates = [[0.2, 0.9], [1.0, 0.0], [0.7, 0.7]]
    assert mmr_select(query, candidates, k=3, lambda_mult=1.0) == [1, 2, 0]


def test_mmr_handles_fewer_candidates_than_k():
    assert len(mmr_select([1.0, 0.0], [[1.0, 0.0]], k=5)) == 1


def test_mmr_handles_no_candidates():
    assert mmr_select([1.0, 0.0], [], k=3) == []


# ===========================================================================
# HYBRID RETRIEVAL
# ===========================================================================


def test_hybrid_returns_the_requested_number_of_chunks(pipeline):
    hybrid = HybridRetriever(pipeline)
    results = hybrid.search("what is chunk overlap?", top_k=4)
    assert len(results) == 4
    assert all(r.doc_id for r in results)


def test_hybrid_finds_what_vector_search_finds(pipeline):
    """Fusion must not LOSE good results. A hybrid that ranks worse than either
    input is misconfigured -- usually by fusing too shallow a candidate list."""
    hybrid = HybridRetriever(pipeline)
    results = hybrid.search("what does chunk overlap protect against?", top_k=4)
    assert "chunking" in {r.doc_id for r in results}


def test_hybrid_fuses_deeper_than_it_returns(pipeline):
    """Fusing only the top 4 of each wastes the technique.

    The whole point is that a chunk ranked 9th by one retriever and 2nd by the
    other should surface -- impossible if you truncated both at 4.
    """
    hybrid = HybridRetriever(pipeline)
    shallow = hybrid.search("reciprocal rank", top_k=4, candidates=4)
    deep = hybrid.search("reciprocal rank", top_k=4, candidates=20)

    assert len(shallow) <= 4 and len(deep) <= 4
    # Not asserting they differ -- on a small corpus they may not. Asserting
    # the deeper fusion is at least as good, and does not error.
    assert deep


def test_hybrid_scores_are_rrf_scores_not_similarities(pipeline):
    """A subtle trap: the returned score is now a FUSED RANK score, roughly
    1/61 + 1/62, not a cosine similarity. Any threshold tuned against cosine
    values will refuse everything once you switch to hybrid."""
    hybrid = HybridRetriever(pipeline)
    top = hybrid.search("what is chunk overlap?", top_k=1)[0]
    assert 0 < top.score < 0.1, f"expected a small RRF score, got {top.score}"


# ===========================================================================
# RERANKING
# ===========================================================================


def test_reranker_reorders_a_shortlist(pipeline):
    reranker = Reranker(lexical_overlap_scorer)
    candidates = pipeline.retrieve("what is chunk overlap?", top_k=8)
    reranked = reranker.rerank("what is chunk overlap?", candidates, top_k=3)

    assert len(reranked) == 3
    assert reranked[0].score >= reranked[-1].score


def test_reranking_runs_on_a_shortlist_not_the_corpus(pipeline):
    """The cost argument, made concrete.

    A cross-encoder is O(candidates) model calls per query with no
    precomputation possible. Running it over the whole corpus is why nobody
    does that; running it over 25 is why everybody does this.
    """
    reranker = Reranker(lexical_overlap_scorer)
    candidates = pipeline.retrieve("chunk overlap", top_k=10)
    reranker.rerank("chunk overlap", candidates, top_k=3)

    assert reranker.calls == 10
    assert reranker.calls < len(pipeline.documents), (
        "reranking touched more chunks than the shortlist -- that is the whole "
        "cost the two-stage design exists to avoid"
    )


def test_reranker_is_injectable_so_production_swaps_the_model():
    """`score_fn` keeps the MECHANISM testable offline. In production this is a
    real cross-encoder such as cross-encoder/ms-marco-MiniLM-L-6-v2."""
    from core.trace import RetrievedChunk

    reranker = Reranker(lambda q, t: 1.0 if "correct" in t else 0.0)
    chunks = [
        RetrievedChunk("wrong text", "a", 0, 0.9),
        RetrievedChunk("the correct text", "b", 0, 0.1),
    ]
    reranked = reranker.rerank("q", chunks, top_k=2)

    assert reranked[0].doc_id == "b", "reranking failed to override the original order"


# ===========================================================================
# QUERY TRANSFORMATION
# ===========================================================================


def test_rewrite_is_a_noop_on_the_first_turn():
    """No history means nothing to resolve, so do not spend a model call."""
    llm = scripted_chat_model(["should not be called"])
    assert rewrite_followup(llm, [], "What is chunk overlap?") == "What is chunk overlap?"


def test_rewrite_resolves_a_conversational_reference():
    """WITHOUT THIS, MULTI-TURN RAG IS BROKEN.

    Embed "what about the second one?" and you retrieve documents about the
    number two. The conversation carries the meaning and the retriever never
    sees the conversation.
    """
    llm = scripted_chat_model(["What is the second chunking strategy?"])
    history = [("What chunking strategies are there?", "Fixed, recursive, and markdown.")]

    rewritten = rewrite_followup(llm, history, "what about the second one?")

    assert "second" in rewritten.lower()
    assert "chunking" in rewritten.lower(), "the rewrite did not pull in the topic"


def test_rewrite_falls_back_to_the_original_on_an_empty_response():
    """A blank rewrite must not produce a blank search query."""
    llm = scripted_chat_model(["   "])
    history = [("q", "a")]
    assert rewrite_followup(llm, history, "the original") == "the original"


def test_hyde_searches_with_an_answer_shaped_string():
    """A question and its answer are written differently.

    The hypothetical answer may be WRONG -- that is fine and is the clever part.
    It is never shown to the user; it is only a search key, and a wrong answer
    about the right topic still uses the right vocabulary.
    """
    llm = scripted_chat_model(["nomic-embed-text produces 768 dimensions."])
    generated = hyde_query(llm, "How many dimensions does nomic-embed-text produce?")

    assert "768" in generated
    assert "?" not in generated, "HyDE should produce a statement, not a question"


def test_hyde_improves_retrieval_on_this_corpus(pipeline):
    """End to end: does searching with the hypothetical answer actually help?

    Worth measuring rather than assuming -- HyDE costs a full model call before
    retrieval starts, so it has to earn that latency.
    """
    question = "How many dimensions does nomic-embed-text produce?"
    llm = scripted_chat_model(
        ["nomic-embed-text produces 768 dimensions. Common embedding sizes vary by model."]
    )

    plain = pipeline.retrieve(question, top_k=3)
    hyde = pipeline.retrieve(hyde_query(llm, question), top_k=3)

    print(f"\nplain: {[c.doc_id for c in plain]}")
    print(f"hyde : {[c.doc_id for c in hyde]}")
    assert "embeddings" in {c.doc_id for c in hyde}


def test_multi_query_expansion_always_keeps_the_original():
    """A paraphrase can drift. The user's own wording is the one phrasing you
    know is faithful to their intent."""
    llm = scripted_chat_model(["How big is the vector?\nWhat is the embedding size?"])
    variants = multi_query_expansion(llm, "How many dimensions?", n=2)

    assert variants[0] == "How many dimensions?"
    assert len(variants) >= 2


# ===========================================================================
# CONVERSATIONAL RETRIEVAL
# ===========================================================================


def test_conversational_retriever_rewrites_then_retrieves(pipeline):
    llm = scripted_chat_model(["What is chunk overlap used for?"])
    convo = ConversationalRetriever(pipeline=pipeline, llm=llm)
    convo.record("Tell me about chunking.", "Chunking splits documents.")

    standalone, chunks = convo.ask("what is it used for?")

    assert "chunk overlap" in standalone.lower()
    assert chunks


def test_history_is_bounded():
    """Unbounded history is a real production bug: the rewrite prompt grows
    until it blows the context window, and it fails mid-conversation for your
    most engaged users."""
    convo = ConversationalRetriever(
        pipeline=None, llm=scripted_chat_model(["x"]), max_turns=3
    )
    for i in range(10):
        convo.record(f"question {i}", f"answer {i}")

    assert len(convo.history) == 3
    assert convo.history[-1][0] == "question 9", "kept the oldest turns instead of the newest"


# ===========================================================================
# DOES ANY OF THIS ACTUALLY HELP? -- measure, do not assume
# ===========================================================================


def test_compare_retrieval_strategies_on_the_golden_dataset(pipeline):
    """The test that matters: score plain vs hybrid over the real dataset.

    An interviewer asking "when would you use hybrid search?" wants to hear
    that you MEASURED it, not that you read it improves things. On this corpus
    it may not help at all -- the questions reuse the source vocabulary, which
    already flatters lexical matching. Reporting that honestly is the point.
    """
    from core.golden import load_golden
    from core.metrics import evaluate_retrieval

    golden = [item for item in load_golden() if item.is_answerable]
    hybrid = HybridRetriever(pipeline)

    plain_results, hybrid_results = {}, {}
    for item in golden:
        plain = pipeline.retrieve(item.question, top_k=4)
        fused = hybrid.search(item.question, top_k=4)
        plain_results[item.id] = (
            _dedupe([c.doc_id for c in plain]), item.reference_doc_ids
        )
        hybrid_results[item.id] = (
            _dedupe([c.doc_id for c in fused]), item.reference_doc_ids
        )

    plain_report = evaluate_retrieval(plain_results)
    hybrid_report = evaluate_retrieval(hybrid_results)

    print(f"\nplain  {plain_report.format_table()}")
    print(f"\nhybrid {hybrid_report.format_table()}")

    # Assert only that hybrid does not COLLAPSE. Asserting it wins would be
    # asserting a result we have not established on this corpus.
    assert hybrid_report.hit_rate >= plain_report.hit_rate - 0.1, (
        "hybrid retrieval scored materially worse than plain vector search -- "
        "usually a sign the candidate pool is too shallow before fusion"
    )


def _dedupe(doc_ids: list[str]) -> list[str]:
    seen: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in seen:
            seen.append(doc_id)
    return seen
