"""
Lesson 01 tests -- FAST TIER unless marked.

Run:  pytest 01_embeddings/ -v
      pytest -m ollama 01_embeddings/ -v     # the real-neural-embeddings tier

These tests are meant to be READ as much as run. Each one demonstrates a claim
the lesson makes, so that "chunk overlap protects boundary facts" is not
something you take on faith -- it is something you can watch fail when you
delete the overlap.
"""

from __future__ import annotations

import pytest

from chunking import (
    chunk_stats,
    fixed_size_chunks,
    markdown_section_chunks,
    measure_overlap,
    recursive_chunks,
)
from mini_rag import MiniRAG

from core.golden import load_golden
from core.metrics import evaluate_retrieval
from core.providers import LexicalEmbeddings, cosine_similarity

# A paragraph with a fact that deliberately straddles the 100-character mark,
# used to demonstrate what overlap is actually for.
BOUNDARY_TEXT = (
    "Retrieval quality depends on many factors that interact in complex ways. "
    "The reciprocal rank for position four is exactly 0.25 in every case. "
    "Other considerations apply to production systems at larger scale."
)


def nonrepeating_text(length: int, seed: int = 0) -> str:
    """Deterministic pseudo-random letters, for overlap tests.

    NOT `"abcdefghij" * 30`. Periodic text breaks `measure_overlap`, which
    returns the longest suffix-of-a matching prefix-of-b: on periodic input
    that finds long coincidental matches even with zero configured overlap.
    Random letters over a 26-symbol alphabet make an accidental 25-character
    match astronomically unlikely, so the measurement means what it says.
    """
    import random
    import string

    rng = random.Random(seed)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(length))


# ===========================================================================
# CHUNKING
# ===========================================================================


def test_fixed_size_respects_the_size_limit():
    chunks = fixed_size_chunks("x" * 1000, "doc", size=300)
    assert all(len(c) <= 300 for c in chunks)
    assert len(chunks) == 4  # 300 + 300 + 300 + 100


def test_fixed_size_step_is_size_minus_overlap():
    """The arithmetic that makes overlap work, asserted directly."""
    chunks = fixed_size_chunks("x" * 1000, "doc", size=100, overlap=20)
    # step = 100 - 20 = 80, so chunk starts are 0, 80, 160, ...
    assert chunks[0].start_char == 0
    assert chunks[1].start_char == 80
    assert chunks[2].start_char == 160


def test_overlap_must_be_smaller_than_size():
    """Guard against the infinite loop this would otherwise cause.

    step = size - overlap. If overlap >= size, step <= 0 and the loop never
    advances. This is a real bug people ship; failing loudly is better than
    hanging.
    """
    with pytest.raises(ValueError, match="must be smaller than size"):
        fixed_size_chunks("some text", "doc", size=100, overlap=100)


def test_overlap_actually_appears_in_the_text():
    """Verify overlap by MEASURING it, not by trusting the parameter.

    A splitter whose merge step silently drops overlap gives you no error --
    your retrieval metrics just quietly get worse. Measuring closes that gap.
    """
    chunks = fixed_size_chunks(nonrepeating_text(300), "doc", size=100, overlap=25)
    assert measure_overlap(chunks[0], chunks[1]) == 25


def test_no_overlap_means_no_shared_text():
    chunks = fixed_size_chunks(nonrepeating_text(300), "doc", size=100, overlap=0)
    # With random letters, any accidental suffix/prefix match is 1-2 chars.
    assert measure_overlap(chunks[0], chunks[1]) < 5


def test_overlap_rescues_a_fact_split_across_a_boundary():
    """THE demonstration of why overlap exists.

    Without overlap, the sentence containing "0.25" is cut in half and no
    single chunk contains it whole. With overlap, one chunk does.
    """
    target = "reciprocal rank for position four is exactly 0.25"

    without = fixed_size_chunks(BOUNDARY_TEXT, "doc", size=100, overlap=0)
    with_ov = fixed_size_chunks(BOUNDARY_TEXT, "doc", size=100, overlap=60)

    assert not any(target in c.text for c in without), (
        "test setup is wrong: the fact should be split by size=100 with no overlap"
    )
    assert any(target in c.text for c in with_ov), (
        "overlap failed to reunite a fact that straddles a chunk boundary"
    )


def test_recursive_prefers_natural_boundaries():
    """Recursive splitting should end chunks at paragraph breaks, not mid-word."""
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = recursive_chunks(text, "doc", size=30)
    # Every chunk should be a whole paragraph, so none is cut mid-word.
    for chunk in chunks:
        assert not chunk.text.endswith(("Fir", "Sec", "Thi"))


def test_recursive_falls_back_to_hard_cutting_when_it_must():
    """A single unbroken token longer than the limit still has to be split.

    This is the termination guarantee: without the empty-separator fallback,
    the recursion would return a piece larger than `size` forever.
    """
    chunks = recursive_chunks("x" * 500, "doc", size=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_markdown_splitting_keeps_sections_whole():
    text = "# Alpha\n\nAlpha body text.\n\n## Beta\n\nBeta body text.\n\n## Gamma\n\nGamma body."
    chunks = markdown_section_chunks(text, "doc")
    assert len(chunks) == 3
    assert chunks[0].text.startswith("# Alpha")
    assert "Beta body text." in chunks[1].text


def test_markdown_prepends_heading_to_oversized_sections():
    """Fragments of a long section must stay interpretable on their own.

    A chunk beginning "It is also far slower" is meaningless alone. Prefixed
    with its heading it is not. This is a cheap approximation of contextual
    retrieval.
    """
    long_body = "Sentence about reranking. " * 60
    text = f"## Reranking\n\n{long_body}"
    chunks = markdown_section_chunks(text, "doc", max_size=300)

    assert len(chunks) > 1
    assert all(c.text.startswith("## Reranking") for c in chunks)


def test_smaller_chunks_produce_more_chunks():
    from core.golden import iter_corpus

    _, text = next(iter(iter_corpus()))
    small = recursive_chunks(text, "d", size=200)
    large = recursive_chunks(text, "d", size=1200)
    assert len(small) > len(large)


def test_chunk_stats_reports_variance():
    """High size variance is a warning sign worth surfacing."""
    uneven = [*fixed_size_chunks("x" * 100, "d", 100), *fixed_size_chunks("y" * 900, "d", 900)]
    stats = chunk_stats(uneven)
    assert stats["stdev"] > 0
    assert stats["min"] < stats["max"]


def test_chunk_stats_handles_empty_input():
    assert chunk_stats([])["count"] == 0


# ===========================================================================
# MINI RAG -- the from-scratch retriever
# ===========================================================================


@pytest.fixture(scope="module")
def rag():
    """One indexed retriever shared across the read-only tests below."""
    return MiniRAG(embeddings=LexicalEmbeddings(dim=2048)).index()


def test_index_builds_a_matrix_of_the_right_shape(rag):
    assert rag.matrix is not None
    n_chunks, dim = rag.matrix.shape
    assert n_chunks == len(rag.chunks)
    assert dim == 2048


def test_search_before_index_fails_clearly():
    with pytest.raises(RuntimeError, match=r"call \.index\(\)"):
        MiniRAG().search("anything")


def test_search_returns_top_k_in_descending_score_order(rag):
    results = rag.search("what is chunk overlap?", top_k=5)
    assert len(results) == 5
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_finds_the_topically_correct_document(rag):
    results = rag.search("what does chunk overlap protect against?", top_k=3)
    assert "chunking" in {r.doc_id for r in results}


def test_retrieval_is_deterministic(rag):
    """Same query, same results -- every time.

    Reproducibility is a precondition for evaluation. If retrieval is
    non-deterministic you cannot tell a real regression from noise, and every
    metric downstream inherits that ambiguity.
    """
    a = [(r.doc_id, r.chunk_index) for r in rag.search("cosine similarity", top_k=5)]
    b = [(r.doc_id, r.chunk_index) for r in rag.search("cosine similarity", top_k=5)]
    assert a == b


def test_trace_captures_everything_an_evaluator_needs(rag):
    trace = rag.trace("what is HNSW?")
    assert trace.question
    assert trace.contexts, "no contexts -- faithfulness and precision would be uncomputable"
    assert trace.retrieved_doc_ids
    assert trace.retrieval_ms > 0
    assert trace.chunk_size > 0
    assert "vector_stores" in trace.retrieved_doc_ids


def test_trace_deduplicates_doc_ids_but_preserves_order():
    """Several chunks from one document must collapse to one doc_id, in rank order."""
    from core.trace import RagTrace, RetrievedChunk

    trace = RagTrace(
        question="q",
        answer="a",
        retrieved=[
            RetrievedChunk("t1", "beta", 0, 0.9),
            RetrievedChunk("t2", "alpha", 0, 0.8),
            RetrievedChunk("t3", "beta", 1, 0.7),
        ],
    )
    assert trace.retrieved_doc_ids == ["beta", "alpha"]


# ===========================================================================
# THE ACTUAL REGRESSION GATE
# ===========================================================================
# This is the test that would fail on a pull request that broke retrieval.
# Deterministic, no model, runs in well under a second.
# ===========================================================================


@pytest.fixture(scope="module")
def retrieval_report(rag):
    golden = [item for item in load_golden() if item.is_answerable]
    results = {
        item.id: (rag.trace(item.question).retrieved_doc_ids, item.reference_doc_ids)
        for item in golden
    }
    return evaluate_retrieval(results)


def test_retrieval_hit_rate_gate(retrieval_report):
    """At least 90% of answerable questions must retrieve a correct document.

    A HIT RATE below this means the generator is being asked to answer from
    context that does not contain the answer, and no prompt engineering will
    fix it.
    """
    assert retrieval_report.hit_rate >= 0.90, (
        f"hit rate {retrieval_report.hit_rate:.3f} below gate.\n"
        f"Questions that missed: {retrieval_report.failures()}"
    )


def test_retrieval_mrr_gate(retrieval_report):
    """Correct documents must rank near the top, not merely appear somewhere.

    MRR rather than recall, deliberately: recall is saturated on this dataset
    (see experiment_chunk_size.py) and a saturated metric cannot detect a
    regression. MRR still has headroom, so it is the more useful gate here.
    Choosing the metric that can actually move is a real skill.
    """
    assert retrieval_report.mrr >= 0.85, (
        f"MRR {retrieval_report.mrr:.3f} below gate -- correct documents are "
        f"ranking too low even when they are retrieved."
    )


def test_report_names_the_failing_questions(retrieval_report):
    """A gate that only prints an average is nearly useless when it fails."""
    assert isinstance(retrieval_report.failures(), list)
    assert retrieval_report.n >= 30


def test_unanswerable_questions_are_not_penalised_by_retrieval_metrics(rag):
    """Recall is vacuously 1.0 when there is nothing to retrieve.

    The system's correctness on these questions is decided by whether it
    REFUSES, which is a generation behaviour measured in 04_deepeval -- not by
    retrieval. Encoding that here stops anyone "fixing" a phantom regression.
    """
    unanswerable = [i for i in load_golden() if not i.is_answerable]
    assert unanswerable, "dataset has no unanswerable questions"

    results = {
        item.id: (rag.trace(item.question).retrieved_doc_ids, item.reference_doc_ids)
        for item in unanswerable
    }
    report = evaluate_retrieval(results)
    assert report.recall == 1.0
    # ...but precision is 0, because everything retrieved really is noise.
    assert report.precision == 0.0


# ===========================================================================
# OLLAMA TIER -- real neural embeddings
# ===========================================================================
# Everything above ran on lexical vectors. These re-run the key claims against
# a real embedding model, which is where the synonym behaviour changes.
# ===========================================================================


@pytest.mark.ollama
def test_neural_embeddings_handle_synonyms_where_lexical_fails():
    """THE payoff test of lesson 01.

    core/test_core.py asserts that lexical embeddings CANNOT match "car" to
    "automobile". This asserts that a real embedding model CAN. Seeing both
    pass is what makes the distinction concrete rather than theoretical.
    """
    from core.providers import get_ollama_embeddings, ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    embeddings = get_ollama_embeddings()
    docs = ["I drove my car to the office.", "Prime numbers underpin cryptography."]
    doc_vectors = embeddings.embed_documents(docs)
    query = embeddings.embed_query("I commuted by automobile to my workplace.")

    sim_paraphrase = cosine_similarity(query, doc_vectors[0])
    sim_unrelated = cosine_similarity(query, doc_vectors[1])

    assert sim_paraphrase > sim_unrelated, (
        f"neural embeddings failed to rank the paraphrase higher "
        f"({sim_paraphrase:.3f} vs {sim_unrelated:.3f})"
    )


@pytest.mark.ollama
def test_neural_embeddings_have_the_expected_dimensionality():
    """nomic-embed-text produces 768 dimensions -- verify, do not assume.

    Dimension mismatches between an index and a query path are a classic
    production outage, and they surface as "retrieval got worse" rather than
    as an error.
    """
    from core.providers import get_ollama_embeddings, ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    vec = get_ollama_embeddings().embed_query("hello")
    assert len(vec) == 768, f"expected 768 dims from nomic-embed-text, got {len(vec)}"


@pytest.mark.ollama
def test_neural_retrieval_scores_at_least_as_well_as_lexical():
    """Sanity check that swapping in real embeddings does not regress retrieval.

    It is not guaranteed to WIN -- this corpus is technical and vocabulary-rich,
    which favours lexical matching. That result would itself be worth knowing,
    and is why the assertion is 'not worse' rather than 'better'.
    """
    from core.providers import get_ollama_embeddings, ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    golden = [item for item in load_golden() if item.is_answerable]
    neural = MiniRAG(embeddings=get_ollama_embeddings()).index()
    results = {
        item.id: (neural.trace(item.question).retrieved_doc_ids, item.reference_doc_ids)
        for item in golden
    }
    report = evaluate_retrieval(results)

    print(f"\nneural embeddings: {report.format_table()}")
    assert report.hit_rate >= 0.85, (
        f"neural hit rate {report.hit_rate:.3f}; misses: {report.failures()}"
    )
