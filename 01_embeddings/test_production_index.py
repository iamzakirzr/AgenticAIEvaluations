"""
Lesson 01 production scenarios. FAST TIER -- no model, no network.

Run:  pytest 01_embeddings/test_production.py -v

The scenario under test is the silent one: an index built with one embedding
model, queried with another. Nothing raises, the numbers look fine, and the
results are noise.
"""

from __future__ import annotations

import pytest
from production_index import (
    IncrementalIndex,
    IndexFingerprint,
    IndexVersionMismatch,
    content_hash,
    describe_embedder,
)

from core.golden import iter_corpus
from core.providers import LexicalEmbeddings


@pytest.fixture(scope="module")
def corpus():
    return dict(iter_corpus())


def build_index(dim: int = 512, **kwargs) -> IncrementalIndex:
    return IncrementalIndex(embedder=LexicalEmbeddings(dim=dim), **kwargs)


class StatelessEmbedder:
    """A per-document embedder, like a real neural model.

    `LexicalEmbeddings` is CORPUS-FITTED: a document's vector depends on IDF
    computed across every other document, so re-embedding a subset is unsound
    and `IncrementalIndex` correctly refuses to do it.

    Real neural embedders (nomic-embed-text, text-embedding-3-small) map each
    document independently, which is what makes incremental indexing safe. This
    stand-in has that property, so the incremental path is genuinely exercised
    in the fast tier without needing Ollama.
    """

    corpus_fitted = False

    def __init__(self, dim: int = 2048) -> None:
        # Large enough that hash collisions are negligible. At dim=64 distinct
        # words share slots and ranking becomes meaningless -- a property of
        # the hashing trick that 01_embeddings/walkthrough.py demonstrates.
        self.dim = dim
        self.embed_calls = 0

    def _vector(self, text: str) -> list[float]:
        import math

        from core.providers import _hash_token, tokenize

        vec = [0.0] * self.dim
        for token in tokenize(text):
            vec[_hash_token(token, self.dim)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls += len(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def build_stateless_index(**kwargs) -> IncrementalIndex:
    return IncrementalIndex(embedder=StatelessEmbedder(dim=2048), **kwargs)


# ===========================================================================
# FINGERPRINTING
# ===========================================================================


def test_fingerprint_probes_the_real_dimension_rather_than_trusting_config():
    """The point is to catch config lying about what the model does."""
    fp = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=256), 700, 120)
    assert fp.dimension == 256
    assert "256" in fp.embedder


def test_identical_setups_produce_identical_fingerprints():
    a = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=512), 700, 120)
    b = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=512), 700, 120)
    assert a == b


def test_chunking_changes_alter_the_fingerprint():
    """Chunk size changes what is in the index, so it is part of its identity."""
    a = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=512), 700, 120)
    b = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=512), 300, 120)
    assert a != b


def test_explain_difference_names_the_field_that_moved():
    """A refusal must say WHICH setting drifted, or nobody can act on it."""
    a = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=512), 700, 120)
    b = IndexFingerprint.from_embedder(LexicalEmbeddings(dim=256), 700, 120)
    message = a.explain_difference(b)
    assert "dimension" in message
    assert "512" in message and "256" in message


def test_describe_embedder_prefers_the_model_name_when_available():
    class FakeOllamaEmbeddings:
        model = "nomic-embed-text"

        def embed_query(self, text):
            return [0.0] * 768

    assert describe_embedder(FakeOllamaEmbeddings()) == "nomic-embed-text"


# ===========================================================================
# THE SILENT OUTAGE
# ===========================================================================


def test_a_mismatched_query_embedder_is_refused_not_served(corpus):
    """THE scenario this module exists for.

    Ingest upgraded its embedding model; the query service did not. Without
    this check the system answers every request from irrelevant passages,
    forever, with no error anywhere.
    """
    index = build_index(dim=512)
    index.build(corpus)

    stale_query_service = LexicalEmbeddings(dim=256)

    with pytest.raises(IndexVersionMismatch, match="does not match the index"):
        index.search("what is chunk overlap?", embedder=stale_query_service)


def test_the_refusal_explains_how_to_fix_it(corpus):
    """An exception that only says 'mismatch' costs an hour of bisecting."""
    index = build_index(dim=512)
    index.build(corpus)

    with pytest.raises(IndexVersionMismatch) as excinfo:
        index.search("q", embedder=LexicalEmbeddings(dim=256))

    message = str(excinfo.value)
    assert "Rebuild the index" in message
    assert "plausible-looking noise" in message


def test_a_matching_embedder_searches_normally(corpus):
    index = build_index(dim=512)
    index.build(corpus)

    results = index.search("what is chunk overlap?", top_k=3)
    assert len(results) == 3
    assert "chunking" in {r.doc_id for r in results}


def test_mismatch_would_otherwise_return_confident_nonsense(corpus):
    """Demonstrates WHY the guard is necessary rather than merely tidy.

    We bypass the check and search with a mismatched embedder directly. The
    call succeeds, returns top_k results with real-looking scores, and ranks
    the wrong document -- exactly the failure that ships silently.
    """
    import numpy as np

    index = build_index(dim=512)
    index.build(corpus)

    # A 512-dim vector from a DIFFERENT vector space (different IDF fitting).
    rogue = LexicalEmbeddings(dim=512)
    rogue.embed_documents(["completely unrelated corpus about marine biology"])
    rogue_vector = np.asarray(rogue.embed_query("what is chunk overlap?"))

    dots = index.matrix @ rogue_vector
    assert dots.shape[0] == len(index.chunks), "no error was raised -- it just works"
    assert np.isfinite(dots).all(), "scores look perfectly normal, which is the danger"


def test_search_before_build_fails_clearly():
    with pytest.raises(RuntimeError, match="call build"):
        build_index().search("anything")


# ===========================================================================
# INCREMENTAL REINDEX
# ===========================================================================


def test_first_build_embeds_everything(corpus):
    index = build_index()
    stats = index.build(corpus)

    assert stats.embedded_documents == len(corpus)
    assert stats.reused_documents == 0
    assert stats.chunks > 0


def test_rebuilding_unchanged_documents_re_embeds_nothing(corpus):
    """The saving that makes reindexing viable at scale.

    Re-embedding an entire corpus because one document changed is the tutorial
    default and is untenable in production. Uses a STATELESS embedder, because
    that is the only case where the optimisation is sound -- see
    test_corpus_fitted_embedders_refuse_incremental_reuse.
    """
    index = build_stateless_index()
    index.build(corpus)

    stats = index.build(corpus)
    assert stats.embedded_documents == 0
    assert stats.reused_documents == len(corpus)


def test_only_the_modified_document_is_re_embedded(corpus):
    index = build_stateless_index()
    index.build(corpus)
    calls_after_first = index.embedder.embed_calls

    updated = dict(corpus)
    updated["chunking"] = corpus["chunking"] + "\n\nA newly appended paragraph.\n"

    stats = index.build(updated)

    assert stats.embedded_documents == 1
    assert stats.reused_documents == len(corpus) - 1
    # And it really did fewer embedding calls, not just report fewer.
    new_calls = index.embedder.embed_calls - calls_after_first
    assert 0 < new_calls < calls_after_first


def test_corpus_fitted_embedders_refuse_incremental_reuse(corpus):
    """A correctness bug caught by test_incremental_rebuild_matches_a_full_rebuild.

    TF-IDF computes a document's vector from corpus-wide statistics (IDF), so
    embedding one changed document alone fits IDF to a one-document corpus and
    produces vectors from a different space than the rest of the index. No
    exception is raised -- retrieval quality just quietly degrades.

    So IncrementalIndex asks the embedder and falls back to a full rebuild.
    Being slow is recoverable; being subtly wrong is not.
    """
    lexical = build_index()
    stateless = build_stateless_index()

    assert lexical.supports_incremental is False
    assert stateless.supports_incremental is True

    lexical.build(corpus)
    stats = lexical.build(corpus)

    assert stats.reused_documents == 0, "a corpus-fitted embedder reused vectors unsafely"
    assert stats.embedded_documents == len(corpus)


def test_an_added_document_becomes_searchable(corpus):
    index = build_stateless_index()
    index.build(corpus)

    extended = dict(corpus)
    extended["pricing"] = (
        "# Pricing\n\nThe enterprise tier costs 4200 dollars per seat per year.\n"
        * 6
    )
    index.build(extended)

    results = index.search("how much does the enterprise tier cost?", top_k=3)
    assert "pricing" in {r.doc_id for r in results}


def test_a_removed_document_stops_being_retrievable(corpus):
    """The subtle half of incremental indexing.

    Forgetting deletions is how a RAG system keeps citing a policy document
    that was withdrawn six months ago -- with a citation, so it looks verified.
    """
    extended = dict(corpus)
    extended["withdrawn_policy"] = (
        "# Withdrawn Policy\n\nRefunds are issued within 90 days, no questions asked.\n"
        * 8
    )

    index = build_stateless_index()
    index.build(extended)
    assert "withdrawn_policy" in {
        r.doc_id for r in index.search("refund policy 90 days", top_k=5)
    }

    stats = index.build(corpus)  # the document is gone from the corpus

    assert stats.removed_documents == 1
    assert "withdrawn_policy" not in {c.doc_id for c in index.chunks}
    assert "withdrawn_policy" not in {
        r.doc_id for r in index.search("refund policy 90 days", top_k=5)
    }


def test_incremental_rebuild_matches_a_full_rebuild(corpus):
    """The correctness property that makes the optimisation safe.

    An incremental index must be INDISTINGUISHABLE from one built from scratch.
    If it is not, you have traded a slow correct system for a fast wrong one.
    """
    updated = dict(corpus)
    updated["retrieval"] = corpus["retrieval"] + "\n\nAn extra note about reranking.\n"

    incremental = build_stateless_index()
    incremental.build(corpus)
    incremental.build(updated)

    from_scratch = build_stateless_index()
    from_scratch.build(updated)

    assert [(c.doc_id, c.index) for c in incremental.chunks] == [
        (c.doc_id, c.index) for c in from_scratch.chunks
    ]
    assert incremental.matrix.shape == from_scratch.matrix.shape

    question = "what is a reranker?"
    a = [(r.doc_id, r.chunk_index) for r in incremental.search(question, top_k=5)]
    b = [(r.doc_id, r.chunk_index) for r in from_scratch.search(question, top_k=5)]
    assert a == b, "incremental and full rebuilds disagree -- the optimisation is unsafe"


def test_changing_chunk_size_invalidates_the_vector_cache(corpus):
    """A correctness bug found in review: content hashing is not enough.

    If chunk_size changes, an unchanged document still produces a DIFFERENT
    NUMBER of chunks. Reusing its cached vectors leaves the matrix and the
    chunk list disagreeing about how many rows there are -- silently, with
    every score afterwards meaningless.
    """
    index = build_stateless_index()
    index.build(corpus)
    assert index.matrix.shape[0] == len(index.chunks)

    # Same content, different chunking.
    index.chunk_size = 200
    index.chunk_overlap = 20
    stats = index.build(corpus)

    assert stats.reused_documents == 0, "stale vectors were reused across a chunking change"
    assert index.matrix.shape[0] == len(index.chunks), (
        "matrix rows and chunk count disagree -- the index is corrupt"
    )

    # And it must still match a from-scratch build at the new setting.
    fresh = build_stateless_index(chunk_size=200, chunk_overlap=20)
    fresh.build(corpus)
    assert len(index.chunks) == len(fresh.chunks)


def test_content_hash_is_stable_across_processes():
    """sha256, not the builtin hash().

    Python randomises str hashing per process, so hash() would report every
    document as modified on every restart -- turning the optimisation into a
    guaranteed full rebuild.
    """
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("hello ")
    assert len(content_hash("x")) == 16


def test_full_rebuild_fallback_still_matches_a_from_scratch_build(corpus):
    """The fallback path must be correct too, not merely slow.

    With a corpus-fitted embedder every build re-embeds everything, so an
    incrementally-driven index and a fresh one must be identical.
    """
    updated = dict(corpus)
    updated["retrieval"] = corpus["retrieval"] + "\n\nAn extra note about reranking.\n"

    reused = build_index()
    reused.build(corpus)
    reused.build(updated)

    fresh = build_index()
    fresh.build(updated)

    question = "what is a reranker?"
    a = [(r.doc_id, r.chunk_index) for r in reused.search(question, top_k=5)]
    b = [(r.doc_id, r.chunk_index) for r in fresh.search(question, top_k=5)]
    assert a == b


def test_changed_documents_classifies_without_embedding(corpus):
    """A cheap scheduled check: 'does the index need rebuilding?'

    Answerable by hashing alone, with no GPU time at all.
    """
    index = build_index()
    index.build(corpus)

    proposed = dict(corpus)
    proposed["chunking"] = corpus["chunking"] + "\nchanged\n"
    proposed["brand_new"] = "# New\n\nSome new content.\n"
    del proposed["hallucination"]

    verdict = index.changed_documents(proposed)

    assert verdict["chunking"] == "modified"
    assert verdict["brand_new"] == "added"
    assert verdict["hallucination"] == "removed"
    assert "retrieval" not in verdict, "an unchanged document should not be listed"


def test_manifest_records_what_a_startup_check_would_verify(corpus, tmp_path):
    index = build_index()
    index.build(corpus)

    path = index.save_manifest(tmp_path / "index_manifest.json")
    import json

    manifest = json.loads(path.read_text())

    assert manifest["fingerprint"]["dimension"] == 512
    assert set(manifest["documents"]) == set(corpus)
    assert manifest["chunks"] == len(index.chunks)


def test_empty_corpus_fails_loudly_rather_than_serving_nothing(corpus):
    """An index that silently builds empty produces a system that refuses
    everything and looks 'safe' while being completely broken."""
    with pytest.raises(ValueError, match="no chunks"):
        build_index().build({})
