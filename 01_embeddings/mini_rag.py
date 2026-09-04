"""
A complete retrieval system in about 80 lines, with no framework at all.

=============================================================================
THE POINT OF THIS FILE
=============================================================================
LangChain, LlamaIndex and every vector database are conveniences layered on a
very small idea:

    1. Split documents into chunks.
    2. Turn each chunk into a vector.
    3. Turn the query into a vector with the SAME model.
    4. Return the chunks whose vectors are most similar to the query's.

That is the entire algorithm. Everything else -- persistence, filtering,
approximate indexes, async, hybrid search -- is engineering around those four
steps, not a change to them.

Build it once by hand and the frameworks stop being magic. When lesson 02
introduces Chroma and LangChain retrievers, you will recognise exactly which
of these four steps each piece is doing, and you will be able to tell when a
framework is doing something surprising.

=============================================================================
WHAT THIS DELIBERATELY DOES NOT DO
=============================================================================
  - No persistence: the index is rebuilt in memory every time.
  - No approximate search: it compares against every chunk (brute force).
    For a corpus this size that is not just acceptable, it is CORRECT --
    core/corpus/vector_stores.md explains that exact search is the right
    default below ~100k chunks because it removes a class of recall bugs.
  - No metadata filtering, no async, no batching.

Those are the things a real vector store adds. Now you know what you are
paying for.
=============================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Lesson directories start with a digit, so they are not importable packages.
# When this file is run as a script, put the project root on sys.path so the
# shared `core` package resolves. (Under pytest, conftest.py already did this.)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chunking import Chunk, recursive_chunks  # noqa: E402  (sibling module)

from core.config import settings  # noqa: E402
from core.golden import iter_corpus  # noqa: E402
from core.providers import LexicalEmbeddings  # noqa: E402
from core.trace import RagTrace, RetrievedChunk  # noqa: E402


class MiniRAG:
    """Brute-force vector retrieval over an in-memory corpus."""

    def __init__(self, embeddings=None, chunk_size: int | None = None, overlap: int | None = None):
        # The embedder is injected, not hard-coded. That single decision is why
        # the same class can run against offline lexical vectors in the fast
        # test tier and against real nomic-embed-text vectors when Ollama is up.
        self.embeddings = embeddings if embeddings is not None else LexicalEmbeddings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.overlap = overlap if overlap is not None else settings.chunk_overlap

        self.chunks: list[Chunk] = []
        # Shape (n_chunks, embedding_dim). Storing as one matrix rather than a
        # list of vectors is what lets us score every chunk with a single
        # matrix-vector product instead of a Python loop.
        self.matrix: np.ndarray | None = None

    # ---- STEP 1 & 2: chunk, then embed ------------------------------------

    def index(self, documents: dict[str, str] | None = None) -> "MiniRAG":
        """Build the index. Returns self so you can chain.

        >>> rag = MiniRAG().index()
        """
        docs = documents if documents is not None else dict(iter_corpus())

        self.chunks = []
        for doc_id, text in sorted(docs.items()):
            self.chunks.extend(recursive_chunks(text, doc_id, self.chunk_size, self.overlap))

        if not self.chunks:
            raise ValueError("no chunks produced -- is the corpus empty?")

        # ONE call with every chunk, not one call per chunk. With a real
        # embedding server the difference is minutes versus seconds, and with
        # LexicalEmbeddings it also matters for correctness: IDF weights are
        # fitted from the whole corpus, so the embedder must see it all at once.
        vectors = self.embeddings.embed_documents([c.text for c in self.chunks])
        self.matrix = np.asarray(vectors, dtype=np.float64)
        return self

    # ---- STEP 3 & 4: embed the query, score every chunk --------------------

    def search(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the top_k most similar chunks, highest score first."""
        if self.matrix is None:
            raise RuntimeError("call .index() before .search()")

        k = top_k if top_k is not None else settings.top_k

        # CRITICAL: the query must be embedded by the SAME model that embedded
        # the documents. Using a different model puts the query in an unrelated
        # vector space and retrieval degenerates to noise. This is a real
        # production bug that happens when an index is rebuilt with an upgraded
        # model but the query path is not.
        query_vector = np.asarray(self.embeddings.embed_query(question), dtype=np.float64)

        scores = self._cosine_scores(query_vector)

        # argsort ascending, then reverse -> indices of the k highest scores.
        ranked = np.argsort(scores)[::-1][:k]

        return [
            RetrievedChunk(
                text=self.chunks[i].text,
                doc_id=self.chunks[i].doc_id,
                chunk_index=self.chunks[i].index,
                score=float(scores[i]),
            )
            for i in ranked
        ]

    def _cosine_scores(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity between the query and EVERY chunk, vectorised.

        The maths, spelled out:

            cos(q, d) = (q . d) / (|q| * |d|)

        `self.matrix @ query_vector` computes the dot product of the query with
        every row at once -- that single line replaces a Python loop over
        thousands of chunks and is roughly two orders of magnitude faster.

        We divide by the norms explicitly rather than assuming normalised
        vectors, because an injected embedder may not normalise. Being
        defensive here costs one multiply and prevents a silent scoring bug
        that only appears with a different model.
        """
        assert self.matrix is not None
        dots = self.matrix @ query_vector
        doc_norms = np.linalg.norm(self.matrix, axis=1)
        query_norm = np.linalg.norm(query_vector)
        denom = doc_norms * query_norm
        # Guard against a zero-vector chunk (happens with empty or
        # stopword-only text) producing a divide-by-zero warning and a NaN,
        # which would then sort unpredictably.
        denom = np.where(denom == 0, 1e-12, denom)
        return dots / denom

    # ---- Convenience: produce a full trace ---------------------------------

    def trace(self, question: str, answer: str = "", top_k: int | None = None) -> RagTrace:
        """Run retrieval and package the result as a RagTrace.

        No generator is involved here -- lesson 01 is about retrieval only.
        Passing an empty answer is deliberate: it lets you compute every
        RETRIEVAL metric (recall@k, MRR, nDCG) before you have a chatbot at
        all, which is the correct order to build a RAG system in.
        """
        started = time.perf_counter()
        retrieved = self.search(question, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return RagTrace(
            question=question,
            answer=answer,
            retrieved=retrieved,
            retrieval_ms=elapsed_ms,
            embed_model=type(self.embeddings).__name__,
            chunk_size=self.chunk_size,
            top_k=top_k if top_k is not None else settings.top_k,
        )
