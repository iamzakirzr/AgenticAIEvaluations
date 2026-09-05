"""
PRODUCTION SCENARIO: index versioning, drift detection and incremental reindex.

=============================================================================
THE OUTAGE THIS PREVENTS
=============================================================================
core/corpus/retrieval.md states the failure plainly:

    "Using a different embedding model for queries than for documents produces
     meaningless results, because the two models place text in unrelated vector
     spaces. This is a frequent bug when an index is rebuilt with an upgraded
     model but the query path is not."

Here is how it actually happens. Someone upgrades `EMBED_MODEL` from
`nomic-embed-text` to something better. The ingestion job runs on a schedule and
picks up the new value. The query service is on an older deploy and still uses
the old one. Now:

  - Nothing raises. Both models return float vectors.
  - If the dimensions happen to match, not even a shape error occurs.
  - Cosine similarity still returns numbers between -1 and 1.
  - Retrieval returns the top_k nearest vectors, as always.
  - THE RESULTS ARE NOISE.

The system keeps answering confidently from irrelevant passages. There is no
error, no alert, and no crash. You discover it from a complaint days later.

THE FIX IS BORING AND COMPLETE: stamp the index with a fingerprint of
everything that affects its vectors, and refuse to serve a query embedded by
anything else. Fail loudly at startup instead of silently at every request.

=============================================================================
THE SECOND SCENARIO: INCREMENTAL REINDEX
=============================================================================
Re-embedding an entire corpus because one document changed is the default in
every tutorial and is untenable in production -- it costs GPU time proportional
to corpus size and takes the index offline.

`IncrementalIndex` hashes each document, so a rebuild only re-embeds documents
whose CONTENT changed, and correctly handles additions and deletions. On a
corpus where one document in fifty changed, that is a 50x saving.

The subtle part is deletion: chunks from a removed document must actually leave
the index. Forgetting that is how a RAG system keeps citing a policy document
that was withdrawn six months ago.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chunking import Chunk, recursive_chunks

from core.config import settings
from core.trace import RetrievedChunk


class IndexVersionMismatch(RuntimeError):
    """Raised when a query would be embedded differently than the index was.

    Deliberately fatal. The alternative -- serving the query anyway -- returns
    plausible-looking nonsense with no error, which is strictly worse than an
    outage because nobody notices.
    """


def content_hash(text: str) -> str:
    """Stable short hash of a document's content.

    sha256 rather than the builtin hash(), which is randomised per process for
    strings (PYTHONHASHSEED) and would therefore report every document as
    changed on every restart.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class IndexFingerprint:
    """Everything that affects the vectors in an index.

    If ANY of these differ between the ingest path and the query path, the
    index is unusable. Dimension is included because it catches the subset of
    mismatches that would otherwise blow up much later inside a matrix
    multiply, with a shape error nobody can trace back to a config change.
    """

    embedder: str
    dimension: int
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_embedder(cls, embedder, chunk_size: int, chunk_overlap: int) -> IndexFingerprint:
        """Derive a fingerprint, probing the embedder for its real dimension.

        We probe rather than trust configuration, because the whole point is to
        detect a mismatch between what config SAYS and what the model DOES.
        """
        probe = embedder.embed_query("dimension probe")
        return cls(
            embedder=describe_embedder(embedder),
            dimension=len(probe),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def explain_difference(self, other: IndexFingerprint) -> str:
        diffs = [
            f"{key}: index={getattr(self, key)!r} query={getattr(other, key)!r}"
            for key in ("embedder", "dimension", "chunk_size", "chunk_overlap")
            if getattr(self, key) != getattr(other, key)
        ]
        return "; ".join(diffs) if diffs else "no difference"


def describe_embedder(embedder) -> str:
    """A stable identity string for an embedding model.

    LangChain embedders expose `.model`; our LexicalEmbeddings does not, so we
    fall back to the class name plus its dimension. Both are stable across
    processes, which is the only requirement.
    """
    model = getattr(embedder, "model", None)
    if model:
        return str(model)
    dim = getattr(embedder, "dim", None)
    return f"{type(embedder).__name__}" + (f"(dim={dim})" if dim else "")


@dataclass
class IndexStats:
    documents: int = 0
    chunks: int = 0
    embedded_documents: int = 0   # how many were actually re-embedded
    reused_documents: int = 0     # how many were skipped as unchanged
    removed_documents: int = 0
    build_seconds: float = 0.0

    def report(self) -> str:
        saved = ""
        if self.reused_documents:
            total = self.embedded_documents + self.reused_documents
            saved = f"  ({self.reused_documents}/{total} documents reused)"
        return (
            f"{self.documents} docs / {self.chunks} chunks in "
            f"{self.build_seconds:.2f}s{saved}"
        )


@dataclass
class IncrementalIndex:
    """A vector index that knows what it was built with and what has changed.

    Two production properties the lesson-01 `MiniRAG` deliberately lacks:

      1. It refuses to answer a query embedded by a different model.
      2. Rebuilding only re-embeds documents whose content actually changed.
    """

    embedder: object
    chunk_size: int = field(default_factory=lambda: settings.chunk_size)
    chunk_overlap: int = field(default_factory=lambda: settings.chunk_overlap)

    fingerprint: IndexFingerprint | None = field(default=None, init=False)
    chunks: list[Chunk] = field(default_factory=list, init=False)
    matrix: np.ndarray | None = field(default=None, init=False)
    doc_hashes: dict[str, str] = field(default_factory=dict, init=False)
    stats: IndexStats = field(default_factory=IndexStats, init=False)

    # Cached vectors per document, so an unchanged document is not re-embedded.
    _vectors: dict[str, list[list[float]]] = field(default_factory=dict, init=False)

    # The chunking parameters the cache was built with. Content hashing alone
    # is NOT a sufficient reuse key: if chunk_size changes, an unchanged
    # document still produces a DIFFERENT NUMBER of chunks, so the cached
    # vectors no longer line up with the re-chunked text. The matrix and the
    # chunk list would silently disagree, and every score after that is
    # meaningless. Tracked separately so a parameter change invalidates the
    # cache the way a content change does.
    _cached_chunking: tuple[int, int] | None = field(default=None, init=False)

    # ---- building ---------------------------------------------------------

    @property
    def supports_incremental(self) -> bool:
        """False for corpus-fitted embedders, where partial re-embedding is UNSAFE.

        =====================================================================
        A CORRECTNESS BUG THIS PROPERTY EXISTS TO PREVENT
        =====================================================================
        `test_incremental_rebuild_matches_a_full_rebuild` caught this while the
        module was being written, which is exactly what it was for.

        TF-IDF (and BM25, and anything using corpus-wide statistics) computes a
        document's vector using information about EVERY OTHER DOCUMENT -- the
        inverse document frequency of each term. Embed one changed document on
        its own and IDF is fitted to a one-document corpus, producing a vector
        from a completely different space than the rest of the index.

        The result is not an error. It is an index where some vectors are
        comparable and some are not, and retrieval quality quietly degrades in
        a way no exception reveals.

        Neural embedders do not have this problem: `nomic-embed-text` maps a
        document to a vector using only that document, so re-embedding a subset
        is safe and identical to a full rebuild.

        So we ask the embedder, and fall back to a full rebuild when the
        optimisation would be unsound. Being slow is recoverable. Being subtly
        wrong is not.
        """
        return not getattr(self.embedder, "corpus_fitted", False)

    def build(self, documents: dict[str, str]) -> IndexStats:
        """(Re)build the index, re-embedding only what changed.

        Falls back to a full re-embed when the embedder is corpus-fitted -- see
        `supports_incremental` for why that is a correctness requirement rather
        than caution.
        """
        started = time.perf_counter()
        stats = IndexStats()

        chunking = (self.chunk_size, self.chunk_overlap)
        if self._cached_chunking is not None and self._cached_chunking != chunking:
            # Chunking changed -> every cached vector is stale, however
            # unchanged the source text is. See _cached_chunking above.
            self._vectors.clear()
        self._cached_chunking = chunking

        if not self.supports_incremental:
            # Drop the vector cache so every document is treated as changed.
            # doc_hashes is deliberately LEFT INTACT so the deletion pass below
            # still sees which documents have disappeared.
            self._vectors.clear()

        incoming = set(documents)
        existing = set(self.doc_hashes)

        # DELETIONS FIRST. A document removed from the corpus must have its
        # chunks removed from the index, or the system keeps citing a policy
        # that was withdrawn.
        for doc_id in existing - incoming:
            self.doc_hashes.pop(doc_id, None)
            self._vectors.pop(doc_id, None)
            stats.removed_documents += 1

        changed: dict[str, list[Chunk]] = {}
        for doc_id, text in sorted(documents.items()):
            digest = content_hash(text)
            if self.doc_hashes.get(doc_id) == digest and doc_id in self._vectors:
                stats.reused_documents += 1
                continue
            changed[doc_id] = recursive_chunks(
                text, doc_id, self.chunk_size, self.chunk_overlap
            )
            self.doc_hashes[doc_id] = digest
            stats.embedded_documents += 1

        # Embed only the changed documents, in ONE batched call.
        if changed:
            flat = [chunk.text for chunks in changed.values() for chunk in chunks]
            vectors = self.embedder.embed_documents(flat)

            cursor = 0
            for doc_id, doc_chunks in changed.items():
                self._vectors[doc_id] = vectors[cursor : cursor + len(doc_chunks)]
                cursor += len(doc_chunks)

        # Rebuild the chunk list and matrix from cache, in a stable order.
        self.chunks = []
        rows: list[list[float]] = []
        for doc_id in sorted(self.doc_hashes):
            doc_chunks = changed.get(doc_id)
            if doc_chunks is None:
                doc_chunks = recursive_chunks(
                    documents[doc_id], doc_id, self.chunk_size, self.chunk_overlap
                )
            self.chunks.extend(doc_chunks)
            rows.extend(self._vectors[doc_id])

        if not self.chunks:
            raise ValueError("index build produced no chunks -- is the corpus empty?")

        self.matrix = np.asarray(rows, dtype=np.float64)
        self.fingerprint = IndexFingerprint.from_embedder(
            self.embedder, self.chunk_size, self.chunk_overlap
        )

        stats.documents = len(self.doc_hashes)
        stats.chunks = len(self.chunks)
        stats.build_seconds = time.perf_counter() - started
        self.stats = stats
        return stats

    # ---- querying ---------------------------------------------------------

    def search(self, question: str, embedder=None, top_k: int | None = None):
        """Search, REFUSING to serve if the query embedder does not match.

        Pass `embedder` explicitly to simulate the real production topology,
        where the ingest job and the query service are separate deployments
        that can drift apart.
        """
        if self.matrix is None or self.fingerprint is None:
            raise RuntimeError("call build() before search()")

        query_embedder = embedder if embedder is not None else self.embedder
        query_fp = IndexFingerprint.from_embedder(
            query_embedder, self.chunk_size, self.chunk_overlap
        )

        if query_fp != self.fingerprint:
            raise IndexVersionMismatch(
                "Refusing to search: the query embedder does not match the index.\n"
                f"  {self.fingerprint.explain_difference(query_fp)}\n"
                "Serving this query would return plausible-looking noise with no "
                "error. Rebuild the index, or roll the query service back to the "
                "model the index was built with."
            )

        k = top_k if top_k is not None else settings.top_k
        query_vector = np.asarray(query_embedder.embed_query(question), dtype=np.float64)

        dots = self.matrix @ query_vector
        norms = np.linalg.norm(self.matrix, axis=1) * np.linalg.norm(query_vector)
        norms = np.where(norms == 0, 1e-12, norms)
        scores = dots / norms

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

    # ---- persistence of the manifest --------------------------------------

    def manifest(self) -> dict:
        """What you would write next to a persisted index, and check on startup."""
        return {
            "fingerprint": asdict(self.fingerprint) if self.fingerprint else None,
            "documents": self.doc_hashes,
            "chunks": len(self.chunks),
        }

    def save_manifest(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2, sort_keys=True) + "\n")
        return path

    def changed_documents(self, documents: dict[str, str]) -> dict[str, str]:
        """Classify what a rebuild would do, WITHOUT doing it.

        Useful as a cheap scheduled job: "does the index need rebuilding?" is
        answerable by hashing, with no GPU time at all.
        """
        verdict: dict[str, str] = {}
        for doc_id, text in documents.items():
            known = self.doc_hashes.get(doc_id)
            if known is None:
                verdict[doc_id] = "added"
            elif known != content_hash(text):
                verdict[doc_id] = "modified"
        for doc_id in self.doc_hashes:
            if doc_id not in documents:
                verdict[doc_id] = "removed"
        return verdict
