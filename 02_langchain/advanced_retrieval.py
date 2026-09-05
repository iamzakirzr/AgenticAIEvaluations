"""
ADVANCED RETRIEVAL: the techniques interviewers actually ask about.

=============================================================================
WHY THIS FILE EXISTS
=============================================================================
Lesson 02 built plain vector search: embed the question, return the k nearest
chunks. That is the baseline, and it is also where most tutorials stop.

Every technique here fixes a specific, nameable failure of that baseline:

  BM25            vector search misses exact rare tokens -- error codes, SKUs,
                  surnames -- because embeddings blur them together.
  Hybrid + RRF    combines lexical and semantic so the two failure modes
                  cancel rather than compound.
  MMR             plain top-k returns five near-identical chunks from the same
                  section. MMR forces diversity.
  Reranking       embedding similarity compares two independent summaries.
                  A cross-encoder reads query and document TOGETHER, which is
                  far more accurate -- and far slower, hence a shortlist.
  Query rewriting "what about the second one?" retrieves nothing useful,
                  because it is meaningless without the conversation.
  HyDE            a question and an answer look different. Searching with a
                  hypothetical ANSWER matches documents better than the
                  question does.

Everything here is implemented from scratch and runs with no model, so you can
read the mechanism instead of a library call. In production you would use
`langchain_community.retrievers.BM25Retriever`, a real cross-encoder from
`sentence-transformers`, and your vector store's built-in MMR -- but you will
configure all three better for having seen what they do.
=============================================================================
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.providers import cosine_similarity, tokenize
from core.trace import RetrievedChunk

# ===========================================================================
# 1. BM25 -- lexical search, from scratch
# ===========================================================================


@dataclass
class BM25:
    """Okapi BM25: the keyword-ranking function search ran on for decades.

    THE FORMULA, and why each piece is there:

        score(q, d) = SUM over query terms t of
                          IDF(t) * ( f(t,d) * (k1 + 1) )
                                   -------------------------------------
                                   f(t,d) + k1 * (1 - b + b * |d| / avgdl)

      f(t,d)   how often term t appears in document d
      |d|      length of d;  avgdl  average document length

      IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
               rare terms score higher. A term in every document is worthless.

      k1 (~1.5) SATURATION. Without it, a document repeating "refund" fifty
                times would outrank one that says it twice and actually answers
                the question. k1 caps how much repetition can help.

      b  (~0.75) LENGTH NORMALISATION. Long documents contain more of every
                term by chance, so they would win everything. b controls how
                hard to penalise length: 1.0 fully normalises, 0.0 not at all.

    WHY YOU STILL WANT THIS IN 2026: embeddings are trained to map similar
    meanings together, which is exactly wrong for an identifier. "ERR_5521" and
    "ERR_5522" mean completely different things and embed almost identically.
    BM25 treats them as unrelated tokens, which is correct.
    """

    k1: float = 1.5
    b: float = 0.75

    doc_ids: list[str] = field(default_factory=list, init=False)
    texts: list[str] = field(default_factory=list, init=False)
    _freqs: list[Counter] = field(default_factory=list, init=False)
    _lengths: list[int] = field(default_factory=list, init=False)
    _doc_freq: Counter = field(default_factory=Counter, init=False)
    _avgdl: float = field(default=0.0, init=False)

    def index(self, documents: list[tuple[str, str]]) -> BM25:
        """Index ``[(doc_id, text), ...]``."""
        self.doc_ids = [doc_id for doc_id, _ in documents]
        self.texts = [text for _, text in documents]
        self._freqs = []
        self._lengths = []
        self._doc_freq = Counter()

        for text in self.texts:
            tokens = tokenize(text)
            freq = Counter(tokens)
            self._freqs.append(freq)
            self._lengths.append(len(tokens))
            # Document frequency counts DOCUMENTS, not occurrences.
            self._doc_freq.update(freq.keys())

        self._avgdl = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        return self

    def idf(self, term: str) -> float:
        n_docs = len(self.texts)
        containing = self._doc_freq.get(term, 0)
        # The +1 inside the log keeps this non-negative. The classic BM25 IDF
        # can go NEGATIVE for a term in more than half the corpus, which lets a
        # very common word actively reduce a document's score -- surprising, and
        # usually not what you want.
        return math.log(1 + (n_docs - containing + 0.5) / (containing + 0.5))

    def score(self, query: str) -> list[float]:
        scores = [0.0] * len(self.texts)
        for term in tokenize(query):
            idf = self.idf(term)
            if idf == 0:
                continue
            for i, freq in enumerate(self._freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                norm = 1 - self.b + self.b * (self._lengths[i] / (self._avgdl or 1))
                scores[i] += idf * (f * (self.k1 + 1)) / (f + self.k1 * norm)
        return scores

    def search(self, query: str, top_k: int = 4) -> list[tuple[int, float]]:
        """Return ``[(index, score), ...]`` best first, dropping zero scores."""
        scored = [(i, s) for i, s in enumerate(self.score(query)) if s > 0]
        return sorted(scored, key=lambda pair: -pair[1])[:top_k]


# ===========================================================================
# 2. RECIPROCAL RANK FUSION
# ===========================================================================


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60, top_k: int | None = None
) -> list[tuple[str, float]]:
    """Merge several ranked lists into one.

        score(d) = SUM over lists of  1 / (k + rank(d))     rank is 1-based

    WHY RRF RATHER THAN AVERAGING SCORES: BM25 scores are unbounded positive
    numbers; cosine similarity is -1 to 1. Averaging them is meaningless, and
    normalising them requires knowing each distribution, which changes per
    query. RRF uses ONLY THE RANKS, so it needs to know nothing about either
    scale. That is why it is the default in every hybrid system.

    WHY k=60: it damps the difference between the top positions so that one
    retriever cannot dominate purely by being confident. With k=60, rank 1
    scores 1/61 and rank 2 scores 1/62 -- close together, so agreement ACROSS
    lists matters more than being first in one. The value is conventional,
    from the original paper, and rarely worth tuning.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    merged = sorted(scores.items(), key=lambda pair: -pair[1])
    return merged[:top_k] if top_k else merged


# ===========================================================================
# 3. MAXIMAL MARGINAL RELEVANCE
# ===========================================================================


def mmr_select(
    query_vector: list[float],
    candidate_vectors: list[list[float]],
    k: int = 4,
    lambda_mult: float = 0.5,
) -> list[int]:
    """Pick k candidates that are relevant to the query AND unlike each other.

        MMR = argmax [ lambda * sim(d, query) - (1 - lambda) * max sim(d, s) ]
                                                              s in selected

    THE PROBLEM IT SOLVES: a document that discusses a topic across five
    consecutive paragraphs produces five near-identical vectors. Plain top-k
    returns all five, the context window fills with the same sentence rephrased,
    and a second document that would have completed the answer never appears.

    lambda_mult = 1.0 is pure relevance (identical to plain top-k).
    lambda_mult = 0.0 is pure diversity (relevance ignored entirely).
    0.5 is a reasonable default.

    Returns indices into ``candidate_vectors``, in selection order.
    """
    if not candidate_vectors:
        return []

    relevance = [cosine_similarity(query_vector, v) for v in candidate_vectors]
    selected: list[int] = []
    remaining = set(range(len(candidate_vectors)))

    while remaining and len(selected) < k:
        best_index, best_score = None, -float("inf")
        for i in remaining:
            if selected:
                redundancy = max(
                    cosine_similarity(candidate_vectors[i], candidate_vectors[j])
                    for j in selected
                )
            else:
                # Nothing selected yet, so nothing to be redundant with.
                redundancy = 0.0
            score = lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
            if score > best_score:
                best_index, best_score = i, score
        selected.append(best_index)  # type: ignore[arg-type]
        remaining.discard(best_index)  # type: ignore[arg-type]

    return selected


# ===========================================================================
# 4. HYBRID RETRIEVER
# ===========================================================================


class HybridRetriever:
    """BM25 + vector search, fused with RRF.

    More robust than either alone because they fail in different directions:
    vector search misses exact rare tokens; lexical search misses paraphrase.
    Fusing means a document only needs to be found by ONE of them.
    """

    def __init__(self, pipeline, rrf_k: int = 60) -> None:
        self.pipeline = pipeline
        self.rrf_k = rrf_k
        self.bm25 = BM25()
        self._key_to_chunk: dict[str, RetrievedChunk] = {}
        self._index()

    def _index(self) -> None:
        documents = []
        for doc in self.pipeline.documents:
            key = f"{doc.metadata['doc_id']}#{doc.metadata['chunk_index']}"
            documents.append((key, doc.page_content))
            self._key_to_chunk[key] = RetrievedChunk(
                text=doc.page_content,
                doc_id=doc.metadata["doc_id"],
                chunk_index=doc.metadata["chunk_index"],
            )
        self.bm25.index(documents)

    def search(self, question: str, top_k: int = 4, candidates: int = 12):
        """Retrieve from both, fuse, return the top_k chunks.

        `candidates` is how deep each retriever goes BEFORE fusion. Fusing only
        the top 4 of each wastes the technique -- the whole point is that a
        document ranked 9th by one retriever and 2nd by the other should
        surface, and it cannot if you truncated at 4.
        """
        vector_hits = self.pipeline.retrieve(question, top_k=candidates)
        vector_ranking = [f"{c.doc_id}#{c.chunk_index}" for c in vector_hits]

        bm25_ranking = [
            self.bm25.doc_ids[i] for i, _ in self.bm25.search(question, top_k=candidates)
        ]

        fused = reciprocal_rank_fusion(
            [vector_ranking, bm25_ranking], k=self.rrf_k, top_k=top_k
        )

        results = []
        for key, score in fused:
            chunk = self._key_to_chunk.get(key)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    text=chunk.text,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    score=score,
                )
            )
        return results


# ===========================================================================
# 5. RERANKING
# ===========================================================================


class Reranker:
    """Two-stage retrieval: fetch many cheaply, then rescore the shortlist.

    A BI-ENCODER (ordinary embedding search) encodes the query and each document
    SEPARATELY, then compares the two summaries. Fast, because documents are
    embedded once at index time -- but the comparison never sees the pair
    together.

    A CROSS-ENCODER reads query and document as one input, so every word of the
    query can attend to every word of the document. Much more accurate, and far
    too slow to run over a whole corpus: it is O(corpus) model calls per query,
    with no precomputation possible.

    Hence the standard shape: retrieve 25-50 candidates with the bi-encoder,
    rerank them with the cross-encoder, keep 3-5.

    `score_fn(query, text) -> float` is injected so the fast tier can drive this
    deterministically. In production you would pass a real cross-encoder such
    as `cross-encoder/ms-marco-MiniLM-L-6-v2`.
    """

    def __init__(self, score_fn) -> None:
        self.score_fn = score_fn
        self.calls = 0

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int = 4):
        rescored = []
        for chunk in chunks:
            self.calls += 1
            rescored.append(
                RetrievedChunk(
                    text=chunk.text,
                    doc_id=chunk.doc_id,
                    chunk_index=chunk.chunk_index,
                    score=float(self.score_fn(question, chunk.text)),
                )
            )
        rescored.sort(key=lambda c: -c.score)
        return rescored[:top_k]


def lexical_overlap_scorer(query: str, text: str) -> float:
    """A stand-in cross-encoder: fraction of query terms present in the text.

    Deterministic and offline, so the reranking MECHANISM is testable without a
    model. It is not a real cross-encoder and makes no claim to be -- swap in
    sentence-transformers for anything real.
    """
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    text_terms = set(tokenize(text))
    return len(query_terms & text_terms) / len(query_terms)


# ===========================================================================
# 6. QUERY TRANSFORMATION
# ===========================================================================

REWRITE_PROMPT = """Given the conversation below, rewrite the final question so \
that it is fully self-contained and can be understood without the conversation.
Resolve every pronoun and reference. Output ONLY the rewritten question.

Conversation:
{history}

Final question: {question}
Rewritten question:"""


def rewrite_followup(llm, history: list[tuple[str, str]], question: str) -> str:
    """Turn a conversational follow-up into a standalone search query.

    WITHOUT THIS, MULTI-TURN RAG IS BROKEN. Embed "what about the second one?"
    and you retrieve documents about the number two. The conversation carries
    the meaning, and the retriever never sees the conversation.

    This is the single highest-impact addition when a single-turn RAG demo
    becomes a real chatbot, and it is very frequently missed.
    """
    if not history:
        return question

    transcript = "\n".join(f"User: {user}\nAssistant: {assistant}" for user, assistant in history)
    response = llm.invoke(REWRITE_PROMPT.format(history=transcript, question=question))
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip() or question


HYDE_PROMPT = """Write a short, factual paragraph that would answer this \
question, as if it were an extract from documentation. Do not say you are \
unsure; invent plausible specifics if necessary. Output only the paragraph.

Question: {question}
Paragraph:"""


def hyde_query(llm, question: str) -> str:
    """HyDE: search with a hypothetical ANSWER instead of the question.

    THE INSIGHT: a question and its answer are written differently. "How many
    dimensions does nomic-embed-text produce?" shares little vocabulary or
    structure with "nomic-embed-text produces 768 dimensions." A hypothetical
    answer is written in the register of the DOCUMENTS, so it lands nearer them
    in vector space.

    THE OBVIOUS OBJECTION, which is worth being able to answer: the hypothetical
    answer may be entirely wrong. That is fine, and is the clever part -- it is
    never shown to the user and never used as an answer. It is used only as a
    SEARCH KEY, and a wrong answer about the right topic still uses the right
    vocabulary.

    The cost is a full model call before retrieval even starts, which roughly
    doubles latency. Worth it when questions are short and documents are prose;
    rarely worth it when questions already look like the documents.
    """
    response = llm.invoke(HYDE_PROMPT.format(question=question))
    text = response.content if hasattr(response, "content") else str(response)
    return text.strip() or question


def multi_query_expansion(llm, question: str, n: int = 3) -> list[str]:
    """Generate paraphrases, retrieve for each, and union the results.

    Improves recall when the user's phrasing does not match the document's.
    Costs one model call plus n retrievals, and the results still need fusing --
    RRF again.
    """
    prompt = (
        f"Write {n} different rephrasings of the question below, one per line, "
        f"using varied vocabulary. Output only the questions.\n\n"
        f"Question: {question}"
    )
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    variants = [line.strip(" -•\t") for line in text.splitlines() if line.strip()]
    # Always include the original: a paraphrase can drift, and the user's own
    # wording is the one phrasing you know is faithful to their intent.
    return [question, *variants[:n]]


# ===========================================================================
# 7. CONVERSATIONAL RETRIEVAL
# ===========================================================================


@dataclass
class ConversationalRetriever:
    """Multi-turn RAG: rewrite the follow-up, then retrieve.

    Keeps a bounded history. Unbounded history is a real production bug -- the
    rewrite prompt grows without limit until it blows the context window, and
    the failure arrives mid-conversation for your most engaged users.
    """

    pipeline: object
    llm: object
    max_turns: int = 5
    history: list[tuple[str, str]] = field(default_factory=list)

    def ask(self, question: str, top_k: int = 4):
        standalone = rewrite_followup(self.llm, self.history, question)
        chunks = self.pipeline.retrieve(standalone, top_k=top_k)
        return standalone, chunks

    def record(self, question: str, answer: str) -> None:
        self.history.append((question, answer))
        # Keep only the most recent turns. Truncating the OLDEST is the right
        # direction: recent turns carry the references a follow-up needs.
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns :]
