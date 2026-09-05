"""
core.providers -- every model this curriculum can talk to, behind one seam.

=============================================================================
THE ONE ABSTRACTION THAT MATTERS
=============================================================================
There are exactly two kinds of model in a RAG system:

    1. An EMBEDDING model:  str        -> list[float]
    2. A CHAT model:        list[msg]  -> str

Everything else -- LangChain chains, LangGraph agents, DeepEval judges, RAGAS
metrics -- is built on those two operations. If you can swap the
implementation behind them, you can point this entire repo at a different
model without touching a lesson file.

That is what this module provides. Three implementations of each:

  EMBEDDINGS                      needs a server?   semantically meaningful?
  ----------------------------    ---------------   ------------------------
  LexicalEmbeddings               no                yes (lexical overlap)
  OllamaEmbeddings (nomic)        yes               yes (true semantics)
  DeterministicFakeEmbedding      no                NO -- random vectors

  CHAT MODELS
  ----------------------------    ---------------   ------------------------
  ScriptedChatModel               no                replays fixed answers
  ChatOllama                      yes               real generation

Why three embedders and not one? Because the fast test tier has to run in CI
with no GPU and no network, but a retrieval test against RANDOM vectors proves
nothing -- recall@k would be pure chance. LexicalEmbeddings solves that: it is
pure numpy, runs in microseconds, and still ranks "what is chunking?" above an
unrelated paragraph. So `pytest` (fast tier) tests real retrieval logic, and
`pytest -m ollama` re-runs the same tests against real neural embeddings.
=============================================================================
"""

from __future__ import annotations

import functools
import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from core.config import settings

# ===========================================================================
# SECTION 1 -- LEXICAL EMBEDDINGS (the "embeddings from scratch" implementation)
# ===========================================================================
# Lesson 01 walks through this line by line. The short version:
#
#   A text embedding is just a vector of numbers where "similar meaning"
#   becomes "small angle between vectors". The simplest thing that achieves
#   that is to count words.
#
#   "the cat sat"  ->  {the: 1, cat: 1, sat: 1}
#
#   Turn that dict into a fixed-length vector, and cosine similarity between
#   two such vectors measures how much vocabulary two texts share. That is a
#   crude but genuine semantic signal -- and it is exactly what search engines
#   ran on for thirty years before neural embeddings existed.
#
#   Two refinements make it usable:
#
#   (a) HASHING. We do not know the vocabulary ahead of time, and we want a
#       FIXED dimension. So instead of building a word->index dictionary, we
#       hash each word to an index in [0, dim). Two different words can collide
#       onto the same slot; with dim=512 and short documents, collisions are
#       rare enough not to matter. This trick is called the "hashing trick".
#
#   (b) IDF WEIGHTING. The word "the" appears in every document, so it carries
#       no discriminating information. Inverse Document Frequency downweights
#       common words and upweights rare ones. Without it, every document looks
#       similar because they all contain "the" and "is".
#
# What this CANNOT do, and why real embeddings exist: it has no idea that
# "car" and "automobile" mean the same thing, because they hash to different
# slots. Lexical matching fails on synonyms and paraphrase. That failure is the
# entire reason neural embedding models were invented -- and lesson 01 has a
# test that demonstrates it concretely.
# ===========================================================================

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric word tokens.

    This is deliberately the simplest tokenizer that works, so you can see
    exactly what a "token" is before meeting subword tokenizers (BPE) in
    lesson 01. Punctuation is dropped, case is folded.

    >>> tokenize("The CAT sat on the mat!")
    ['the', 'cat', 'sat', 'on', 'the', 'mat']
    """
    return _TOKEN_RE.findall(text.lower())


def _hash_token(token: str, dim: int) -> int:
    """Map a token to a vector index using a stable hash.

    NOTE: Python's built-in hash() is randomised per process for str (PYTHONHASHSEED),
    so using it would make embeddings differ between runs -- catastrophic for
    reproducible evaluation. We use a simple deterministic FNV-1a instead.
    """
    h = 2166136261
    for ch in token.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h % dim


class LexicalEmbeddings(Embeddings):
    """Deterministic, dependency-free, semantically meaningful embeddings.

    Implements LangChain's ``Embeddings`` interface, so it is a drop-in
    replacement for ``OllamaEmbeddings`` anywhere in this repo.

    The IDF statistics are learned from whatever corpus you pass to
    ``embed_documents``. That mirrors how real TF-IDF works and also means the
    class has *state*: embed the documents first, then queries.
    """

    # CORPUS-FITTED: this embedder's output for a document depends on the OTHER
    # documents it was fitted alongside, because IDF is a corpus-wide statistic.
    #
    # That has a consequence which is easy to miss and expensive to discover:
    # you CANNOT incrementally re-embed a subset. Embedding one changed document
    # on its own fits IDF to a one-document corpus and produces vectors from a
    # different space than the rest of the index.
    #
    # Neural embedders (nomic-embed-text, text-embedding-3-small) are stateless
    # per document and set this to False, which is what makes incremental
    # indexing safe for them. 01_embeddings/production.py reads this flag and
    # falls back to a full rebuild when it is True.
    corpus_fitted = True

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        # document frequency: how many documents each hashed slot appeared in
        self._doc_freq = np.zeros(dim, dtype=np.float64)
        self._n_docs = 0

    # ---- the vector maths -------------------------------------------------

    def _term_frequencies(self, text: str) -> Counter[int]:
        """Count how often each hashed slot occurs in this text."""
        return Counter(_hash_token(tok, self.dim) for tok in tokenize(text))

    def _idf(self) -> np.ndarray:
        """Inverse document frequency weight for every slot.

        Formula: log((1 + N) / (1 + df)) + 1
        The +1s are smoothing so a slot seen in zero documents does not produce
        a division by zero or a negative weight.
        """
        return np.log((1.0 + self._n_docs) / (1.0 + self._doc_freq)) + 1.0

    def _vectorize(self, text: str) -> list[float]:
        """Turn one string into an L2-normalised TF-IDF vector."""
        vec = np.zeros(self.dim, dtype=np.float64)
        for slot, count in self._term_frequencies(text).items():
            # Sublinear TF: a word appearing 100 times is not 100x as
            # important as appearing once. log damping is standard.
            vec[slot] = 1.0 + math.log(count)
        vec *= self._idf()

        # L2-normalise so that cosine similarity is just a dot product, and so
        # that long documents do not automatically outrank short ones.
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    # ---- LangChain Embeddings interface -----------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Fit IDF on this corpus, then embed every document in it."""
        # Pass 1: accumulate document frequencies.
        self._doc_freq = np.zeros(self.dim, dtype=np.float64)
        self._n_docs = len(texts)
        for text in texts:
            for slot in set(self._term_frequencies(text)):
                self._doc_freq[slot] += 1.0
        # Pass 2: vectorize with the fitted IDF.
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query using the IDF learned from the indexed documents."""
        return self._vectorize(text)


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors: 1.0 = identical direction.

    This is THE similarity function of vector search. Written out explicitly
    because understanding it is non-negotiable:

        cos(a, b) = (a . b) / (|a| * |b|)

    The dot product rewards dimensions where both vectors are large; dividing
    by the magnitudes removes the influence of vector *length*, leaving only
    *direction*. Direction is what encodes meaning; length mostly encodes
    document size, which we do not care about.
    """
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ===========================================================================
# SECTION 2 -- OLLAMA CONNECTIVITY
# ===========================================================================


@functools.lru_cache(maxsize=1)
def ollama_available() -> bool:
    """Return True if an Ollama server answers at the configured URL.

    Cached: pytest calls this in a skip-condition on dozens of tests and we do
    not want dozens of HTTP round-trips. Restart the process to re-check.
    """
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


def installed_ollama_models() -> list[str]:
    """List model tags the local Ollama server has pulled.

    Used by tests to give a precise skip reason ("llama3.1:8b not pulled")
    rather than a generic failure deep inside a library.
    """
    if not ollama_available():
        return []
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=5.0)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def get_chat_model(model: str | None = None, temperature: float | None = None):
    """Return a LangChain chat model backed by Ollama.

    Imported lazily so that the fast test tier never pays the import cost of
    the ollama integration package.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature if temperature is None else temperature,
        # Ollama calls the response length cap num_predict, not max_tokens.
        num_predict=1024,
    )


def get_ollama_embeddings(model: str | None = None):
    """Return real neural embeddings served by Ollama (default nomic-embed-text)."""
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=model or settings.embed_model,
        base_url=settings.ollama_base_url,
    )


def get_openai_compatible_client(async_client: bool = False):
    """An OpenAI SDK client pointed at Ollama's /v1 compatibility endpoint.

    THIS IS THE BRIDGE THAT MAKES RAGAS WORK WITH A LOCAL MODEL.

    RAGAS 0.4 builds judges via ``instructor``, which needs a client speaking
    the OpenAI protocol. Ollama serves one at /v1. The api_key is required by
    the SDK but ignored by Ollama, so any non-empty string works.

    The same client works for anything else expecting OpenAI's API -- vLLM,
    LM Studio, llama.cpp's server, text-generation-inference. That is why this
    repo can point at "any open source chatbot", not just Ollama.
    """
    from openai import AsyncOpenAI, OpenAI

    cls = AsyncOpenAI if async_client else OpenAI
    return cls(
        base_url=settings.ollama_openai_base_url,
        api_key="ollama-does-not-check-this",
        timeout=settings.request_timeout,
    )


# ===========================================================================
# SECTION 3 -- DETERMINISTIC FAKES FOR THE FAST TEST TIER
# ===========================================================================


def scripted_chat_model(responses: Iterable[str]):
    """A chat model that replays a fixed list of answers, in order.

    Used to test everything AROUND the model -- prompt assembly, retrieval,
    graph routing, metric plumbing, report generation -- without needing a
    model at all. When the list is exhausted it cycles back to the start.

    This is how the fast tier can assert on pipeline behaviour deterministically
    in milliseconds.
    """
    return FakeListChatModel(responses=list(responses))


__all__ = [
    "LexicalEmbeddings",
    "cosine_similarity",
    "get_chat_model",
    "get_ollama_embeddings",
    "get_openai_compatible_client",
    "installed_ollama_models",
    "ollama_available",
    "scripted_chat_model",
    "tokenize",
]
