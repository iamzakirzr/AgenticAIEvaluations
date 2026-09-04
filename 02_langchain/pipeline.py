"""
The RAG pipeline, built with LangChain 1.4.

=============================================================================
WHAT LANGCHAIN ACTUALLY ADDS OVER 01_embeddings/mini_rag.py
=============================================================================
Lesson 01 built retrieval in 80 lines of numpy. So what is the framework for?

  1. INTEGRATIONS. One `Embeddings` interface, and the same code runs against
     Ollama, OpenAI, HuggingFace, Cohere. You write the swap once.
  2. TEXT SPLITTERS. RecursiveCharacterTextSplitter is the hand-written
     splitter from lesson 01, battle-tested and maintained.
  3. VECTOR STORE ABSTRACTION. Move from in-memory to Chroma to pgvector by
     changing one constructor.
  4. LCEL COMPOSITION. Chains built from `|` operators get streaming, async,
     batching and callback instrumentation for free -- and lesson 06 relies on
     those callbacks to trace the pipeline.

What it does NOT do is change the four-step algorithm. If lesson 01 made sense,
nothing here should be surprising.

=============================================================================
A WARNING ABOUT LANGCHAIN VERSIONS
=============================================================================
This file targets langchain 1.4.0 / langchain-core 1.6.1. LangChain 1.x is a
substantial rewrite of 0.3.x, and MOST TUTORIALS ONLINE ARE FOR 0.3 OR EARLIER.
If you copy a snippet from a blog post and it fails on imports, that is why.
The pins in pyproject.toml are exact for this reason.
=============================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompts import GROUNDED_PROMPT, format_context, invalid_citations

from core.config import settings
from core.golden import iter_corpus
from core.providers import LexicalEmbeddings
from core.trace import RagTrace, RetrievedChunk


class RagPipeline:
    """Retrieve, then generate -- with the whole trace preserved.

    The design rule this class exists to demonstrate: **never return only the
    answer string.** core/trace.py explains why at length; the short version is
    that faithfulness, context precision and context recall are all
    uncomputable from an answer alone, so a pipeline that throws away its
    retrieved chunks cannot be evaluated properly.

    Both the embedder and the chat model are INJECTED. That is what lets the
    fast test tier drive this exact class with a scripted model and offline
    lexical vectors, while `make chat` drives it with Ollama.
    """

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        embeddings=None,
        prompt=None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
    ) -> None:
        self.embeddings = embeddings if embeddings is not None else LexicalEmbeddings(dim=2048)
        self.llm = llm  # may be None: retrieval-only mode still works
        self.prompt = prompt if prompt is not None else GROUNDED_PROMPT
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.top_k = top_k if top_k is not None else settings.top_k

        self.store: InMemoryVectorStore | None = None
        self.documents: list[Document] = []

    # ---- INGESTION --------------------------------------------------------

    def ingest(self, corpus: dict[str, str] | None = None) -> RagPipeline:
        """Load, split and index the corpus. Returns self for chaining."""
        docs = corpus if corpus is not None else dict(iter_corpus())

        # This is lesson 01's `recursive_chunks`, maintained by someone else.
        # The separator priority list is the same: paragraphs, then newlines,
        # then sentences, then words, then a hard cut.
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            # Measure length in characters. Switching to a token counter here
            # is the right move when you are optimising context-window usage,
            # because 700 characters of prose and 700 characters of JSON are
            # very different token counts.
            length_function=len,
        )

        self.documents = []
        for doc_id, text in sorted(docs.items()):
            for index, piece in enumerate(splitter.split_text(text)):
                # METADATA IS NOT OPTIONAL. doc_id is what makes recall@k and
                # MRR computable without an LLM. A pipeline that indexes bare
                # strings has thrown away its own evaluability.
                self.documents.append(
                    Document(
                        page_content=piece,
                        metadata={"doc_id": doc_id, "chunk_index": index},
                    )
                )

        if not self.documents:
            raise ValueError("ingestion produced no documents -- is the corpus empty?")

        self.store = InMemoryVectorStore.from_documents(self.documents, self.embeddings)
        return self

    # ---- RETRIEVAL --------------------------------------------------------

    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Vector search, converted into our own trace type.

        We deliberately use `similarity_search_with_score` rather than the
        plain variant, because the score is diagnostic: a top result at 0.31
        versus 0.89 tells you whether retrieval was confident, and thresholding
        on it is one of the mitigations listed in
        core/corpus/hallucination.md ("refuse when the best chunk falls below
        a relevance threshold").
        """
        if self.store is None:
            raise RuntimeError("call .ingest() before .retrieve()")

        k = top_k if top_k is not None else self.top_k
        hits = self.store.similarity_search_with_score(question, k=k)

        return [
            RetrievedChunk(
                text=doc.page_content,
                doc_id=doc.metadata.get("doc_id", "unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                score=float(score),
            )
            for doc, score in hits
        ]

    # ---- GENERATION -------------------------------------------------------

    def answer(self, question: str, top_k: int | None = None) -> RagTrace:
        """The full pipeline: retrieve, build the prompt, generate, trace it.

        Written as explicit steps rather than one LCEL pipe, because the point
        of this lesson is that you can SEE the stages. `chain()` below shows
        the idiomatic LCEL version of exactly this.
        """
        # --- retrieve ---
        t0 = time.perf_counter()
        retrieved = self.retrieve(question, top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # --- generate ---
        answer_text = ""
        generation_ms = 0.0
        if self.llm is not None:
            context = format_context([c.text for c in retrieved])
            messages = self.prompt.format_messages(context=context, question=question)

            t1 = time.perf_counter()
            response = self.llm.invoke(messages)
            generation_ms = (time.perf_counter() - t1) * 1000
            answer_text = response.content if hasattr(response, "content") else str(response)

        trace = RagTrace(
            question=question,
            answer=answer_text,
            retrieved=retrieved,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            chat_model=getattr(self.llm, "model", type(self.llm).__name__) if self.llm else "",
            embed_model=type(self.embeddings).__name__,
            chunk_size=self.chunk_size,
            top_k=top_k if top_k is not None else self.top_k,
        )

        # Deterministic citation check, run on every answer at no cost.
        # Catches a model citing [7] when only 4 passages were supplied.
        bad = invalid_citations(answer_text, len(retrieved))
        trace.metadata["invalid_citations"] = sorted(bad)

        return trace

    # ---- THE IDIOMATIC LCEL VERSION ---------------------------------------

    def chain(self):
        """The same pipeline expressed as an LCEL chain.

        Reads as: take a question -> retrieve and format context alongside it
        -> fill the prompt -> call the model -> parse to a string.

            chain = (
                {"context": retriever | format, "question": passthrough}
                | prompt
                | llm
                | StrOutputParser()
            )

        WHY YOU WOULD USE THIS: composing with `|` gives you `.stream()`,
        `.batch()`, `.ainvoke()` and callback instrumentation for every stage
        without writing any of it. Lesson 06 hooks LangWatch into exactly these
        callbacks to trace the pipeline.

        WHY `answer()` ABOVE DOES NOT USE IT: the chain returns a string, which
        discards the retrieved chunks -- the precise mistake core/trace.py warns
        against. To evaluate properly you need both, so production code here
        keeps the explicit version and uses the chain for streaming only.
        """
        if self.llm is None:
            raise RuntimeError("chain() requires an llm")

        from langchain_core.runnables import RunnableLambda, RunnableParallel

        retrieve_and_format = RunnableLambda(
            lambda q: format_context([c.text for c in self.retrieve(q)])
        )

        return (
            RunnableParallel(
                context=retrieve_and_format,
                question=RunnableLambda(lambda q: q),
            )
            | self.prompt
            | self.llm
            | StrOutputParser()
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_offline_pipeline(responses: list[str] | None = None) -> RagPipeline:
    """A fully deterministic pipeline: no server, no network, no GPU.

    Real retrieval over the real corpus with real (lexical) vectors, and a
    scripted model standing in for the generator. This is what the fast test
    tier drives, and it is enough to test prompt assembly, citation checking,
    retrieval quality and trace completeness.
    """
    from core.providers import scripted_chat_model

    llm = scripted_chat_model(responses or ["A grounded answer citing [1]."])
    return RagPipeline(llm=llm, embeddings=LexicalEmbeddings(dim=2048)).ingest()


def build_ollama_pipeline(use_neural_embeddings: bool = True) -> RagPipeline:
    """The real thing: Ollama for generation, nomic-embed-text for vectors."""
    from core.providers import get_chat_model, get_ollama_embeddings

    embeddings = get_ollama_embeddings() if use_neural_embeddings else LexicalEmbeddings(dim=2048)
    return RagPipeline(llm=get_chat_model(), embeddings=embeddings).ingest()
