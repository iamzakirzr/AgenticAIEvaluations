"""
Lesson 02 tests -- FAST TIER unless marked.

Run:  pytest 02_langchain/ -v
      pytest -m ollama 02_langchain/ -v

The theme of this file: you can test almost everything about a RAG pipeline
WITHOUT a language model. Prompt assembly, retrieval quality, citation
validation, trace completeness, refusal-path plumbing -- all deterministic.
Only the question "is the generated prose any good?" needs a model, and that
is what lessons 04 and 05 are for.

Knowing which half is which is the difference between a test suite that runs
in one second and one that runs in twenty minutes and gets switched off.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pipeline import RagPipeline, build_offline_pipeline
from prompts import (
    GROUNDED_PROMPT,
    NAIVE_PROMPT,
    extract_citations,
    format_context,
    invalid_citations,
)

from core.golden import load_golden
from core.metrics import evaluate_retrieval
from core.providers import LexicalEmbeddings, scripted_chat_model


@pytest.fixture(scope="module")
def pipeline():
    return build_offline_pipeline(["Chunk overlap protects boundary facts [1]."])


# ===========================================================================
# INGESTION
# ===========================================================================


def test_ingestion_produces_documents_with_provenance(pipeline):
    """Every chunk must carry doc_id, or retrieval metrics become impossible.

    A pipeline that indexes bare strings has thrown away its own evaluability:
    recall@k and MRR need to know which source a chunk came from.
    """
    assert len(pipeline.documents) > 20
    for doc in pipeline.documents:
        assert doc.metadata.get("doc_id"), "chunk is missing doc_id metadata"
        assert "chunk_index" in doc.metadata


def test_retrieve_before_ingest_fails_clearly():
    with pytest.raises(RuntimeError, match=r"call \.ingest\(\)"):
        RagPipeline(embeddings=LexicalEmbeddings()).retrieve("anything")


def test_empty_corpus_fails_loudly():
    """Silent success on an empty index would produce a system that refuses
    everything and looks 'safe' while being completely broken."""
    with pytest.raises(ValueError, match="no documents"):
        RagPipeline(embeddings=LexicalEmbeddings()).ingest({})


def test_smaller_chunks_produce_more_documents():
    small = RagPipeline(embeddings=LexicalEmbeddings(), chunk_size=200, chunk_overlap=20).ingest()
    large = RagPipeline(embeddings=LexicalEmbeddings(), chunk_size=1500, chunk_overlap=20).ingest()
    assert len(small.documents) > len(large.documents)


# ===========================================================================
# PROMPT ASSEMBLY -- no model needed
# ===========================================================================


def test_context_passages_are_numbered_for_citation():
    """Numbering is what makes citation hallucination deterministically checkable."""
    formatted = format_context(["first passage", "second passage"])
    assert "[1] first passage" in formatted
    assert "[2] second passage" in formatted


def test_empty_context_is_explicit_not_blank():
    """A blank context silently invites the model to answer from memory.

    Saying so out loud gives the model something to refuse against.
    """
    assert "no passages" in format_context([]).lower()


def test_grounded_prompt_contains_the_refusal_instruction():
    """The claim from core/corpus/hallucination.md, pinned as a test.

    If someone 'tidies up' the prompt and removes this line, hallucination on
    unanswerable questions rises and no other test would catch it.
    """
    rendered = GROUNDED_PROMPT.format_messages(context="x", question="y")
    system = rendered[0].content
    assert "does not contain this information" in system
    assert "ONLY" in system


def test_grounded_prompt_instructs_correcting_false_premises():
    """Targets sycophancy on the adversarial golden items."""
    system = GROUNDED_PROMPT.format_messages(context="x", question="y")[0].content
    assert "contradicts" in system.lower()


def test_naive_prompt_is_the_control_and_lacks_those_instructions():
    """The A/B pair must actually differ, or the experiment in lesson 04 is void."""
    naive = NAIVE_PROMPT.format_messages(context="x", question="y")[0].content
    assert "does not contain this information" not in naive
    assert len(naive) < 200


def test_prompt_carries_the_question_and_context_through():
    messages = GROUNDED_PROMPT.format_messages(context="[1] the sky is blue", question="colour?")
    human = messages[-1].content
    assert "the sky is blue" in human
    assert "colour?" in human


# ===========================================================================
# CITATION VALIDATION -- a deterministic hallucination check
# ===========================================================================


def test_extract_citations_parses_markers():
    assert extract_citations("Supported by [1] and [3].") == {1, 3}
    assert extract_citations("No citations here.") == set()


def test_invalid_citations_detects_a_fabricated_source():
    """The whole point: catching a made-up citation with no LLM at all.

    Only 3 passages were supplied, so [7] cannot exist. This is citation
    hallucination, caught by an integer comparison -- cheaper, faster and more
    reliable than any judge.
    """
    assert invalid_citations("As shown in [7].", n_passages=3) == {7}
    assert invalid_citations("As shown in [2].", n_passages=3) == set()


def test_citation_zero_is_invalid():
    assert invalid_citations("See [0].", n_passages=3) == {0}


def test_pipeline_records_the_citation_check(pipeline):
    trace = pipeline.answer("what is chunk overlap?")
    assert "invalid_citations" in trace.metadata


def test_pipeline_flags_a_model_that_fabricates_a_citation():
    """Drive the pipeline with a model that deliberately cites out of range."""
    bad_pipeline = RagPipeline(
        llm=scripted_chat_model(["This is explained in passage [99]."]),
        embeddings=LexicalEmbeddings(dim=2048),
    ).ingest()

    trace = bad_pipeline.answer("what is chunk overlap?", top_k=3)
    assert trace.metadata["invalid_citations"] == [99]


# ===========================================================================
# THE TRACE
# ===========================================================================


def test_answer_returns_a_full_trace_not_a_string(pipeline):
    """core/trace.py's central argument, asserted.

    Without contexts, faithfulness and context precision/recall are all
    uncomputable. A pipeline that returns only a string cannot be evaluated.
    """
    trace = pipeline.answer("what is chunk overlap?")
    assert trace.answer
    assert trace.contexts, "no retrieved contexts -- half the metrics become impossible"
    assert trace.retrieved_doc_ids
    assert trace.retrieval_ms > 0
    assert trace.generation_ms > 0


def test_trace_records_the_configuration_that_produced_it(pipeline):
    """Two eval runs are incomparable unless you know their settings."""
    trace = pipeline.answer("anything")
    assert trace.chunk_size > 0
    assert trace.top_k > 0
    assert trace.embed_model


def test_top_k_is_respected(pipeline):
    assert len(pipeline.answer("chunking", top_k=2).retrieved) == 2
    assert len(pipeline.answer("chunking", top_k=6).retrieved) == 6


def test_retrieval_only_mode_works_without_a_model():
    """You should be able to measure and tune retrieval before owning a chatbot.

    This is the correct order to build a RAG system in: get retrieval right
    first, because no prompt fixes a missing document.
    """
    retrieval_only = RagPipeline(llm=None, embeddings=LexicalEmbeddings(dim=2048)).ingest()
    trace = retrieval_only.answer("what is HNSW?")
    assert trace.answer == ""
    assert trace.contexts
    assert "vector_stores" in trace.retrieved_doc_ids


# ===========================================================================
# RETRIEVAL QUALITY GATE (LangChain path)
# ===========================================================================


def test_langchain_retrieval_matches_the_from_scratch_quality(pipeline):
    """Sanity check that the framework did not make retrieval worse.

    Lesson 01's hand-written retriever scored MRR ~0.99. LangChain's splitter
    and vector store should land in the same region. A large gap would mean a
    configuration mismatch, most likely in the splitter separators.
    """
    golden = [item for item in load_golden() if item.is_answerable]
    results = {
        item.id: (pipeline.answer(item.question).retrieved_doc_ids, item.reference_doc_ids)
        for item in golden
    }
    report = evaluate_retrieval(results)

    print("\n" + report.format_table())
    assert report.hit_rate >= 0.90, f"misses: {report.failures()}"
    assert report.mrr >= 0.85


def test_unanswerable_questions_still_retrieve_something(pipeline):
    """Vector search ALWAYS returns top_k; it has no concept of 'no match'.

    This is the single most important thing to understand about why RAG
    hallucinates. The retriever hands the model 4 irrelevant passages with a
    straight face, and unless the prompt tells it to refuse, the model will
    write something plausible from them.

    The defence is the prompt (lesson 02) and the measurement is refusal rate
    (lesson 04) -- not anything the retriever can do.
    """
    unanswerable = [i for i in load_golden() if not i.is_answerable]
    trace = pipeline.answer(unanswerable[0].question)
    assert len(trace.retrieved) == pipeline.top_k, (
        "retrieval returned fewer than top_k -- if it ever returns 0, the "
        "refusal path is being handled by the retriever, not the prompt"
    )


# ===========================================================================
# THE HTTP SERVICE
# ===========================================================================


@pytest.fixture(scope="module")
def client():
    import server

    # Force the deterministic pipeline so this test never depends on Ollama.
    server._pipeline = build_offline_pipeline(["Offline answer citing [1]."])
    return TestClient(server.app)


def test_health_reports_the_active_configuration(client):
    body = client.get("/health").json()
    for key in ("ollama", "chat_model", "embed_model", "chunk_size", "top_k"):
        assert key in body


def test_ask_returns_sources_not_just_an_answer(client):
    """The API contract that makes the service evaluable from outside."""
    body = client.post("/ask", json={"question": "what is chunk overlap?"}).json()

    assert body["answer"]
    assert body["sources"], "an answer with no sources cannot be checked by anyone"
    first = body["sources"][0]
    for key in ("n", "doc_id", "chunk_index", "score", "text"):
        assert key in first
    assert body["timing_ms"]["total"] >= 0


def test_ask_respects_top_k(client):
    body = client.post("/ask", json={"question": "chunking", "top_k": 2}).json()
    assert len(body["sources"]) == 2


def test_ui_renders_and_exposes_the_sources_panel(client):
    html = client.get("/").text
    assert "Retrieved passages" in html
    assert "unanswerable" in html  # the example-question links


# ===========================================================================
# OLLAMA TIER -- real generation
# ===========================================================================


@pytest.mark.ollama
def test_real_model_answers_a_simple_question_from_context():
    from pipeline import build_ollama_pipeline

    from core.providers import ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    pipeline = build_ollama_pipeline()
    trace = pipeline.answer("How many dimensions does nomic-embed-text produce?")
    print("\n" + trace.summary())
    assert "768" in trace.answer, f"expected 768 in the answer, got: {trace.answer!r}"


@pytest.mark.ollama
def test_real_model_refuses_an_unanswerable_question():
    """The grounding instruction, tested against a real model.

    NOTE this is a single sample, so treat a pass as weak evidence. Lesson 04
    measures refusal rate across the whole unanswerable set, which is the
    number you would actually report.
    """
    from pipeline import build_ollama_pipeline

    from core.providers import ollama_available

    if not ollama_available():
        pytest.skip("Ollama not running")

    trace = build_ollama_pipeline().answer("What is the capital of France?")
    print("\n" + trace.summary())
    lowered = trace.answer.lower()
    refused = "does not contain" in lowered or "not contain this information" in lowered
    assert refused, (
        f"model leaked parametric knowledge instead of refusing: {trace.answer!r}\n"
        "This is exactly the failure the unanswerable golden category exists to catch."
    )
