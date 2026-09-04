"""
Lesson 06 tests -- FAST TIER. No LangWatch account, no network, no model.

Run:  pytest 06_langwatch/ -v

=============================================================================
TEST YOUR INSTRUMENTATION
=============================================================================
Almost nobody does this, and the consequence is always the same: during an
incident you go looking for a field and discover it was never being recorded.

Because LangWatch is built on OpenTelemetry, we can hand it our own
TracerProvider with an in-memory exporter and assert on the exact spans and
attributes our code emits. Instrumentation is code; code gets tested.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for path in (
    str(_HERE),
    str(_HERE.parent / "02_langchain"),
    str(_HERE.parent / "03_langgraph"),
):
    if path not in sys.path:
        sys.path.insert(0, path)

from agent import bind_default_retriever, build_offline_agent
from instrumented import (
    answer_with_tracing,
    run_agent_with_tracing,
    setup_offline_tracing,
    should_sample,
)
from pipeline import build_offline_pipeline
from scripted_model import final_answer, tool_call


@pytest.fixture(scope="module")
def exporter():
    """One offline tracer for the whole module."""
    return setup_offline_tracing()


@pytest.fixture
def spans(exporter):
    """Clear captured spans before each test, return the exporter."""
    exporter.clear()
    return exporter


@pytest.fixture(scope="module")
def pipeline():
    return build_offline_pipeline(["Chunk overlap protects boundary facts [1]."])


def span_named(exporter, name: str):
    for span in exporter.get_finished_spans():
        if span.name == name:
            return span
    raise AssertionError(
        f"no span named {name!r}. Captured: {[s.name for s in exporter.get_finished_spans()]}"
    )


def attr(span, key: str):
    """Read a LangWatch attribute, unwrapping its JSON envelope.

    LangWatch stores input/output as JSON like {"type": "text", "value": "..."},
    so a raw attribute read gives you a string of JSON rather than the value.
    Worth knowing before you write a dashboard query against it.
    """
    raw = span.attributes.get(key)
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(parsed, dict) and "value" in parsed:
        return parsed["value"]
    return parsed


# ===========================================================================
# SPAN STRUCTURE
# ===========================================================================


def test_rag_answer_emits_a_span_per_stage(spans, pipeline):
    """Retrieval and generation get SEPARATE spans.

    A single opaque "rag" span tells you a request was slow. Two tell you which
    half was slow -- the same retriever-versus-generator split that the lesson
    02 UI shows and the lesson 04/05 context metrics measure.
    """
    answer_with_tracing(pipeline, "what is chunk overlap?")

    names = {s.name for s in spans.get_finished_spans()}
    assert {"retrieve", "generate", "rag_answer"} <= names, f"got {names}"


def test_retrieval_span_records_the_contexts_not_just_the_timing(spans, pipeline):
    """Recording only latency is the most common instrumentation mistake.

    It tells you something is wrong without telling you what. The retrieved
    contexts are also what a LangWatch online faithfulness evaluator reads, so
    omitting them makes production evaluation impossible.
    """
    answer_with_tracing(pipeline, "what is chunk overlap?")
    span = span_named(spans, "retrieve")

    assert attr(span, "langwatch.input") == "what is chunk overlap?"
    output = attr(span, "langwatch.output")
    assert output, "retrieved contexts were not recorded -- online eval impossible"


def test_retrieval_span_records_provenance_and_confidence(spans, pipeline):
    """doc_ids and top_score are the two highest-value custom attributes.

    doc_ids let you reconstruct retrieval quality from production traces.
    top_score is the earliest warning that the corpus no longer covers incoming
    traffic -- visible long before anyone complains about answer quality.
    """
    answer_with_tracing(pipeline, "what is chunk overlap?")
    span = span_named(spans, "retrieve")

    metadata = _metadata(span)
    assert metadata["doc_ids"], "no doc_ids recorded"
    assert metadata["n_chunks"] > 0
    assert "top_score" in metadata
    assert metadata["retrieval_ms"] >= 0


def test_generation_span_records_the_model_identity(spans, pipeline):
    """Two traces are not comparable unless you know which model produced each."""
    answer_with_tracing(pipeline, "what is chunk overlap?")
    span = span_named(spans, "generate")

    assert attr(span, "langwatch.output"), "no answer recorded"
    assert "model" in _metadata(span)


def test_span_types_are_set_for_dashboard_grouping(spans, pipeline):
    """LangWatch groups and renders by span type; leaving it default loses that."""
    answer_with_tracing(pipeline, "q")
    assert span_named(spans, "retrieve").attributes.get("langwatch.span.type") == "rag"
    assert span_named(spans, "generate").attributes.get("langwatch.span.type") == "llm"


def _metadata(span) -> dict:
    """Read the JSON-encoded custom attributes back off a span.

    They land under the plain `metadata` key as a JSON string -- see
    `otel_metadata` in instrumented.py for why they must be encoded at all.
    """
    raw = span.attributes.get("metadata")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def test_dict_attributes_are_silently_dropped_by_opentelemetry(spans):
    """Pinning the trap that made three tests above fail.

    OTel span attributes accept only primitives and sequences of primitives.
    Handing `span.update(metadata={...})` a raw dict produces NO error -- the
    span exports happily, just without the fields. You find out during an
    incident, when the attribute you went looking for was never recorded.

    This test proves the encoded form survives and the raw dict does not, so
    nobody 'simplifies' otel_metadata() away.

    NOTE it uses the shared `spans` fixture rather than calling
    setup_offline_tracing() again. Calling setup twice installs a SECOND
    TracerProvider, and every later test in the module then writes to an
    exporter its fixture is not holding -- which presents as "no spans were
    captured" in unrelated tests. Global tracing state is process-wide; set it
    up once.
    """
    from instrumented import otel_metadata

    exporter = spans

    import langwatch

    with langwatch.trace(name="attr_probe", disable_sending=True):
        with langwatch.span(name="encoded", type="rag") as span:
            span.update(metadata=otel_metadata(doc_ids=["a", "b"], n=2))
        with langwatch.span(name="raw_dict", type="rag") as span:
            span.update(metadata={"doc_ids": ["a", "b"], "n": 2})

    encoded = span_named(exporter, "encoded")
    raw = span_named(exporter, "raw_dict")

    assert _metadata(encoded) == {"doc_ids": ["a", "b"], "n": 2}
    assert _metadata(raw) == {}, (
        "OpenTelemetry now accepts dict attributes -- otel_metadata() can be simplified"
    )


# ===========================================================================
# ONLINE EVALUATIONS -- the reference-free ones
# ===========================================================================


def test_deterministic_evaluations_run_without_a_reference_answer(spans, pipeline):
    """The whole point of the online/offline split.

    In production nobody labelled the user's question, so reference-based
    metrics (context recall, answer correctness, recall@k) are unavailable.
    These three need only the trace itself, cost nothing, and can therefore run
    on 100% of traffic.
    """
    result = answer_with_tracing(pipeline, "what is chunk overlap?")
    assert result.answer  # the trace completed


def test_citation_validity_flags_a_fabricated_source(spans):
    """A hallucination detector that runs in production for free."""
    from pipeline import RagPipeline

    from core.providers import LexicalEmbeddings, scripted_chat_model

    liar = RagPipeline(
        llm=scripted_chat_model(["As shown in passage [99]."]),
        embeddings=LexicalEmbeddings(dim=2048),
    ).ingest()

    result = answer_with_tracing(liar, "what is chunk overlap?")
    assert result.metadata["invalid_citations"] == [99]


def test_refusal_is_recorded_as_information_not_failure(spans):
    """Refusing is correct behaviour on an unanswerable question.

    So it is scored but NOT marked as a failure. In production the refusal rate
    is the number to watch: a spike means retrieval broke, a collapse means the
    model started making things up.
    """
    from pipeline import RagPipeline

    from core.providers import LexicalEmbeddings, scripted_chat_model

    from core.golden import REFUSAL

    honest = RagPipeline(
        llm=scripted_chat_model([REFUSAL]), embeddings=LexicalEmbeddings(dim=2048)
    ).ingest()

    result = answer_with_tracing(honest, "What is the capital of France?")
    assert "does not contain" in result.answer.lower()


# ===========================================================================
# SAMPLING
# ===========================================================================


def test_sampling_is_deterministic_for_the_same_question():
    """CRITICAL: sampling must be reproducible.

    With random(), re-running a request samples differently, so you cannot
    reproduce a scored trace while investigating it -- and two services
    handling the same request disagree about whether it was sampled.

    Hashing the question makes the decision stable forever.
    """
    question = "what is chunk overlap?"
    decisions = {should_sample(question, rate=0.5) for _ in range(20)}
    assert len(decisions) == 1, "sampling was not deterministic for a fixed input"


def test_sampling_rate_is_roughly_honoured():
    """A hash-based sampler should still approximate the requested rate."""
    questions = [f"question number {i}" for i in range(1000)]
    sampled = sum(1 for q in questions if should_sample(q, rate=0.1))
    assert 50 <= sampled <= 160, f"sampled {sampled}/1000 at rate 0.1 -- distribution is skewed"


def test_rate_zero_and_one_are_absolute():
    assert not should_sample("anything", rate=0.0)
    assert should_sample("anything", rate=1.0)


# ===========================================================================
# AGENT TRACING
# ===========================================================================


def test_agent_run_emits_a_span_per_tool_call(spans):
    """Agents need MORE instrumentation, not less, because the trajectory varies.

    A span per tool call is what lets you answer "why did this take 40 seconds?"
    -- almost always because the agent looped, which is invisible from the final
    answer.
    """
    bind_default_retriever()
    agent = build_offline_agent(
        [
            tool_call("search_knowledge_base", call_id="c1", query="chunk overlap"),
            tool_call("reciprocal_rank", call_id="c2", position=4),
            final_answer("Overlap protects boundary facts [1]. RR at 4 is 0.25."),
        ]
    )
    run_agent_with_tracing(agent, "explain chunk overlap and reciprocal rank")

    names = [s.name for s in spans.get_finished_spans()]
    assert "tool:search_knowledge_base" in names
    assert "tool:reciprocal_rank" in names
    assert "agent_run" in names


def test_agent_tool_spans_record_arguments(spans):
    """Argument correctness is a distinct failure from tool selection.

    Dropping arguments from the span makes it undiagnosable from a trace.
    """
    bind_default_retriever()
    agent = build_offline_agent(
        [tool_call("reciprocal_rank", position=4), final_answer("0.25")]
    )
    run_agent_with_tracing(agent, "q")

    span = span_named(spans, "tool:reciprocal_rank")
    recorded = attr(span, "langwatch.input")
    assert recorded, "tool arguments were not recorded"


def test_looping_agent_is_visible_in_its_trace(spans):
    """A production agent that loops must be diagnosable from the trace alone."""
    bind_default_retriever()
    agent = build_offline_agent([tool_call("list_documents")], max_steps=4)
    result = run_agent_with_tracing(agent, "this loops")

    assert result.hit_step_limit
    tool_spans = [s for s in spans.get_finished_spans() if s.name.startswith("tool:")]
    assert len(tool_spans) > 1, "a looping agent produced only one tool span"


# ===========================================================================
# SAAS TIER
# ===========================================================================


@pytest.mark.saas
def test_real_langwatch_setup_requires_an_api_key():
    """Only runs if you have configured LANGWATCH_API_KEY.

    Everything above runs offline. This is the single test that touches the
    hosted platform, and it is marked `saas` so it never runs in CI.
    """
    import os

    api_key = os.environ.get("LANGWATCH_API_KEY")
    if not api_key:
        pytest.skip(
            "LANGWATCH_API_KEY not set. Sign up at langwatch.ai, then: "
            "export LANGWATCH_API_KEY=sk-lw-..."
        )

    import langwatch

    langwatch.setup(api_key=api_key)
    pipeline = build_offline_pipeline(["A traced answer [1]."])
    result = answer_with_tracing(pipeline, "what is chunk overlap?", offline=False)
    assert result.answer
    print("\nTrace sent. Check your LangWatch dashboard.")
