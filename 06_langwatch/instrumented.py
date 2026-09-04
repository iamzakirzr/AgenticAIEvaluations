"""
Tracing the RAG pipeline and the agent with LangWatch.

=============================================================================
OBSERVABILITY IS NOT EVALUATION -- and the difference decides your metrics
=============================================================================
Lessons 04 and 05 did OFFLINE evaluation: a fixed golden dataset, known
reference answers, run before you ship.

This lesson is about ONLINE evaluation: real user traffic, in production,
where you have something you never have offline -- the actual questions people
ask -- and lack the one thing every offline metric depends on:

    IN PRODUCTION THERE IS NO REFERENCE ANSWER.

Nobody labelled the user's question. So the metrics split cleanly:

    WORKS ONLINE (reference-free)        NEEDS A LABEL (offline only)
    ------------------------------       -----------------------------
    Faithfulness                         Context Recall
    Answer Relevancy                     Context Precision (w/ reference)
    Context Relevancy                    Factual Correctness
    Toxicity / Bias / PII                Answer Correctness
    Refusal rate (regex)                 recall@k, MRR, nDCG
    Latency, cost, token counts          Noise Sensitivity

That table is the practical reason to know both offline and online evaluation:
they can measure different things, and a production dashboard that promises
"answer correctness" on unlabelled traffic is measuring something else.

=============================================================================
WHY THIS LESSON IS TESTABLE WITHOUT A LANGWATCH ACCOUNT
=============================================================================
LangWatch is a hosted platform -- the dashboards, annotation queues and online
evaluators live on their servers and need an API key.

But `langwatch` 1.3.1 is built on OpenTelemetry. That means we can hand it our
OWN TracerProvider with an in-memory exporter, set `disable_sending=True`, and
assert on the exact spans our code produces -- no account, no network, no key.

That is a genuinely useful pattern beyond this lesson: **instrumentation is
code, and code should be tested.** Most teams never test their tracing and
discover in an incident that the field they needed was never being recorded.
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for path in (str(_ROOT), str(_ROOT / "02_langchain"), str(_ROOT / "03_langgraph")):
    if path not in sys.path:
        sys.path.insert(0, path)

import langwatch  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from core.trace import RagTrace  # noqa: E402


# ===========================================================================
# A TRAP THAT SILENTLY LOSES YOUR DATA
# ===========================================================================


def otel_metadata(**fields) -> str:
    """JSON-encode custom attributes so OpenTelemetry actually keeps them.

    OTel span attributes may only be primitives (bool/str/int/float) or
    sequences of primitives. Pass a DICT and it is DROPPED -- not an error, just
    a log line you will never read:

        WARNING opentelemetry.attributes: Invalid type dict for attribute
        'metadata' value. Expected one of ['bool','str','bytes','int','float']
        or a sequence of those types

    The span still exports. It just silently lacks the fields you needed. This
    is the classic instrumentation failure: everything looks fine until an
    incident, when the attribute you went looking for was never there.

    We encode to JSON, which round-trips structure and lands as one readable
    attribute. (Flattening to scalar keys via `span.set_attributes()` also
    works, and is better if you intend to filter on individual fields in a
    dashboard query.)
    """
    return json.dumps(fields, default=str)


# ===========================================================================
# SETUP
# ===========================================================================


def setup_offline_tracing() -> InMemorySpanExporter:
    """Point LangWatch at an in-memory exporter so nothing leaves the process.

    Returns the exporter, whose `.get_finished_spans()` is what the tests
    assert on.

    NOTE `disable_sending=True` must ALSO be passed per-trace (see
    `traced_rag_answer` below). Without it, LangWatch attaches its own OTLP
    exporter and you get `401 Unauthorized` noise on every span -- which is how
    we found out, and is exactly the kind of thing you want to discover in a
    test rather than in CI logs.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # SimpleSpanProcessor exports synchronously on span end. The batching
    # processor you would use in production would make assertions racy.
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    langwatch.setup(api_key="offline-testing-no-network", tracer_provider=provider)
    return exporter


def setup_production_tracing(api_key: str) -> None:
    """The real thing. Requires a LangWatch account.

        export LANGWATCH_API_KEY=sk-lw-...
        python -c "import langwatch; langwatch.setup()"

    Once configured, traces appear in the LangWatch dashboard where you can run
    online evaluators, build annotation queues, and -- most valuably -- turn
    real production traffic into new golden-dataset items.
    """
    langwatch.setup(api_key=api_key)


# ===========================================================================
# INSTRUMENTING THE RAG PIPELINE
# ===========================================================================


@contextmanager
def traced_rag_answer(question: str, offline: bool = True):
    """Open a trace for one question, with a span per pipeline stage.

    THE SPAN STRUCTURE IS THE DESIGN DECISION HERE:

        trace: rag_answer
          |- span: retrieve   (type="rag")   input=question  output=chunks
          |- span: generate   (type="llm")   input=prompt    output=answer

    Separate spans for retrieval and generation, because that is the split that
    matters when something goes wrong -- exactly the same
    retriever-versus-generator distinction the chatbot UI surfaces in lesson 02
    and the context metrics measure in lessons 04 and 05.

    A single opaque "rag" span would tell you a request was slow. These two tell
    you WHICH HALF was slow.
    """
    with langwatch.trace(name="rag_answer", disable_sending=offline) as trace:
        trace.update(metadata={"question": question})  # trace metadata IS handled
        yield trace


def answer_with_tracing(pipeline, question: str, offline: bool = True) -> RagTrace:
    """Run the lesson-02 pipeline with full span instrumentation.

    Records on each span the things you will actually want during an incident:
    what went in, what came out, how long it took, and how many chunks came
    back. Recording only latency is the most common instrumentation mistake --
    it tells you something is wrong without ever telling you what.
    """
    with traced_rag_answer(question, offline=offline) as trace:
        # ---- retrieval span ------------------------------------------------
        with langwatch.span(name="retrieve", type="rag") as span:
            started = time.perf_counter()
            chunks = pipeline.retrieve(question)
            retrieval_ms = (time.perf_counter() - started) * 1000

            span.update(
                input=question,
                output=[c.text for c in chunks],
                # Custom attributes. `contexts` is what a LangWatch online
                # faithfulness evaluator would read, so recording it is what
                # makes production evaluation possible at all.
                metadata=otel_metadata(
                    doc_ids=[c.doc_id for c in chunks],
                    scores=[round(c.score, 4) for c in chunks],
                    n_chunks=len(chunks),
                    retrieval_ms=round(retrieval_ms, 2),
                    # The top score is a cheap, powerful production signal: a
                    # sudden drop across traffic means the index or the query
                    # distribution changed, and it is visible long before
                    # anyone files a complaint about answer quality.
                    top_score=round(chunks[0].score, 4) if chunks else 0.0,
                ),
            )

        # ---- generation span ------------------------------------------------
        with langwatch.span(name="generate", type="llm") as span:
            started = time.perf_counter()
            result = pipeline.answer(question)
            generation_ms = (time.perf_counter() - started) * 1000

            span.update(
                input=question,
                output=result.answer,
                metadata=otel_metadata(
                    model=result.chat_model or "scripted",
                    generation_ms=round(generation_ms, 2),
                    invalid_citations=result.metadata.get("invalid_citations", []),
                ),
            )

        # ---- attach reference-free evaluations ------------------------------
        # These run on EVERY production request because they are deterministic
        # and free. Anything needing a judge should be sampled instead (see
        # should_sample below).
        attach_deterministic_evaluations(result)

        return result


def attach_deterministic_evaluations(result: RagTrace) -> None:
    """Score a live trace using only checks that need no reference answer.

    This is the online half of the metric table at the top of this file. Each
    check is computable from the trace alone, costs nothing, and never flakes --
    so it can run on 100% of traffic.

    API NOTE (langwatch 1.3.1): evaluations attach to a SPAN, not a trace.
    `trace.add_evaluation(...)` is deprecated and now raises

        ValueError: No span or trace found, could not add evaluation to span

    because it forwards to the span-based implementation with span=None. So we
    open a dedicated span of type "evaluation" and attach there -- which also
    renders as its own grouped section in the LangWatch dashboard, rather than
    scattering scores across unrelated spans.
    """
    with langwatch.span(name="online_evaluations", type="evaluation") as span:
        # 1. Citation validity. From lesson 02: a citation numbered higher than
        #    the passages supplied is a fabricated source, caught by an integer
        #    comparison rather than a judge.
        bad_citations = result.metadata.get("invalid_citations", [])
        span.add_evaluation(
            name="citation_validity",
            passed=not bad_citations,
            score=0.0 if bad_citations else 1.0,
            details=f"fabricated citations: {bad_citations}" if bad_citations else "all valid",
        )

        # 2. Refusal detection. From lesson 04: because the prompt mandates an
        #    exact refusal string, a regex is a complete detector. In production
        #    this is the number to watch -- a spike means retrieval broke, a
        #    collapse means the model started making things up.
        refused = "does not contain this information" in result.answer.lower()
        span.add_evaluation(
            name="refused",
            passed=True,  # refusing is not a failure; it is information
            score=1.0 if refused else 0.0,
            label="refused" if refused else "answered",
        )

        # 3. Retrieval confidence. Not a quality metric on its own, but the
        #    earliest available warning that the corpus does not cover incoming
        #    traffic.
        top_score = max((c.score for c in result.retrieved), default=0.0)
        span.add_evaluation(
            name="retrieval_confidence",
            passed=top_score > 0.1,
            score=float(top_score),
            details=f"best chunk similarity {top_score:.4f}",
        )


# ===========================================================================
# SAMPLING
# ===========================================================================


def should_sample(question: str, rate: float = 0.1) -> bool:
    """Decide whether to run EXPENSIVE (judged) evaluation on this request.

    WHY SAMPLING IS NOT OPTIONAL: a judged metric costs a full LLM call. Running
    faithfulness on every production request roughly doubles your inference bill
    and adds latency to a user-facing path.

    WHY THE HASH RATHER THAN random(): sampling must be DETERMINISTIC per
    question. With `random()`, re-running the same request samples differently,
    so you cannot reproduce a scored trace when investigating it -- and two
    services processing the same request disagree about whether it was sampled.

    Hashing the question means the same input is always sampled or always not.
    """
    import hashlib

    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    # First 8 hex digits -> an integer in [0, 16^8), scaled to [0, 1).
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < rate


# ===========================================================================
# INSTRUMENTING THE AGENT
# ===========================================================================


def run_agent_with_tracing(agent, question: str, offline: bool = True):
    """Trace an agent run, with one span per tool call.

    Agents need MORE instrumentation than pipelines, not less, because the
    trajectory is variable. Recording a span per tool call is what makes it
    possible to answer "why did this request take 40 seconds?" -- almost always
    because the agent looped, which is invisible from the final answer alone.
    """
    with langwatch.trace(name="agent_run", disable_sending=offline) as trace:
        trace.update(metadata={"question": question})  # trace metadata IS handled

        result = agent.run(question)

        for index, call in enumerate(result.tool_calls):
            with langwatch.span(name=f"tool:{call.name}", type="tool") as span:
                span.update(
                    input=call.args,
                    output=call.result[:2000],
                    metadata=otel_metadata(step=index + 1, tool=call.name),
                )

        # Trajectory health as evaluations -- all deterministic, from lesson 03.
        # Attached to a span, not the trace: see attach_deterministic_evaluations.
        with langwatch.span(name="trajectory_evaluations", type="evaluation") as span:
            span.add_evaluation(
                name="terminated_cleanly",
                passed=not result.hit_step_limit,
                score=0.0 if result.hit_step_limit else 1.0,
                details=f"{result.steps} steps",
            )
            repeats = result.repeated_calls()
            span.add_evaluation(
                name="no_repeated_calls",
                passed=not repeats,
                score=0.0 if repeats else 1.0,
                details=f"repeated: {repeats}" if repeats else "no loops",
            )
            span.add_evaluation(
                name="refused",
                passed=True,
                score=1.0 if result.refused else 0.0,
                label="refused" if result.refused else "answered",
            )

        return result
