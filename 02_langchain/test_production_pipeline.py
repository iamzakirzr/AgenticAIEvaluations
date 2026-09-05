"""
Lesson 02 production scenarios. FAST TIER -- no model, no network, no sleeping.

Run:  pytest 02_langchain/test_production.py -v

Every guard here is deterministic, which is the point: a safety property moved
out of the model's judgement and into code is cheaper, testable, and cannot be
talked out of its decision by a prompt injection.
"""

from __future__ import annotations

import pytest
from pipeline import RagPipeline
from production_pipeline import (
    GuardStats,
    ProductionRagPipeline,
    ThresholdPolicy,
    redact,
    suggest_threshold,
)

from core.golden import REFUSAL, load_golden
from core.providers import LexicalEmbeddings, scripted_chat_model
from core.resilience import Budget, BudgetExceeded, CircuitBreaker, RetryPolicy
from core.trace import RagTrace, RetrievedChunk

NO_SLEEP = lambda _: None


class AlwaysFails:
    """A model that is down. Named to match core.resilience.default_retryable."""

    model = "primary-that-is-down"

    def __init__(self, exc_type=None):
        self.calls = 0
        self.exc_type = exc_type or ConnectionError

    def invoke(self, messages):
        self.calls += 1
        raise self.exc_type("model server unavailable")


def build(**kwargs) -> ProductionRagPipeline:
    base = RagPipeline(
        llm=scripted_chat_model(["A grounded answer citing [1]."]),
        embeddings=LexicalEmbeddings(dim=2048),
    ).ingest()
    kwargs.setdefault("retry_policy", RetryPolicy(max_attempts=2))
    return ProductionRagPipeline(base, **kwargs)


# ===========================================================================
# PII REDACTION
# ===========================================================================


def test_redacts_the_common_identifier_types():
    result = redact(
        "Email ada@example.com or call +44 7700 900123. Card 4111 1111 1111 1111."
    )
    assert "ada@example.com" not in result.text
    assert "4111" not in result.text
    assert "[EMAIL]" in result.text
    assert "[CARD]" in result.text
    assert result.had_pii


def test_redaction_preserves_sentence_shape():
    """Placeholders rather than deletion.

    Removing the text outright often makes the question unanswerable; keeping
    the shape lets the model still understand what was asked.
    """
    result = redact("Please send the invoice to billing@acme.co.uk today.")
    assert result.text == "Please send the invoice to [EMAIL] today."


def test_card_numbers_are_labelled_as_cards_not_phones():
    """Pattern ORDER matters. A 16-digit card also matches a long digit run,
    so the more specific pattern has to win or you lose the label that decides
    your retention policy."""
    result = redact("4111111111111111")
    assert "[CARD]" in result.text
    assert "[PHONE]" not in result.text


def test_clean_text_is_untouched_and_reports_no_pii():
    result = redact("What is chunk overlap for?")
    assert result.text == "What is chunk overlap for?"
    assert not result.had_pii
    assert result.found == {}


def test_redaction_counts_are_reported_for_alerting():
    """'We redacted 400 emails today' is an operational signal.

    A sudden spike usually means a new integration started forwarding raw
    customer records into the chat box.
    """
    result = redact("a@b.com and c@d.com and e@f.com")
    assert result.found["EMAIL"] == 3


def test_pipeline_redacts_before_the_prompt_sees_the_question():
    """The ordering property that makes redaction worth anything.

    If redaction happened after prompt assembly, the raw value would already
    be in the prompt, the logs and the trace.
    """
    pipeline = build()
    trace = pipeline.answer("My email is ada@example.com, what is chunk overlap?")

    assert "ada@example.com" not in trace.question
    assert "[EMAIL]" in trace.question
    assert trace.metadata["pii_found"]["EMAIL"] == 1
    assert pipeline.stats.pii_redacted == 1


def test_redaction_can_be_disabled_explicitly():
    pipeline = build(redact_pii=False)
    trace = pipeline.answer("mail me at ada@example.com about chunk overlap")
    assert "ada@example.com" in trace.question


# ===========================================================================
# RELEVANCE THRESHOLD
# ===========================================================================


def make_trace(scores: list[float]) -> RagTrace:
    return RagTrace(
        question="q",
        answer="",
        retrieved=[RetrievedChunk(f"c{i}", "doc", i, s) for i, s in enumerate(scores)],
    )


def test_threshold_refuses_when_nothing_clears_the_bar():
    policy = ThresholdPolicy(min_top_score=0.25)
    assert policy.should_refuse(make_trace([0.10, 0.08, 0.02]))
    assert not policy.should_refuse(make_trace([0.30, 0.08]))


def test_threshold_refuses_on_empty_retrieval():
    assert ThresholdPolicy().should_refuse(make_trace([]))


def test_requiring_several_supporting_chunks_guards_against_a_lucky_match():
    """One chunk clearing the bar can be a single keyword coincidence."""
    policy = ThresholdPolicy(min_top_score=0.25, min_supporting_chunks=2)
    assert policy.should_refuse(make_trace([0.90, 0.05, 0.01]))
    assert not policy.should_refuse(make_trace([0.90, 0.40, 0.01]))


def test_refusal_happens_before_generation_so_it_costs_nothing():
    """The operational argument for a threshold, not just the safety one.

    A refusal that never calls the model costs zero tokens and returns in
    milliseconds.
    """
    llm = AlwaysFails()  # would raise if it were ever called
    base = RagPipeline(llm=llm, embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(
        base, threshold=ThresholdPolicy(min_top_score=0.99)
    )

    trace = pipeline.answer("what is chunk overlap?")

    assert trace.answer == REFUSAL
    assert llm.calls == 0, "the model was called despite the threshold refusing"
    assert trace.generation_ms == 0.0
    assert trace.metadata["refused_reason"] == "below relevance threshold"
    assert pipeline.stats.refused_low_relevance == 1


def test_a_confident_question_is_answered_normally():
    pipeline = build(threshold=ThresholdPolicy(min_top_score=0.05))
    trace = pipeline.answer("what does chunk overlap protect against?")

    assert trace.answer != REFUSAL
    assert trace.metadata["top_score"] > 0.05


def test_suggest_threshold_picks_a_value_from_data_and_prices_it():
    """A threshold picked by intuition either refuses everything or nothing.

    This derives the smallest threshold that refuses every unanswerable
    question, and reports the false-refusal rate that buys -- both directions
    measured together, as core/corpus/hallucination.md insists.
    """
    answerable = [0.40, 0.35, 0.50, 0.12]
    unanswerable = [0.10, 0.08, 0.05]

    threshold, false_refusals = suggest_threshold(answerable, unanswerable)

    assert threshold > 0.10
    assert threshold < 0.12
    assert false_refusals == 0.0

    # Now with an answerable question that scores below the worst unanswerable:
    _, false2 = suggest_threshold([0.40, 0.09], [0.10])
    assert false2 == 0.5, "the cost of the threshold was not reported"


def test_suggest_threshold_handles_no_unanswerable_examples():
    assert suggest_threshold([0.5], []) == (0.0, 0.0)


def test_threshold_derived_from_the_real_golden_set():
    """End to end on real data: do unanswerable questions actually score lower?

    If they do not, a relevance threshold cannot help you on this corpus, and
    finding that out is the point of measuring rather than assuming.
    """
    base = RagPipeline(
        llm=scripted_chat_model(["x"]), embeddings=LexicalEmbeddings(dim=2048)
    ).ingest()

    golden = load_golden()
    answerable, unanswerable = [], []
    for item in golden:
        chunks = base.retrieve(item.question)
        top = max((c.score for c in chunks), default=0.0)
        (answerable if item.is_answerable else unanswerable).append(top)

    threshold, false_rate = suggest_threshold(answerable, unanswerable)
    print(
        f"\nsuggested threshold {threshold:.4f} "
        f"(false refusal rate {false_rate:.0%} on answerable questions)"
    )
    assert 0.0 <= false_rate <= 1.0
    assert threshold > 0.0


# ===========================================================================
# FALLBACK, RETRY, CIRCUIT BREAKER
# ===========================================================================


def test_fallback_answers_when_the_primary_model_is_down():
    primary = AlwaysFails()
    base = RagPipeline(llm=primary, embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(
        base,
        fallback_llm=scripted_chat_model(["Degraded but useful answer [1]."]),
        threshold=ThresholdPolicy(min_top_score=0.0),
        retry_policy=RetryPolicy(max_attempts=2),
    )

    trace = pipeline.answer("what is chunk overlap?", sleep=NO_SLEEP)

    assert "Degraded but useful" in trace.answer
    assert pipeline.stats.fallback_used == 1


def test_the_trace_records_which_model_actually_answered():
    """NOT OPTIONAL. If a fallback silently serves 30% of traffic, your quality
    metrics are the average of two different systems and every conclusion drawn
    from them is wrong."""
    base = RagPipeline(llm=AlwaysFails(), embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(
        base,
        fallback_llm=scripted_chat_model(["fallback answer"]),
        threshold=ThresholdPolicy(min_top_score=0.0),
        retry_policy=RetryPolicy(max_attempts=1),
    )

    trace = pipeline.answer("what is chunk overlap?", sleep=NO_SLEEP)
    assert "fallback" in trace.metadata["model_used"]


def test_without_a_fallback_the_failure_propagates():
    """Failing loudly beats returning an empty answer that looks like a refusal."""
    base = RagPipeline(llm=AlwaysFails(), embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(
        base, threshold=ThresholdPolicy(min_top_score=0.0),
        retry_policy=RetryPolicy(max_attempts=1),
    )

    with pytest.raises(ConnectionError):
        pipeline.answer("what is chunk overlap?", sleep=NO_SLEEP)
    assert pipeline.stats.failures == 1


def test_transient_failures_are_retried_before_falling_back():
    """A blip should not demote you to the fallback model."""

    class FlakyThenFine:
        model = "flaky"

        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("transient reset")
            from langchain_core.messages import AIMessage

            return AIMessage(content="recovered answer")

    llm = FlakyThenFine()
    base = RagPipeline(llm=llm, embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(
        base,
        fallback_llm=scripted_chat_model(["should not be used"]),
        threshold=ThresholdPolicy(min_top_score=0.0),
        retry_policy=RetryPolicy(max_attempts=3),
    )

    trace = pipeline.answer("what is chunk overlap?", sleep=NO_SLEEP)

    assert trace.answer == "recovered answer"
    assert pipeline.stats.fallback_used == 0, "fell back despite recovering on retry"
    assert pipeline.stats.retries >= 1


def test_the_circuit_breaker_stops_hammering_a_dead_service():
    primary = AlwaysFails()
    base = RagPipeline(llm=primary, embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(
        base,
        fallback_llm=scripted_chat_model(["fallback"]),
        threshold=ThresholdPolicy(min_top_score=0.0),
        retry_policy=RetryPolicy(max_attempts=2),
        breaker=CircuitBreaker(failure_threshold=3, reset_after=60),
    )

    for _ in range(4):
        pipeline.answer("what is chunk overlap?", sleep=NO_SLEEP)

    calls_after_open = primary.calls
    pipeline.answer("what is chunk overlap?", sleep=NO_SLEEP)

    assert primary.calls == calls_after_open, (
        "the open circuit still called through to a service known to be down"
    )


# ===========================================================================
# BUDGET
# ===========================================================================


def test_budget_stops_a_runaway_batch():
    pipeline = build(budget=Budget(max_calls=2), threshold=ThresholdPolicy(min_top_score=0.0))

    pipeline.answer("what is chunk overlap?")
    pipeline.answer("what is chunk overlap?")

    with pytest.raises(BudgetExceeded, match="call budget"):
        pipeline.answer("what is chunk overlap?")


def test_budget_records_token_usage():
    budget = Budget(max_tokens=10**6)
    pipeline = build(budget=budget, threshold=ThresholdPolicy(min_top_score=0.0))
    pipeline.answer("what is chunk overlap?")

    assert budget.calls == 1
    assert budget.input_tokens > 0
    assert budget.output_tokens > 0


# ===========================================================================
# BATCHING
# ===========================================================================


def test_batch_answers_preserve_order_and_survive_partial_failure():
    """One failure must not lose the other results.

    This is what lets an evaluation sweep report "45 scored, 3 failed" rather
    than nothing at all.
    """
    pipeline = build(threshold=ThresholdPolicy(min_top_score=0.0))

    questions = [
        "what is chunk overlap?",
        "",  # empty question -- retrieval still runs, no crash expected
        "what is HNSW?",
    ]
    results = pipeline.answer_many(questions, max_workers=3)

    assert [q for q, _, _ in results] == questions
    assert len(results) == 3
    succeeded = [t for _, t, e in results if e is None and t is not None]
    assert len(succeeded) >= 2


def test_batch_respects_the_worker_bound():
    """Unbounded parallelism against a local model server just builds a queue,
    inflates latency and can exhaust file descriptors."""
    import threading

    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    class CountingLLM:
        model = "counting"

        def invoke(self, messages):
            with lock:
                live["now"] += 1
                live["peak"] = max(live["peak"], live["now"])
            time_to_work = 0.01
            import time as _t

            _t.sleep(time_to_work)
            with lock:
                live["now"] -= 1
            from langchain_core.messages import AIMessage

            return AIMessage(content="ok")

    base = RagPipeline(llm=CountingLLM(), embeddings=LexicalEmbeddings(dim=2048)).ingest()
    pipeline = ProductionRagPipeline(base, threshold=ThresholdPolicy(min_top_score=0.0))

    pipeline.answer_many(["what is chunk overlap?"] * 12, max_workers=3)

    assert live["peak"] <= 3, f"concurrency bound exceeded: {live['peak']} in flight"


def test_guard_stats_report_is_readable():
    stats = GuardStats(requests=10, pii_redacted=2, refused_low_relevance=1)
    text = stats.report()
    assert "10 requests" in text
    assert "2 redacted" in text
