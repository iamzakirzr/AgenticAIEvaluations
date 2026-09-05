"""
Lesson 06 production scenarios. FAST TIER -- no account, no network, no model.

Run:  pytest 06_langwatch/test_production_observability.py -v

The non-negotiable property under test: nothing raw ever reaches a span. Once a
customer's card number is a span attribute it is in a buffer another thread is
already shipping to a third party, and no downstream filter can catch it.
"""

from __future__ import annotations

import json

from production_observability import (
    Alert,
    DatasetCandidate,
    ExportPolicy,
    OnlineMonitor,
    SamplingPolicy,
    hash_identifier,
    prepare_for_export,
    propose_dataset_candidates,
    write_candidates,
)

from core.golden import GoldenItem
from core.trace import RagTrace, RetrievedChunk


def trace(
    question="what is chunk overlap?",
    answer="Overlap protects boundary facts [1].",
    scores=(0.4, 0.2),
    latency=100.0,
    **kwargs,
) -> RagTrace:
    t = RagTrace(
        question=question,
        answer=answer,
        retrieved=[
            RetrievedChunk(f"chunk text {i}", "chunking", i, s)
            for i, s in enumerate(scores)
        ],
        retrieval_ms=latency / 2,
        generation_ms=latency / 2,
        chat_model="llama3.1:8b",
        embed_model="nomic-embed-text",
        chunk_size=700,
        top_k=4,
    )
    t.metadata.update(kwargs)
    return t


# ===========================================================================
# REDACTION BEFORE EXPORT
# ===========================================================================


def test_pii_never_reaches_the_export_payload():
    """THE non-negotiable. Redaction happens before the span is built."""
    payload = prepare_for_export(
        trace(
            question="My card is 4111 1111 1111 1111, what is chunk overlap?",
            answer="I cannot help with card numbers. Contact ada@example.com.",
        )
    )

    serialised = json.dumps(payload)
    assert "4111" not in serialised
    assert "ada@example.com" not in serialised
    assert "[CARD]" in payload["question"]
    assert "[EMAIL]" in payload["answer"]


def test_redaction_counts_are_exported_but_values_are_not():
    """'We redacted 3 emails' is safe to export AND is an alerting signal."""
    payload = prepare_for_export(
        trace(question="mail a@b.com or c@d.com", answer="ok")
    )
    assert payload["pii_redacted"]["EMAIL"] == 2
    assert "a@b.com" not in json.dumps(payload)


def test_user_ids_are_hashed_by_default():
    """Lets you count 'this user hit the bug 11 times' without exporting who."""
    payload = prepare_for_export(trace(), user_id="alice@example.com")

    assert payload["user"] != "alice@example.com"
    assert len(payload["user"]) == 16
    assert "alice" not in json.dumps(payload)


def test_hashing_is_stable_so_correlation_still_works():
    a = hash_identifier("user-42", salt="s")
    b = hash_identifier("user-42", salt="s")
    assert a == b
    assert hash_identifier("user-43", salt="s") != a


def test_the_salt_changes_the_hash():
    """Without a secret salt, a hash over a small identifier space is trivially
    reversed with a rainbow table."""
    assert hash_identifier("user-42", salt="") != hash_identifier("user-42", salt="secret")


def test_long_fields_are_truncated_with_a_visible_marker():
    """Traces carrying 50KB of context are expensive to store, slow to render,
    and rarely more informative than the first 2KB."""
    payload = prepare_for_export(
        trace(answer="x" * 10_000), policy=ExportPolicy(max_field_chars=100)
    )
    assert len(payload["answer"]) < 200
    assert "truncated" in payload["answer"]


def test_contexts_can_be_withheld_for_a_confidential_corpus():
    with_contexts = prepare_for_export(trace())
    without = prepare_for_export(trace(), policy=ExportPolicy(include_contexts=False))

    assert "contexts" in with_contexts
    assert "contexts" not in without


def test_redaction_can_be_disabled_but_is_on_by_default():
    """Defaults must be the safe direction. Exporting less is recoverable;
    exporting a card number to a third party is not."""
    assert ExportPolicy().redact_pii is True
    assert ExportPolicy().hash_user_ids is True

    raw = prepare_for_export(
        trace(question="mail a@b.com"), policy=ExportPolicy(redact_pii=False)
    )
    assert "a@b.com" in raw["question"]


def test_pii_in_retrieved_contexts_is_redacted_and_counted():
    """Review finding: counts previously covered only question + answer.

    Retrieved chunks are USUALLY your own documents -- but "usually" is not a
    guarantee. A support corpus can contain customer data, and a count that
    ignores contexts under-reports exactly the case you would most want
    alerted on.
    """
    t = trace(question="what is chunk overlap?", answer="see below")
    t.retrieved[0].text = "Ticket from ada@example.com regarding card 4111111111111111."

    payload = prepare_for_export(t)

    assert "ada@example.com" not in json.dumps(payload)
    assert "4111111111111111" not in json.dumps(payload)
    assert payload["pii_redacted"].get("EMAIL", 0) >= 1
    assert payload["pii_redacted"].get("CARD", 0) >= 1


def test_operational_fields_survive_export():
    payload = prepare_for_export(trace())
    for key in ("retrieval_ms", "generation_ms", "chat_model", "doc_ids", "top_score"):
        assert key in payload


# ===========================================================================
# SAMPLING
# ===========================================================================


def test_head_sampling_is_deterministic():
    policy = SamplingPolicy(rate=0.5)
    decisions = {policy.should_evaluate(trace(question="stable question"))[0] for _ in range(20)}
    assert len(decisions) == 1


def test_errors_are_always_kept_regardless_of_the_dice():
    """Pure random sampling at 10% throws away nine of every ten incidents."""
    policy = SamplingPolicy(rate=0.0)  # head sampling keeps nothing
    sampled, reason = policy.should_evaluate(trace(), errored=True)

    assert sampled
    assert "errored" in reason


def test_refusals_are_always_kept():
    policy = SamplingPolicy(rate=0.0)
    sampled, reason = policy.should_evaluate(
        trace(answer="The provided context does not contain this information.")
    )
    assert sampled
    assert "refusal" in reason


def test_slow_requests_are_always_kept():
    policy = SamplingPolicy(rate=0.0, always_keep_slow_ms=1000)
    sampled, reason = policy.should_evaluate(trace(latency=4000))
    assert sampled
    assert "slow" in reason


def test_low_confidence_retrievals_are_always_kept():
    policy = SamplingPolicy(rate=0.0, always_keep_low_confidence=0.1)
    sampled, reason = policy.should_evaluate(trace(scores=(0.02, 0.01)))
    assert sampled
    assert "low retrieval confidence" in reason


def test_ordinary_traffic_is_sampled_at_the_configured_rate():
    policy = SamplingPolicy(rate=0.1, always_keep_slow_ms=10**9, always_keep_low_confidence=-1)
    kept = sum(
        1
        for i in range(500)
        if policy.should_evaluate(trace(question=f"ordinary question {i}"))[0]
    )
    assert 20 <= kept <= 200, f"sampled {kept}/500 at rate 0.1"


def test_the_sampling_reason_is_reported_for_debuggability():
    _, reason = SamplingPolicy(rate=1.0).should_evaluate(trace())
    assert reason


# ===========================================================================
# ALERTING
# ===========================================================================


def feed(monitor: OnlineMonitor, n: int, **kwargs):
    for i in range(n):
        monitor.record(trace(question=f"q{i}", **kwargs))


def test_monitor_stays_silent_until_it_has_enough_data():
    monitor = OnlineMonitor(min_samples=30)
    feed(monitor, 5)
    assert monitor.check() == []


def test_a_refusal_spike_alerts_even_from_a_zero_baseline():
    """A relative-only test cannot fire when the reference rate is 0.

    Multiplying zero by any factor is still zero, so a 0% -> 50% refusal spike
    -- retrieval breaking completely -- would stay silent. That gap was real
    and this test caught it; the monitor now also has an absolute floor.
    """
    monitor = OnlineMonitor(window=100, min_samples=20, change_factor=2.0)
    feed(monitor, 50)  # reference half: zero refusals
    for i in range(50):
        monitor.record(
            trace(
                question=f"r{i}",
                answer="The provided context does not contain this information."
                if i % 2
                else "a normal answer",
            )
        )
    alerts = {a.metric for a in monitor.check()}
    # With half the recent window refusing and none before, this must fire.
    assert "refusal_rate" in alerts


def test_a_refusal_collapse_alerts_and_is_the_dangerous_direction():
    """The alert people forget to write.

    A one-sided alert on 'refusals went up' never fires when the model stops
    refusing and starts confabulating -- which is the worse outcome.
    """
    monitor = OnlineMonitor(window=100, min_samples=20, change_factor=2.0)

    for i in range(50):  # reference half: refusing often
        monitor.record(
            trace(question=f"a{i}", answer="The provided context does not contain this information.")
        )
    for i in range(50):  # recent half: never refusing
        monitor.record(trace(question=f"b{i}", answer="A confident answer."))

    alerts = monitor.check()
    collapse = [a for a in alerts if "COLLAPSED" in a.message]

    assert collapse, f"no collapse alert fired; got {[a.message for a in alerts]}"
    assert collapse[0].severity == "critical"


def test_an_error_rate_spike_alerts():
    monitor = OnlineMonitor(window=100, min_samples=20)
    for i in range(100):
        monitor.record(trace(question=f"q{i}"), errored=i % 3 == 0)

    assert any(a.metric == "error_rate" for a in monitor.check())


def test_a_retrieval_confidence_drop_alerts():
    """The earliest warning that the index or the query distribution changed."""
    monitor = OnlineMonitor(window=100, min_samples=20)
    for i in range(50):
        monitor.record(trace(question=f"a{i}", scores=(0.8, 0.7)))
    for i in range(50):
        monitor.record(trace(question=f"b{i}", scores=(0.05, 0.02)))

    assert any(a.metric == "retrieval_confidence" for a in monitor.check())


def test_a_pii_spike_alerts():
    monitor = OnlineMonitor(window=100, min_samples=20)
    for i in range(100):
        monitor.record(trace(question=f"q{i}", pii_found={"EMAIL": 2}))

    assert any(a.metric == "pii_rate" for a in monitor.check())


def test_a_healthy_stream_produces_no_alerts():
    """A monitor that cries wolf gets muted, and then it is worth nothing."""
    monitor = OnlineMonitor(window=100, min_samples=20)
    feed(monitor, 100)
    assert monitor.check() == []


# ===========================================================================
# TRACES BACK INTO THE DATASET
# ===========================================================================


def test_refusals_are_promoted_as_unanswerable_candidates():
    traces = [
        trace(
            question="What is your refund policy?",
            answer="The provided context does not contain this information.",
        )
    ]
    candidates = propose_dataset_candidates(traces)

    assert len(candidates) == 1
    assert candidates[0].suggested_category == "unanswerable"
    assert "refused" in candidates[0].reason


def test_low_confidence_answers_are_promoted_as_adversarial():
    """Answered anyway despite weak retrieval -- where hallucinations live."""
    traces = [
        trace(question="Something barely covered", answer="A confident answer.", scores=(0.01,))
    ]
    candidates = propose_dataset_candidates(traces)

    assert candidates[0].suggested_category == "adversarial"
    assert "low retrieval confidence" in candidates[0].reason


def test_the_most_interesting_outcome_wins_for_a_repeated_question():
    """A question that was ever refused matters more than one merely asked often."""
    traces = [
        trace(question="Same question"),
        trace(question="Same question", answer="The provided context does not contain this information."),
        trace(question="Same question"),
    ]
    candidates = propose_dataset_candidates(traces)

    assert len(candidates) == 1
    assert candidates[0].suggested_category == "unanswerable"
    assert candidates[0].occurrences == 3


def test_ordinary_questions_need_repetition_to_be_promoted():
    """Otherwise the dataset fills with one-off noise."""
    once = propose_dataset_candidates([trace(question="asked once")], min_occurrences=2)
    assert once == []

    twice = propose_dataset_candidates(
        [trace(question="asked twice"), trace(question="asked twice")], min_occurrences=2
    )
    assert len(twice) == 1


def test_questions_already_in_the_dataset_are_skipped():
    existing = [
        GoldenItem(
            id="sh-01",
            question="What is chunk overlap for?",
            reference_answer="...",
            reference_doc_ids=["chunking"],
        )
    ]
    traces = [trace(question="  what is CHUNK overlap for?  ")] * 3
    assert propose_dataset_candidates(traces, existing=existing) == []


def test_the_stub_refuses_to_invent_a_reference_answer():
    """THE most important property here.

    Auto-filling the reference from the system's own output builds a dataset
    that certifies current behaviour as correct -- circular, and it hides every
    existing bug forever.
    """
    candidate = DatasetCandidate(
        question="q", suggested_category="unanswerable", reason="r",
        observed_answer="whatever the system happened to say",
    )
    stub = candidate.to_golden_stub("prod-001")

    assert stub["reference_answer"] == "TODO: a human must write this"
    assert "whatever the system happened to say" not in json.dumps(stub)


def test_candidates_are_ordered_by_frequency():
    traces = [trace(question="rare", answer="The provided context does not contain this information.")]
    traces += [
        trace(question="common", answer="The provided context does not contain this information.")
    ] * 5
    candidates = propose_dataset_candidates(traces)
    assert candidates[0].question == "common"


def test_candidates_are_written_as_jsonl_stubs(tmp_path):
    candidates = propose_dataset_candidates(
        [trace(question="Refund policy?", answer="The provided context does not contain this information.")]
    )
    path = write_candidates(candidates, tmp_path / "candidates.jsonl")

    lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert lines[0]["id"] == "prod-001"
    assert lines[0]["category"] == "unanswerable"


def test_writing_no_candidates_produces_an_empty_file(tmp_path):
    path = write_candidates([], tmp_path / "none.jsonl")
    assert path.read_text() == ""


def test_alert_dataclass_carries_context_for_a_pager():
    alert = Alert("refusal_rate", "went up", "critical", 0.5, 0.1)
    assert alert.severity == "critical"
    assert alert.current == 0.5 and alert.reference == 0.1
