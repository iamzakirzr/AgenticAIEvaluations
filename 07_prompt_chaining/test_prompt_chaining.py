"""
Tests for lesson 07 -- prompt chaining.

Read these as the lesson's second half. Each one pins a claim the README makes,
and several exist because the naive version of the test was flaky or vacuous.

Run:  pytest 07_prompt_chaining -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chains import (
    HAS_CITATION,
    IS_LABEL,
    NON_EMPTY,
    ChainTrace,
    Contract,
    ContractViolation,
    branching_chain,
    build_guarded_chain,
    end_to_end_reliability,
    guard,
    map_reduce_chain,
    parallel_chain,
    required_link_reliability,
    sequential_chain,
    traced,
)
from keyed_model import KeyedChatModel
from langchain_core.runnables import RunnableLambda

# ---------------------------------------------------------------------------
# Reliability arithmetic -- the claim that should change your design
# ---------------------------------------------------------------------------


def test_compounding_reliability_is_multiplicative():
    """Four 95% links are an 81% chain, not a 95% one.

    THE point of the lesson. People reason additively ("95% minus a bit") and
    are consistently wrong by 14 percentage points at four links.
    """
    assert end_to_end_reliability([0.95] * 4) == pytest.approx(0.81450625)
    # And the shape of the curve: doubling the links roughly doubles the loss.
    assert end_to_end_reliability([0.95] * 2) == pytest.approx(0.9025)
    assert end_to_end_reliability([0.95] * 8) == pytest.approx(0.6634, abs=1e-4)


def test_a_single_bad_link_dominates():
    """Reliability is bounded above by the worst link, always.

    Which is why "improve every prompt" is the wrong response to a flaky chain.
    Find the worst link. Nothing you do to the others can beat it.
    """
    chain = [0.99, 0.99, 0.60, 0.99]
    assert end_to_end_reliability(chain) <= min(chain)


def test_required_reliability_per_link_is_brutal():
    """99% end-to-end over 5 links needs 99.8% per link."""
    per_link = required_link_reliability(0.99, 5)
    assert per_link == pytest.approx(0.99799, abs=1e-5)
    # Round-trips, so the arithmetic is self-consistent.
    assert end_to_end_reliability([per_link] * 5) == pytest.approx(0.99)


def test_reliability_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        end_to_end_reliability([1.2])
    with pytest.raises(ValueError):
        required_link_reliability(0.9, 0)


# ---------------------------------------------------------------------------
# Contracts -- catching a failure at the link that caused it
# ---------------------------------------------------------------------------


def test_contract_names_the_link_not_the_symptom():
    """The error message must identify the LINK. That is the entire value."""
    with pytest.raises(ContractViolation) as exc:
        IS_LABEL.enforce("Category: FACTUAL")
    message = str(exc.value)
    assert message.startswith("classification:")
    assert "routes nowhere" in message
    assert "Category: FACTUAL" in message  # the offending value, for triage


def test_contract_catches_the_empty_string_that_would_vanish():
    """An empty link output is the nastiest chain bug there is.

    It does not raise. It gets formatted into the next prompt as nothing at
    all, and the downstream model answers a question it was never asked.
    """
    with pytest.raises(ContractViolation):
        NON_EMPTY.enforce("   ")
    assert NON_EMPTY.enforce("real content") == "real content"


def test_citation_contract_is_a_free_faithfulness_proxy():
    """No model, no judge, no cost -- and it catches an ungrounded answer shape."""
    assert HAS_CITATION.check("Chunking splits documents [2].")
    assert not HAS_CITATION.check("Chunking splits documents.")


def test_guard_composes_with_a_pipe():
    """A contract is a Runnable, so it drops into a chain like anything else."""
    chain = RunnableLambda(lambda s: s.upper()) | guard(
        Contract("shouty", lambda v: v.isupper(), "must be upper case")
    )
    assert chain.invoke("abc") == "ABC"


# ---------------------------------------------------------------------------
# Tracing -- a failure needs an address
# ---------------------------------------------------------------------------


def test_trace_records_every_link_in_order():
    trace = ChainTrace()
    chain = traced("first", trace)(RunnableLambda(lambda x: x + 1)) | traced("second", trace)(
        RunnableLambda(lambda x: x * 10)
    )
    assert chain.invoke(1) == 20
    assert trace.names == ["first", "second"]
    assert trace.of("second").input == 2  # what the second link actually saw


def test_trace_records_the_failing_link_and_re_raises():
    """Swallowing the exception would be worse than not tracing at all."""
    trace = ChainTrace()

    def boom(_: object) -> object:
        raise RuntimeError("model timed out")

    chain = traced("ok", trace)(RunnableLambda(lambda x: x)) | traced("bad", trace)(
        RunnableLambda(boom)
    )

    with pytest.raises(RuntimeError):
        chain.invoke("x")

    assert [link.name for link in trace.failed] == ["bad"]
    assert "model timed out" in trace.of("bad").error
    assert trace.of("ok").ok  # the successful link is still recorded


def test_a_link_that_never_ran_is_distinguishable_from_one_that_failed():
    """`None` vs a failed LinkTrace. A chain-level assertion conflates them."""
    trace = ChainTrace()
    traced("ran", trace)(RunnableLambda(lambda x: x)).invoke(1)
    assert trace.of("ran") is not None
    assert trace.of("never_ran") is None


# ---------------------------------------------------------------------------
# The four shapes
# ---------------------------------------------------------------------------


def _model() -> KeyedChatModel:
    return KeyedChatModel(
        rules=[
            (r"Classify as", "factual"),
            (r"Answer using only the context", "Chunk overlap protects boundary facts [1]."),
            (r"take no side", "There are trade-offs on both sides."),
            (r"Answer concisely", "A first draft."),
            (r"List flaws", "- too vague"),
            (r"Rewrite the answer", "A revised answer [1]."),
            (r"Summarise in one sentence", "A summary sentence."),
            (r"List 3 keywords", "a, b, c"),
            (r"One word sentiment", "neutral"),
            (r"Combine these", "A combined paragraph."),
        ],
        calls=[],
    )


def test_sequential_chain_passes_all_intermediates_forward():
    """RunnablePassthrough.assign ADDS a key; a bare `|` would drop the draft."""
    model = _model()
    result = sequential_chain(model).invoke({"question": "what is chunking?"})
    assert result == "A revised answer [1]."
    # Three model calls, and the revise prompt saw BOTH earlier outputs.
    assert len(model.calls) == 3
    revise_prompt = next(c for c in model.calls if "Rewrite the answer" in c)
    assert "A first draft." in revise_prompt
    assert "- too vague" in revise_prompt


def test_parallel_chain_returns_a_dict_keyed_by_branch():
    """Deterministic ONLY because the fake is keyed on the prompt.

    With a counter-based fake this assertion passes or fails on thread
    scheduling -- see keyed_model.py for why that matters.
    """
    result = parallel_chain(_model()).invoke({"text": "some text"})
    assert result == {
        "summary": "A summary sentence.",
        "keywords": "a, b, c",
        "sentiment": "neutral",
    }


def test_every_branch_is_reachable():
    """Branch coverage. The default arm is the one nobody tests.

    A novel label must produce the refusal, not an exception. Without the
    default arm RunnableBranch raises, and a production router that raises on
    unexpected input is an outage rather than a degraded answer.
    """
    chain = branching_chain(_model())
    assert "[1]" in chain.invoke({"question": "q", "label": "factual"})
    assert "trade-offs" in chain.invoke({"question": "q", "label": "opinion"})
    for unexpected in ("unknown", "FACTUAL", "", None):
        assert "does not contain this information" in chain.invoke(
            {"question": "q", "label": unexpected}
        )


def test_branch_routing_costs_no_model_call():
    """Routing is a Python predicate. Paying a model to pick a prompt is waste."""
    model = _model()
    branching_chain(model).invoke({"question": "q", "label": "factual"})
    assert len(model.calls) == 1  # the answer only -- nothing spent on routing


def test_map_reduce_keeps_the_intermediates():
    """Returning only the final paragraph makes dropped facts unattributable."""
    result = map_reduce_chain(_model()).invoke({"documents": ["doc one", "doc two", "doc three"]})
    assert result["answer"] == "A combined paragraph."
    assert len(result["summaries"]) == 3  # one per document, available for recall checks


# ---------------------------------------------------------------------------
# The full guarded chain
# ---------------------------------------------------------------------------


def test_guarded_chain_traces_the_whole_path():
    trace = ChainTrace()
    answer = build_guarded_chain(_model(), trace).invoke({"question": "what is chunking?"})
    assert answer == "Chunk overlap protects boundary facts [1]."
    assert trace.names == ["classify", "route", "answer"]
    assert trace.of("classify").output == "factual"
    assert not trace.failed


def test_guarded_chain_fails_at_the_classifier_when_the_classifier_is_wrong():
    """The whole point: the blame lands on link 1, not on the final answer.

    Without the contract this chain returns a refusal -- a plausible-looking
    output that sends you debugging the retriever for an hour.
    """
    broken = KeyedChatModel(rules=[(r"Classify as", "Category: FACTUAL")], calls=[])
    trace = ChainTrace()

    with pytest.raises(ContractViolation) as exc:
        build_guarded_chain(broken, trace).invoke({"question": "q"})

    assert str(exc.value).startswith("classification:")
    assert [link.name for link in trace.failed] == ["classify"]
    assert trace.of("answer") is None  # never reached -- the blast radius stopped


def test_trace_measures_something_real():
    """Guard against the trace being decorative: durations must be populated."""
    trace = ChainTrace()
    build_guarded_chain(_model(), trace).invoke({"question": "q"})
    assert trace.total_ms > 0.0
    assert all(link.duration_ms >= 0.0 for link in trace.links)
