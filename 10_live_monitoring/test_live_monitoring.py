"""
Tests for lesson 10 -- live monitoring of a deployed agent.

Each test pins one operational claim. The ones worth reading twice are
`test_a_refusal_collapse_pages` (the failure every one-sided alert misses) and
`test_burn_rate_distinguishes_an_outage_from_a_slow_bleed`.

Run:  pytest 10_live_monitoring -v
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from live_monitor import (
    ErrorBudget,
    LiveEvent,
    LiveMonitor,
    promote_candidates,
    total_variation_distance,
)


def _event(**kwargs) -> LiveEvent:
    base = {
        "request_id": kwargs.pop("request_id", "r"),
        "timestamp": time.time(),
        "latency_ms": 100.0,
        "server": "corpus",
    }
    base.update(kwargs)
    return LiveEvent(**base)


def _fill(monitor: LiveMonitor, n: int, **kwargs) -> None:
    for index in range(n):
        monitor.record(_event(request_id=f"r{index}", **kwargs))


# ---------------------------------------------------------------------------
# Error budget and burn rate
# ---------------------------------------------------------------------------


def test_burn_rate_is_error_rate_over_budget():
    budget = ErrorBudget(slo=0.99)
    assert budget.budget == pytest.approx(0.01)
    assert budget.burn_rate(0.01) == pytest.approx(1.0)  # spends the budget exactly
    assert budget.burn_rate(0.144) == pytest.approx(14.4)


def test_a_100_percent_slo_is_a_configuration_error():
    """"We want 100% uptime" sets an objective that can only be violated.

    Any failure at all is infinite burn. Better to surface that than to divide
    by zero in the alerting path at the worst possible moment.
    """
    budget = ErrorBudget(slo=1.0)
    assert budget.burn_rate(0.001) == float("inf")
    assert budget.burn_rate(0.0) == 0.0


def test_burn_rate_distinguishes_an_outage_from_a_slow_bleed():
    budget = ErrorBudget()
    # 20% errors right now: the budget is gone in hours. Page.
    outage = budget.evaluate(fast_rate=0.20, slow_rate=0.05)
    assert outage.severity == "page"
    assert "14" in outage.detail or "20.0x" in outage.detail

    # 8% sustained: not an outage, but the month's budget will not survive it.
    bleed = budget.evaluate(fast_rate=0.0, slow_rate=0.08)
    assert bleed.severity == "ticket"

    # 0.5% errors is INSIDE a 1% budget. Silence is correct.
    assert budget.evaluate(fast_rate=0.005, slow_rate=0.005) is None


def test_one_incident_produces_one_alert():
    """During a real outage both windows breach. Two alerts for one incident is
    how alert fatigue begins."""
    alert = ErrorBudget().evaluate(fast_rate=0.5, slow_rate=0.5)
    assert alert is not None and alert.severity == "page"


def test_a_threshold_alert_would_miss_the_slow_bleed():
    """The comparison that justifies burn rate at all.

    A naive "alert above 10% errors" stays silent at 8% forever, while 8%
    against a 1% budget is an 8x burn that exhausts a 30-day budget in under
    four days.
    """
    def naive_alert(error_rate: float, threshold: float = 0.10) -> bool:
        return error_rate > threshold

    observed = 0.08
    assert not naive_alert(observed)  # silent, forever
    assert ErrorBudget().evaluate(fast_rate=0.0, slow_rate=observed) is not None


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def test_tvd_is_zero_for_identical_and_one_for_disjoint():
    assert total_variation_distance(Counter({"a": 5}), Counter({"a": 9})) == 0.0
    assert total_variation_distance(Counter({"a": 5}), Counter({"b": 5})) == pytest.approx(1.0)


def test_tvd_handles_a_category_that_vanished():
    """The case KL divergence cannot express without an epsilon fudge -- and it
    is the interesting case: a server stopped being used entirely."""
    before = Counter({"corpus": 50, "web": 50})
    after = Counter({"corpus": 100})
    assert total_variation_distance(before, after) == pytest.approx(0.5)


def test_tvd_on_an_empty_side_is_zero_not_a_crash():
    """A monitor must never be able to break the thing it watches."""
    assert total_variation_distance(Counter(), Counter({"a": 1})) == 0.0


# ---------------------------------------------------------------------------
# The monitor's checks
# ---------------------------------------------------------------------------


def test_a_quiet_service_does_not_alert():
    monitor = LiveMonitor()
    _fill(monitor, 5, errored=True)
    assert monitor.check() == []  # 5 samples is not evidence of anything


def test_an_error_spike_pages():
    monitor = LiveMonitor(window=200, min_samples=40, fast_window=60)
    _fill(monitor, 140)  # healthy history
    _fill(monitor, 60, errored=True)  # the incident
    names = {a.name: a for a in monitor.check()}
    assert names["error_budget_burn"].severity == "page"


def test_a_refusal_spike_raises_a_ticket():
    monitor = LiveMonitor(window=200, min_samples=40)
    _fill(monitor, 100)
    _fill(monitor, 100, refused=True)
    assert any(a.name == "refusal_spike" for a in monitor.check())


def test_a_refusal_collapse_pages():
    """THE failure a one-sided alert never catches.

    Refusals dropping to zero means the system stopped abstaining and started
    answering from parametric memory. Latency improves, errors are unchanged,
    the refusal graph goes DOWN -- which on most dashboards is drawn green.
    """
    monitor = LiveMonitor(window=200, min_samples=40)
    for index in range(100):  # reference half: refuses 40% of the time
        monitor.record(_event(request_id=f"a{index}", refused=index % 10 < 4))
    _fill(monitor, 100, refused=False)  # recent half: never refuses

    collapse = next(a for a in monitor.check() if a.name == "refusal_collapse")
    assert collapse.severity == "page"
    assert "confabulation" in collapse.detail


def test_tool_mix_drift_fires_with_nothing_else_wrong():
    """No errors, no refusals, normal latency -- and the agent is now answering
    internal questions from the web server. Only the mix shows it."""
    monitor = LiveMonitor(window=200, min_samples=40)
    _fill(monitor, 100, server="corpus")
    _fill(monitor, 100, server="web")

    alerts = {a.name for a in monitor.check()}
    assert "tool_mix_drift" in alerts
    assert "error_budget_burn" not in alerts


def test_latency_regression_uses_p95_not_the_mean():
    monitor = LiveMonitor(window=200, min_samples=40)
    _fill(monitor, 100, latency_ms=100.0)
    _fill(monitor, 100, latency_ms=400.0)
    assert any(a.name == "latency_regression" for a in monitor.check())


def test_a_steady_service_produces_no_alerts():
    """The test that keeps the monitor usable. A monitor that alerts on healthy
    traffic gets muted, and a muted monitor is worse than none."""
    monitor = LiveMonitor(window=200, min_samples=40)
    for index in range(200):
        monitor.record(
            _event(
                request_id=f"r{index}",
                server="corpus" if index % 4 else "web",
                refused=index % 10 == 0,
                latency_ms=100.0 + (index % 7),
            )
        )
    assert monitor.check() == []


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------


def test_the_snapshot_states_its_own_limit():
    """The dashboard must say what it cannot measure, on the dashboard."""
    monitor = LiveMonitor()
    _fill(monitor, 10)
    snapshot = monitor.snapshot()
    assert "no reference answers in production" in snapshot["quality_note"]
    for forbidden in ("faithfulness", "accuracy", "correctness"):
        assert forbidden not in snapshot


def test_an_empty_snapshot_says_so_rather_than_reporting_zeroes():
    assert LiveMonitor().snapshot() == {"samples": 0, "note": "no traffic yet"}


def test_the_snapshot_reports_percentiles_and_mix():
    monitor = LiveMonitor()
    for index in range(100):
        monitor.record(
            _event(
                request_id=f"r{index}",
                latency_ms=float(index),
                server="corpus" if index < 70 else "web",
                tools_used=("corpus_search",),
            )
        )
    snapshot = monitor.snapshot()
    assert snapshot["p50_ms"] == 50
    assert snapshot["p95_ms"] == 95
    assert snapshot["server_mix"] == {"corpus": 70, "web": 30}
    assert snapshot["tool_calls_per_request"] == 1.0


# ---------------------------------------------------------------------------
# Adapting a real gateway response
# ---------------------------------------------------------------------------


def test_record_response_reads_a_real_gateway_body():
    monitor = LiveMonitor()
    event = monitor.record_response(
        {
            "request_id": "abc123",
            "answer": "Overlap protects boundary facts [chunking#4].",
            "tools_used": ["corpus_search"],
            "servers_used": ["corpus"],
            "refused": False,
            "latency_ms": 42.5,
        },
        question="what is overlap?",
    )
    assert event.server == "corpus"
    assert event.latency_ms == 42.5
    assert event.answer_chars > 0


def test_record_response_survives_a_malformed_body():
    """A monitor that raises on an unexpected payload takes down the service it
    was installed to protect. It must degrade, never throw."""
    monitor = LiveMonitor()
    event = monitor.record_response({}, question="q", errored=True)
    assert event.server == "none"
    assert event.errored
    assert event.latency_ms == 0.0


# ---------------------------------------------------------------------------
# Closing the loop
# ---------------------------------------------------------------------------


def test_candidates_are_ranked_by_how_much_they_would_teach_you():
    monitor = LiveMonitor()
    monitor.record(_event(request_id="1", question="normal question"))
    monitor.record(_event(request_id="2", question="refused question", refused=True))
    monitor.record(_event(request_id="3", question="blocked question", blocked=True))
    monitor.record(_event(request_id="4", question="errored question", errored=True))

    candidates = promote_candidates(monitor)
    assert [c.reason for c in candidates] == [
        "output guard blocked",
        "request errored",
        "refused -- coverage gap?",
    ]
    # The ordinary question is not promoted: labelling more of what already
    # works costs human time and teaches the suite nothing.
    assert "normal question" not in {c.question for c in candidates}


def test_candidates_are_deduplicated_by_question():
    """A popular failing question otherwise floods the queue forty times."""
    monitor = LiveMonitor()
    for index in range(40):
        monitor.record(_event(request_id=str(index), question="Same Question?", refused=True))
    assert len(promote_candidates(monitor)) == 1


def test_candidates_are_not_auto_promoted_into_the_golden_set():
    """Every candidate carries a REASON and no label.

    There is deliberately no `expected_answer` field to fill in. Auto-promoting
    a model's own output into the reference set teaches the system to grade its
    own homework, and it fails silently and permanently.
    """
    monitor = LiveMonitor()
    monitor.record(_event(request_id="1", question="q", refused=True))
    candidate = promote_candidates(monitor)[0]
    assert not hasattr(candidate, "expected_answer")
    assert not hasattr(candidate, "label")
    assert candidate.reason


def test_promotion_on_empty_traffic_is_empty():
    assert promote_candidates(LiveMonitor()) == []


def test_the_long_tail_rule_is_off_below_a_sample_floor():
    """A percentile over one sample IS that sample.

    Without the floor, a quiet service promotes every single answer as
    "unusually long" and the label queue becomes a copy of the traffic log.
    Found by the lesson 09/10 integration test failing.
    """
    quiet = LiveMonitor()
    quiet.record(_event(request_id="1", question="only question", answer_chars=500))
    assert promote_candidates(quiet) == []

    busy = LiveMonitor()
    for index in range(30):
        busy.record(_event(request_id=str(index), question=f"q{index}", answer_chars=100))
    busy.record(_event(request_id="tail", question="rambling one", answer_chars=9000))
    assert [c.question for c in promote_candidates(busy)] == ["rambling one"]
