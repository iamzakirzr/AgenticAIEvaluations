"""
Tests for the production primitives. ALL FAST TIER -- no model, no sleeping.

Run:  pytest core/test_production_core.py -v

Every retry test injects a fake `sleep`, so the suite stays instantaneous. A
resilience test that actually waits for backoff takes seconds, and a slow test
is a test somebody eventually deletes.
"""

from __future__ import annotations

import json

import pytest

from core.baseline import (
    Baseline,
    MetricSample,
    compare,
    current_fingerprint,
    measure_repeatedly,
    record_baseline,
)
from core.resilience import (
    Budget,
    BudgetExceeded,
    CircuitBreaker,
    CircuitOpen,
    OperationTimeout,
    RetryPolicy,
    RetryStats,
    call_with_timeout,
    default_retryable,
    map_bounded,
    retry,
)


class Transient(Exception):
    """Stands in for a 503. Named so default_retryable does not match it."""


class ConnectError(Exception):
    """Name matches the httpx class default_retryable recognises."""


# ===========================================================================
# RETRY
# ===========================================================================


def test_retry_returns_on_first_success():
    calls = []
    result = retry(lambda: (calls.append(1), "ok")[1], sleep=lambda _: None)
    assert result == "ok"
    assert len(calls) == 1


def test_retry_recovers_from_a_transient_failure():
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectError("connection reset")
        return "recovered"

    stats = RetryStats()
    result = retry(flaky, RetryPolicy(max_attempts=5), stats=stats, sleep=lambda _: None)

    assert result == "recovered"
    assert attempts["n"] == 3
    assert stats.retries == 2


def test_retry_does_not_retry_non_retryable_errors():
    """THE most important property in the module.

    A ValueError means a bug or a malformed response. Retrying it burns time
    and money to fail identically three more times, and buries the real error.
    """
    attempts = {"n": 0}

    def broken():
        attempts["n"] += 1
        raise ValueError("this is a bug, not a blip")

    with pytest.raises(ValueError):
        retry(broken, RetryPolicy(max_attempts=5), sleep=lambda _: None)

    assert attempts["n"] == 1, "a non-retryable error was retried"


def test_retry_gives_up_and_reraises_after_max_attempts():
    attempts = {"n": 0}

    def always_down():
        attempts["n"] += 1
        raise ConnectError("still down")

    with pytest.raises(ConnectError):
        retry(always_down, RetryPolicy(max_attempts=3), sleep=lambda _: None)

    assert attempts["n"] == 3


def test_backoff_grows_exponentially_and_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=4.0, jitter=0.0)
    assert policy.delay_for(2) == 1.0
    assert policy.delay_for(3) == 2.0
    assert policy.delay_for(4) == 4.0
    assert policy.delay_for(5) == 4.0, "max_delay did not cap the backoff"


def test_jitter_spreads_retries_to_avoid_a_thundering_herd():
    """Without jitter, N workers that fail together retry together, forever.

    That converts a brief blip into a sustained outage. Randomising each delay
    is what breaks the synchronisation.
    """
    policy = RetryPolicy(base_delay=1.0, jitter=0.25)
    delays = {round(policy.delay_for(3), 6) for _ in range(50)}
    assert len(delays) > 1, "jitter produced identical delays"
    assert all(1.5 <= d <= 2.5 for d in delays), f"jitter escaped its bounds: {delays}"


def test_default_retryable_classifies_errors_sensibly():
    assert default_retryable(ConnectError("reset"))
    assert default_retryable(TimeoutError("slow"))
    assert not default_retryable(ValueError("bad input"))
    assert not default_retryable(KeyError("missing"))


# ===========================================================================
# TIMEOUT
# ===========================================================================


def test_timeout_returns_a_fast_result():
    assert call_with_timeout(lambda: "quick", seconds=5) == "quick"


def test_timeout_fires_on_a_hanging_call():
    import time as _time

    with pytest.raises(OperationTimeout, match="deadline"):
        call_with_timeout(lambda: _time.sleep(2), seconds=0.05)


def test_timeout_propagates_the_original_exception():
    """A failing call must fail with ITS error, not a timeout."""
    with pytest.raises(ValueError, match="real error"):
        call_with_timeout(lambda: (_ for _ in ()).throw(ValueError("real error")), seconds=5)


# ===========================================================================
# CIRCUIT BREAKER
# ===========================================================================


def test_circuit_opens_after_consecutive_failures():
    breaker = CircuitBreaker(failure_threshold=3, reset_after=60)

    for _ in range(3):
        with pytest.raises(ConnectError):
            breaker.call(lambda: (_ for _ in ()).throw(ConnectError("down")))

    assert breaker.state == "open"

    # The next call must fail IMMEDIATELY, without invoking fn at all.
    called = {"n": 0}
    with pytest.raises(CircuitOpen):
        breaker.call(lambda: (called.__setitem__("n", called["n"] + 1), "ok")[1])
    assert called["n"] == 0, "an open circuit still called through to the service"


def test_a_success_resets_the_failure_count():
    """Only CONSECUTIVE failures should open the circuit.

    Otherwise a service with a 1% error rate eventually trips the breaker for
    no reason.
    """
    breaker = CircuitBreaker(failure_threshold=3, reset_after=60)

    for _ in range(2):
        with pytest.raises(ConnectError):
            breaker.call(lambda: (_ for _ in ()).throw(ConnectError("blip")))

    breaker.call(lambda: "fine")
    assert breaker.state == "closed"

    with pytest.raises(ConnectError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectError("blip")))
    assert breaker.state == "closed", "failure count did not reset after a success"


def test_circuit_half_opens_after_the_reset_window():
    breaker = CircuitBreaker(failure_threshold=1, reset_after=0.05)

    with pytest.raises(ConnectError):
        breaker.call(lambda: (_ for _ in ()).throw(ConnectError("down")))
    assert breaker.state == "open"

    import time as _time

    _time.sleep(0.06)
    assert breaker.state == "half_open"

    # A successful probe closes the circuit.
    assert breaker.call(lambda: "back up") == "back up"
    assert breaker.state == "closed"


# ===========================================================================
# BUDGET
# ===========================================================================


def test_call_budget_stops_a_runaway_loop():
    budget = Budget(max_calls=3)
    for _ in range(3):
        budget.check()
        budget.record(input_tokens=10, output_tokens=5)

    with pytest.raises(BudgetExceeded, match="call budget"):
        budget.check()


def test_token_budget_accounts_for_both_directions():
    budget = Budget(max_tokens=100)
    budget.record(input_tokens=60, output_tokens=30)
    budget.check()  # 90 < 100, still fine
    budget.record(input_tokens=20, output_tokens=0)
    with pytest.raises(BudgetExceeded, match="token budget"):
        budget.check()


def test_cost_budget_uses_the_configured_rates():
    budget = Budget(max_cost=0.01, input_cost_per_1k=1.0, output_cost_per_1k=2.0)
    budget.record(input_tokens=5000, output_tokens=1000)  # $5.00 + $2.00
    assert budget.cost == pytest.approx(7.0)
    with pytest.raises(BudgetExceeded, match="cost budget"):
        budget.check()


def test_zero_rates_make_cost_a_noop_for_local_models():
    """Local inference is free, so cost accounting should not get in the way --
    but call and token ceilings must still apply."""
    budget = Budget(max_cost=1.0, max_calls=2)
    budget.record(input_tokens=10**6, output_tokens=10**6)
    assert budget.cost == 0.0
    budget.check()  # cost is 0, so only the call ceiling matters
    budget.record()
    with pytest.raises(BudgetExceeded, match="call budget"):
        budget.check()


def test_check_before_record_is_the_correct_order():
    """Checking after the call has already spent the money you were saving."""
    budget = Budget(max_calls=1)
    budget.check()
    budget.record()
    with pytest.raises(BudgetExceeded):
        budget.check()


# ===========================================================================
# BOUNDED PARALLELISM
# ===========================================================================


def test_map_bounded_preserves_input_order():
    """Results must line up with the dataset that produced them."""
    results = map_bounded(lambda x: x * 2, [1, 2, 3, 4, 5], max_workers=3)
    assert [item for item, _, _ in results] == [1, 2, 3, 4, 5]
    assert [value for _, value, _ in results] == [2, 4, 6, 8, 10]


def test_one_failure_does_not_destroy_the_other_results():
    """THE reason not to use ThreadPoolExecutor.map for an eval sweep.

    `.map()` re-raises the first exception and throws away every other result.
    One judge failure on item 3 should not lose the other 4 scores -- that is
    what lets you report "4 scored, 1 judge failure" instead of nothing.
    """

    def sometimes(x):
        if x == 3:
            raise ConnectError("judge died on this one")
        return x * 10

    results = map_bounded(sometimes, [1, 2, 3, 4], max_workers=2)

    ok = [(i, v) for i, v, e in results if e is None]
    failed = [(i, e) for i, _, e in results if e is not None]

    assert len(ok) == 3
    assert len(failed) == 1
    assert failed[0][0] == 3
    assert isinstance(failed[0][1], ConnectError)


def test_map_bounded_handles_an_empty_input():
    assert map_bounded(lambda x: x, [], max_workers=4) == []


# ===========================================================================
# BASELINE AND THE REGRESSION GATE
# ===========================================================================


def test_metric_sample_statistics():
    sample = MetricSample(name="faithfulness", values=[0.8, 0.9, 0.85])
    assert sample.n == 3
    assert sample.mean == pytest.approx(0.85)
    assert sample.stdev > 0


def test_single_observation_reports_zero_spread_honestly():
    """One value tells you nothing about variance. 0.0 is the honest answer,
    and `compare()` warns when it gates on that."""
    assert MetricSample(name="m", values=[0.9]).stdev == 0.0


def test_measure_repeatedly_counts_failures_instead_of_scoring_them():
    """A judge failure must NEVER be recorded as 0.0.

    Doing so converts an infrastructure problem into a fake quality regression
    and corrupts the baseline you compare against afterwards.
    """
    state = {"n": 0}

    def flaky_metric():
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("judge returned malformed JSON")
        return 0.9

    sample = measure_repeatedly("faithfulness", flaky_metric, n=3)

    assert sample.values == [0.9, 0.9]
    assert sample.failures == 1
    assert sample.mean == pytest.approx(0.9), "a failure dragged the mean down"


def test_a_drop_inside_the_noise_band_is_not_a_regression():
    """The whole point of variance-aware gating.

    The metric moved by 0.01 while its own run-to-run spread is ~0.04. That is
    not evidence of anything, and failing the build on it teaches everyone to
    ignore CI.
    """
    baseline = Baseline().add(MetricSample("faithfulness", [0.90, 0.86, 0.94]))
    current = [MetricSample("faithfulness", [0.89, 0.85, 0.93])]

    result = compare(current, baseline, fingerprint=baseline.fingerprint)

    assert result.passed
    assert not result.regressions


def test_a_drop_outside_the_noise_band_is_a_regression():
    baseline = Baseline().add(MetricSample("faithfulness", [0.90, 0.91, 0.89]))
    current = [MetricSample("faithfulness", [0.60, 0.61, 0.59])]

    result = compare(current, baseline, fingerprint=baseline.fingerprint)

    assert not result.passed
    assert [c.name for c in result.regressions] == ["faithfulness"]
    assert "outside the" in result.regressions[0].reason


def test_a_noisy_metric_cannot_trigger_a_regression():
    """If a metric's own spread exceeds the change you care about, it CANNOT
    gate your CI -- and discovering that is a result, not a failure."""
    baseline = Baseline().add(MetricSample("wobbly", [0.5, 0.9, 0.3, 0.95]))
    current = [MetricSample("wobbly", [0.4, 0.8, 0.35, 0.85])]

    result = compare(current, baseline, fingerprint=baseline.fingerprint)
    assert result.passed, "a metric this noisy should not be able to fail a build"


def test_lower_is_better_metrics_invert_the_comparison():
    """Noise sensitivity, latency and cost all get WORSE as they go UP.

    Treating every metric as higher-is-better silently inverts the gate on
    these -- it would celebrate a latency increase as an improvement.
    """
    baseline = Baseline().add(
        MetricSample("noise_sensitivity", [0.10, 0.11, 0.09], higher_is_better=False)
    )
    worse = [MetricSample("noise_sensitivity", [0.50, 0.52, 0.48], higher_is_better=False)]
    better = [MetricSample("noise_sensitivity", [0.01, 0.02, 0.01], higher_is_better=False)]

    assert not compare(worse, baseline, fingerprint=baseline.fingerprint).passed
    result = compare(better, baseline, fingerprint=baseline.fingerprint)
    assert result.passed
    assert result.comparisons[0].improved


def test_min_delta_prevents_a_perfectly_stable_metric_from_flapping():
    """stdev 0.0 would give a zero-width noise band, so a 0.001 wobble fails.

    min_delta is the floor that stops that.
    """
    baseline = Baseline().add(MetricSample("exact", [1.0, 1.0, 1.0]))
    current = [MetricSample("exact", [0.999, 0.999, 0.999])]

    assert compare(current, baseline, min_delta=0.02, fingerprint=baseline.fingerprint).passed


def test_a_configuration_change_disables_the_gate_rather_than_failing_it():
    """THE most important safety property in core/baseline.py.

    Scores produced with a different model or chunk size are not comparable.
    Failing the build would punish a developer for a change they made on
    purpose; silently comparing anyway would report a fake regression. The
    honest response is to print the numbers and decline to judge them.
    """
    baseline = Baseline().add(MetricSample("faithfulness", [0.90, 0.91, 0.89]))
    baseline.fingerprint = {**current_fingerprint(), "chat_model": "llama3.1:8b"}

    current = [MetricSample("faithfulness", [0.40, 0.41, 0.39])]  # much worse
    changed = {**baseline.fingerprint, "chat_model": "qwen2.5:14b"}

    result = compare(current, baseline, fingerprint=changed)

    assert result.fingerprint_changed
    assert result.passed, "a config change must not fail the build"
    assert "chat_model" in result.fingerprint_diff
    assert "Gate disabled" in result.markdown()


def test_missing_baseline_reports_clearly_and_gates_nothing():
    result = compare([MetricSample("faithfulness", [0.9])], baseline=None)
    assert result.passed
    assert any("No baseline" in w for w in result.warnings)


def test_a_new_metric_is_reported_but_not_gated():
    baseline = Baseline().add(MetricSample("faithfulness", [0.9, 0.9, 0.9]))
    current = [
        MetricSample("faithfulness", [0.9, 0.9, 0.9]),
        MetricSample("brand_new", [0.1]),
    ]
    result = compare(current, baseline, fingerprint=baseline.fingerprint)

    assert result.passed
    new = next(c for c in result.comparisons if c.name == "brand_new")
    assert "not in baseline" in new.reason


def test_single_run_baselines_warn_that_the_band_is_a_guess():
    baseline = Baseline().add(MetricSample("faithfulness", [0.90]))
    result = compare(
        [MetricSample("faithfulness", [0.89])], baseline, fingerprint=baseline.fingerprint
    )
    assert any("only one observation" in w for w in result.warnings)


def test_judge_failures_surface_as_a_warning_not_a_score():
    baseline = Baseline().add(MetricSample("faithfulness", [0.9, 0.9, 0.9]))
    current = [MetricSample("faithfulness", values=[0.9], failures=4)]

    result = compare(current, baseline, fingerprint=baseline.fingerprint)
    assert any("judge/infrastructure failure" in w for w in result.warnings)


def test_markdown_report_is_pr_comment_shaped():
    baseline = Baseline().add(MetricSample("faithfulness", [0.90, 0.91, 0.89]))
    result = compare(
        [MetricSample("faithfulness", [0.60, 0.61, 0.59])],
        baseline,
        fingerprint=baseline.fingerprint,
    )
    md = result.markdown()

    assert "| metric |" in md
    assert "REGRESSION" in md
    assert "faithfulness" in md


# ===========================================================================
# PERSISTENCE
# ===========================================================================


def test_baseline_round_trips_through_json(tmp_path):
    path = tmp_path / "baseline.json"
    record_baseline(
        [MetricSample("faithfulness", [0.9, 0.88, 0.92]), MetricSample("mrr", [0.99])],
        path=path,
        note="first recorded baseline",
    )

    loaded = Baseline.load(path)
    assert loaded is not None
    assert set(loaded.metrics) == {"faithfulness", "mrr"}
    assert loaded.metrics["faithfulness"].values == [0.9, 0.88, 0.92]
    assert loaded.note == "first recorded baseline"
    assert loaded.fingerprint["chat_model"]


def test_saved_baseline_is_diff_friendly(tmp_path):
    """Sorted keys and a trailing newline, so re-recording produces a readable
    git diff rather than a whole-file rewrite."""
    path = tmp_path / "baseline.json"
    record_baseline([MetricSample("m", [1.0])], path=path)
    text = path.read_text()

    assert text.endswith("\n")
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)


def test_loading_a_missing_baseline_returns_none_rather_than_raising(tmp_path):
    assert Baseline.load(tmp_path / "nope.json") is None
