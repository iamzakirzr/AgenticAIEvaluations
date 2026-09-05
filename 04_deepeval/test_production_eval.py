"""
Lesson 04 production scenarios. FAST TIER -- the judge is simulated.

Run:  pytest 04_deepeval/test_production_eval.py -v

The runner is tested with an INJECTED scoring function, so every property that
matters -- failure budgets, category slicing, variance, partial results -- is
verified deterministically without a GPU. The real judge is swapped in by
changing one argument.
"""

from __future__ import annotations

import json

import pytest
from production_eval import (
    EvalRunner,
    EvaluationReport,
    EvaluationRunInvalid,
    ItemScore,
    gate,
    samples_from_runs,
)

from core.baseline import MetricSample, record_baseline
from core.golden import load_golden
from core.resilience import Budget


@pytest.fixture(scope="module")
def items():
    return load_golden()


def fake_answer(question: str) -> str:
    return f"answer to {question[:20]}"


def perfect_scorer(item, trace, metric):
    return 1.0, "perfect"


def make_report(scores: list[ItemScore], **kwargs) -> EvaluationReport:
    return EvaluationReport(scores=scores, **kwargs)


# ===========================================================================
# JUDGE FAILURES ARE NEVER SCORED AS ZERO
# ===========================================================================


def test_a_judge_failure_records_none_not_zero(items):
    """THE rule from core/corpus/llm_as_judge.md.

    Scoring a broken judge as 0.0 converts an infrastructure problem into a
    fake quality regression and corrupts every baseline afterwards.
    """

    def sometimes_broken(item, trace, metric):
        if item.id.endswith("2"):
            raise RuntimeError("judge returned malformed JSON")
        return 0.9, "ok"

    runner = EvalRunner(fake_answer, sometimes_broken, ["faithfulness"], max_workers=2)
    report = runner.run(items[:12])

    failed = [s for s in report.scores if s.failed]
    assert failed, "test setup produced no failures"
    assert all(s.score is None for s in failed)
    assert report.mean("faithfulness") == pytest.approx(0.9), (
        "a judge failure dragged the mean down -- it was scored as 0"
    )


def test_failures_are_counted_and_reported_separately(items):
    def always_broken(item, trace, metric):
        raise RuntimeError("judge down")

    report = EvalRunner(fake_answer, always_broken, ["faithfulness"]).run(items[:5])

    assert report.failure_count() == 5
    assert report.failure_rate() == 1.0
    assert report.mean("faithfulness") is None, "a mean was invented from zero data"


def test_a_run_that_mostly_failed_is_rejected_as_invalid(items):
    """The dangerous outcome is publishing a mean from the few calls that worked.

    That number looks like a quality signal and is an artefact of which calls
    survived.
    """

    def mostly_broken(item, trace, metric):
        if hash(item.id) % 4:
            raise RuntimeError("judge down")
        return 0.95, "ok"

    report = EvalRunner(fake_answer, mostly_broken, ["faithfulness"]).run(items)

    with pytest.raises(EvaluationRunInvalid, match="artefact of which calls"):
        report.assert_valid(max_failure_rate=0.2)


def test_a_healthy_run_passes_validation(items):
    report = EvalRunner(fake_answer, perfect_scorer, ["faithfulness"]).run(items[:10])
    report.assert_valid(max_failure_rate=0.2)  # must not raise


def test_one_item_blowing_up_does_not_lose_the_others(items):
    """`answer_fn` itself failing must not destroy the whole sweep."""

    def flaky_answer(question: str):
        if "capital of France" in question:
            raise ConnectionError("pipeline died on this one")
        return "ok"

    runner = EvalRunner(flaky_answer, perfect_scorer, ["faithfulness"])
    report = runner.run(items)

    assert report.failure_count() >= 1
    assert len(report.values("faithfulness")) > 30, "a single failure lost the sweep"


# ===========================================================================
# SLICING
# ===========================================================================


def test_scores_slice_by_question_category(items):
    """The slice that matters most.

    A system can hold its overall mean steady while its refusal rate on
    unanswerable questions collapses to zero.
    """

    def category_dependent(item, trace, metric):
        return (0.95, "good") if item.is_answerable else (0.10, "hallucinated")

    report = EvalRunner(fake_answer, category_dependent, ["faithfulness"]).run(items)

    slices = report.by_category("faithfulness")
    assert slices["single_hop"] > 0.9
    assert slices["unanswerable"] < 0.2

    overall = report.mean("faithfulness")
    assert overall > 0.8, (
        "the overall mean looks healthy while unanswerable questions are failing "
        "-- which is exactly why you must slice"
    )


def test_worst_items_names_what_to_look_at(items):
    """A gate that prints only an average is nearly useless when it fails."""

    def varied(item, trace, metric):
        return (0.1 if item.id == "sh-01" else 0.9), "r"

    report = EvalRunner(fake_answer, varied, ["faithfulness"]).run(items[:10])
    worst = report.worst_items("faithfulness", n=3)

    assert worst[0].item_id == "sh-01"
    assert worst[0].score == pytest.approx(0.1)


def test_multiple_metrics_are_tracked_independently(items):
    def two_metrics(item, trace, metric):
        return (0.9, "r") if metric == "faithfulness" else (0.4, "r")

    report = EvalRunner(
        fake_answer, two_metrics, ["faithfulness", "answer_relevancy"]
    ).run(items[:6])

    assert report.metric_names() == ["answer_relevancy", "faithfulness"]
    assert report.mean("faithfulness") == pytest.approx(0.9)
    assert report.mean("answer_relevancy") == pytest.approx(0.4)


# ===========================================================================
# BUDGET
# ===========================================================================


def test_a_budget_stops_an_expensive_sweep(items):
    runner = EvalRunner(
        fake_answer, perfect_scorer, ["faithfulness"],
        max_workers=1, budget=Budget(max_calls=5),
    )
    report = runner.run(items)

    # Items beyond the budget are recorded as failures, not silently dropped --
    # a shorter sweep that looks complete is worse than one that says it stopped.
    assert any("BudgetExceeded" in s.error for s in report.scores)
    assert len(report.values("faithfulness")) <= 6


def test_budget_report_is_attached_to_the_evaluation_report(items):
    runner = EvalRunner(
        fake_answer, perfect_scorer, ["faithfulness"], budget=Budget(max_calls=1000)
    )
    report = runner.run(items[:3])
    assert "calls" in report.budget_report


# ===========================================================================
# VARIANCE AND GATING
# ===========================================================================


def test_repeated_runs_give_the_gate_a_noise_band(items):
    """A single judged run has no error bar. Gating on it is a coin flip."""
    state = {"n": 0}

    def drifting(item, trace, metric):
        state["n"] += 1
        return 0.90 + (0.02 if state["n"] % 3 else -0.02), "r"

    runner = EvalRunner(fake_answer, drifting, ["faithfulness"])
    reports = runner.run_repeatedly(items[:8], n=3)

    samples = samples_from_runs(reports)
    assert len(samples) == 1
    assert samples[0].n == 3, "each run should contribute one observation"
    assert samples[0].stdev >= 0


def test_samples_from_runs_measures_run_to_run_variance_not_item_spread():
    """A subtle but important distinction.

    Each run contributes its MEAN, so stdev measures run-to-run variance --
    which is the noise the gate must not fire inside. Pooling every individual
    item score would instead measure how much items differ from each other,
    a different and much larger number that would make the gate useless.
    """
    r1 = make_report([ItemScore("a", "single_hop", "m", 0.0), ItemScore("b", "single_hop", "m", 1.0)])
    r2 = make_report([ItemScore("a", "single_hop", "m", 0.0), ItemScore("b", "single_hop", "m", 1.0)])

    samples = samples_from_runs([r1, r2])

    assert samples[0].values == [0.5, 0.5], "item spread leaked into the run sample"
    assert samples[0].stdev == 0.0, (
        "two identical runs must show zero run-to-run variance, however "
        "different the individual items were"
    )


def test_gate_passes_when_scores_match_the_baseline(items, tmp_path):
    baseline_path = tmp_path / "baseline.json"
    record_baseline([MetricSample("faithfulness", [0.90, 0.90, 0.90])], path=baseline_path)

    reports = EvalRunner(fake_answer, lambda i, t, m: (0.90, "r"), ["faithfulness"]).run_repeatedly(
        items[:5], n=3
    )
    result = gate(reports, baseline_path=baseline_path)

    assert result.passed


def test_gate_catches_a_real_regression(items, tmp_path):
    baseline_path = tmp_path / "baseline.json"
    record_baseline([MetricSample("faithfulness", [0.90, 0.91, 0.89])], path=baseline_path)

    reports = EvalRunner(fake_answer, lambda i, t, m: (0.45, "r"), ["faithfulness"]).run_repeatedly(
        items[:5], n=3
    )
    result = gate(reports, baseline_path=baseline_path)

    assert not result.passed
    assert result.regressions[0].name == "faithfulness"


def test_gate_writes_a_report_artifact(items, tmp_path):
    baseline_path = tmp_path / "baseline.json"
    record_baseline([MetricSample("faithfulness", [0.9, 0.9, 0.9])], path=baseline_path)

    reports = EvalRunner(fake_answer, perfect_scorer, ["faithfulness"]).run_repeatedly(
        items[:3], n=2
    )
    result = gate(reports, baseline_path=baseline_path)

    from core.config import ARTIFACTS_DIR

    written = (ARTIFACTS_DIR / "regression_report.md").read_text()
    assert "| metric |" in written
    assert result.markdown() in written


# ===========================================================================
# ARTIFACTS
# ===========================================================================


def test_json_artifact_has_everything_ci_needs(items, tmp_path):
    def category_dependent(item, trace, metric):
        return (0.95, "good") if item.is_answerable else (0.10, "bad")

    report = EvalRunner(fake_answer, category_dependent, ["faithfulness"]).run(items)
    path = report.save_json(tmp_path / "report.json")
    data = json.loads(path.read_text())

    assert "faithfulness" in data["metrics"]
    assert "by_category" in data["metrics"]["faithfulness"]
    assert data["metrics"]["faithfulness"]["by_category"]["unanswerable"] < 0.2
    assert len(data["items"]) == len(items)


def test_markdown_report_flags_excluded_failures(items):
    def half_broken(item, trace, metric):
        if item.id == "sh-01":
            raise RuntimeError("judge died")
        return 0.9, "ok"

    report = EvalRunner(fake_answer, half_broken, ["faithfulness"]).run(items[:6])
    md = report.markdown()

    assert "| metric | mean |" in md
    assert "EXCLUDED from the means, not counted as zero" in md


def test_report_to_samples_carries_failure_counts(items):
    def half_broken(item, trace, metric):
        if item.id == "sh-01":
            raise RuntimeError("judge died")
        return 0.9, "ok"

    report = EvalRunner(fake_answer, half_broken, ["faithfulness"]).run(items[:6])
    samples = report.to_samples()

    assert samples[0].failures == 1
    assert 0.9 == pytest.approx(samples[0].mean)


def test_empty_run_does_not_crash_the_reporting():
    report = EvaluationReport()
    assert report.metric_names() == []
    assert report.mean("anything") is None
    assert report.failure_rate() == 0.0
    assert samples_from_runs([]) == []
