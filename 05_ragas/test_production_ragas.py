"""
Lesson 05 production scenarios. FAST TIER except where marked.

Run:  pytest 05_ragas/test_production_ragas.py -v
      pytest -m judge 05_ragas/test_production_ragas.py -v

The centrepiece is the NaN behaviour. `raise_exceptions=False` is the setting
you want in production, and it is also the setting that silently destroys your
metrics if you average the results naively.
"""

from __future__ import annotations

import math

import pytest
from production_ragas import (
    EXAMPLE_HOSTED_PRICING,
    LOCAL_PRICING,
    MetricSummary,
    Pricing,
    is_failure,
    production_run_config,
    projected_cost,
    rows_from_result,
    summarise,
    summary_report,
)

NAN = float("nan")


# ===========================================================================
# THE NaN TRAP
# ===========================================================================


def test_naive_averaging_of_ragas_output_is_destroyed_by_one_nan():
    """Demonstrates the failure this module exists to prevent.

    `raise_exceptions=False` turns a failed row into NaN. One NaN destroys the
    mean of the entire metric, and nothing warns you.
    """
    scores = [0.9, 0.8, NAN, 0.95]
    naive = sum(scores) / len(scores)
    assert math.isnan(naive), "test premise is wrong -- NaN should poison the mean"


def test_nan_is_truthy_so_a_falsiness_check_does_not_catch_it():
    """Why `if score:` is not a sufficient guard.

    float('nan') is truthy, so the obvious defensive check lets it straight
    through into your aggregate.
    """
    assert bool(NAN) is True
    assert is_failure(NAN) is True, "is_failure must catch what truthiness does not"


def test_is_failure_classifies_every_no_measurement_case():
    assert is_failure(None)
    assert is_failure(NAN)
    assert is_failure(float("inf"))
    assert is_failure("error")
    assert not is_failure(0.0), "a genuine score of zero is a measurement, not a failure"
    assert not is_failure(1.0)


def test_summarise_excludes_failures_instead_of_averaging_them():
    rows = [
        {"faithfulness": 0.9},
        {"faithfulness": 0.8},
        {"faithfulness": NAN},
        {"faithfulness": 0.7},
    ]
    summary = summarise(rows)["faithfulness"]

    assert summary.n == 3
    assert summary.failures == 1
    assert summary.mean == pytest.approx(0.8)
    assert not math.isnan(summary.mean)


def test_a_metric_that_failed_everywhere_reports_none_not_zero():
    """Reporting 0.0 would look like catastrophic quality rather than an outage."""
    summary = summarise([{"faithfulness": NAN}, {"faithfulness": NAN}])["faithfulness"]

    assert summary.mean is None
    assert summary.failure_rate == 1.0


def test_a_zero_score_is_kept_as_a_real_measurement():
    """The inverse mistake: treating a genuine 0.0 as a failure would hide the
    worst results in your dataset."""
    summary = summarise([{"faithfulness": 0.0}, {"faithfulness": 1.0}])["faithfulness"]
    assert summary.n == 2
    assert summary.failures == 0
    assert summary.mean == pytest.approx(0.5)


def test_summarise_handles_several_metrics_per_row():
    rows = [
        {"faithfulness": 0.9, "context_recall": 0.5},
        {"faithfulness": NAN, "context_recall": 0.7},
    ]
    summaries = summarise(rows)

    assert summaries["faithfulness"].failures == 1
    assert summaries["context_recall"].failures == 0
    assert summaries["context_recall"].mean == pytest.approx(0.6)


def test_noise_sensitivity_is_marked_lower_is_better():
    """Getting the direction wrong makes a gate celebrate a system getting worse."""
    summaries = summarise([{"noise_sensitivity": 0.2}, {"faithfulness": 0.9}])

    assert summaries["noise_sensitivity"].higher_is_better is False
    assert summaries["faithfulness"].higher_is_better is True
    assert "lower is better" in summaries["noise_sensitivity"].report()


def test_summary_report_warns_when_a_metric_failed_too_often():
    rows = [{"faithfulness": NAN}] * 8 + [{"faithfulness": 0.95}] * 2
    text = summary_report(summarise(rows), max_failure_rate=0.2)

    assert "WARNING" in text
    assert "80%" in text
    assert "rows that happened to succeed" in text


def test_summary_report_is_quiet_on_a_healthy_run():
    text = summary_report(summarise([{"faithfulness": 0.9}] * 10))
    assert "WARNING" not in text
    assert "faithfulness" in text


def test_metric_summary_arithmetic():
    summary = MetricSummary(name="m", values=[0.5, 0.7], failures=2)
    assert summary.total == 4
    assert summary.failure_rate == 0.5
    assert summary.mean == pytest.approx(0.6)


def test_empty_summary_is_safe():
    summary = MetricSummary(name="m")
    assert summary.mean is None
    assert summary.failure_rate == 0.0
    assert summarise([]) == {}


# ===========================================================================
# RunConfig
# ===========================================================================


def test_production_run_config_lowers_concurrency_for_a_local_model():
    """RAGAS defaults to max_workers=16, which is wrong for one Ollama server.

    Sixteen concurrent requests do not go 16x faster -- they queue, inflate
    latency, and start tripping the timeout.
    """
    default = __import__("ragas.run_config", fromlist=["RunConfig"]).RunConfig()
    tuned = production_run_config()

    assert default.max_workers == 16, "ragas defaults changed; revisit the rationale"
    assert tuned.max_workers <= 4


def test_production_run_config_shortens_the_retry_ladder():
    """max_retries=10 with backoff to max_wait=60 can hang one row for minutes."""
    tuned = production_run_config()
    assert tuned.max_retries <= 3
    assert tuned.max_wait <= 30


def test_seed_is_preserved_because_reproducibility_is_not_optional():
    assert production_run_config().seed == 42


def test_run_config_is_overridable():
    tuned = production_run_config(max_workers=1, max_retries=0, timeout=30)
    assert (tuned.max_workers, tuned.max_retries, tuned.timeout) == (1, 0, 30)


# ===========================================================================
# COST
# ===========================================================================


def test_local_inference_costs_nothing_but_still_accounts():
    assert LOCAL_PRICING.cost(1_000_000, 1_000_000) == 0.0


def test_projected_cost_reports_the_number_that_changes_decisions():
    """Per-run cost rarely changes a decision. 'per month' does."""
    projection = projected_cost(
        input_tokens=100_000, output_tokens=20_000, runs_per_day=1
    )

    assert projection["per_run"] > 0
    assert projection["per_month"] == pytest.approx(projection["per_run"] * 30)


def test_pricing_arithmetic_is_per_thousand_tokens():
    pricing = Pricing(input_per_1k=1.0, output_per_1k=2.0)
    assert pricing.cost(1000, 1000) == pytest.approx(3.0)
    assert pricing.cost(500, 0) == pytest.approx(0.5)


def test_hosted_pricing_constant_is_marked_illustrative():
    """Guards against someone quoting these rates as authoritative.

    Published prices change. A hard-coded rate with no caveat next to it ends
    up in a slide deck as fact.
    """
    from pathlib import Path

    import production_ragas

    source = Path(production_ragas.__file__).read_text()
    declaration = source.split("EXAMPLE_HOSTED_PRICING")[0]

    assert "not authoritative" in declaration, (
        "the illustrative-rates caveat was removed from above EXAMPLE_HOSTED_PRICING"
    )
    assert EXAMPLE_HOSTED_PRICING.input_per_1k > 0
    assert EXAMPLE_HOSTED_PRICING.output_per_1k > EXAMPLE_HOSTED_PRICING.input_per_1k


# ===========================================================================
# RESULT NORMALISATION
# ===========================================================================


def test_rows_from_result_prefers_the_scores_attribute():
    class FakeResult:
        scores = [{"faithfulness": 0.9}, {"faithfulness": 0.8}]

    assert rows_from_result(FakeResult()) == [
        {"faithfulness": 0.9},
        {"faithfulness": 0.8},
    ]


def test_rows_from_result_survives_an_unexpected_result_shape():
    """Which attribute is populated has moved between RAGAS versions.

    A summariser that raises AttributeError on upgrade is worse than one that
    degrades to an empty result the caller can notice.
    """

    class Opaque:
        scores = None

        def to_pandas(self):
            raise RuntimeError("pandas not installed")

    assert rows_from_result(Opaque()) == []


# ===========================================================================
# JUDGED TIER
# ===========================================================================


@pytest.mark.judge
async def test_a_real_evaluation_produces_nan_safe_summaries():
    """End to end with a live judge, through the production path."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "02_langchain"))

    from pipeline import build_ollama_pipeline
    from ragas_adapters import rag_trace_to_sample
    from ragas_setup import build_ragas_llm, ragas_ready

    ready, reason = ragas_ready()
    if not ready:
        pytest.skip(reason)

    from ragas.metrics.collections import Faithfulness

    from core.golden import load_golden

    pipeline = build_ollama_pipeline()
    metric = Faithfulness(llm=build_ragas_llm())
    items = [i for i in load_golden() if i.category == "single_hop"][:4]

    rows = []
    for item in items:
        sample = rag_trace_to_sample(pipeline.answer(item.question), item)
        try:
            result = await metric.ascore(
                user_input=sample.user_input,
                response=sample.response,
                retrieved_contexts=list(sample.retrieved_contexts),
            )
            rows.append({"faithfulness": result.value})
        except Exception as exc:
            print(f"judge failure on {item.id}: {exc}")
            rows.append({"faithfulness": NAN})

    summaries = summarise(rows)
    print("\n" + summary_report(summaries))

    summary = summaries["faithfulness"]
    assert summary.total == len(items)
    if summary.mean is not None:
        assert not math.isnan(summary.mean)
