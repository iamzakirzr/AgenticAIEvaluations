"""Run metrics on one test case, record every score, then assert.

Why not `deepeval.assert_test`? It stops at the verdict. This runner keeps
each metric's score, reason and error so the results file and the regression
gate can see *how close* a case was, which is what tells you a threshold is
wrong or a metric is drifting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from deepeval.metrics import BaseConversationalMetric, BaseMetric
from deepeval.test_case import ConversationalTestCase, LLMTestCase

from evalkit.metrics.suites import LOWER_IS_BETTER


@dataclass
class MetricOutcome:
    suite: str
    case_id: str
    metric: str
    score: float | None
    threshold: float
    success: bool
    reason: str | None
    error: str | None = None
    lower_is_better: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def metric_name(metric: BaseMetric | BaseConversationalMetric) -> str:
    return metric.__name__


def run_metrics(
    suite: str,
    case_id: str,
    test_case: LLMTestCase | ConversationalTestCase,
    metrics: list[BaseMetric] | list[BaseConversationalMetric],
) -> list[MetricOutcome]:
    outcomes = []
    for metric in metrics:
        name = metric_name(metric)
        try:
            metric.measure(test_case)
            outcomes.append(
                MetricOutcome(
                    suite,
                    case_id,
                    name,
                    metric.score,
                    metric.threshold,
                    bool(metric.is_successful()),
                    metric.reason,
                    lower_is_better=name in LOWER_IS_BETTER,
                )
            )
        except Exception as exc:
            # A metric that errors is a failure, never a skip: a judge that
            # cannot produce valid JSON is a broken judge.
            outcomes.append(
                MetricOutcome(
                    suite,
                    case_id,
                    name,
                    None,
                    metric.threshold,
                    False,
                    None,
                    error=f"{type(exc).__name__}: {exc}",
                    lower_is_better=name in LOWER_IS_BETTER,
                )
            )
    return outcomes


def format_failures(outcomes: list[MetricOutcome]) -> str:
    lines = []
    for o in outcomes:
        if o.success:
            continue
        if o.error:
            lines.append(f"  {o.metric}: ERROR {o.error}")
        else:
            op = "<=" if o.lower_is_better else ">="
            lines.append(f"  {o.metric}: {o.score:.3f} (needs {op} {o.threshold}) - {o.reason}")
    return "\n".join(lines)


def assert_passed(outcomes: list[MetricOutcome]) -> None:
    failed = [o for o in outcomes if not o.success]
    if failed:
        case = outcomes[0].case_id
        raise AssertionError(
            f"{case}: {len(failed)}/{len(outcomes)} metric(s) failed\n{format_failures(outcomes)}"
        )
