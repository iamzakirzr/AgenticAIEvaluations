"""
PRODUCTION SCENARIO: an evaluation run you can actually put in a pipeline.

=============================================================================
THE GAP BETWEEN "I RAN A METRIC" AND "I HAVE AN EVAL PIPELINE"
=============================================================================
`test_deepeval_rag.py` shows metrics working. A pipeline needs five more things
that no tutorial covers:

  1. A JUDGE FAILURE BUDGET. A local judge fails sometimes. One failure is
     noise. Forty percent failures means the RUN IS INVALID and its scores must
     not be published -- but they will look like a quality regression unless
     something explicitly checks. core/corpus/llm_as_judge.md: failures are
     recorded separately and never scored as zero.

  2. SLICING BY CATEGORY. An overall mean across single_hop and unanswerable
     questions is nearly meaningless; they measure different behaviours. The
     mean can hold steady while refusal collapses.

  3. VARIANCE. One judged run has no error bar. `core.baseline` gates on
     mean +/- observed noise, and needs N runs to know the noise.

  4. RESUMABILITY / CACHING. A judged sweep over 48 items takes minutes. If it
     dies at item 40 you do not want to redo the first 39. DeepEval ships
     CacheConfig for exactly this.

  5. A MACHINE-READABLE ARTIFACT. CI needs JSON to gate on and markdown to post.

=============================================================================
DEEPEVAL'S PRODUCTION CONFIG OBJECTS (verified against 4.2.1)
=============================================================================
`deepeval.evaluate()` takes four config objects that the quickstart never
mentions:

    AsyncConfig(run_async=True, throttle_value=0, max_concurrent=20)
        Concurrency. max_concurrent=20 against a single local Ollama server is
        far too high -- requests queue, latency inflates, and nothing goes
        faster. For local models, 2-4 is realistic.

    CacheConfig(write_cache=True, use_cache=False)
        NOTE use_cache defaults to FALSE. Turning it on is what makes a rerun
        cheap, and is the single most useful flag for iterating on a suite.

    ErrorConfig(ignore_errors=False, skip_on_missing_params=False)
        ignore_errors=True keeps a sweep alive when individual metrics throw.
        Essential for a nightly run -- but ONLY if you count what was ignored,
        otherwise a run where every judge failed reports a cheerful mean of
        nothing.

    DisplayConfig(...)
        Progress output. Turn the indicator off in CI, where it just produces
        thousands of lines of ANSI escapes.
=============================================================================
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for path in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.baseline import Baseline, MetricSample, compare, write_report
from core.config import ARTIFACTS_DIR
from core.golden import GoldenItem
from core.resilience import Budget, map_bounded


class EvaluationRunInvalid(RuntimeError):
    """The run failed so often that its scores cannot be trusted.

    Raised rather than returned, because the dangerous outcome is publishing a
    mean computed from the handful of items where the judge happened to work.
    That number looks like a quality signal and is an artefact of which calls
    survived.
    """


@dataclass
class ItemScore:
    """One metric on one dataset item."""

    item_id: str
    category: str
    metric: str
    score: float | None = None   # None means the judge failed -- never 0.0
    reason: str = ""
    error: str = ""

    @property
    def failed(self) -> bool:
        return self.score is None


@dataclass
class EvaluationReport:
    """Everything a pipeline needs: aggregates, slices, failures, artifacts."""

    scores: list[ItemScore] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    judge_model: str = ""
    budget_report: str = ""

    # ---- aggregates -------------------------------------------------------

    def metric_names(self) -> list[str]:
        return sorted({s.metric for s in self.scores})

    def values(self, metric: str, category: str | None = None) -> list[float]:
        return [
            s.score
            for s in self.scores
            if s.metric == metric
            and s.score is not None
            and (category is None or s.category == category)
        ]

    def mean(self, metric: str, category: str | None = None) -> float | None:
        values = self.values(metric, category)
        return sum(values) / len(values) if values else None

    def failure_count(self, metric: str | None = None) -> int:
        return sum(
            1 for s in self.scores if s.failed and (metric is None or s.metric == metric)
        )

    def failure_rate(self, metric: str | None = None) -> float:
        total = sum(1 for s in self.scores if metric is None or s.metric == metric)
        return self.failure_count(metric) / total if total else 0.0

    def by_category(self, metric: str) -> dict[str, float]:
        """Scores sliced by question category.

        The slice that matters most: a system can hold its overall mean steady
        while its refusal rate on unanswerable questions collapses to zero.
        """
        out: dict[str, float] = {}
        for category in sorted({s.category for s in self.scores}):
            value = self.mean(metric, category)
            if value is not None:
                out[category] = value
        return out

    def worst_items(self, metric: str, n: int = 5) -> list[ItemScore]:
        """The lowest-scoring items. What you actually read when a gate fails."""
        scored = [s for s in self.scores if s.metric == metric and s.score is not None]
        return sorted(scored, key=lambda s: s.score)[:n]

    # ---- validity ----------------------------------------------------------

    def assert_valid(self, max_failure_rate: float = 0.2) -> None:
        """Refuse to publish scores from a run that mostly failed."""
        rate = self.failure_rate()
        if rate > max_failure_rate:
            raise EvaluationRunInvalid(
                f"{rate:.0%} of judged calls failed (limit {max_failure_rate:.0%}). "
                f"These scores are an artefact of which calls happened to "
                f"succeed, not a measure of quality. Investigate the judge "
                f"before trusting any number from this run -- see "
                f"04_deepeval/calibrate.py."
            )

    # ---- artifacts ----------------------------------------------------------

    def to_samples(self) -> list[MetricSample]:
        """Convert into the shape `core.baseline` gates on."""
        samples: list[MetricSample] = []
        for metric in self.metric_names():
            samples.append(
                MetricSample(
                    name=metric,
                    values=self.values(metric),
                    higher_is_better=True,
                    failures=self.failure_count(metric),
                )
            )
        return samples

    def to_json(self) -> dict:
        return {
            "judge_model": self.judge_model,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "budget": self.budget_report,
            "metrics": {
                metric: {
                    "mean": self.mean(metric),
                    "n": len(self.values(metric)),
                    "failures": self.failure_count(metric),
                    "by_category": self.by_category(metric),
                }
                for metric in self.metric_names()
            },
            "items": [
                {
                    "item_id": s.item_id,
                    "category": s.category,
                    "metric": s.metric,
                    "score": s.score,
                    "error": s.error,
                }
                for s in self.scores
            ],
        }

    def save_json(self, path: Path | None = None) -> Path:
        target = path or (ARTIFACTS_DIR / "deepeval_report.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_json(), indent=2, sort_keys=True) + "\n")
        return target

    def markdown(self) -> str:
        lines = [f"### Evaluation report ({self.judge_model})", ""]
        lines.append("| metric | mean | n | judge failures |")
        lines.append("|---|---:|---:|---:|")
        for metric in self.metric_names():
            mean = self.mean(metric)
            # "n/a", never 0.000: a metric whose every call failed has no mean,
            # and printing one would read as catastrophic quality rather than
            # an outage.
            shown = f"{mean:.3f}" if mean is not None else "n/a"
            lines.append(
                f"| {metric} | {shown} | {len(self.values(metric))} "
                f"| {self.failure_count(metric)} |"
            )

        for metric in self.metric_names():
            slices = self.by_category(metric)
            if len(slices) > 1:
                lines += ["", f"**{metric} by category**", "", "| category | mean |", "|---|---:|"]
                lines += [f"| {c} | {v:.3f} |" for c, v in slices.items()]

        if self.failure_count():
            note = (
                f"> {self.failure_count()} judged call(s) failed and were EXCLUDED "
                f"from the means, not counted as zero."
            )
            lines += ["", note]
        return "\n".join(lines)


# ===========================================================================
# THE RUNNER
# ===========================================================================


@dataclass
class EvalRunner:
    """Score a dataset with one or more metrics, resiliently.

    `score_fn(item, trace, metric_name) -> (score, reason)` is injected, which
    keeps this runner independent of DeepEval specifically -- the same runner
    drives the RAGAS suite in lesson 05. That is deliberate: the SPINE is
    shared, the metric dialects are not.
    """

    answer_fn: Callable[[str], object]
    score_fn: Callable[[GoldenItem, object, str], tuple[float, str]]
    metrics: Sequence[str]
    max_workers: int = 2
    max_failure_rate: float = 0.2
    budget: Budget | None = None
    judge_model: str = "unknown"

    def run(self, items: Sequence[GoldenItem]) -> EvaluationReport:
        started = time.perf_counter()
        report = EvaluationReport(judge_model=self.judge_model)

        def score_one(item: GoldenItem) -> list[ItemScore]:
            if self.budget is not None:
                self.budget.check()

            trace = self.answer_fn(item.question)
            results: list[ItemScore] = []
            for metric in self.metrics:
                try:
                    value, reason = self.score_fn(item, trace, metric)
                    results.append(
                        ItemScore(item.id, item.category, metric, float(value), reason)
                    )
                except Exception as exc:
                    # score stays None. NEVER 0.0 -- that would convert an
                    # infrastructure failure into a fake quality regression.
                    results.append(
                        ItemScore(
                            item.id,
                            item.category,
                            metric,
                            None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            if self.budget is not None:
                self.budget.record(output_tokens=200 * len(self.metrics))
            return results

        # Bounded parallelism, keeping partial results. One item's failure must
        # not lose the other 47 scores.
        for item, produced, error in map_bounded(
            score_one, list(items), max_workers=self.max_workers
        ):
            if error is not None:
                for metric in self.metrics:
                    report.scores.append(
                        ItemScore(
                            item.id,
                            item.category,
                            metric,
                            None,
                            error=f"{type(error).__name__}: {error}",
                        )
                    )
                continue
            report.scores.extend(produced)  # type: ignore[arg-type]

        report.elapsed_seconds = time.perf_counter() - started
        if self.budget is not None:
            report.budget_report = self.budget.report()
        return report

    def run_repeatedly(self, items: Sequence[GoldenItem], n: int = 3) -> list[EvaluationReport]:
        """N runs, so the regression gate has a noise band to work with.

        A single judged run gives a number with no error bar, and gating on
        that is how CI becomes a coin flip.
        """
        return [self.run(items) for _ in range(n)]


def samples_from_runs(reports: Sequence[EvaluationReport]) -> list[MetricSample]:
    """Collapse N runs into one sample per metric, preserving every observation.

    Each run contributes its MEAN, so the resulting stdev measures run-to-run
    variance -- which is exactly the noise the gate must not fire inside.
    Pooling every individual item score instead would measure how much items
    differ from each other, which is a different (and much larger) number.
    """
    if not reports:
        return []

    by_metric: dict[str, list[float]] = defaultdict(list)
    failures: dict[str, int] = defaultdict(int)

    for report in reports:
        for metric in report.metric_names():
            mean = report.mean(metric)
            if mean is not None:
                by_metric[metric].append(mean)
            failures[metric] += report.failure_count(metric)

    return [
        MetricSample(name=metric, values=values, failures=failures[metric])
        for metric, values in sorted(by_metric.items())
    ]


def gate(
    reports: Sequence[EvaluationReport],
    baseline_path: Path | None = None,
    sensitivity: float = 2.0,
    min_delta: float = 0.02,
    report_path: Path | None = None,
):
    """Compare N runs against the recorded baseline and write the CI report.

    `report_path` exists so tests can write somewhere disposable. Defaulting to
    the shared artifacts directory was a real bug: a unit test's fixture data
    was left in `.artifacts/regression_report.md`, CI picked the file up, and a
    PR comment reported "faithfulness improved 0.900 -> 1.000" -- a number that
    came from a test, not a measurement.

    That is the exact failure this whole repo is about, committed by the repo
    itself. Tests must never write to a path that a pipeline reads.
    """
    samples = samples_from_runs(reports)
    result = compare(
        samples,
        Baseline.load(baseline_path),
        sensitivity=sensitivity,
        min_delta=min_delta,
    )
    write_report(result, path=report_path)
    return result
