"""
PRODUCTION SCENARIO: running RAGAS `evaluate()` like a pipeline, not a demo.

=============================================================================
THE GOTCHA THAT MATTERS MOST: raise_exceptions=False PRODUCES NaN
=============================================================================
`ragas.evaluate()` takes `raise_exceptions`. The tempting production setting is
False -- you do not want one bad row to kill a 500-item nightly sweep.

But look at what it does instead: a failed row's score becomes **NaN**, and NaN
propagates through arithmetic silently.

    scores = [0.9, 0.8, float('nan'), 0.95]
    sum(scores) / len(scores)     -> nan       (the whole mean is destroyed)
    statistics.fmean(scores)      -> nan

So a run where ONE row failed reports `nan` for the entire metric. Worse, some
aggregation paths drop NaN instead, which silently computes the mean over only
the rows that happened to succeed -- exactly the artefact
`core/corpus/llm_as_judge.md` warns about, now with no visible failure at all.

Neither behaviour is acceptable, and neither announces itself. So this module
NEVER lets NaN through: `summarise()` partitions scores into valid values and
counted failures, and reports them separately.

=============================================================================
RunConfig: THE FOUR NUMBERS THAT DECIDE WHETHER A SWEEP FINISHES
=============================================================================
Verified against ragas 0.4.3, the defaults are:

    RunConfig(timeout=180, max_retries=10, max_wait=60, max_workers=16, seed=42)

Every one of them is wrong for a local model:

  max_workers=16   Sixteen concurrent requests to one Ollama server does not
                   go 16x faster. Requests queue on the server, per-request
                   latency inflates, and timeouts start firing. 2-4 is real.

  max_retries=10   With exponential backoff up to max_wait=60, ten retries on a
                   genuinely-down server means a single row can hang for many
                   minutes before failing. Fail faster and report it.

  timeout=180      Reasonable for a hosted API; generous for a local 8B model
                   producing structured output, but keep it -- claim-by-claim
                   faithfulness on a long answer genuinely takes a while.

  seed=42          Good. Keep it. Reproducibility is not optional in eval.

=============================================================================
COST
=============================================================================
RAGAS tracks token usage through a `token_usage_parser`, and the official API
is `result.total_tokens()` / `result.total_cost(cost_per_input_token=...)`.
Local inference is free, so cost is 0 here -- but the ACCOUNTING still matters:
token counts tell you how expensive the same suite would be against a hosted
judge, which is the number you need before proposing one.
=============================================================================
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.compat import bootstrap

bootstrap()

from ragas.run_config import RunConfig


def production_run_config(
    max_workers: int = 3,
    max_retries: int = 3,
    timeout: int = 180,
    seed: int = 42,
) -> RunConfig:
    """A RunConfig tuned for a local model rather than a hosted API.

    See the module docstring for why each default is changed. The seed is kept
    at 42: reproducibility is the one default that is already right.
    """
    return RunConfig(
        timeout=timeout,
        max_retries=max_retries,
        max_wait=30,
        max_workers=max_workers,
        seed=seed,
    )


# ===========================================================================
# NaN-SAFE SUMMARISATION
# ===========================================================================


def is_failure(value: Any) -> bool:
    """True for anything that is not a usable score.

    RAGAS signals a failed row as NaN when `raise_exceptions=False`. `None` is
    also possible depending on the path. Both mean 'no measurement', and both
    must be counted rather than averaged.

    NOTE the NaN check must come via math.isnan, NOT `value != value`-style
    cleverness or a truthiness test: `float('nan')` is truthy, so
    `if score:` happily lets it through.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isnan(float(value)) or math.isinf(float(value))
    return True


@dataclass
class MetricSummary:
    """One metric's outcome across a dataset, with failures kept separate."""

    name: str
    values: list[float] = field(default_factory=list)
    failures: int = 0
    higher_is_better: bool = True

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def total(self) -> int:
        return self.n + self.failures

    @property
    def failure_rate(self) -> float:
        return self.failures / self.total if self.total else 0.0

    @property
    def mean(self) -> float | None:
        """None -- never NaN, and never a silently-partial average."""
        return sum(self.values) / len(self.values) if self.values else None

    def report(self) -> str:
        mean = self.mean
        shown = f"{mean:.3f}" if mean is not None else "n/a"
        direction = "" if self.higher_is_better else "  (lower is better)"
        line = f"{self.name:<32} {shown:>7}   n={self.n}"
        if self.failures:
            line += f"  FAILED={self.failures} ({self.failure_rate:.0%})"
        return line + direction


# Metrics where an INCREASE is a regression. Getting this wrong makes a gate
# celebrate a system getting worse.
LOWER_IS_BETTER = {"noise_sensitivity", "noise_sensitivity_relevant"}


def summarise(rows: Iterable[dict[str, Any]]) -> dict[str, MetricSummary]:
    """Turn per-row RAGAS scores into NaN-safe per-metric summaries.

    Args:
        rows: one dict per dataset item, mapping metric name -> score. This is
            the shape `EvaluationResult.scores` gives you, and it is also
            trivially constructible from the `.to_pandas()` dataframe, which
            keeps this function testable without running a real evaluation.
    """
    summaries: dict[str, MetricSummary] = {}
    for row in rows:
        for metric, value in row.items():
            summary = summaries.setdefault(
                metric,
                MetricSummary(
                    name=metric, higher_is_better=metric not in LOWER_IS_BETTER
                ),
            )
            if is_failure(value):
                summary.failures += 1
            else:
                summary.values.append(float(value))
    return summaries


def summary_report(summaries: dict[str, MetricSummary], max_failure_rate: float = 0.2) -> str:
    lines = ["RAGAS evaluation summary", "-" * 60]
    lines += [summaries[name].report() for name in sorted(summaries)]

    troubled = [s for s in summaries.values() if s.failure_rate > max_failure_rate]
    if troubled:
        lines.append("")
        lines.append("WARNING: these metrics failed too often to be trusted:")
        for s in troubled:
            lines.append(
                f"  {s.name}: {s.failures}/{s.total} rows failed ({s.failure_rate:.0%}). "
                f"The mean is computed from the rows that happened to succeed, "
                f"which is not a measure of quality."
            )
    return "\n".join(lines)


# ===========================================================================
# COST
# ===========================================================================


@dataclass(frozen=True)
class Pricing:
    """Per-1000-token rates. Zero for local inference, which is the point.

    Keeping the accounting even at zero cost tells you what the same suite
    WOULD cost against a hosted judge -- the number you need before proposing
    one.
    """

    input_per_1k: float = 0.0
    output_per_1k: float = 0.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1000 * self.input_per_1k
            + output_tokens / 1000 * self.output_per_1k
        )


# Illustrative rates so `projected_cost` produces a meaningful number when you
# are deciding whether to move a suite onto a hosted judge. Update before
# quoting anything: published prices change, and these are not authoritative.
EXAMPLE_HOSTED_PRICING = Pricing(input_per_1k=0.003, output_per_1k=0.015)
LOCAL_PRICING = Pricing()


def projected_cost(
    input_tokens: int,
    output_tokens: int,
    runs_per_day: int = 1,
    pricing: Pricing = EXAMPLE_HOSTED_PRICING,
) -> dict[str, float]:
    """What one sweep costs, and what it costs as a habit.

    The per-run number is almost never the one that changes a decision. "This
    nightly suite costs $18/month" is.
    """
    per_run = pricing.cost(input_tokens, output_tokens)
    return {
        "per_run": per_run,
        "per_day": per_run * runs_per_day,
        "per_month": per_run * runs_per_day * 30,
    }


# ===========================================================================
# THE EVALUATION CALL
# ===========================================================================


def evaluate_dataset(
    dataset,
    metrics: Sequence[Any],
    llm=None,
    embeddings=None,
    run_config: RunConfig | None = None,
    token_usage_parser=None,
):
    """Call `ragas.evaluate()` with production settings.

    The two flags that matter:

      raise_exceptions=False   one bad row must not kill a 500-item sweep.
                               ALWAYS pair this with `summarise()`, because it
                               is what turns failures into NaN.

      show_progress=False      a progress bar in CI produces thousands of lines
                               of ANSI escapes and hides the real output.

    Requires a live model, so this is exercised by the `judge`-marked tests
    rather than the fast tier.
    """
    from ragas import evaluate

    return evaluate(
        dataset=dataset,
        metrics=list(metrics),
        llm=llm,
        embeddings=embeddings,
        run_config=run_config or production_run_config(),
        token_usage_parser=token_usage_parser,
        raise_exceptions=False,
        show_progress=False,
    )


def rows_from_result(result) -> list[dict[str, Any]]:
    """Normalise an EvaluationResult into the row dicts `summarise()` expects.

    Tries the documented `.scores` first and falls back to the pandas export,
    because which one is populated has moved between RAGAS versions -- and a
    summariser that raises AttributeError on an upgrade is worse than one that
    tries both.
    """
    scores = getattr(result, "scores", None)
    if scores:
        return [dict(row) for row in scores]

    try:
        frame = result.to_pandas()
    except Exception:
        return []

    metric_columns = [
        c
        for c in frame.columns
        if c
        not in {
            "user_input",
            "response",
            "retrieved_contexts",
            "reference",
            "reference_contexts",
        }
    ]
    return frame[metric_columns].to_dict(orient="records")
