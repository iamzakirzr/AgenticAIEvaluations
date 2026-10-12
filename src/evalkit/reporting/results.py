"""Collect metric outcomes during a run and write them to reports/.

With pytest-xdist each worker writes its own file; `load_outcomes` merges
them. Results are per run: the directory is cleared at session start by the
conftest so stale files never leak into an aggregate.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

from evalkit.runner import MetricOutcome


class Recorder:
    def __init__(self) -> None:
        self.outcomes: list[MetricOutcome] = []

    def add(self, outcomes: list[MetricOutcome]) -> None:
        self.outcomes.extend(outcomes)

    def write(self, reports_dir: Path) -> Path | None:
        if not self.outcomes:
            return None
        reports_dir.mkdir(parents=True, exist_ok=True)
        worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
        path = reports_dir / f"outcomes-{worker}.json"
        path.write_text(json.dumps([o.to_dict() for o in self.outcomes], indent=2))
        return path


def load_outcomes(reports_dir: Path) -> list[MetricOutcome]:
    outcomes = []
    for path in sorted(reports_dir.glob("outcomes-*.json")):
        outcomes += [MetricOutcome(**row) for row in json.loads(path.read_text())]
    return outcomes


def aggregate(outcomes: list[MetricOutcome]) -> dict[str, dict[str, dict]]:
    """suite -> metric -> {mean, pass_rate, n, errors, lower_is_better}."""
    groups: dict[tuple[str, str], list[MetricOutcome]] = defaultdict(list)
    for o in outcomes:
        groups[(o.suite, o.metric)].append(o)
    summary: dict[str, dict[str, dict]] = defaultdict(dict)
    for (suite, metric), rows in sorted(groups.items()):
        scored = [r.score for r in rows if r.score is not None]
        summary[suite][metric] = {
            "mean": round(mean(scored), 4) if scored else None,
            "pass_rate": round(sum(r.success for r in rows) / len(rows), 4),
            "n": len(rows),
            "errors": sum(r.error is not None for r in rows),
            "lower_is_better": rows[0].lower_is_better,
        }
    return dict(summary)


def markdown_summary(summary: dict[str, dict[str, dict]]) -> str:
    lines = ["| Suite | Metric | Mean | Pass rate | N | Errors |", "|---|---|---|---|---|---|"]
    for suite, metrics in summary.items():
        for metric, s in metrics.items():
            m = "n/a" if s["mean"] is None else f"{s['mean']:.3f}"
            arrow = " (lower is better)" if s["lower_is_better"] else ""
            lines.append(
                f"| {suite} | {metric}{arrow} | {m} | {s['pass_rate']:.0%} | "
                f"{s['n']} | {s['errors']} |"
            )
    return "\n".join(lines)
