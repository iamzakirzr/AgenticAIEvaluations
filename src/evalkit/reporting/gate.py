"""Regression gate: compare this run's aggregate scores to a recorded baseline.

Usage:
    python -m evalkit.reporting.gate                 # compare, exit 1 on regression
    python -m evalkit.reporting.gate --record        # make this run the new baseline

A per-case pass/fail tells you a case is below its threshold. The gate tells
you the system got *worse*, which can happen while every case still passes.
Moving the baseline is how a regression disappears, so --record belongs in a
reviewed commit, never in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.config import get_settings
from evalkit.reporting.results import aggregate, load_outcomes, markdown_summary


@dataclass
class GateResult:
    regressions: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.regressions


def compare(current: dict, baseline: dict, tolerance: float = 0.05) -> GateResult:
    result = GateResult()
    for suite, metrics in baseline.items():
        for metric, base in metrics.items():
            key = f"{suite}/{metric}"
            now = current.get(suite, {}).get(metric)
            if now is None:
                # Not run this time (e.g. a filtered run). Reported, not failed:
                # a gate that fails on `-k rag` is a gate people bypass.
                result.missing.append(key)
                continue
            if base["mean"] is None:
                continue
            result.checked += 1
            if now["mean"] is None:
                result.regressions.append(f"{key}: every case errored (baseline {base['mean']})")
                continue
            delta = now["mean"] - base["mean"]
            worse = delta > tolerance if base["lower_is_better"] else delta < -tolerance
            if worse:
                result.regressions.append(
                    f"{key}: mean {base['mean']:.3f} -> {now['mean']:.3f} (tolerance {tolerance})"
                )
            if now["pass_rate"] < base["pass_rate"] - tolerance:
                result.regressions.append(
                    f"{key}: pass rate {base['pass_rate']:.0%} -> {now['pass_rate']:.0%}"
                )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports-dir", type=Path, default=get_settings().reports_dir)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args(argv)

    baseline_path = args.baseline or args.reports_dir.parent / "baselines" / "baseline.json"
    outcomes = load_outcomes(args.reports_dir)
    if not outcomes:
        print(f"No outcomes in {args.reports_dir}. Run the llm tier first.", file=sys.stderr)
        return 2
    current = aggregate(outcomes)
    (args.reports_dir / "summary.json").write_text(json.dumps(current, indent=2))
    (args.reports_dir / "summary.md").write_text(markdown_summary(current) + "\n")
    print(markdown_summary(current))

    if args.record:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(current, indent=2) + "\n")
        print(f"\nRecorded baseline: {baseline_path}. Review the diff before committing.")
        return 0
    if not baseline_path.exists():
        print(f"\nNo baseline at {baseline_path}; nothing to compare. Use --record.")
        return 0

    result = compare(current, json.loads(baseline_path.read_text()), args.tolerance)
    print(f"\nChecked {result.checked} metric(s) against {baseline_path}.")
    for key in result.missing:
        print(f"  not run: {key}")
    for line in result.regressions:
        print(f"  REGRESSION {line}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
