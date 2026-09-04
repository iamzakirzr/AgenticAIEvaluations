"""
core.baseline -- variance-aware regression gating.

=============================================================================
THE PROBLEM THIS SOLVES, AND WHY NAIVE THRESHOLDS FAIL
=============================================================================
The obvious way to gate a pull request on quality is a fixed threshold:

    assert faithfulness >= 0.80

It does not survive contact with an LLM judge, for two reasons.

  1. IT CANNOT DETECT A REGRESSION. A change that drops faithfulness from 0.95
     to 0.82 is a serious regression and this assertion passes happily.

  2. IT FIRES ON NOISE. A judged metric varies between identical runs. If the
     true value sits near 0.80, the build goes red and green at random, and
     within a fortnight the team adds `continue-on-error: true`.

The fix is to compare against a RECORDED BASELINE and to know how much your
metric moves when NOTHING has changed. That second part is what almost nobody
measures, and it is what makes the gate trustworthy:

    regression  <=>  drop  >  max(min_delta, sensitivity x noise)

where `noise` is the standard deviation observed across repeated runs of the
unchanged system. A drop inside the noise band is not evidence of anything.

=============================================================================
WHY N RUNS, NOT ONE
=============================================================================
A single judged run gives you a number with no error bar. Running the same
evaluation N times and recording the spread turns "faithfulness is 0.87" into
"faithfulness is 0.87 +/- 0.04", which is the difference between a metric you
can gate on and a number you can only nod at.

N=3 is usually enough to detect gross instability. If your metric's stdev is
larger than the regressions you care about, that metric CANNOT gate your CI --
and finding that out is a result, not a failure. See
`04_deepeval/calibrate.py` for the judge-quality half of the same argument.

=============================================================================
WHY THE CONFIG FINGERPRINT MATTERS
=============================================================================
Two eval runs are only comparable if they used the same models, the same
chunking and the same dataset. A baseline that does not record those will
silently compare llama3.1:8b against qwen2.5:14b and report a "regression"
that is really a model change.

`Baseline.fingerprint` stores them, and `compare()` REFUSES to gate across a
fingerprint change -- it reports the comparison as informational instead. That
refusal is the single most important safety property in this file.
=============================================================================
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from core.config import ARTIFACTS_DIR, settings

DEFAULT_BASELINE_PATH = Path(__file__).resolve().parent.parent / "baselines" / "baseline.json"


# ===========================================================================
# MEASUREMENT
# ===========================================================================


@dataclass
class MetricSample:
    """One metric measured possibly several times.

    ``values`` holds every observation. Keeping them all (rather than only the
    mean) is what makes the noise band computable later.
    """

    name: str
    values: list[float] = field(default_factory=list)
    higher_is_better: bool = True

    # Number of judge/infrastructure failures seen while producing these
    # values. Recorded SEPARATELY and never folded into the scores -- a broken
    # judge must not masquerade as a quality regression.
    failures: int = 0

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.values) if self.values else 0.0

    @property
    def stdev(self) -> float:
        """Sample standard deviation; 0.0 when there are fewer than 2 values.

        Returning 0.0 for n=1 is honest -- one observation tells you nothing
        about spread -- but it means the noise band collapses to `min_delta`.
        `compare()` warns when it is gating on a single-run baseline.
        """
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MetricSample":
        return cls(
            name=raw["name"],
            values=[float(v) for v in raw.get("values", [])],
            higher_is_better=bool(raw.get("higher_is_better", True)),
            failures=int(raw.get("failures", 0)),
        )


def measure_repeatedly(
    name: str,
    run: "callable[[], float]",
    n: int = 3,
    higher_is_better: bool = True,
) -> MetricSample:
    """Run a scoring function N times and collect the spread.

    Exceptions are counted as failures rather than scored, for the reason
    core/corpus/llm_as_judge.md gives: recording a broken judge as 0.0 converts
    an infrastructure problem into a fake quality regression.
    """
    sample = MetricSample(name=name, higher_is_better=higher_is_better)
    for _ in range(n):
        try:
            sample.values.append(float(run()))
        except Exception:  # noqa: BLE001 - counted, never scored
            sample.failures += 1
    return sample


# ===========================================================================
# BASELINE
# ===========================================================================


def current_fingerprint(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything that must match for two runs to be comparable."""
    fingerprint = {
        "chat_model": settings.chat_model,
        "judge_model": settings.judge_model,
        "embed_model": settings.embed_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": settings.top_k,
        "temperature": settings.temperature,
    }
    if extra:
        fingerprint.update(extra)
    return fingerprint


@dataclass
class Baseline:
    """A recorded reference run, with enough context to be comparable."""

    metrics: dict[str, MetricSample] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=current_fingerprint)
    git_sha: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    note: str = ""

    def add(self, sample: MetricSample) -> "Baseline":
        self.metrics[sample.name] = sample
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {name: s.to_dict() for name, s in self.metrics.items()},
            "fingerprint": self.fingerprint,
            "git_sha": self.git_sha,
            "created_at": self.created_at,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Baseline":
        return cls(
            metrics={
                name: MetricSample.from_dict(value)
                for name, value in raw.get("metrics", {}).items()
            },
            fingerprint=raw.get("fingerprint", {}),
            git_sha=raw.get("git_sha", ""),
            created_at=raw.get("created_at", ""),
            note=raw.get("note", ""),
        )

    def save(self, path: Path | None = None) -> Path:
        target = path or DEFAULT_BASELINE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys + trailing newline so a re-recorded baseline produces a
        # readable git diff instead of a whole-file rewrite.
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    @classmethod
    def load(cls, path: Path | None = None) -> "Baseline | None":
        target = path or DEFAULT_BASELINE_PATH
        if not target.exists():
            return None
        return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))


# ===========================================================================
# COMPARISON
# ===========================================================================


@dataclass
class MetricComparison:
    name: str
    baseline_mean: float
    current_mean: float
    noise_band: float
    delta: float
    regressed: bool
    improved: bool
    reason: str

    @property
    def symbol(self) -> str:
        if self.regressed:
            return "REGRESSION"
        if self.improved:
            return "improved"
        return "stable"


@dataclass
class GateResult:
    comparisons: list[MetricComparison]
    fingerprint_changed: bool
    fingerprint_diff: dict[str, tuple[Any, Any]]
    warnings: list[str] = field(default_factory=list)

    @property
    def regressions(self) -> list[MetricComparison]:
        return [c for c in self.comparisons if c.regressed]

    @property
    def passed(self) -> bool:
        """A fingerprint change NEVER fails the gate -- it disables it.

        Comparing scores across different models or chunk sizes is meaningless,
        so the honest response is to report the numbers and decline to judge
        them, not to fail a build for a change the developer made on purpose.
        """
        if self.fingerprint_changed:
            return True
        return not self.regressions

    def markdown(self) -> str:
        """A report suitable for pasting into a PR comment."""
        lines: list[str] = []

        if self.fingerprint_changed:
            lines.append(
                "> **Gate disabled: configuration changed.** Scores below are "
                "informational only -- they were produced under different "
                "settings than the baseline, so a difference is not a regression."
            )
            lines.append("")
            lines.append("| setting | baseline | current |")
            lines.append("|---|---|---|")
            for key, (was, now) in sorted(self.fingerprint_diff.items()):
                lines.append(f"| `{key}` | `{was}` | `{now}` |")
            lines.append("")

        lines.append("| metric | baseline | current | delta | noise band | verdict |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for c in sorted(self.comparisons, key=lambda x: x.name):
            lines.append(
                f"| {c.name} | {c.baseline_mean:.3f} | {c.current_mean:.3f} | "
                f"{c.delta:+.3f} | ±{c.noise_band:.3f} | {c.symbol} |"
            )

        for warning in self.warnings:
            lines.append("")
            lines.append(f"> WARNING: {warning}")

        if self.regressions:
            lines.append("")
            lines.append("**Regressions:**")
            for c in self.regressions:
                lines.append(f"- `{c.name}`: {c.reason}")

        return "\n".join(lines)


def compare(
    current: Iterable[MetricSample],
    baseline: Baseline | None,
    sensitivity: float = 2.0,
    min_delta: float = 0.02,
    fingerprint: dict[str, Any] | None = None,
) -> GateResult:
    """Decide whether ``current`` regressed against ``baseline``.

    Args:
        sensitivity: how many standard deviations of observed noise a drop must
            exceed. 2.0 is a reasonable default -- roughly, do not cry wolf at
            anything that could plausibly be sampling variation.
        min_delta: an absolute floor, so that a metric which happened to be
            perfectly stable across N runs (stdev 0.0) does not fail on a
            0.001 wobble.

    The noise band is `max(min_delta, sensitivity * pooled_stdev)`, where the
    pooled stdev is the larger of the baseline's and the current run's -- being
    pessimistic about noise is the safe direction for a gate.
    """
    current_samples = {s.name: s for s in current}
    warnings: list[str] = []

    # --- fingerprint ------------------------------------------------------
    now_fp = fingerprint if fingerprint is not None else current_fingerprint()
    fingerprint_diff: dict[str, tuple[Any, Any]] = {}
    if baseline is not None:
        for key in set(baseline.fingerprint) | set(now_fp):
            was, is_now = baseline.fingerprint.get(key), now_fp.get(key)
            if was != is_now:
                fingerprint_diff[key] = (was, is_now)

    if baseline is None:
        warnings.append(
            "No baseline recorded yet. Nothing was gated. "
            "Record one with `make baseline` once you trust the current scores."
        )
        return GateResult(
            comparisons=[
                MetricComparison(
                    name=s.name,
                    baseline_mean=0.0,
                    current_mean=s.mean,
                    noise_band=0.0,
                    delta=0.0,
                    regressed=False,
                    improved=False,
                    reason="no baseline",
                )
                for s in current_samples.values()
            ],
            fingerprint_changed=False,
            fingerprint_diff={},
            warnings=warnings,
        )

    comparisons: list[MetricComparison] = []
    for name, sample in current_samples.items():
        reference = baseline.metrics.get(name)
        if reference is None:
            comparisons.append(
                MetricComparison(
                    name=name,
                    baseline_mean=0.0,
                    current_mean=sample.mean,
                    noise_band=0.0,
                    delta=0.0,
                    regressed=False,
                    improved=False,
                    reason="new metric, not in baseline",
                )
            )
            continue

        if reference.n < 2 and sample.n < 2:
            warnings.append(
                f"`{name}` has only one observation on both sides, so its noise "
                f"band falls back to min_delta ({min_delta}). Re-record the "
                f"baseline with n>=3 to gate on it meaningfully."
            )

        noise = max(min_delta, sensitivity * max(reference.stdev, sample.stdev))
        raw_delta = sample.mean - reference.mean

        # For a metric where lower is better (noise sensitivity, latency, cost)
        # an INCREASE is the regression. Normalise so `drop` is always "worse".
        drop = -raw_delta if sample.higher_is_better else raw_delta

        regressed = drop > noise
        improved = -drop > noise

        if regressed:
            reason = (
                f"moved {raw_delta:+.3f} (baseline {reference.mean:.3f} -> "
                f"{sample.mean:.3f}), which is outside the ±{noise:.3f} noise band"
            )
        elif improved:
            reason = f"improved by {abs(raw_delta):.3f}"
        else:
            reason = f"within the ±{noise:.3f} noise band"

        if sample.failures:
            warnings.append(
                f"`{name}` had {sample.failures} judge/infrastructure failure(s). "
                f"Those are excluded from the score, not counted as zeros -- but "
                f"a high failure rate makes the mean unrepresentative."
            )

        comparisons.append(
            MetricComparison(
                name=name,
                baseline_mean=reference.mean,
                current_mean=sample.mean,
                noise_band=noise,
                delta=raw_delta,
                regressed=regressed,
                improved=improved,
                reason=reason,
            )
        )

    return GateResult(
        comparisons=comparisons,
        fingerprint_changed=bool(fingerprint_diff),
        fingerprint_diff=fingerprint_diff,
        warnings=warnings,
    )


def write_report(result: GateResult, path: Path | None = None) -> Path:
    """Write the markdown report where CI can upload or post it."""
    target = path or (ARTIFACTS_DIR / "regression_report.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result.markdown() + "\n", encoding="utf-8")
    return target


def git_sha() -> str:
    """Current commit, or empty string outside a git checkout."""
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - absence of git is not an error here
        return ""


def record_baseline(
    samples: Sequence[MetricSample],
    path: Path | None = None,
    note: str = "",
) -> Path:
    """Save the current run as the new reference."""
    baseline = Baseline(git_sha=git_sha(), note=note)
    for sample in samples:
        baseline.add(sample)
    return baseline.save(path)
