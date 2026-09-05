#!/usr/bin/env python
"""
The CI entry point: run the suite N times, compare against the baseline, exit.

Usage:
    python scripts/run_regression_gate.py                    # gate
    python scripts/run_regression_gate.py --record           # record a baseline
    python scripts/run_regression_gate.py --repetitions 5
    python scripts/run_regression_gate.py --deterministic    # no model needed

=============================================================================
EXIT CODES ARE THE INTERFACE
=============================================================================
    0   no regression (or the gate was disabled for a good reason)
    1   a regression outside the noise band
    2   the run itself was invalid (too many judge failures)

Code 2 exists because "the judge broke" and "quality dropped" require
completely different responses, and a CI system that reports both as `1` sends
someone to bisect a code change that never happened.

=============================================================================
WHY IT RUNS WITHOUT A MODEL
=============================================================================
With `--deterministic` (the automatic fallback when Ollama is unreachable) the
gate runs on RETRIEVAL metrics only: recall@k, MRR, nDCG, plus the refusal rate
measured by regex. Those need no LLM, so this exact script is usable as a PR
gate, and the judged metrics simply join in when a model is available.

That progressive-enhancement shape is the useful bit: one gate, one baseline
format, one report, and the tier of evidence scales with the hardware you have.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (str(ROOT), str(ROOT / "02_langchain"), str(ROOT / "04_deepeval")):
    if path not in sys.path:
        sys.path.insert(0, path)

from core.baseline import (
    Baseline,
    MetricSample,
    compare,
    current_fingerprint,
    record_baseline,
    write_report,
)
from core.config import ARTIFACTS_DIR
from core.golden import load_golden
from core.metrics import evaluate_retrieval
from core.providers import ollama_available

REFUSAL_MARKER = "does not contain this information"


def build_pipeline(deterministic: bool):
    from pipeline import RagPipeline, build_offline_pipeline

    if deterministic:
        return build_offline_pipeline(["A deterministic placeholder answer [1]."])

    from core.providers import get_chat_model, get_ollama_embeddings

    return RagPipeline(llm=get_chat_model(), embeddings=get_ollama_embeddings()).ingest()


def measure_once(pipeline, items, deterministic: bool = False) -> dict[str, float]:
    """One full pass over the dataset, returning aggregate metrics.

    Every metric here is reference-based and deterministic given the pipeline's
    output, so the only variance across repetitions comes from the MODEL, which
    is exactly the noise the gate needs to know about.

    `deterministic` suppresses metrics that would be MEANINGLESS without a real
    generator -- see the refusal metric below. Reporting a number that measures
    a scripted placeholder rather than the system is the precise failure this
    repository is about, and a gate report is the worst place to do it.
    """
    answerable = [i for i in items if i.is_answerable]
    unanswerable = [i for i in items if not i.is_answerable]

    retrieval_inputs = {}
    refusals = 0

    for item in answerable:
        trace = pipeline.answer(item.question)
        retrieval_inputs[item.id] = (trace.retrieved_doc_ids, item.reference_doc_ids)

    for item in unanswerable:
        trace = pipeline.answer(item.question)
        if REFUSAL_MARKER in (trace.answer or "").lower():
            refusals += 1

    report = evaluate_retrieval(retrieval_inputs)

    metrics = {
        "retrieval_hit_rate": report.hit_rate,
        "retrieval_mrr": report.mrr,
        "retrieval_ndcg": report.ndcg,
        "retrieval_precision": report.precision,
    }
    # The single most important BEHAVIOURAL metric in the suite: does the system
    # refuse when it should? A system that stops refusing looks perfectly
    # healthy on every retrieval metric.
    #
    # But it is only meaningful with a REAL generator. In deterministic mode the
    # answer is a fixed scripted string that never contains the refusal wording,
    # so this would always report 0.000 -- a number describing the test harness,
    # not the system. Publishing it in a gate report would invite someone to
    # gate on it, and it can never move.
    if unanswerable and not deterministic:
        metrics["refusal_rate_unanswerable"] = refusals / len(unanswerable)
    return metrics


def collect(repetitions: int, deterministic: bool) -> list[MetricSample]:
    pipeline = build_pipeline(deterministic)
    items = load_golden()

    runs: list[dict[str, float]] = []
    for index in range(repetitions):
        print(f"  run {index + 1}/{repetitions}...", flush=True)
        runs.append(measure_once(pipeline, items, deterministic=deterministic))

    names = sorted({name for run in runs for name in run})
    return [
        MetricSample(name=name, values=[run[name] for run in runs if name in run])
        for name in names
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--record", action="store_true", help="save as the new baseline")
    parser.add_argument("--deterministic", action="store_true", help="never use a model")
    parser.add_argument("--sensitivity", type=float, default=2.0)
    parser.add_argument("--min-delta", type=float, default=0.02)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.2,
        help="above this share of failed measurements the run exits 2 as invalid",
    )
    args = parser.parse_args()

    deterministic = args.deterministic or not ollama_available()
    if deterministic and not args.deterministic:
        print("Ollama is not reachable -- falling back to deterministic metrics only.")

    # A deterministic run has no model variance, so repeating it measures
    # nothing. Say so rather than burning time producing identical numbers.
    repetitions = args.repetitions
    if deterministic and repetitions > 1:
        print(
            f"Deterministic mode: reducing {repetitions} repetitions to 1 "
            f"(there is no model variance to measure)."
        )
        repetitions = 1

    print(f"Collecting metrics over {repetitions} run(s)...")
    samples = collect(repetitions, deterministic)

    for sample in samples:
        spread = f" +/- {sample.stdev:.4f}" if sample.n > 1 else ""
        failures = f"  FAILED={sample.failures}" if sample.failures else ""
        print(f"  {sample.name:<32} {sample.mean:.4f}{spread}{failures}")

    # EXIT 2: the run itself was invalid. Distinct from a regression, because
    # "the judge broke" and "quality dropped" need completely different
    # responses -- reporting both as 1 sends someone to bisect a code change
    # that never happened.
    invalid = [
        s for s in samples
        if s.total_observations and s.failures / s.total_observations > args.max_failure_rate
    ]
    if invalid:
        for sample in invalid:
            rate = sample.failures / sample.total_observations
            print(
                f"\nRUN INVALID: {sample.name} failed {rate:.0%} of the time "
                f"({sample.failures}/{sample.total_observations}). Its mean is an "
                f"artefact of which calls happened to succeed, not a measure of "
                f"quality. Investigate the judge before trusting any number here "
                f"-- see 04_deepeval/calibrate.py.",
                file=sys.stderr,
            )
        return 2

    if args.record:
        path = record_baseline(
            samples,
            path=args.baseline,
            note=f"{repetitions} run(s), deterministic={deterministic}",
        )
        print(f"\nBaseline written to {path}")
        return 0

    result = compare(
        samples,
        Baseline.load(args.baseline),
        sensitivity=args.sensitivity,
        min_delta=args.min_delta,
        fingerprint=current_fingerprint({"deterministic": deterministic}),
    )
    report_path = write_report(result)

    print("\n" + result.markdown())
    print(f"\nReport written to {report_path}")

    if not result.passed:
        print("\nGATE FAILED: a metric moved outside its noise band.", file=sys.stderr)
        return 1

    if result.fingerprint_changed:
        print("\nGate disabled (configuration changed) -- scores are informational.")
    else:
        print("\nGate passed.")
    return 0


if __name__ == "__main__":
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    raise SystemExit(main())
