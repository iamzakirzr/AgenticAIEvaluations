"""
Tests for the CI entry point. FAST TIER.

Run:  pytest scripts/ -v

The gate script is the thing CI actually executes, so its behaviour is worth
pinning -- especially which metrics it reports, since a gate report is the
worst possible place to publish a number that only looks like a measurement.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
for path in (str(_ROOT), str(_ROOT / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_gate_module():
    """Import the script by path -- it is a CLI entry point, not a package."""
    spec = importlib.util.spec_from_file_location(
        "run_regression_gate", _ROOT / "scripts" / "run_regression_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def gate():
    return load_gate_module()


@pytest.fixture(scope="module")
def golden():
    from core.golden import load_golden

    return load_golden()


def test_deterministic_mode_omits_the_refusal_metric(gate, golden):
    """A number that measures the test harness must not reach a gate report.

    In deterministic mode the answer is a fixed scripted string that never
    contains the refusal wording, so refusal_rate would always be 0.000 --
    describing the harness, not the system. Publishing it would invite someone
    to gate on a metric that can never move.
    """
    from pipeline import build_offline_pipeline

    pipeline = build_offline_pipeline(["A deterministic placeholder answer [1]."])
    metrics = gate.measure_once(pipeline, golden, deterministic=True)

    assert "refusal_rate_unanswerable" not in metrics
    assert "retrieval_mrr" in metrics


def test_a_real_generator_does_report_the_refusal_metric(gate, golden):
    """With a generator that can actually refuse, the metric is meaningful."""
    from pipeline import build_offline_pipeline

    from core.golden import REFUSAL

    honest = build_offline_pipeline([REFUSAL])
    metrics = gate.measure_once(honest, golden, deterministic=False)

    assert metrics["refusal_rate_unanswerable"] == 1.0


def test_the_refusal_metric_catches_a_system_that_stops_refusing(gate, golden):
    """The behaviour the metric exists for.

    A system that hallucinates on unanswerable questions looks perfectly
    healthy on every retrieval metric.
    """
    from pipeline import RagPipeline

    from core.providers import LexicalEmbeddings, scripted_chat_model

    leaky = RagPipeline(
        llm=scripted_chat_model(["The capital of France is Paris."]),
        embeddings=LexicalEmbeddings(dim=2048),
    ).ingest()

    metrics = gate.measure_once(leaky, golden, deterministic=False)

    assert metrics["refusal_rate_unanswerable"] == 0.0
    # ...while retrieval still looks fine, which is the whole point.
    assert metrics["retrieval_hit_rate"] > 0.9


def test_retrieval_metrics_are_always_reported(gate, golden):
    from pipeline import build_offline_pipeline

    metrics = gate.measure_once(build_offline_pipeline(["x"]), golden, deterministic=True)
    for name in ("retrieval_hit_rate", "retrieval_mrr", "retrieval_ndcg", "retrieval_precision"):
        assert name in metrics
        assert 0.0 <= metrics[name] <= 1.0
