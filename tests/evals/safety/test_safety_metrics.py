"""Safety on curated probes: bias, toxicity, PII leakage, forbidden content.

These are fixed, reviewable probes that run every night. Generated adversarial
attacks live in test_redteam.py.
"""

import pytest
from deepeval.test_case import LLMTestCase

from evalkit.config import get_settings
from evalkit.datasets import case_id, load_goldens
from evalkit.metrics import suites
from evalkit.runner import run_metrics

pytestmark = pytest.mark.llm

GOLDENS = load_goldens(get_settings().goldens_dir / "safety.json")


@pytest.mark.parametrize("golden", GOLDENS, ids=case_id)
def test_safety(golden, chat_app, judge, thresholds, record):
    answer = chat_app.reply([], golden.input)
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=answer,
        expected_output=golden.expected_output,
        name=golden.name,
    )
    metrics = suites.safety_deterministic(thresholds) + suites.safety_judged(judge, thresholds)
    record(run_metrics("safety", case_id(golden), test_case, metrics))
