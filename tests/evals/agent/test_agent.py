"""Agent evals: right tools, right arguments, task done, efficient trajectory."""

import pytest

from evalkit.builders import agent_test_case
from evalkit.config import get_settings
from evalkit.datasets import case_id, load_goldens
from evalkit.metrics import suites
from evalkit.runner import run_metrics

pytestmark = pytest.mark.llm

GOLDENS = load_goldens(get_settings().goldens_dir / "agent.json")


@pytest.mark.parametrize("golden", GOLDENS, ids=case_id)
def test_agent(golden, agent_app, judge, thresholds, record):
    test_case = agent_test_case(golden, agent_app.run(golden.input))
    metrics = suites.agent_deterministic(thresholds) + suites.agent_judged(judge, thresholds)
    record(run_metrics("agent", case_id(golden), test_case, metrics))
