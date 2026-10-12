"""RAG quality: retrieval, grounding, correctness and refusal behaviour.

One test per golden in data/goldens/rag.json. The metric set depends on the
golden's category (see evalkit.metrics.suites).
"""

import pytest

from evalkit.builders import rag_test_case
from evalkit.config import get_settings
from evalkit.datasets import case_id, load_goldens
from evalkit.metrics import suites
from evalkit.runner import run_metrics

pytestmark = pytest.mark.llm

GOLDENS = load_goldens(get_settings().goldens_dir / "rag.json")


@pytest.mark.parametrize("golden", GOLDENS, ids=case_id)
def test_rag_quality(golden, rag_app, judge, thresholds, record):
    category = golden.additional_metadata["category"]
    test_case = rag_test_case(golden, rag_app.query(golden.input))
    metrics = suites.rag_deterministic(category, thresholds) + suites.rag_judged(
        category, judge, thresholds, has_context=bool(golden.context)
    )
    record(run_metrics("rag", case_id(golden), test_case, metrics))
