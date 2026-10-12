"""Suite composition: the right metrics, for the right category, at the configured thresholds."""

from deepeval.metrics import (
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    StepEfficiencyMetric,
    ToolCorrectnessMetric,
)

from evalkit.config import get_thresholds
from evalkit.judges import OfflineJudge
from evalkit.metrics import CitationMetric, RefusalMetric, suites

T = get_thresholds()
J = OfflineJudge()


def names(metrics):
    return [m.__name__ for m in metrics]


def test_answerable_gets_retrieval_metrics_and_must_not_refuse():
    judged = suites.rag_judged("answerable", J, T, has_context=True)
    assert any(isinstance(m, ContextualRecallMetric) for m in judged)
    assert any(isinstance(m, HallucinationMetric) for m in judged)
    refusal = next(
        m for m in suites.rag_deterministic("answerable", T) if isinstance(m, RefusalMetric)
    )
    assert refusal.should_refuse is False


def test_hallucination_needs_ground_truth_context():
    judged = suites.rag_judged("answerable", J, T, has_context=False)
    assert not any(isinstance(m, HallucinationMetric) for m in judged)


def test_unanswerable_is_scored_on_refusal_not_recall():
    judged = suites.rag_judged("unanswerable", J, T, has_context=False)
    assert not any(isinstance(m, ContextualRecallMetric) for m in judged)
    assert "Refusal Quality [GEval]" in names(judged)
    refusal = next(
        m for m in suites.rag_deterministic("unanswerable", T) if isinstance(m, RefusalMetric)
    )
    assert refusal.should_refuse is True


def test_false_premise_has_no_refusal_expectation():
    deterministic = suites.rag_deterministic("false_premise", T)
    assert [type(m) for m in deterministic] == [CitationMetric]


def test_thresholds_come_from_config():
    faith = next(
        m for m in suites.rag_judged("answerable", J, T, False) if isinstance(m, FaithfulnessMetric)
    )
    assert faith.threshold == T.get("rag", "faithfulness")


def test_agent_suites():
    det = suites.agent_deterministic(T)
    assert isinstance(det[0], ToolCorrectnessMetric)
    assert isinstance(det[0].model, OfflineJudge)
    assert any(isinstance(m, StepEfficiencyMetric) for m in suites.agent_judged(J, T))


def test_chat_and_safety_suites_construct():
    assert "Knowledge Retention" in names(suites.chat_judged(J, T))
    assert {"Bias", "Toxicity"} <= set(names(suites.safety_judged(J, T)))
    assert names(suites.safety_deterministic(T)) == ["Forbidden Content"]


def test_lower_is_better_names_match_deepeval():
    assert {m.__name__ for m in suites.safety_judged(J, T)} >= {"Bias", "Toxicity"}
    hallucination = HallucinationMetric(model=J)
    assert hallucination.__name__ in suites.LOWER_IS_BETTER
