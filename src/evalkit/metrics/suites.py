"""Metric suites per evaluation area, with thresholds from config/thresholds.yaml.

Which metrics apply depends on the golden's category. Scoring an
unanswerable question with ContextualRecall, for example, measures nothing:
there is no correct context to recall.
"""

from __future__ import annotations

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ArgumentCorrectnessMetric,
    BaseConversationalMetric,
    BaseMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    ConversationCompletenessMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    KnowledgeRetentionMetric,
    PIILeakageMetric,
    RoleAdherenceMetric,
    StepEfficiencyMetric,
    TaskCompletionMetric,
    ToolCorrectnessMetric,
    ToxicityMetric,
    TurnRelevancyMetric,
)
from deepeval.models import DeepEvalBaseLLM

from evalkit.config import Thresholds
from evalkit.judges.offline import OfflineJudge
from evalkit.metrics import geval
from evalkit.metrics.deterministic import (
    CitationMetric,
    ForbiddenContentMetric,
    RedundantToolCallMetric,
    RefusalMetric,
    StepBudgetMetric,
)

ANSWERABLE = {"answerable", "multi_hop"}
DECLINE = {"unanswerable", "false_premise"}

# DeepEval passes these when score <= threshold.
LOWER_IS_BETTER = {"Bias", "Toxicity", "Hallucination"}

AGENT_MAX_STEPS = 6


def rag_deterministic(category: str, t: Thresholds) -> list[BaseMetric]:
    metrics: list[BaseMetric] = [CitationMetric(t.get("rag", "citation"))]
    if category in ANSWERABLE:
        metrics.append(RefusalMetric(should_refuse=False))
    elif category == "unanswerable":
        metrics.append(RefusalMetric(should_refuse=True))
    return metrics


def rag_judged(
    category: str, judge: DeepEvalBaseLLM, t: Thresholds, has_context: bool
) -> list[BaseMetric]:
    if category in DECLINE:
        return [
            geval.refusal_quality(judge, t.get("rag", "refusal_quality")),
            FaithfulnessMetric(t.get("rag", "faithfulness"), model=judge),
        ]
    metrics: list[BaseMetric] = [
        FaithfulnessMetric(t.get("rag", "faithfulness"), model=judge),
        AnswerRelevancyMetric(t.get("rag", "answer_relevancy"), model=judge),
        ContextualPrecisionMetric(t.get("rag", "contextual_precision"), model=judge),
        ContextualRecallMetric(t.get("rag", "contextual_recall"), model=judge),
        ContextualRelevancyMetric(t.get("rag", "contextual_relevancy"), model=judge),
        geval.answer_correctness(judge, t.get("rag", "answer_correctness")),
    ]
    if has_context:
        # Needs ground-truth `context` on the golden, not the retrieved chunks.
        metrics.append(HallucinationMetric(t.get("rag", "hallucination"), model=judge))
    return metrics


def agent_deterministic(t: Thresholds, max_steps: int = AGENT_MAX_STEPS) -> list[BaseMetric]:
    return [
        # Name-level tool matching is exact and needs no LLM; OfflineJudge only
        # satisfies the constructor (and fails loudly if it is ever called).
        ToolCorrectnessMetric(
            threshold=t.get("agent", "tool_correctness"), model=OfflineJudge(), async_mode=False
        ),
        StepBudgetMetric(max_steps, t.get("agent", "step_budget")),
        RedundantToolCallMetric(t.get("agent", "redundant_tool_calls")),
    ]


def agent_judged(judge: DeepEvalBaseLLM, t: Thresholds) -> list[BaseMetric]:
    return [
        ArgumentCorrectnessMetric(t.get("agent", "argument_correctness"), model=judge),
        TaskCompletionMetric(t.get("agent", "task_completion"), model=judge),
        StepEfficiencyMetric(t.get("agent", "step_efficiency"), model=judge),
    ]


def chat_judged(judge: DeepEvalBaseLLM, t: Thresholds) -> list[BaseConversationalMetric]:
    return [
        KnowledgeRetentionMetric(t.get("chat", "knowledge_retention"), model=judge),
        RoleAdherenceMetric(t.get("chat", "role_adherence"), model=judge),
        ConversationCompletenessMetric(t.get("chat", "conversation_completeness"), model=judge),
        TurnRelevancyMetric(t.get("chat", "turn_relevancy"), model=judge),
        geval.support_resolution(judge, t.get("chat", "support_resolution")),
    ]


def safety_deterministic(t: Thresholds) -> list[BaseMetric]:
    return [ForbiddenContentMetric(threshold=t.get("safety", "forbidden_content"))]


def safety_judged(judge: DeepEvalBaseLLM, t: Thresholds) -> list[BaseMetric]:
    return [
        BiasMetric(t.get("safety", "bias"), model=judge),
        ToxicityMetric(t.get("safety", "toxicity"), model=judge),
        PIILeakageMetric(t.get("safety", "pii_leakage"), model=judge),
    ]
