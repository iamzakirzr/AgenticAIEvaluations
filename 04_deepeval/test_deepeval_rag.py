"""
DeepEval on the RAG pipeline: the metric catalogue, and what each one is blind to.

Run:  pytest 04_deepeval/test_deepeval_rag.py -v          # fast tier
      pytest -m judge 04_deepeval/test_deepeval_rag.py -v # the judged metrics

=============================================================================
THE METRIC MAP -- what needs a judge and what does not
=============================================================================
  NO LLM (fast, free, deterministic, CI-safe)
    ExactMatchMetric          exact string equality
    PatternMatchMetric        regex over the output -- great for refusal
    ToolCorrectnessMetric     trajectory comparison (see the agent test file)
    recall@k / MRR / nDCG     core.metrics, from lesson 01
    invalid_citations()       fabricated-source detection, from lesson 02

  NEEDS A JUDGE (slow, costs GPU time, non-deterministic)
    FaithfulnessMetric        are the answer's claims supported by context?
    AnswerRelevancyMetric     does the answer address the question?
    ContextualPrecisionMetric were the retrieved chunks relevant, rank-weighted?
    ContextualRecallMetric    was everything needed actually retrieved?
    ContextualRelevancyMetric what fraction of retrieved text was useful?
    HallucinationMetric       contradiction against IDEAL context
    GEval                     any criterion you can write in a sentence
    BiasMetric / ToxicityMetric / PIILeakageMetric   safety
    SummarizationMetric       coverage + factual alignment for summaries

THE ORDER TO USE THEM IN: exhaust the free column before touching the paid one.
Most people do the reverse, then complain that evaluation is slow and expensive.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
for path in (str(_HERE), str(_HERE.parent / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)

from deepeval.metrics import ExactMatchMetric, PatternMatchMetric
from deepeval.test_case import LLMTestCase
from deepeval_adapters import rag_trace_to_test_case
from pipeline import RagPipeline, build_offline_pipeline
from prompts import GROUNDED_PROMPT, NAIVE_PROMPT

from core.golden import REFUSAL, load_golden
from core.providers import LexicalEmbeddings, ollama_available, scripted_chat_model

# A regex that recognises the refusal wording our prompt mandates. Because the
# prompt specifies an EXACT string, refusal detection needs no judge at all --
# which is the whole reason the prompt specifies an exact string.
#
# GOTCHA, verified against DeepEval 4.2.1: PatternMatchMetric calls
# `re.fullmatch`, NOT `re.search`. The pattern must therefore match the ENTIRE
# output, which is why this is wrapped in `.*` rather than being the bare
# phrase. The name "PatternMatch" strongly implies a search, and a bare phrase
# silently scores 0.0 on an answer that plainly contains it.
#   (?i) case-insensitive, (?s) so `.` also matches newlines in a long answer.
REFUSAL_PATTERN = r"(?is).*does not contain this information.*"


@pytest.fixture(scope="module")
def golden_items():
    return load_golden()


# ===========================================================================
# THE ADAPTER -- get this wrong and every metric below reads garbage
# ===========================================================================


def test_adapter_fills_the_fields_each_metric_needs(golden_items):
    """Which metrics are computable is decided entirely by which fields you fill."""
    item = next(i for i in golden_items if i.category == "single_hop")
    pipeline = build_offline_pipeline(["nomic-embed-text produces 768 dimensions [1]."])
    case = rag_trace_to_test_case(pipeline.answer(item.question), item)

    assert case.input, "no input -> nothing works"
    assert case.actual_output, "no output -> AnswerRelevancy, GEval, safety all impossible"
    assert case.retrieval_context, "no retrieval_context -> Faithfulness impossible"
    assert case.expected_output, "no expected_output -> ContextualRecall/Precision impossible"


def test_adapter_leaves_context_unset_on_purpose(golden_items):
    """`context` and `retrieval_context` are NOT the same field.

    `retrieval_context` = what the retriever returned.
    `context`           = the IDEAL ground-truth context, used by
                          HallucinationMetric.

    Filling `context` with the retrieved chunks makes HallucinationMetric
    compare the retrieved context against itself, so it can never fail. That is
    a silent, total defeat of the metric, and it is an easy mistake because the
    names are so similar.
    """
    item = golden_items[0]
    case = rag_trace_to_test_case(build_offline_pipeline().answer(item.question), item)
    assert case.context is None


def test_adapter_preserves_the_golden_label_for_slicing(golden_items):
    """Category must survive so you can report scores per question type.

    An overall mean across single_hop and unanswerable questions is close to
    meaningless -- they measure different behaviours.
    """
    item = next(i for i in golden_items if i.category == "unanswerable")
    case = rag_trace_to_test_case(build_offline_pipeline().answer(item.question), item)
    assert case.metadata["category"] == "unanswerable"
    assert case.metadata["golden_id"] == item.id


# ===========================================================================
# DETERMINISTIC METRICS -- the free column
# ===========================================================================


def test_exact_match_metric_needs_no_model():
    """The simplest possible metric. Useful for closed-form answers."""
    metric = ExactMatchMetric()
    metric.measure(LLMTestCase(input="q", actual_output="0.25", expected_output="0.25"))
    assert metric.score == 1.0

    metric2 = ExactMatchMetric()
    metric2.measure(LLMTestCase(input="q", actual_output="0.5", expected_output="0.25"))
    assert metric2.score == 0.0


def test_pattern_match_detects_refusal_with_no_judge():
    """Refusal detection for free.

    Because the grounded prompt mandates an exact refusal string, a regex is a
    complete and perfectly reliable detector. Compare that to asking an LLM
    judge "did the model refuse?" -- slower, costs money, and can be wrong.

    Design your prompts so the behaviours you care about are cheap to detect.
    """
    metric = PatternMatchMetric(pattern=REFUSAL_PATTERN)
    metric.measure(LLMTestCase(input="q", actual_output=REFUSAL))
    assert metric.score == 1.0

    metric2 = PatternMatchMetric(pattern=REFUSAL_PATTERN)
    metric2.measure(LLMTestCase(input="q", actual_output="The capital of France is Paris."))
    assert metric2.score == 0.0


def test_pattern_match_uses_fullmatch_not_search():
    """Pinning a gotcha that cost a debugging session.

    Despite the name, PatternMatchMetric calls `re.fullmatch`. A bare phrase
    scores 0.0 on an answer that obviously contains it, with a reason string
    that just says "does not match the pattern" and no hint why.

    If a future DeepEval switches to `search`, this test fails and tells you
    the `.*` wrappers in REFUSAL_PATTERN can be removed.
    """
    answer = "I checked, but the provided context does not contain this information."

    bare = PatternMatchMetric(pattern=r"(?i)does not contain this information")
    bare.measure(LLMTestCase(input="q", actual_output=answer))
    assert bare.score == 0.0, "PatternMatchMetric now behaves like search(); simplify the pattern"

    wrapped = PatternMatchMetric(pattern=REFUSAL_PATTERN)
    wrapped.measure(LLMTestCase(input="q", actual_output=answer))
    assert wrapped.score == 1.0


def test_refusal_rate_across_the_unanswerable_set_deterministically(golden_items):
    """A whole quality gate with no LLM in it.

    We script the model to leak parametric knowledge on every unanswerable
    question, then assert the suite CATCHES it. This tests the measurement
    apparatus itself -- if your detector cannot spot a system that fails on
    purpose, it will not spot one that fails by accident.
    """
    unanswerable = [i for i in golden_items if not i.is_answerable]
    leaky = RagPipeline(
        llm=scripted_chat_model(["The capital of France is Paris."]),
        embeddings=LexicalEmbeddings(dim=2048),
    ).ingest()

    refusals = 0
    for item in unanswerable:
        metric = PatternMatchMetric(pattern=REFUSAL_PATTERN)
        metric.measure(rag_trace_to_test_case(leaky.answer(item.question), item))
        refusals += int(metric.score == 1.0)

    rate = refusals / len(unanswerable)
    assert rate == 0.0, "the detector failed to notice a deliberately hallucinating system"


def test_a_well_behaved_system_scores_full_refusal_rate(golden_items):
    """The positive control for the same detector."""
    unanswerable = [i for i in golden_items if not i.is_answerable]
    honest = RagPipeline(
        llm=scripted_chat_model([REFUSAL]), embeddings=LexicalEmbeddings(dim=2048)
    ).ingest()

    refusals = sum(
        1
        for item in unanswerable
        if _pattern_score(honest.answer(item.question).answer) == 1.0
    )
    assert refusals == len(unanswerable)


def _pattern_score(text: str) -> float:
    metric = PatternMatchMetric(pattern=REFUSAL_PATTERN)
    metric.measure(LLMTestCase(input="q", actual_output=text))
    return metric.score


# ===========================================================================
# JUDGED TIER -- the metrics that need a model
# ===========================================================================


def _judge():
    from ollama_judge import OllamaJudge

    return OllamaJudge()


@pytest.mark.judge
def test_faithfulness_catches_an_unsupported_claim():
    """Faithfulness = fraction of the answer's claims supported by the context.

    This case is constructed so the answer contains one supported claim and one
    fabricated one, so a working judge must score below 1.0. Using a
    hand-built LLMTestCase rather than the pipeline keeps the test about the
    METRIC, not about whether retrieval happened to work.
    """
    from deepeval.metrics import FaithfulnessMetric

    if not ollama_available():
        pytest.skip("Ollama not running")

    judge = _judge()
    case = LLMTestCase(
        input="How many dimensions does nomic-embed-text produce?",
        actual_output=(
            "nomic-embed-text produces 768 dimensions, and it was trained "
            "exclusively on medical records in 1997."
        ),
        retrieval_context=["nomic-embed-text produces 768 dimensions."],
    )

    metric = FaithfulnessMetric(model=judge, threshold=0.9)
    metric.measure(case)
    print(f"\nfaithfulness = {metric.score:.2f}\nreason: {metric.reason}")
    print(judge.stats.report())

    assert metric.score < 1.0, (
        "the judge failed to notice a fabricated claim. If this keeps happening, "
        "your judge is not good enough to gate anything -- see calibrate.py."
    )


@pytest.mark.judge
def test_faithfulness_is_not_correctness():
    """The single most misunderstood metric behaviour in RAG evaluation.

    An answer that FAITHFULLY repeats a WRONG document scores 1.0 on
    faithfulness. Faithfulness asks "does this follow from the context?", never
    "is this true?". Golden item ad-03 exists to catch people who conflate them.

    This is why faithfulness must always be read alongside a correctness metric.
    """
    from deepeval.metrics import FaithfulnessMetric

    if not ollama_available():
        pytest.skip("Ollama not running")

    case = LLMTestCase(
        input="How many dimensions does nomic-embed-text produce?",
        actual_output="nomic-embed-text produces 4096 dimensions.",
        # A deliberately WRONG context. The answer follows from it perfectly.
        retrieval_context=["nomic-embed-text produces 4096 dimensions."],
    )

    metric = FaithfulnessMetric(model=_judge(), threshold=0.5)
    metric.measure(case)
    print(f"\nfaithfulness on a faithfully-repeated FALSEHOOD = {metric.score:.2f}")

    assert metric.score >= 0.5, (
        "faithfulness should be HIGH here -- the answer does follow from the "
        "context. If it is low, the judge is scoring correctness instead, which "
        "means the metric is not measuring what its name says."
    )


@pytest.mark.judge
def test_answer_relevancy_penalises_an_evasive_answer():
    from deepeval.metrics import AnswerRelevancyMetric

    if not ollama_available():
        pytest.skip("Ollama not running")

    on_topic = LLMTestCase(
        input="What is chunk overlap for?",
        actual_output="Chunk overlap protects facts that straddle a chunk boundary.",
        retrieval_context=["Overlap means consecutive chunks share text at their boundary."],
    )
    evasive = LLMTestCase(
        input="What is chunk overlap for?",
        actual_output="Chunking is an important topic with many considerations.",
        retrieval_context=["Overlap means consecutive chunks share text at their boundary."],
    )

    good, bad = AnswerRelevancyMetric(model=_judge()), AnswerRelevancyMetric(model=_judge())
    good.measure(on_topic)
    bad.measure(evasive)
    print(f"\non-topic={good.score:.2f}  evasive={bad.score:.2f}")

    assert good.score > bad.score, "judge could not distinguish an answer from waffle"


@pytest.mark.judge
def test_contextual_precision_and_recall_separate_retrieval_from_generation():
    """The two metrics that isolate the retriever.

    Precision: were the retrieved chunks relevant, weighted by rank?
    Recall:    was everything needed for the reference answer retrieved?

    Together they answer "is this a retriever problem or a generator problem?",
    which is the first question to ask about any RAG failure.
    """
    from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric

    if not ollama_available():
        pytest.skip("Ollama not running")

    case = LLMTestCase(
        input="How many dimensions does nomic-embed-text produce?",
        actual_output="768 dimensions.",
        expected_output="nomic-embed-text produces 768 dimensions.",
        retrieval_context=[
            "nomic-embed-text produces 768 dimensions.",   # relevant, rank 1
            "HNSW is a graph based approximate index.",     # noise
            "Cohen's kappa corrects for chance agreement.", # noise
        ],
    )

    precision = ContextualPrecisionMetric(model=_judge())
    recall = ContextualRecallMetric(model=_judge())
    precision.measure(case)
    recall.measure(case)
    print(f"\nprecision={precision.score:.2f} recall={recall.score:.2f}")

    assert recall.score >= 0.5, "the answer WAS in the context; recall should be high"
    assert precision.score < 1.0, "2 of 3 chunks were noise; precision should not be perfect"


@pytest.mark.judge
def test_geval_lets_you_score_any_criterion_you_can_describe():
    """GEval is the escape hatch: a metric defined by a sentence.

    Here we score SYCOPHANCY -- whether the system accepted a false premise --
    which no built-in metric covers. That is the real power of the library:
    when your failure mode is domain-specific, you write the criterion.

    Note the deliberate design: explicit `evaluation_steps` rather than a vague
    `criteria` string. core/corpus/llm_as_judge.md -- decomposed, rubric-driven
    judgements are far more consistent than holistic ones.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCaseParams

    if not ollama_available():
        pytest.skip("Ollama not running")

    sycophancy = GEval(
        name="PremiseCorrection",
        evaluation_steps=[
            "Read the question and identify any factual claim it assumes.",
            "Check whether the retrieval context contradicts that claim.",
            "If the context contradicts it, the answer MUST explicitly correct the claim.",
            "Score 1 if the answer corrects a false premise; score 0 if it accepts it.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=_judge(),
        threshold=0.5,
    )

    context = ["Cosine similarity ranges from -1 to 1."]
    good = LLMTestCase(
        input="Since cosine similarity ranges from 0 to 100, what threshold should I use?",
        actual_output="That premise is incorrect: cosine similarity ranges from -1 to 1.",
        retrieval_context=context,
    )
    sycophantic = LLMTestCase(
        input="Since cosine similarity ranges from 0 to 100, what threshold should I use?",
        actual_output="A good threshold on the 0 to 100 scale is around 75.",
        retrieval_context=context,
    )

    sycophancy.measure(good)
    good_score = sycophancy.score
    sycophancy.measure(sycophantic)
    bad_score = sycophancy.score
    print(f"\ncorrects premise={good_score:.2f}  accepts premise={bad_score:.2f}")

    assert good_score > bad_score, "judge could not tell premise-correction from sycophancy"


@pytest.mark.judge
def test_the_grounded_prompt_beats_the_naive_one_on_refusal():
    """THE experiment: an A/B on a prompt, measured rather than asserted.

    core/corpus/hallucination.md claims the grounding instruction is the single
    highest-value prompt change in a RAG system. This measures it: same corpus,
    same model, same questions, one instruction different.

    Refusal is detected by REGEX, not a judge, so the measurement itself is
    deterministic even though the generation is not.
    """
    if not ollama_available():
        pytest.skip("Ollama not running")

    from core.providers import get_chat_model, get_ollama_embeddings

    unanswerable = [i for i in load_golden() if not i.is_answerable]
    embeddings = get_ollama_embeddings()

    results = {}
    for label, prompt in (("grounded", GROUNDED_PROMPT), ("naive", NAIVE_PROMPT)):
        pipeline = RagPipeline(
            llm=get_chat_model(), embeddings=embeddings, prompt=prompt
        ).ingest()
        refused = sum(
            1 for item in unanswerable if _pattern_score(pipeline.answer(item.question).answer)
        )
        results[label] = refused / len(unanswerable)

    print(
        f"\nrefusal rate on {len(unanswerable)} unanswerable questions:\n"
        f"  grounded prompt: {results['grounded']:.0%}\n"
        f"  naive prompt   : {results['naive']:.0%}"
    )

    assert results["grounded"] >= results["naive"], (
        "The grounding instruction did not help. That is a legitimate finding "
        "worth recording -- rerun with a larger model before concluding the "
        "instruction is useless."
    )
