"""
RAGAS metrics on the same pipeline DeepEval scored in lesson 04.

Run:  pytest 05_ragas/ -v            # fast tier: setup, adapters, API shape
      pytest -m judge 05_ragas/ -v   # the judged metrics (needs Ollama)

=============================================================================
WHAT THIS LESSON ADDS OVER LESSON 04
=============================================================================
RAGAS and DeepEval overlap heavily on RAG metrics. Three things make RAGAS
worth learning as well:

  1. NoiseSensitivity -- has no DeepEval equivalent, and measures something
     genuinely distinct: how often irrelevant retrieved context causes the
     system to make incorrect claims. Lower is better.

  2. FactualCorrectness -- decomposes both the answer and the reference into
     claims and computes precision/recall/F1 over them, so you get a
     directional read (are we ADDING wrong claims, or MISSING right ones?)
     rather than one blended number.

  3. Explicit argument APIs -- the new collections classes take typed keyword
     arguments, which makes it obvious which inputs a metric actually uses.
     That clarity is genuinely useful when you are trying to work out why a
     metric returned NaN.
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

from adapters import rag_trace_to_sample, sample_to_kwargs, traces_to_dataset
from pipeline import build_offline_pipeline
from ragas_setup import build_ragas_embeddings, build_ragas_llm, ragas_ready

from core.golden import load_golden


@pytest.fixture(scope="module")
def golden_items():
    return load_golden()


@pytest.fixture(scope="module")
def pipeline():
    return build_offline_pipeline(["nomic-embed-text produces 768 dimensions [1]."])


def judged_or_skip():
    ready, reason = ragas_ready()
    if not ready:
        pytest.skip(reason)


# ===========================================================================
# THE IMPORT ITSELF -- a real compatibility problem, pinned
# ===========================================================================


def test_ragas_imports_only_because_of_the_compat_shim():
    """`import ragas` is broken on a modern LangChain stack without core.compat.

    ragas 0.4.3 has a module-level import of
    langchain_community.chat_models.vertexai, which langchain-community 0.4.x
    removed, and it declares langchain-community unpinned. A clean install
    therefore fails on import.

    This test documents the dependency and will start failing usefully if
    someone removes the bootstrap() call from ragas_setup.py.
    """
    import sys as _sys

    from core import compat

    assert compat.is_shim_needed(), (
        "the real vertexai module is importable again -- delete the shim"
    )
    assert "langchain_community.chat_models.vertexai" in _sys.modules, (
        "bootstrap() did not run before ragas was imported"
    )

    import ragas

    assert ragas.__version__.startswith("0.4")


def test_the_classic_metric_import_path_is_deprecated():
    """Every RAGAS tutorial you will find online uses the OLD import path.

    `from ragas.metrics import Faithfulness` still resolves in 0.4.3 but emits
    a DeprecationWarning pointing at `ragas.metrics.collections`. Knowing this
    saves you from copying a snippet that will break at v1.0 -- and from being
    confused about why the docs and the blog posts disagree.
    """
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from ragas.metrics import Faithfulness  # noqa: F401

    messages = " ".join(str(w.message) for w in caught)
    assert "deprecated" in messages.lower()
    assert "collections" in messages, "the warning should name the replacement path"


def test_the_current_metric_path_exports_the_full_catalogue():
    """`ragas.metrics.collections` is the API to actually use."""
    import ragas.metrics.collections as collections

    exported = set(collections.__all__)
    for expected in (
        "Faithfulness",
        "AnswerRelevancy",
        "ContextPrecisionWithReference",
        "ContextRecall",
        "NoiseSensitivity",
        "FactualCorrectness",
        "ResponseGroundedness",
        "ToolCallAccuracy",
        "TopicAdherence",
    ):
        assert expected in exported, f"{expected} missing from ragas.metrics.collections"
    assert len(exported) > 30


# ===========================================================================
# THE ADAPTER
# ===========================================================================


def test_adapter_maps_our_trace_onto_ragas_field_names(pipeline, golden_items):
    """The vocabulary translation, asserted.

    input -> user_input, actual_output -> response, expected_output ->
    reference, retrieval_context -> retrieved_contexts. Getting one wrong
    produces a metric that silently returns NaN rather than an error.
    """
    item = next(i for i in golden_items if i.category == "single_hop")
    sample = rag_trace_to_sample(pipeline.answer(item.question), item)

    assert sample.user_input == item.question
    assert sample.response
    assert sample.retrieved_contexts
    assert sample.reference == item.reference_answer


def test_adapter_leaves_reference_contexts_unset(pipeline, golden_items):
    """Same trap as DeepEval's `context`, under a different name.

    reference_contexts means "the IDEAL context". Filling it with what the
    retriever actually returned makes any metric that uses it compare a thing
    against itself.
    """
    sample = rag_trace_to_sample(pipeline.answer(golden_items[0].question), golden_items[0])
    assert sample.reference_contexts is None


def test_dataset_builds_from_many_traces(pipeline, golden_items):
    """The classic bulk form, which `evaluate()` and most tutorials use."""
    items = [i for i in golden_items if i.is_answerable][:5]
    dataset = traces_to_dataset([(pipeline.answer(i.question), i) for i in items])
    assert len(dataset) == 5


def test_sample_to_kwargs_bridges_the_two_api_styles(pipeline, golden_items):
    sample = rag_trace_to_sample(pipeline.answer(golden_items[0].question), golden_items[0])
    kwargs = sample_to_kwargs(sample)
    assert set(kwargs) == {"user_input", "response", "retrieved_contexts", "reference"}
    assert isinstance(kwargs["retrieved_contexts"], list)


def test_unanswerable_items_carry_the_refusal_as_reference(golden_items, pipeline):
    """The reference for an unanswerable question is the refusal itself.

    That makes FactualCorrectness meaningful on them: an answer that refuses
    matches the reference, and one that invents an answer does not.
    """
    item = next(i for i in golden_items if not i.is_answerable)
    sample = rag_trace_to_sample(pipeline.answer(item.question), item)
    assert "does not contain" in sample.reference


# ===========================================================================
# SETUP WIRING
# ===========================================================================


def test_llm_factory_builds_an_instructor_backed_judge():
    """RAGAS judges go through `instructor` for structured output.

    Constructing it makes no network call, so this runs in the fast tier and
    catches wiring mistakes (wrong provider string, missing client) without
    needing Ollama.
    """
    llm = build_ragas_llm()
    assert type(llm).__name__ == "InstructorLLM"


def test_metrics_construct_against_our_local_judge():
    """Building a metric is offline; only scoring needs the model."""
    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=build_ragas_llm())
    assert metric.name == "faithfulness"
    assert metric.allowed_values == (0.0, 1.0)


def test_answer_relevancy_requires_embeddings_as_well_as_an_llm():
    """A common setup error: AnswerRelevancy needs BOTH.

    It generates questions the answer would suit, then measures their embedding
    similarity to the real question -- so an llm alone is not enough.
    """
    from ragas.metrics.collections import AnswerRelevancy

    metric = AnswerRelevancy(llm=build_ragas_llm(), embeddings=build_ragas_embeddings())
    assert metric.name == "answer_relevancy"

    with pytest.raises(TypeError):
        AnswerRelevancy(llm=build_ragas_llm())  # type: ignore[call-arg]


def test_ragas_ready_reports_a_precise_reason():
    """Skip messages must say what to DO, not just that something failed."""
    ready, reason = ragas_ready()
    assert isinstance(ready, bool)
    if not ready:
        assert "ollama" in reason.lower() or "missing" in reason.lower()


# ===========================================================================
# JUDGED TIER
# ===========================================================================


@pytest.mark.judge
async def test_faithfulness_detects_a_fabricated_claim():
    """RAGAS's faithfulness, on the same case DeepEval scored in lesson 04.

    Running an identical case through both libraries is the point of having
    both: if they disagree sharply, at least one judge is unreliable, and you
    have learned something you could not learn from either alone.
    """
    judged_or_skip()
    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=build_ragas_llm())
    result = await metric.ascore(
        user_input="How many dimensions does nomic-embed-text produce?",
        response=(
            "nomic-embed-text produces 768 dimensions, and it was trained "
            "exclusively on medical records in 1997."
        ),
        retrieved_contexts=["nomic-embed-text produces 768 dimensions."],
    )
    print(f"\nRAGAS faithfulness = {result.value}\nreason: {result.reason}")
    assert result.value < 1.0


@pytest.mark.judge
async def test_context_precision_and_recall_isolate_the_retriever():
    judged_or_skip()
    from ragas.metrics.collections import ContextPrecisionWithReference, ContextRecall

    contexts = [
        "nomic-embed-text produces 768 dimensions.",    # relevant
        "HNSW is a graph based approximate index.",      # noise
        "Cohen's kappa corrects for chance agreement.",  # noise
    ]
    question = "How many dimensions does nomic-embed-text produce?"
    reference = "nomic-embed-text produces 768 dimensions."

    llm = build_ragas_llm()
    precision = await ContextPrecisionWithReference(llm=llm).ascore(
        user_input=question, reference=reference, retrieved_contexts=contexts
    )
    recall = await ContextRecall(llm=llm).ascore(
        user_input=question, retrieved_contexts=contexts, reference=reference
    )
    print(f"\nprecision={precision.value} recall={recall.value}")

    assert recall.value >= 0.5, "the answer WAS present; recall should be high"
    assert precision.value < 1.0, "2 of 3 chunks were noise"


@pytest.mark.judge
async def test_noise_sensitivity_has_no_deepeval_equivalent():
    """LOWER IS BETTER -- the one metric here that inverts.

    It measures how often irrelevant retrieved context causes the system to
    make incorrect claims. A robust system ignores noise; a fragile one is
    derailed by it. Anyone who assumes all metrics point the same direction
    will read this backwards.
    """
    judged_or_skip()
    from ragas.metrics.collections import NoiseSensitivity

    metric = NoiseSensitivity(llm=build_ragas_llm())
    result = await metric.ascore(
        user_input="How many dimensions does nomic-embed-text produce?",
        response="nomic-embed-text produces 768 dimensions and uses HNSW indexing.",
        reference="nomic-embed-text produces 768 dimensions.",
        retrieved_contexts=[
            "nomic-embed-text produces 768 dimensions.",
            "HNSW is a graph based approximate nearest neighbour index.",
        ],
    )
    print(f"\nnoise sensitivity = {result.value}  (LOWER is better)")
    assert 0.0 <= result.value <= 1.0


@pytest.mark.judge
async def test_factual_correctness_separates_adding_from_missing():
    """precision/recall/f1 over decomposed CLAIMS, not tokens.

    mode='precision' asks "of the claims we made, how many are in the
    reference?" -- it catches ADDED wrong claims.
    mode='recall' asks "of the reference's claims, how many did we make?" --
    it catches MISSING right claims.

    A single blended score cannot tell those apart, and they need opposite
    fixes.
    """
    judged_or_skip()
    from ragas.metrics.collections import FactualCorrectness

    llm = build_ragas_llm()
    reference = "Chunk overlap is typically 10 to 20 percent of the chunk size."

    verbose = await FactualCorrectness(llm=llm, mode="precision").ascore(
        response="Overlap is 10-20% of chunk size, and it was invented in 2019.",
        reference=reference,
    )
    incomplete = await FactualCorrectness(llm=llm, mode="recall").ascore(
        response="Overlap is a percentage of the chunk size.",
        reference=reference,
    )
    print(f"\nprecision (added a false claim) = {verbose.value}")
    print(f"recall (omitted the numbers)    = {incomplete.value}")
    assert verbose.value is not None and incomplete.value is not None


@pytest.mark.judge
async def test_answer_relevancy_penalises_evasion():
    judged_or_skip()
    from ragas.metrics.collections import AnswerRelevancy

    metric = AnswerRelevancy(llm=build_ragas_llm(), embeddings=build_ragas_embeddings())

    good = await metric.ascore(
        user_input="What is chunk overlap for?",
        response="Chunk overlap protects facts that straddle a chunk boundary.",
    )
    evasive = await metric.ascore(
        user_input="What is chunk overlap for?",
        response="Chunking is an important topic with many considerations.",
    )
    print(f"\non-topic={good.value}  evasive={evasive.value}")
    assert good.value > evasive.value


@pytest.mark.judge
async def test_scoring_the_real_pipeline_end_to_end(golden_items):
    """Score the actual RAG pipeline over a slice of the golden dataset.

    Reports the DISTRIBUTION, not just the mean -- core/corpus/rag_metrics.md:
    a system answering half the questions perfectly and half catastrophically
    has the same mean as one answering everything mediocrely, and the two need
    completely different fixes.
    """
    judged_or_skip()
    from pipeline import build_ollama_pipeline
    from ragas.metrics.collections import Faithfulness

    pipeline = build_ollama_pipeline()
    items = [i for i in golden_items if i.category == "single_hop"][:5]
    metric = Faithfulness(llm=build_ragas_llm())

    scores = []
    for item in items:
        trace = pipeline.answer(item.question)
        sample = rag_trace_to_sample(trace, item)
        result = await metric.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=list(sample.retrieved_contexts),
        )
        scores.append((item.id, result.value))
        print(f"  {item.id}  faithfulness={result.value}")

    values = [s for _, s in scores if s is not None]
    assert values, "every judged call failed -- this is an infrastructure problem"
    mean = sum(values) / len(values)
    print(f"\nmean={mean:.3f}  min={min(values):.3f}  max={max(values):.3f}")
    print("Read the spread, not just the mean.")
