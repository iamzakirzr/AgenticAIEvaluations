# RAG Evaluation Metrics

## The two failure modes

Every RAG failure is either a retrieval failure or a generation failure. The
purpose of having several metrics rather than one overall score is to tell them
apart. If retrieval metrics are healthy and faithfulness is low, the generator
is hallucinating. If retrieval metrics are poor, no amount of prompt engineering
on the generator will help.

## Retrieval metrics that need no LLM

These are computed from ranks and known-correct source identifiers, so they are
fast, free and perfectly deterministic. They belong in continuous integration.

**Recall at K** is the fraction of the relevant documents that appear anywhere
in the top K results. It answers: did we even fetch the answer?

**Precision at K** is the fraction of the top K results that are relevant.

**Mean Reciprocal Rank**, abbreviated MRR, is the average across queries of one
divided by the rank of the first relevant result. If the first relevant document
is in position 1 the reciprocal rank is 1.0, in position 2 it is 0.5, in position
4 it is 0.25. MRR rewards putting the right answer at the top, which matters
because generators attend most strongly to the beginning of the context.

**Hit rate** is the fraction of queries where at least one relevant document
appears in the top K. It is Recall at K collapsed to a yes-or-no per query.

**Normalised Discounted Cumulative Gain**, or nDCG, extends this to graded
relevance, where documents can be partially relevant rather than simply relevant
or not, and discounts gains logarithmically by rank.

## Metrics that require an LLM judge

**Faithfulness**, also called groundedness, measures whether the answer is
supported by the retrieved context. It is computed by breaking the answer into
individual claims and checking each one against the context, then reporting the
fraction of claims that are supported. It detects hallucination. Crucially it
does not check whether the answer is correct, only whether it follows from the
context provided. An answer that faithfully repeats a wrong document scores 1.0.

**Answer Relevancy**, called Response Relevancy in newer RAGAS versions,
measures whether the answer actually addresses the question. The usual
implementation asks a model to generate several questions that the given answer
would be a good response to, embeds them, and measures their average similarity
to the real question. It penalises evasive and incomplete answers. It does not
check factual correctness.

**Context Precision** measures the proportion of retrieved chunks that were
relevant, weighted so that relevant chunks appearing at higher ranks score
better. Low context precision means the context window is being filled with
noise.

**Context Recall** measures whether all the information needed to produce the
reference answer was present in the retrieved context. It is computed by
attributing each claim in the reference answer to the retrieved context. It
requires a reference answer, which is why a golden dataset is mandatory for it.

**Context Entity Recall** is a stricter variant that checks which named entities
from the reference answer appear in the retrieved context. It is particularly
useful for fact-heavy domains.

**Noise Sensitivity** measures how often the system produces incorrect claims
when irrelevant documents are present in the context. A robust system ignores
noise; a fragile one is derailed by it. Lower is better, unlike most metrics.

**Answer Correctness** compares the generated answer against the reference
answer directly, combining factual overlap with semantic similarity. It is the
closest thing to an overall accuracy score, and because it blends two different
things, a change in it is harder to diagnose than a change in the component
metrics.

## Metric interpretation traps

A high faithfulness score with a low answer correctness score usually means
retrieval returned the wrong document and the generator faithfully summarised
it. Looking at faithfulness alone would have made this look like a success.

Averaging scores across a dataset hides bimodal behaviour. A system that answers
half the questions perfectly and half catastrophically has the same mean as one
that answers every question mediocrely, and the two require completely different
fixes. Always inspect the distribution, not just the mean.
