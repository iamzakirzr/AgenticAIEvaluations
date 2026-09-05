# Glossary

Every term in this repo, in plain English, with the QA equivalent where one
exists. Skim it once; come back when something confuses you.

---

## The basics

**Token** — Roughly 4 characters, or ¾ of an English word. Models see tokens,
not characters. The unit of cost, of context limits, and of latency.

**Context window** — How many tokens the model can consider at once (input +
output). Exceed it and the call errors or silently truncates.

**Temperature** — Randomness in output selection. `0` = always pick the most
likely next token (as close to deterministic as you get). *Always 0 in tests.*
Non-zero temperature on a judge makes your metric flaky for no benefit.

**Inference** — Running the model. Distinct from *training*, which is finished
before you ever touch it.

**Prompt** — The text you send. Usually a *system* part (instructions) and a
*user* part (the question).

**Parametric knowledge** — What the model learnt during training, baked into its
weights. The opposite of what you supply in the prompt. "Parametric leakage" is
when it answers from training data instead of your documents.

**Hallucination** — Output presented as fact that isn't supported by the source
of truth. Three flavours:
- *Intrinsic* — contradicts the provided context.
- *Extrinsic* — adds information absent from the context (may even be true).
- *Citation* — cites a source that doesn't contain the claim. Worst kind,
  because the citation makes it look verified.

---

## Retrieval

**Embedding** — Text converted to a list of numbers (a point in space) such that
similar meanings land nearby. `nomic-embed-text` produces 768 numbers.

**Dimension** — How many numbers in that list. Must match between the index and
the query path, or results are noise. *This mismatch raises no error* — see
`01_embeddings/production_index.py`.

**Cosine similarity** — The angle between two vectors. Range −1 to 1; 1.0 means
identical direction. The standard way to measure "how similar is this text".
Preferred over Euclidean distance because it ignores length, which mostly
reflects document size rather than meaning.

**Chunk** — A slice of a document. Documents are split because one vector can't
faithfully represent fifty pages of mixed topics.

**Chunk overlap** — Text shared between neighbouring chunks, so a fact straddling
a boundary survives whole in at least one of them. Typically 10–20% of chunk size.

**Vector store / vector database** — Stores embeddings and answers
nearest-neighbour queries. Chroma, FAISS, Qdrant, pgvector.

**Exact vs approximate search** — Exact compares against every vector (always
correct, linear cost). Approximate (ANN) skips most of them for speed and
returns *most* of the right answers. Below ~100k chunks, exact is the right
default because it removes a class of recall bugs.

**HNSW** — The most common approximate index. Three parameters: `M`
(neighbours per node), `ef_construction` (build effort), `ef_search` (query
effort — the only one changeable after the index is built).

**top_k** — How many chunks to retrieve. Typically 3–10. Higher = better recall,
worse precision.

**BM25** — Classic keyword search scoring. Good at exact rare terms (error
codes, SKUs) where embeddings blur distinctions.

**Hybrid search** — Running BM25 *and* vector search, then merging. More robust
than either, because they fail differently.

**RRF (Reciprocal Rank Fusion)** — The standard way to merge two ranked lists:
score each document by `1/(k + rank)` summed across lists, conventionally
`k=60`. Uses only ranks, so it safely combines scores on incomparable scales.

**MMR (Maximal Marginal Relevance)** — Picks results that are relevant to the
query *but different from each other*, to avoid five near-identical chunks.
A `lambda` of 1.0 is pure relevance, 0.0 pure diversity.

**Reranker / cross-encoder** — A second-stage model that reads the query and a
candidate *together* and scores relevance. Much more accurate than embedding
similarity, much slower — so it runs on a shortlist of 25–50, not the corpus.

**HyDE** — Hypothetical Document Embeddings. Ask the model to write a fake
answer, then search using *that* instead of the question, because the fake
answer looks more like the documents than the question does.

**Query rewriting** — Turning "what about the second one?" into a standalone
question using conversation history. Without it, multi-turn RAG retrieves
nonsense on every follow-up.

**Lost in the middle** — Models use information at the start and end of a long
context better than the middle. Why raising `top_k` can make answers *worse*.

---

## Evaluation

**Golden dataset** — Your labelled test data. Question + expected answer +
which documents should be retrieved. *This is your fixture file.*

**Ground truth / reference** — The correct answer. Absent in production.

**Judge / LLM-as-a-judge** — Using a model to score another model's output. The
only practical way to measure "helpfulness" at scale, and the biggest source of
false confidence in the field.

**Metric** — A scoring function. Some need a judge; many don't.

### Metrics that need NO model

**recall@k** — Of the documents that *should* have been found, what fraction
appeared in the top k? "Did we even fetch the answer?"

**precision@k** — Of the k results returned, what fraction were relevant? "How
much of what we fetched was worth fetching?"

**hit rate** — Did *at least one* correct document appear? recall@k collapsed to
yes/no.

**MRR (Mean Reciprocal Rank)** — Average of `1/rank of first correct result`.
Position 1 → 1.0, position 2 → 0.5, position 4 → 0.25. Rewards ranking the right
thing *first*, which matters because of "lost in the middle".

**nDCG** — Like MRR but accounts for multiple relevant results, discounting each
by log of its position. The standard information-retrieval metric.

**Exact match / pattern match** — String equality or regex. Underrated: if your
prompt mandates an exact refusal string, a regex is a *complete* refusal
detector, for free.

### Metrics that need a judge

**Faithfulness (groundedness)** — Are the answer's claims supported by the
retrieved context? Breaks the answer into claims and checks each. **It does not
check correctness** — an answer that faithfully repeats a *wrong* document
scores 1.0. The most misunderstood metric in the field.

**Answer relevancy** — Does the answer actually address the question? Catches
evasion and waffle. Doesn't check truth.

**Context precision** — Were the retrieved chunks relevant, weighted so that
relevant chunks ranked higher score better?

**Context recall** — Was everything needed to produce the reference answer
actually retrieved? Requires a reference answer.

**Noise sensitivity** — How often irrelevant context causes wrong claims.
**Lower is better** — the one metric that inverts.

**Answer correctness / factual correctness** — Compares against the reference.
RAGAS splits it into precision (did we *add* wrong claims?) and recall (did we
*miss* right ones?) — which need opposite fixes.

**G-Eval** — DeepEval's "write your own criterion in a sentence" metric. The
escape hatch for domain-specific failures no built-in metric covers.

### Judge quality

**Calibration** — Measuring whether your judge agrees with *human* labels.
Almost nobody does it. Doing it is a differentiator.

**Cohen's kappa** — Agreement between two raters, corrected for chance. −1 to 1.
`>0.8` almost perfect, `>0.6` substantial, `<0.4` don't gate anything on it.
**Raw percentage agreement lies on imbalanced data**: a judge that always says
"pass" scores 90% agreement and kappa 0.0.

**Position bias** — Preferring whichever answer is shown first. Fix: run each
comparison twice with the order swapped.

**Verbosity bias** — Rating longer answers higher regardless of quality.

**Self-preference bias** — A model rates *its own* output more favourably.
Why the generator and judge should be configured separately.

**Leniency bias** — Judges cluster scores at the top, turning a 5-point scale
into a 2-point one.

---

## Agents

**Tool** — A function the model can call. What the model sees is a JSON schema
built from the function's name, type hints and **docstring** — so the docstring
is a *prompt*, not documentation.

**Trajectory** — The sequence of tool calls. Agent evaluation scores this, not
just the final answer: reaching the right answer after nine wasted calls is not
the same as going straight there.

**Tool correctness** — Did it call the right tools? Deterministic, no judge.

**Task completion** — Did it achieve the user's goal, judged from the whole
trace? Needs a judge, but no labels — so it works on production traffic.

**Step efficiency** — Penalises unnecessary steps.

**Loop detection** — Catching an agent repeating the same call. Invisible to
outcome-only evaluation, because a looping agent times out rather than
answering wrongly.

**Checkpointer** — Persists graph state, keyed by `thread_id`. What gives an
agent memory across turns. `InMemorySaver` is development-only.

**thread_id** — The conversation identifier. Same id = same conversation.

**interrupt() / human-in-the-loop** — Suspends the graph, persists state, and
returns control so a human can approve. Resuming continues from exactly that
point, possibly in another process.

**Reducer** — How LangGraph merges a node's output into state. `add_messages`
appends *and de-duplicates by message id* — which bites you if you replay an
identical message object.

---

## Operations

**Baseline** — Recorded reference scores that new runs are compared against.

**Noise band** — How much a metric moves when *nothing changed*. A drop inside
it is not evidence of anything. `max(min_delta, sensitivity × stdev)`.

**Regression gate** — CI check that fails when a metric drops outside its noise
band.

**Fingerprint** — The configuration (models, chunk size, top_k, temperature)
that produced a score. Two runs are only comparable if it matches.

**Offline vs online evaluation** — Offline: fixed dataset with labels, before
you ship. Online: real traffic, no labels, after you ship.

**Head vs tail sampling** — Head: decide before you know how the request went
(cheap, uniform). Tail: keep it *because* something interesting happened
(errors, refusals, slow). You need both.

**Trace / span** — A record of one request and its stages. Spans nest.

**Circuit breaker** — Fails fast after N consecutive failures instead of
retrying into a dead service. Three states: closed, open, half-open.

**Jitter** — Randomising retry delays so N workers that fail together don't
retry together. Prevents a "thundering herd".

**Budget** — A hard ceiling on tokens, cost, calls or time for one run. The only
thing that reliably stops a runaway loop.

---

## The five tools

**LangChain** — Orchestration. Models, embeddings, vector stores, prompts,
splitters behind common interfaces. **v1.x is a rewrite — most tutorials online
target 0.3 and won't run.**

**LCEL** — LangChain Expression Language. Composing with `|`, which gives you
streaming, batching and async for free.

**LangGraph** — State machines for agents. Nodes, edges, conditional edges,
reducers, checkpointers.

**DeepEval** — Eval library shaped like pytest. `LLMTestCase`, metrics,
`assert_test`. Also does synthetic dataset generation.

**RAGAS** — Eval library shaped around datasets. `SingleTurnSample`,
`EvaluationDataset`, `evaluate()`. **The current metric API is
`ragas.metrics.collections`; the path every tutorial uses is deprecated.**

**LangWatch** — Hosted tracing and online evaluation, built on OpenTelemetry —
which is why its instrumentation can be unit-tested offline.

**Ollama** — Runs open-source models locally. Serves *two* APIs: its native one
at `/api/*` and an OpenAI-compatible one at `/v1`. Different libraries need
different ones.
