# 04 — DeepEval: the Metric Catalogue, and Whether to Trust It

**Goal:** score the RAG pipeline and the agent with DeepEval, using a **local
Ollama judge** — then measure whether that judge is worth listening to.

---

## Run it

```bash
pytest 04_deepeval/ -v                    # 28 fast tests, no model needed
pytest -m judge 04_deepeval/ -v           # the judged metrics (needs Ollama)
python 04_deepeval/calibrate.py --demo    # the kappa argument, offline
python 04_deepeval/calibrate.py           # calibrate a real judge
```

---

## Files

| File | What it is |
|---|---|
| `ollama_judge.py` | A `DeepEvalBaseLLM` backed by Ollama, with schema enforcement and honest failure handling |
| `adapters.py` | `RagTrace` / `AgentTrace` → `LLMTestCase` |
| `test_deepeval_rag.py` | The RAG metric catalogue |
| `test_deepeval_agent.py` | Agent metrics, mostly deterministic |
| `calibrate.py` | **Cohen's kappa against human labels — read this one** |
| `test_calibration.py` | The calibration maths, tested |

---

## The metric map: free column first

| No LLM — fast, free, deterministic | Needs a judge — slow, costs GPU, noisy |
|---|---|
| `ExactMatchMetric` | `FaithfulnessMetric` |
| `PatternMatchMetric` (refusal detection) | `AnswerRelevancyMetric` |
| `ToolCorrectnessMetric` | `ContextualPrecision` / `Recall` / `Relevancy` |
| recall@k, MRR, nDCG (lesson 01) | `HallucinationMetric` |
| `invalid_citations()` (lesson 02) | `GEval` (any criterion you can write) |
| loop / step-limit detection (lesson 03) | `Bias`, `Toxicity`, `PIILeakage` |

**Exhaust the left column before touching the right one.** Most people do the
reverse, then complain that evaluation is slow and expensive.

---

## Connecting Ollama as the judge

DeepEval's docs cover the four abstract methods. The part they don't make
obvious: **metrics never call `generate` directly.** They call
`generate_with_schema(prompt, schema=SomePydanticModel)`, and the caller then
either uses a returned pydantic object directly or parses your string as JSON.

We return the **validated pydantic object**, because the string-parsing path is
exactly where small local models fall over.

The leverage is in `_chat`: Ollama's native `/api/chat` accepts a `format`
parameter containing a JSON schema and **constrains decoding** so output must
satisfy it. That turns "usually valid JSON" into "always valid JSON" — the
single highest-value thing you can do to make an 8B judge usable.

> Note this judge uses Ollama's **native** API, while lesson 05 uses its
> **OpenAI-compatible `/v1`** endpoint. That is not inconsistency: `format`
> lives on the native API, and RAGAS goes through `instructor`, which speaks
> the OpenAI protocol. Two libraries, two different correct answers.

### Judge failures are never scored as zero

```
"A judge failure must never be silently recorded as a score of zero. That
 converts an infrastructure problem into a fake quality regression."
```

`OllamaJudge` raises `JudgeFailure` and counts every failure on `JudgeStats`,
reported *separately* from quality scores. A 0.0 for "the judge broke" is
indistinguishable from a 0.0 for "the answer was terrible", and mixing them
poisons every baseline you compare against later.

---

## Three real traps found while building this

**1. `ToolCorrectnessMetric` demands an OpenAI key it never uses.**
It's a purely deterministic trajectory comparison, but its `__init__`
constructs a default OpenAI model and raises without `OPENAI_API_KEY`. Passing
any `DeepEvalBaseLLM` fixes it. We pass `ExplodingJudge`, which *raises if ever
called* — turning "I believe this is deterministic" into a test that fails the
day it stops being true.

**2. `PatternMatchMetric` uses `fullmatch`, not `search`.**
Despite the name. A bare phrase scores 0.0 on an answer that plainly contains
it, with a reason string that gives no hint why. Hence
`r"(?is).*does not contain this information.*"`. Pinned by
`test_pattern_match_uses_fullmatch_not_search`.

**3. `context` and `retrieval_context` are different fields.**
`retrieval_context` = what your retriever returned. `context` = the *ideal*
ground truth, used by `HallucinationMetric`. Filling `context` with retrieved
chunks makes that metric compare the context against itself, so it can never
fail — a silent, total defeat of the metric.

---

## Faithfulness is not correctness

The most misunderstood behaviour in RAG evaluation, and it has its own test:

```python
retrieval_context = ["nomic-embed-text produces 4096 dimensions."]  # WRONG
actual_output     = "nomic-embed-text produces 4096 dimensions."
# faithfulness -> HIGH
```

Faithfulness asks *"does this follow from the context?"*, never *"is this
true?"*. An answer that faithfully repeats a wrong document scores 1.0. Golden
item `ad-03` exists to catch people who conflate them, and
`test_faithfulness_is_not_correctness` asserts the behaviour rather than
describing it.

**A high faithfulness score with low answer correctness means retrieval fetched
the wrong document and the generator summarised it perfectly.**

---

## The part that actually differentiates you: calibration

Every judged metric produces a confident-looking decimal. None of them tell you
whether the judge is any good.

```bash
python 04_deepeval/calibrate.py --demo
```

```
lazy judge (always says pass)
  raw agreement : 90%   <- looks great either way
  Cohen's kappa : 0.000   (poor -- DO NOT gate anything on this judge)

useful judge (caught the failure)
  raw agreement : 90%   <- looks great either way
  Cohen's kappa : 0.615   (substantial -- usable as a gate)
```

Both judges are wrong exactly once. Both score 90%. One missed the only real
failure; the other caught it and raised one false alarm. **Only kappa
distinguishes them** — which is why "my judge agrees with me 90% of the time"
is not evidence of anything.

`calibrate.py` ships 8 hand-labelled faithfulness examples including the cases
that actually discriminate: an extrinsic hallucination that is *true in the
world*, a direct contradiction, and a claim derived by arithmetic from a
supported rule.

| kappa | what to do |
|---|---|
| > 0.6 | you may gate CI on this metric |
| 0.4–0.6 | track the trend, don't fail builds |
| < 0.4 | the metric is noise — bigger judge, simpler criterion, or fall back to deterministic metrics |

**Reporting "my local judge scored kappa 0.31, so I did not gate on it" is a
stronger result than a green dashboard you never checked.** Almost nobody does
this step. Doing it is the difference between running an eval and understanding
one.

---

Next: **[05_ragas](../05_ragas/)** — the same pipeline, a different library, and
where the two disagree.
