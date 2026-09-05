# 06 — LangWatch: Tracing, and Evaluating What You Cannot Label

**Goal:** instrument the pipeline and the agent so you can evaluate **production
traffic**, not just a golden dataset.

> **This is the one lesson that involves a hosted service.** LangWatch's
> dashboards, annotation queues and online evaluators live on their servers and
> need an API key. **Everything in this lesson is still testable offline** —
> read on.

---

## Run it

```bash
pytest 06_langwatch/ -v      # 15 fast tests, no account, no network, no model
pytest -m saas 06_langwatch/ # the single test that hits the real platform
```

---

## Observability is not evaluation

Lessons 04 and 05 did **offline** evaluation: fixed dataset, known reference
answers, run before you ship.

This lesson is **online** evaluation: real traffic, where you gain the questions
people actually ask and lose the one thing every offline metric depends on.

> **In production there is no reference answer.** Nobody labelled the user's
> question.

So the metrics split cleanly:

| Works online (reference-free) | Needs a label (offline only) |
|---|---|
| Faithfulness | Context Recall |
| Answer Relevancy | Context Precision (with reference) |
| Context Relevancy | Factual Correctness |
| Toxicity / Bias / PII | Answer Correctness |
| Refusal rate (regex) | recall@k, MRR, nDCG |
| Latency, cost, tokens | Noise Sensitivity |

A production dashboard promising "answer correctness" on unlabelled traffic is
measuring something else.

---

## Why this is testable without an account

`langwatch` 1.3.1 is built on **OpenTelemetry**. So we hand it our own
`TracerProvider` with an `InMemorySpanExporter`, pass `disable_sending=True`,
and assert on the exact spans our code emits.

**Instrumentation is code, and code should be tested.** Most teams never test
their tracing and find out during an incident that the field they needed was
never recorded. Every span-structure test here exists to prevent that.

---

## The span structure is the design decision

```
trace: rag_answer
  ├─ span: retrieve            (type="rag")         input=question, output=chunks
  ├─ span: generate            (type="llm")         input=prompt,   output=answer
  └─ span: online_evaluations  (type="evaluation")  the reference-free scores
```

Separate spans for retrieval and generation, because that is the split that
matters when something breaks — the same retriever-vs-generator distinction the
lesson 02 UI shows and the lesson 04/05 context metrics measure.

**A single opaque `rag` span tells you a request was slow. Two tell you which
half was slow.**

For agents, one span **per tool call**. Agents need *more* instrumentation, not
less, because the trajectory varies — it's how you answer "why did this take 40
seconds?" (almost always: it looped).

---

## Three real traps, all found by writing the tests

**1. OpenTelemetry silently drops dict attributes.**

Span attributes may only be primitives or sequences of primitives. Pass a dict
and there is **no error** — the span exports happily, just without your fields:

```
WARNING opentelemetry.attributes: Invalid type dict for attribute 'metadata'
value. Expected one of ['bool','str','bytes','int','float'] or a sequence...
```

You find out during an incident. `otel_metadata()` JSON-encodes instead, and
`test_dict_attributes_are_silently_dropped_by_opentelemetry` pins it.

**2. `trace.add_evaluation()` is deprecated and now raises.**

Evaluations attach to a **span**, not a trace. The trace-level method forwards
with `span=None` and dies with `ValueError: No span or trace found`. We open a
dedicated `type="evaluation"` span, which also groups cleanly in the dashboard.

**3. Calling `setup_offline_tracing()` twice breaks unrelated tests.**

Tracing state is process-global. A second `TracerProvider` means later tests
write to an exporter their fixture isn't holding — presenting as "no spans
captured" in tests that never touched setup. Set it up once.

---

## Deterministic online evaluations — free, on 100% of traffic

Three checks need no judge and no label, so they run on every request:

| Check | What it catches | Cost |
|---|---|---|
| `citation_validity` | fabricated sources (from lesson 02) | integer comparison |
| `refused` | refusal rate — the single best production health signal | regex |
| `retrieval_confidence` | corpus no longer covering incoming traffic | already computed |

Refusal is scored but **not marked as a failure** — refusing is correct on an
unanswerable question. In production, a *spike* means retrieval broke; a
*collapse* means the model started making things up.

---

## Sampling must be deterministic

Judged metrics cost a full LLM call each, so you sample. But use a **hash of the
question**, never `random()`:

```python
bucket = int(sha256(question)[:8], 16) / 0xFFFFFFFF
return bucket < rate
```

With `random()`, re-running a request samples differently, so you cannot
reproduce a scored trace while investigating it — and two services handling the
same request disagree about whether it was sampled. Hashing makes the decision
stable forever. Pinned by `test_sampling_is_deterministic_for_the_same_question`.

---

## Using the real platform

```bash
export LANGWATCH_API_KEY=sk-lw-...   # sign up at langwatch.ai
pytest -m saas 06_langwatch/
```

The genuinely valuable thing the hosted product adds is not the dashboard — it
is the loop back to lesson 04: **real production questions become new golden
dataset items.** A golden set written entirely by you tests what you imagined
users would ask. Production traffic tests what they actually ask, and the gap
between those two is where systems fail.

---

Back to the **[main README](../README.md)** for the full learning path.
