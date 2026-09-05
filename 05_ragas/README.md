# 05 — RAGAS: a Second Opinion on the Same Pipeline

**Goal:** score the lesson-02 pipeline with RAGAS 0.4, wired to the same local
Ollama model — and see where the two libraries disagree.

---

## Run it

```bash
pytest 05_ragas/ -v            # 13 fast tests, no model needed
pytest -m judge 05_ragas/ -v   # the judged metrics (needs Ollama)
```

---

## Two things that will break your RAGAS code

### 1. `import ragas` fails outright on a modern LangChain stack

```
ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
```

ragas 0.4.3 has a **module-level** import of a class that
`langchain-community 0.4.x` removed, and it declares `langchain-community` as an
**unpinned** dependency. A clean install is broken out of the box.

`core/compat.py` installs a stub so the import succeeds, and must run *before*
the first `import ragas` in the process. Read that file — it explains the three
possible fixes and why the shim wins (downgrading `langchain-community` drags
`langchain-core` below 1.0 and breaks lessons 02 and 03).

This is worth internalising as a category: **an eval library pins loosely
against a fast-moving orchestration library, and a transitive upgrade breaks
you.** You will hit it in a real job.

### 2. Every RAGAS tutorial online uses the deprecated API

The classic form, which is everywhere:

```python
from ragas.metrics import faithfulness, context_precision   # instances
evaluate(dataset, metrics=[faithfulness])
```

still half-works in 0.4.3, but warns:

> `DeprecationWarning: Importing X from 'ragas.metrics' is deprecated and will
> be removed in v1.0. Please use 'ragas.metrics.collections' instead.`

The current API is **`ragas.metrics.collections`** — 39 metric *classes* you
instantiate with an llm and call with explicit typed arguments:

```python
from ragas.metrics.collections import Faithfulness
metric = Faithfulness(llm=judge)
result = await metric.ascore(user_input=..., response=..., retrieved_contexts=[...])
```

Better API, but it means nearly every example you find is out of date.
`test_the_classic_metric_import_path_is_deprecated` pins the warning so you find
out from a test rather than from a v1.0 upgrade.

---

## Connecting Ollama — differently than lesson 04 did

| | Lesson 04 (DeepEval) | Lesson 05 (RAGAS) |
|---|---|---|
| Endpoint | Ollama **native** `/api/chat` | Ollama **OpenAI-compatible** `/v1` |
| Why | needs `format` for schema-constrained decoding | goes through `instructor`, which speaks the OpenAI protocol |

Same model, same server, two endpoints. **That is not inconsistency — it is the
concrete argument against a "unified" wrapper.** Forcing both libraries through
one abstraction would have broken one of them.

The upside of the RAGAS path: because it only needs an OpenAI-compatible
endpoint, the same code runs against vLLM, LM Studio, llama.cpp's server, or any
hosted API. Change the base URL, nothing else.

```python
llm_factory(model="llama3.1:8b", provider="openai",
            client=OpenAI(base_url="http://localhost:11434/v1", api_key="ignored"))
```

`provider="openai"` does **not** mean OpenAI the company — it means "speaks the
OpenAI protocol". The injected client decides where requests go.

---

## The vocabulary table, worth memorising

| Concept | DeepEval | RAGAS |
|---|---|---|
| the question | `input` | `user_input` |
| what the system said | `actual_output` | `response` |
| the reference answer | `expected_output` | `reference` |
| what the retriever found | `retrieval_context` | `retrieved_contexts` |
| ideal / ground-truth context | `context` | `reference_contexts` |

Five concepts, ten names, zero overlap. **This is why lessons 04 and 05 have
separate adapters instead of one shared abstraction** — a unified interface
would have to pick one vocabulary and would mislead anyone reading it with the
other library's docs open.

Note the last row especially: in *both* libraries the same mistake exists —
filling the ideal-context field with what your retriever actually returned turns
those metrics into tautologies.

---

## What RAGAS gives you that DeepEval does not

**`NoiseSensitivity`** — no DeepEval equivalent. Measures how often irrelevant
retrieved context causes incorrect claims. **Lower is better**, which trips up
anyone assuming all metrics point the same way.

**`FactualCorrectness`** with `mode="precision" | "recall" | "f1"` — decomposes
both answer and reference into claims:
- `precision` catches **added** wrong claims
- `recall` catches **missing** right claims

A single blended score cannot tell those apart, and they need opposite fixes.

**`ResponseGroundedness`**, **`TopicAdherence`**, **`ToolCallAccuracy`** for
agent and scoped-bot evaluation.

---

## Why run both libraries at all

Running an identical case through both is the point. If they agree, your
confidence is justified. **If they disagree sharply, at least one judge is
unreliable — and you have learned something neither library could tell you
alone.**

That is a cheap, powerful sanity check, and it costs one extra adapter.

---

## A bug this lesson caught in our own code

`core.compat.is_shim_needed()` originally just tried to import the module. But
once the shim is installed, *our own stub* is in `sys.modules`, so the import
succeeds and the function reports "shim not needed" **precisely because the shim
is working**.

The symptom: a test that passed or failed depending on whether anything imported
`ragas` earlier in the same pytest session. Order-dependent tests look flaky
rather than wrong, which is the worst failure mode to debug.

Fixed with a marker attribute on the stub, pinned by
`test_is_shim_needed_ignores_our_own_stub`.

---

Next: **[06_langwatch](../06_langwatch/)** — tracing, so you can evaluate what
happens in production rather than only what happens in tests.
