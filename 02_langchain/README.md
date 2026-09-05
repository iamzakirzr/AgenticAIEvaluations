# 02 — LangChain: the RAG Pipeline and the Chatbot

**Goal:** rebuild lesson 01's retriever with LangChain 1.4, add a real
generator, and serve it as a chatbot whose UI shows its own retrieved sources.

> **Version warning.** This targets `langchain 1.4.0` / `langchain-core 1.6.1`.
> LangChain 1.x is a substantial rewrite of 0.3.x and **most tutorials online
> are for 0.3 or earlier**. If a snippet from a blog post fails on imports,
> that is why. All pins in `pyproject.toml` are exact.

---

## Run it

```bash
pytest 02_langchain/ -v          # 25 fast tests, no model needed
pytest -m ollama 02_langchain/   # real generation
make chat                        # http://localhost:8000
```

`make chat` works without Ollama — retrieval stays real, generation is scripted,
so you can still explore the sources panel offline.

---

## Files

| File | What it is |
|---|---|
| `pipeline.py` | The RAG pipeline. Explicit stages + the idiomatic LCEL version |
| `prompts.py` | Two prompts: grounded vs naive, as an A/B pair |
| `server.py` | FastAPI + a single-file UI that shows retrieved passages |
| `test_langchain.py` | Tests, almost all of which need no model |

---

## What LangChain actually adds over `mini_rag.py`

Lesson 01 built retrieval in 80 lines. The framework buys you four things:

1. **Integrations** — one `Embeddings` interface across Ollama, OpenAI,
   HuggingFace, Cohere.
2. **Maintained text splitters** — `RecursiveCharacterTextSplitter` is lesson
   01's hand-written splitter, battle-tested.
3. **Vector store abstraction** — swap in-memory → Chroma → pgvector by
   changing a constructor.
4. **LCEL composition** — `|` gives you `.stream()`, `.batch()`, `.ainvoke()`
   and callback instrumentation for free. Lesson 06 hooks LangWatch into
   exactly those callbacks.

It does **not** change the four-step algorithm. Nothing here should be
surprising if lesson 01 made sense.

### On the vector store

This uses `InMemoryVectorStore` from `langchain-core` — no database dependency.
That is a deliberate choice, not laziness: it does **brute-force exact search**,
which `core/corpus/vector_stores.md` explains is correct below ~100k chunks
because it removes an entire class of recall bugs. Adding Chroma buys
persistence and approximate search; it costs those bugs. Swap it when you need
persistence:

```bash
uv pip install --python .venv/bin/python langchain-chroma
```
```python
from langchain_chroma import Chroma
self.store = Chroma.from_documents(self.documents, self.embeddings,
                                   persist_directory=".chroma")
```

---

## The prompt is an evaluation concern

`prompts.py` holds **two** prompts so you can test a claim rather than believe
it. `core/corpus/hallucination.md` asserts that telling a model to answer only
from context is the single highest-value change in a RAG system. `GROUNDED_PROMPT`
has that instruction; `NAIVE_PROMPT` is the control. Lesson 04 runs both across
the unanswerable golden items and measures the difference.

Each rule in the grounded prompt does a specific job, annotated inline:

| Rule | Targets | Golden items it defends against |
|---|---|---|
| Answer only from context | parametric leakage | `un-01`…`un-06` |
| Exact refusal string | undetectable hedging | all `unanswerable` |
| Correct false premises | sycophancy | `ad-01`…`ad-05` |
| Cite passages by number | citation hallucination | checked deterministically |
| Be concise | verbosity bias in judges | all judged metrics |

### Citation checking with no LLM

Because passages are numbered, a citation of `[7]` when only 4 were supplied is
a **fabricated source detectable by integer comparison**. `invalid_citations()`
runs on every answer, costs nothing, and never flakes. Look for transformations
like this constantly — a deterministic check beats a judged one on cost, speed
*and* reliability.

---

## The chatbot's one important design decision

**The UI shows the retrieved passages and their scores next to every answer.**

Most chatbot demos hide them. Showing them turns the app into a debugging tool:
when an answer is wrong you can see instantly whether retrieval fetched the
wrong passage (**retriever problem**) or fetched the right one and the model
ignored it (**generator problem**). That is the same split lessons 04 and 05
measure — here you get it by eye, in a second, for free.

If you demo this repo to an interviewer, open the sources panel.

---

## The thing to actually understand from this lesson

`test_unanswerable_questions_still_retrieve_something` asserts that vector
search **always** returns `top_k` results. It has no concept of "no match".

That is the single most important fact about why RAG hallucinates. For "What is
the capital of France?" the retriever hands the model four irrelevant passages
about chunking with a completely straight face. Nothing in the retriever can
prevent that.

The defence is the prompt. The measurement is refusal rate. Neither is
retrieval's job — which is why the next lessons exist.

---

Next: **[03_langgraph](../03_langgraph/)** — agents, where a single answer
becomes a trajectory you have to evaluate.
