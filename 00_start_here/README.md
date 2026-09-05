# Start Here — AI Evaluation for SDETs and QA Engineers

**You already know 70% of this. You just don't know it's the same thing yet.**

This repo is a curriculum for learning to test AI systems. If you come from QA
or SDET work, your instincts transfer almost completely — the gap is
vocabulary, not rigour. This page closes that gap, then hands you a 4-week plan.

```bash
make setup     # one-time, needs `uv`
make hello     # your first evaluation: 60 lines, 2 seconds, no AI model needed
```

---

## 1. The translation table

Read this once. Everything else in the repo will make sense afterwards.

| What you already do | What it's called in AI eval | Notes |
|---|---|---|
| Test case | **Golden item** | question + expected answer + expected sources |
| Fixture file / test data | **Golden dataset** | usually JSONL, version-controlled |
| Assertion | **Metric** | returns a *score* 0–1, not a boolean |
| Expected value | **Reference / ground truth** | often absent in production |
| System under test | **The pipeline / agent** | same thing |
| Test oracle | **The judge** (or the reference answer) | the judge may be *another AI* |
| Flaky test | **Non-deterministic judge** | same disease, same cure: find the source of variance |
| Boundary value analysis | **Adversarial / edge-case questions** | false premises, unanswerable inputs |
| Negative testing | **Unanswerable questions** | correct behaviour is *refusal* |
| Code coverage | **Dataset category coverage** | are you testing all the behaviours? |
| Regression suite | **Baseline + regression gate** | with a *noise band*, see below |
| Smoke test | **Fast tier** | no model needed, runs in seconds |
| Load test | **Concurrency + budget tests** | tokens cost money; requests queue |
| Test pyramid | **Deterministic → judged tiers** | cheap tests first, expensive last |
| Mocking a slow service | **Scripted model / fake embedder** | this repo does it everywhere |
| Observability in prod | **Tracing + online eval** | but *no reference answer exists* |

### The three things that are genuinely new

1. **Assertions return a score, not a boolean.** "Is this answer good?" has no
   exact expected value, so you get 0.87 instead of `True`. Everything awkward
   about AI testing follows from this.

2. **Your oracle can be wrong.** Sometimes the thing grading the output is
   *another AI model*. An untested oracle is a number that looks rigorous and
   may measure nothing. Lesson 04 shows how to measure your oracle.

3. **In production there is no expected value.** Nobody labelled the user's
   question. Half your metrics stop working. Lesson 06 covers what survives.

### The one habit that makes you good at this immediately

You already refuse to write `assert response.time < 1000` on a timing-dependent
value, because it flaps. Apply exactly that instinct here:

> **A metric that cannot move cannot detect a regression.**

Run `make hello` and you'll see recall sitting at a perfect 1.000. That is not
good news — it's an assertion that passes no matter what the code does. This
repo gates on MRR instead, and says so out loud.

---

## 2. How AI actually works — the 20 minutes you need

You do not need linear algebra. You need five ideas.

### 2.1 A model is a function with frozen numbers inside

`f(text) -> text`. Training set billions of internal numbers ("weights") so that
the function produces useful output. Training is finished before you ever touch
it. **Inference** — running the function — is what you test.

Consequences that matter to a tester:

- The weights don't change when you use it. Same model, same input, same
  settings → *nearly* the same output.
- "Nearly", because output is **sampled** from a probability distribution.
  `temperature=0` makes it pick the most likely token every time, which is as
  close to deterministic as you get. **Set temperature to 0 in every test.**
- The model knows nothing about your company. That's what RAG is for.

### 2.2 Tokens: the unit of everything

Models don't see characters or words. Text is chopped into **tokens** —
roughly 4 characters, or ¾ of a word, in English.

Tokens are the unit of **cost** (you're billed per token), of **limits** (the
"context window" is a token budget), and of **latency**. When you see
`max_tokens`, `num_predict`, or a bill, it's tokens.

### 2.3 Embeddings: text as coordinates

An **embedding** turns text into a list of numbers — a point in space — such
that texts with similar meaning land near each other.

That's the whole trick behind search. To find relevant documents you convert
everything to points, then find the points nearest your question. "Nearest" is
measured by **cosine similarity** (the angle between two vectors; 1.0 = same
direction, −1 = opposite).

`01_embeddings/walkthrough.py` builds this from scratch and prints every
intermediate step. It is the single best 15 minutes in this repo.

### 2.4 RAG: the pattern you'll be testing

**R**etrieval-**A**ugmented **G**eneration. Four steps, and that's genuinely all:

```
1. Split your documents into chunks.
2. Convert each chunk to an embedding; store them.
3. Convert the user's question to an embedding; find the nearest chunks.
4. Paste those chunks into the prompt and ask the model to answer from them.
```

Why it exists: the model doesn't know your data, and retraining it is
impractical. So you *show* it the relevant text at question time.

**The failure mode you must internalise:** step 3 *always* returns results. Ask
"what is the capital of France?" of a knowledge base about chunking, and it
returns the four least-irrelevant chunks with a straight face. Nothing errors.
If the prompt doesn't explicitly permit refusal, the model writes something
plausible from garbage.

That's not a model being stupid. That's a pipeline with no "no match" concept.
`make hello` demonstrates it in step 7.

### 2.5 Agents: a while-loop whose exit condition is a model

An **agent** is a model that can call functions ("tools"). The loop:

```
ask the model → did it request a tool? → run it → feed the result back → repeat
                       ↓ no
                     done
```

There is no planner and no controller. That's why agents fail in ways pipelines
can't: infinite loops, wrong tool, right tool with wrong arguments, stopping
early. `03_langgraph` builds this in ~40 lines and tests every one of those
failures without a model.

---

## 3. The five tools, in one line each

| Tool | What it is | You'll use it to |
|---|---|---|
| **LangChain** | Orchestration — glue for models, embeddings, vector stores, prompts | Build the RAG pipeline |
| **LangGraph** | State machines for agents (same authors) | Build the agent loop, memory, approval gates |
| **DeepEval** | Eval library, pytest-shaped | Score outputs; it feels like a test framework |
| **RAGAS** | Eval library, dataset-shaped | Score the same outputs a second way |
| **LangWatch** | Hosted tracing + online eval | See what production is actually doing |

**Why two eval libraries?** Because when they disagree sharply, at least one
judge is unreliable — and that's something neither can tell you alone. It's the
same reason you'd cross-check a flaky assertion against a different method.

---

## 4. The 4-week plan

Each block is ~2–4 hours. Do the reading, run the code, then break something
on purpose and watch the number move — that last part is where the learning is.

### Week 1 — Ground yourself

| Day | Do this | You'll be able to say |
|---|---|---|
| 1 | `make hello`, read this page | "Evaluation is testing with scored assertions" |
| 2 | `make lesson-embeddings` (all 7 steps) | "An embedding is text as coordinates; cosine measures the angle" |
| 3 | `make experiment-chunking`, then read the caveat | "Chunk size trades precision against recall, and recall is saturated here" |
| 4 | `01_embeddings/README.md` + run its tests with `-v` | "Lexical search fails on synonyms; that's why neural embeddings exist" |
| 5 | Change `CHUNK_SIZE=200` in `.env`, re-run. Explain the movement. | "I can predict which metric a config change will move" |

### Week 2 — Build and serve

| Day | Do this | You'll be able to say |
|---|---|---|
| 1 | `02_langchain/README.md`, read `pipeline.py` | "RAG is four steps; the framework adds integrations, not magic" |
| 2 | `make chat` — ask a normal question, an unanswerable one, a false premise | "I can tell a retriever failure from a generator failure by eye" |
| 3 | Read `prompts.py`. Delete the refusal instruction, re-run the tests. | "The grounding instruction is the highest-value line in the prompt" |
| 4 | `03_langgraph/README.md`, read `agent.py` | "An agent is a while-loop; trajectories need their own metrics" |
| 5 | Run `pytest 03_langgraph/ -v` and read each test name | "I can name five agent failure modes and test all of them offline" |

### Week 3 — Measure, and measure the measurer

| Day | Do this | You'll be able to say |
|---|---|---|
| 1 | `04_deepeval/README.md`; the metric map | "I know which metrics need a judge and which are free" |
| 2 | `make calibrate` (the `--demo` runs offline) | "90% agreement can mean kappa 0.0 — I check kappa" |
| 3 | Install Ollama, `make models`, `make test-judge` | "I've run judged metrics against a local model and seen them fail" |
| 4 | `05_ragas/README.md`; the vocabulary table | "Same concepts, ten different field names" |
| 5 | Run the same case through both libraries; compare | "When two judges disagree, I don't trust either yet" |

### Week 4 — Production and pipelines

| Day | Do this | You'll be able to say |
|---|---|---|
| 1 | `PRODUCTION.md` end to end | "I know the failures that take eval systems down" |
| 2 | `06_langwatch/README.md`; run its tests | "In production there's no reference answer, so these metrics survive and those don't" |
| 3 | `make gate-fast`, then `make baseline`, then `make gate` | "I gate on a noise band, not a fixed threshold" |
| 4 | Read the four workflows in `.github/workflows/` | "Judged metrics never gate a PR, and here's why" |
| 5 | `INTERVIEW.md` — answer out loud, then check | Ready to interview |

---

## 5. What to do when you're stuck

- **Every test name is a sentence.** `pytest -v` is a table of contents.
- **Every module has a header explaining *why*,** not just what. Read those first.
- **Break things deliberately.** Change a number, re-run, explain the movement.
  If nothing moves, that metric was never going to catch a regression.
- **`make test` is your safety net.** 337 tests, ~15 seconds, no model needed.

---

## 6. The honest bit

Two things about this repo you should know before you quote it in an interview:

1. **The judged tier has not been run here.** No GPU in the environment it was
   built in. The code is verified against the installed libraries by
   introspection, but "imports correctly" is not "produces trustworthy
   numbers". Run `make test-judge` yourself.

2. **The corpus is 8 documents and the questions reuse its vocabulary.** That's
   why recall saturates. It's a teaching corpus, not a hard one. Swapping in
   your own domain documents is the highest-value change you can make — and
   doing it is a better interview story than anything you can recite.

---

Next: **[glossary.md](glossary.md)** for any term you hit, then
**[01_embeddings](../01_embeddings/)**.
