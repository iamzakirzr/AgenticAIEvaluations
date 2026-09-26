# Agentic AI Evaluations

A hands-on curriculum for **building and evaluating** RAG pipelines, chatbots and
agents — with LangChain, LangGraph, LangWatch, DeepEval and RAGAS, running
entirely against **local open-source models via Ollama**.

Eleven lessons, ending with an agent on a public endpoint, watched live. Each
builds something, then measures it. **513 tests run with no model, no GPU and
no API key**, so you can explore and break things freely before ever loading a
model.

```bash
make setup && make test        # 513 tests, ~70s, nothing to install beyond Python
make test-quick                # 430 of them in ~18s
make hello                     # your first evaluation, 2 seconds
make chat                      # the chatbot at http://localhost:8000
make serve-agent               # the public agent + live dashboard on :8001
```

---

## The uncomfortable thing this repo is built around

**Installing an eval library and printing a number is not evaluation.** Anyone
can `pip install deepeval` and copy the quickstart. Three things separate that
from work worth paying for, and each has a lesson here:

1. **A dataset designed to make your system fail** — unanswerable questions,
   false premises, multi-hop synthesis. A golden set of happy-path questions
   reports great scores and catches nothing.
2. **Knowing which metric can actually move.** Recall is saturated on this
   corpus (spread 0.012 across a 16× chunk-size sweep). Gating CI on a saturated
   metric is theatre, so the gate uses MRR instead.
3. **Calibrating the judge.** A metric from an uncalibrated judge is a number
   that looks rigorous and may measure nothing. Almost nobody checks. Lesson 04
   does.

---

## New to this? Start at zero

**[00_start_here](00_start_here/)** is the on-ramp for QA and SDET engineers:
a translation table from concepts you already own (test oracle, boundary
values, flaky tests, coverage, CI gates) to their AI-evaluation equivalents,
how AI/ML actually works with no maths, a full glossary, and a 4-week plan.

```bash
make hello     # your first evaluation: 60 lines, 2 seconds, no model needed
```

Then **[INTERVIEW.md](INTERVIEW.md)** — 15 scenario questions with
weak-vs-strong answer contrasts, a 60-second project pitch, and questions to
ask the interviewer.

---

## The learning path

| # | Lesson | You build | You learn |
|---|---|---|---|
| 01 | **[embeddings](01_embeddings/)** | A retriever in ~80 lines of numpy | Tokenization → hashing → IDF → cosine → ranking, and where lexical search breaks |
| 02 | **[langchain](02_langchain/)** | The RAG pipeline + a chatbot | LangChain 1.4, prompt design as an eval concern, citation checking |
| 03 | **[langgraph](03_langgraph/)** | A tool-using agent | Why an agent is a while-loop, and why trajectories need their own metrics |
| 04 | **[deepeval](04_deepeval/)** | An Ollama judge + calibration | The metric catalogue, and Cohen's kappa |
| 05 | **[ragas](05_ragas/)** | The same pipeline, scored again | A second opinion, and where two libraries disagree |
| 06 | **[langwatch](06_langwatch/)** | Tracing + online evaluation | Evaluating traffic you cannot label |
| 07 | **[prompt_chaining](07_prompt_chaining/)** | Four LCEL chain shapes, traced and contract-checked | Why four 95% links are an 81% chain, and how to give a failure an address |
| 08 | **[mcp](08_mcp/)** | Three MCP servers, one agent | Five traps in multi-server MCP, and tool-selection accuracy |
| 09 | **[serving](09_serving/)** | The agent on a public endpoint | What a public agent endpoint actually exposes, and the guards |
| 10 | **[live_monitoring](10_live_monitoring/)** | Burn-rate alerting + a live dashboard | Why you cannot measure quality live, and what you can do instead |

Each lesson also has **advanced topics** beyond the core walkthrough:

| Lesson | Advanced module | Covers |
|---|---|---|
| 02 | `advanced_retrieval.py` | BM25, RRF, hybrid search, MMR, reranking, query rewriting, HyDE, multi-query, conversational retrieval |
| 03 | `advanced_graph.py` | parallel fan-out/fan-in, reducers, subgraphs, streaming |
| 04 | `advanced_deepeval.py` | red teaming, synthetic data, multi-turn metrics, `assert_test` |
| 05 | `advanced_ragas.py` | test-set generation and the review queue |
| 08 | `measure_threshold.py` | whether a similarity threshold separates signal from noise (it does not, here) |
| 09 | `Dockerfile` | deploying the gateway, and what matters more than the image |
| 10 | `dashboard.py` | the live page, and what it deliberately refuses to show |

Then **[PRODUCTION.md](PRODUCTION.md)** — every lesson also ships a
`production_*.py` scenario covering the failures that actually take eval systems
down: embedding/index version drift, PII leaving your process, judge-failure
budgets, NaN-poisoned means, variance-aware regression gates, human approval
gates that fail closed, and four CI workflows.

**Read them in order.** Each lesson's `README.md` is the written explanation; the
code is heavily commented and meant to be read alongside it.

---

## Setup

```bash
make setup          # creates .venv, installs everything (needs `uv`)
make test           # 513 fast tests -- no model required
```

For the judged tiers you need [Ollama](https://ollama.com):

```bash
ollama serve
make models         # pulls llama3.1:8b and nomic-embed-text
make test-judge     # the expensive judged metrics
```

**Any OpenAI-compatible server works** — vLLM, LM Studio, llama.cpp, TGI, or a
hosted API. Point `OLLAMA_BASE_URL` at it. See `.env.example` for every knob.

---

## The two-tier test strategy

This is the design decision worth stealing:

| Tier | Marker | Needs | Speed | Runs |
|---|---|---|---|---|
| **Fast** | *(none)* | nothing | ~70s | every push |
| **Ollama** | `-m ollama` | a local model | minutes | on demand |
| **Judged** | `-m judge` | a local model as judge | slow, noisy | nightly / manual |
| **SaaS** | `-m saas` | a LangWatch account | — | never in CI |

An eval suite that takes 20 minutes and fails randomly gets switched off within
a fortnight. One that runs in two seconds and never lies gets trusted — and the
slow judged tier stays credible because it isn't asked to do a job it's bad at.

**What the fast tier actually covers**, with no model: retrieval quality
(recall@k, MRR, nDCG), chunking behaviour, golden-dataset integrity, prompt
structure, citation validation, agent trajectories, loop detection, judge
calibration maths, span instrumentation — and every production scenario:
retry/circuit-breaker/budget behaviour, index version drift, PII redaction,
approval gates, NaN handling and the regression gate itself. That is most of
what matters.

---

## What's in the repo

```
00_start_here/       the on-ramp: QA-to-eval bridge, ML primer, glossary, hello_eval
core/                shared spine -- config, providers, traces, dataset, metrics
  resilience.py      retry, timeout, circuit breaker, budget, bounded map
  baseline.py        variance-aware regression gating
  corpus/            8-document knowledge base (the thing being retrieved)
  golden.jsonl       48 labelled questions across 4 categories
01_embeddings/ … 06_langwatch/     the lessons, each with a production_*.py
07_prompt_chaining/  LCEL shapes, link tracing, boundary contracts
08_mcp/              three real MCP servers + one agent over all of them
  servers/           corpus, web and evaluator servers (real subprocesses)
09_serving/          the public gateway: auth, allowlists, budgets, guards
10_live_monitoring/  burn-rate alerting, drift, and the label queue
scripts/             run_regression_gate.py -- the CI entry point
.github/workflows/   4 workflows: PR gate, nightly judged, drift canary, baseline
```

### The golden dataset is the most important file

48 items, four categories, three of which exist to make the system **fail**:

| Category | n | Purpose |
|---|---|---|
| `single_hop` | 26 | baseline competence |
| `multi_hop` | 10 | needs synthesis across documents |
| `unanswerable` | 6 | **correct behaviour is refusal** |
| `adversarial` | 6 | false premises — catches sycophancy |

`un-02` asks "What is the capital of France?" The model certainly knows. The
corpus doesn't contain it. Answering proves the system ignores its own grounding
instruction — and a dataset without such items would score a hallucinating
system perfectly.

Every item records `reference_doc_ids`, which is what makes recall@k and MRR
computable **with no LLM at all**. Most tutorial datasets omit it. It is the
cheapest thing you can add that makes a dataset genuinely useful.

---

## Design decisions, and why

**Shared spine, separate adapters.** `core/` holds only what two or more lessons
use. DeepEval and RAGAS get *separate* adapters rather than one unified metric
interface, because they genuinely disagree about what a test case is:

| Concept | DeepEval | RAGAS |
|---|---|---|
| the question | `input` | `user_input` |
| what the system said | `actual_output` | `response` |
| the reference answer | `expected_output` | `reference` |
| what the retriever found | `retrieval_context` | `retrieved_contexts` |

A unified interface would flatten these and lose information from both. **Adapt
at the edges; don't unify in the middle.**

**Every pipeline returns a trace, never a string.** Faithfulness, context
precision and context recall are all uncomputable from an answer alone. A
pipeline that discards its retrieved chunks cannot be evaluated.

**Design for deterministic observability.** Numbered passages make citation
hallucination an integer comparison. An exact refusal string makes refusal a
regex. `refuse` as an agent *tool* makes refusal a boolean. Each turns a judged
question into a free one.

---

## Real findings from building this

Everything below was discovered by writing the code, verified against installed
versions, and pinned as a test:

- **`ragas 0.4.3` will not import** on a modern LangChain stack — a module-level
  import of a class `langchain-community 0.4.x` removed, declared as an unpinned
  dependency. `core/compat.py`
- **Every RAGAS tutorial online uses a deprecated API.** `ragas.metrics` now
  warns; `ragas.metrics.collections` is current. `05_ragas`
- **`ToolCorrectnessMetric` demands an OpenAI key it never uses** — it's a
  deterministic comparison whose `__init__` builds a default OpenAI model.
  `04_deepeval`
- **`PatternMatchMetric` uses `fullmatch`, not `search`**, despite the name. A
  bare phrase scores 0.0 on text that plainly contains it. `04_deepeval`
- **OpenTelemetry silently drops dict-valued span attributes** — no error, the
  field simply isn't there when you go looking during an incident. `06_langwatch`
- **`add_messages` de-duplicates by message id**, so replaying an identical
  message object ends an agent loop early with no error. `03_langgraph`
- **Recall is saturated on this dataset**, so the CI gate uses MRR. Noticing your
  own metric has no headroom is the difference between running an eval and
  understanding one. `01_embeddings`
- **Two MCP servers exporting the same tool name collide silently.** You get two
  tools called `search`, no error, and tool choice resolved by list order.
  `08_mcp`
- **One unreachable MCP server takes down `get_tools()` for every server** --
  an `ExceptionGroup` out of the shared TaskGroup. Three servers, three single
  points of failure, unless you load them independently. `08_mcp`
- **MCP tools have no sync implementation**: `agent.invoke()` raises
  `NotImplementedError` at the first tool call, after binding and planning
  succeed. Most LangChain examples use `.invoke()`. `08_mcp`
- **A sessionless MCP tool call spawns a fresh server process** -- measured,
  different pid and a reset counter -- so server-side state silently vanishes
  between calls. `08_mcp`
- **No similarity threshold separates real questions from gibberish here.**
  Gibberish tops out at 0.433, real questions bottom out at 0.208. At a 0.25
  cut-off you refuse 16.7% of genuine questions and still answer 10% of
  nonsense. `08_mcp/measure_threshold.py`
- **In-process rate limiting multiplies by worker count** -- four workers serve
  four times your stated limit. Nearly every FastAPI tutorial has this bug.
  `09_serving`

---

## The measurement that matters most

```bash
python 04_deepeval/calibrate.py --demo
```

```
lazy judge (always says pass)        raw agreement: 90%   kappa: 0.000
useful judge (caught the failure)    raw agreement: 90%   kappa: 0.615
```

Both judges are wrong exactly once. One missed the only real failure; the other
caught it and raised one false alarm. **Only kappa distinguishes them.**

"My judge agrees with me 90% of the time" is not evidence of anything.

| kappa | verdict |
|---|---|
| > 0.6 | may gate CI |
| 0.4–0.6 | track the trend, don't fail builds |
| < 0.4 | the metric is noise |

Reporting *"my local judge scored kappa 0.31, so I did not gate on it"* is a
stronger result than a green dashboard nobody checked.

---

## Versions

All pins are exact. Every library here has made breaking changes across minor
versions, and most tutorials you'll find target the old ones.

`langchain 1.4.0` · `langgraph 1.2.11` · `langwatch 1.3.1` · `deepeval 4.2.1` ·
`ragas 0.4.3` · Python 3.11

---

## Where to go next

The honest limitations of this repo, and what you would do about them:

1. **The corpus is 8 documents.** Retrieval is easy at this scale. Swap in your
   own domain corpus and the metrics get interesting immediately.
2. **The golden questions reuse the source vocabulary**, which flatters lexical
   retrieval. Adding paraphrased questions would create real headroom.
3. **The judged tier has never been calibrated on your hardware.** Run
   `calibrate.py` before trusting a single judged number.
4. **Production traffic beats an imagined dataset.** Lessons 06 and 10 both end
   in the same loop back to lesson 04: real questions become new golden items.
   An eval dataset that never grows is describing a system that no longer exists.
5. **The tool-selection set is 8 items.** The keyword baseline scores 75% on it,
   so there is headroom -- but eight labelled cases rank two routers coarsely.
   Add yours.
6. **Lesson 09 is not a security review.** It implements the minimum: auth, a
   per-key tool allowlist, budgets, timeouts and output guards. A genuinely
   public endpoint belongs behind a WAF, real identity, and egress control.
7. **The MCP servers here run over stdio.** That means one subprocess set per
   replica. At scale, move to HTTP-transport MCP servers.
