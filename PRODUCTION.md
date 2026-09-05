# Production Scenarios

Every lesson has a `production_*.py` module and a matching test file. They cover
the failures that take evaluation systems down — not a fake microservice.

**All of it is testable with no model, no GPU and no API key.** 337 tests, ~15s.

```bash
make test          # everything, including every production scenario
make gate-fast     # the regression gate, retrieval metrics only
make baseline      # record a new reference (review the diff!)
make calibrate     # is your judge worth listening to?
```

---

## The shared primitives

| Module | What it gives you |
|---|---|
| `core/resilience.py` | retry + jitter, timeout, circuit breaker, budget, bounded parallel map |
| `core/baseline.py` | variance-aware regression gating |

### Retry only what is retryable

`RetryPolicy` takes an explicit `retry_on` predicate. A 503 is worth retrying; a
`ValueError` is a bug and retrying it burns time and money to fail identically
three more times, hiding the real error behind a timeout.

Jitter is not decoration: without it, N workers that fail together retry
together forever, turning a blip into a sustained outage.

### The regression gate

The naive gate is `assert faithfulness >= 0.80`. It fails twice over: it cannot
detect a drop from 0.95 to 0.82, and it flaps when the true value sits near the
threshold.

```
regression  <=>  drop  >  max(min_delta, sensitivity × observed_noise)
```

Three properties worth stealing:

1. **N runs, not one.** A single judged run is a number with no error bar.
2. **A config fingerprint DISABLES the gate rather than failing it.** Scores from
   a different model or chunk size are not comparable, so the honest response is
   to print them and decline to judge — not to fail a build for a change someone
   made on purpose.
3. **Judge failures are counted, never scored as zero.** A 0.0 for "the judge
   broke" is indistinguishable from a 0.0 for "the answer was terrible", and
   mixing them poisons every future comparison.

---

## Per-lesson scenarios

### 01 — `production_index.py`: the silent outage

Ingest upgrades its embedding model; the query service is on an older deploy.
**Nothing raises.** Both models return float vectors, cosine similarity still
returns numbers in range, retrieval still returns top_k. The results are noise,
and you find out from a complaint days later.

The fix is boring and complete: stamp the index with a fingerprint of everything
affecting its vectors and refuse to serve a query embedded by anything else.

Also: incremental reindex that only re-embeds changed documents — **and knows
when that is unsafe.** TF-IDF is corpus-fitted, so embedding one changed document
alone fits IDF to a one-document corpus and produces vectors from a different
space. `supports_incremental` detects this and falls back to a full rebuild.
Being slow is recoverable; being subtly wrong is not.

### 02 — `production_pipeline.py`: deterministic guards

| Guard | Why |
|---|---|
| PII redaction | happens **before** the prompt, logs or traces see the text |
| Relevance threshold | refuses without calling the model — safer *and* free |
| Model fallback | records **which model answered**; a silent 30% fallback makes every metric the average of two systems |
| Bounded concurrency | unbounded parallelism against one Ollama server just builds a queue |

`suggest_threshold()` derives the threshold from data and reports the
false-refusal rate it costs. Both directions measured together, or you optimise
your way into either a liar or a system that refuses everything.

### 03 — `production_agent.py`: memory and approval

`thread_id` **is** the conversation. `InMemorySaver` is development-only — an
agent that forgets every conversation on deploy is reported as "it keeps asking
me things I already told it".

`interrupt()` does not block. It suspends the graph, persists state, and returns
an `__interrupt__` payload; `Command(resume=...)` continues from exactly that
point, possibly in another process hours later. Two consequences that catch
people out: **it requires a checkpointer**, and **the node re-runs from the top
on resume** — so the approval check must come before any side effect.

Approval **fails closed**. `None`, `""`, a typo, a timeout sentinel: all mean not
approved. A guard that fails open is worse than no guard, because it creates the
belief that someone is checking.

### 04 — `production_eval.py`: an eval run you can put in a pipeline

Judge-failure budget (a run that mostly failed is *invalid*, not low-scoring),
category slicing, N-run variance, and baseline gating.

The slice that matters: a system can hold its overall mean steady while its
refusal rate on unanswerable questions collapses to zero.

### 05 — `production_ragas.py`: the NaN trap

`raise_exceptions=False` is the setting you want in production. It is also the
setting that silently destroys your metrics:

```python
[0.9, 0.8, float('nan'), 0.95]  ->  mean is nan
bool(float('nan'))              ->  True   # `if score:` does not catch it
```

One failed row and the entire metric reports `nan` — or worse, an aggregation
path drops it and quietly averages only the rows that succeeded.

`RunConfig` defaults are wrong for a local model: `max_workers=16` and
`max_retries=10` with backoff to `max_wait=60`. `production_run_config()` and the
module docstring explain each change.

### 06 — `production_observability.py`: traces you can legally keep

**Redact before the span is created** — not before export, not in a processor.
Once the raw string is a span attribute it is in a buffer another thread is
already shipping.

Sampling: head sampling by deterministic hash (so you can reproduce a scored
trace), plus **tail sampling** that always keeps errors, refusals, slow requests
and low-confidence retrievals. Pure random sampling at 10% throws away nine of
every ten incidents.

Alerting is **two-directional**. Everyone writes "refusals went up". The alert
people forget is refusals *collapsing* — the model stopped refusing and started
confabulating, which is the worse outcome.

Trace-to-dataset promotion never invents a reference answer. Auto-filling it from
the system's own output builds a dataset that certifies current behaviour as
correct — circular, and it hides every existing bug forever.

---

## CI workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `evals.yml` | every PR | fast tier + artifacts + an updated-in-place PR comment |
| `nightly-judged.yml` | 02:00 daily | judged tier, **calibrates the judge first**, opens an issue on regression |
| `dependency-drift.yml` | weekly | installs **unpinned** and reports what breaks |
| `record-baseline.yml` | manual | records a baseline **via a reviewed PR** |

Two design decisions worth copying:

**The PR gate never runs judged metrics.** Gate a PR on a noisy LLM judge and the
build goes red on sampling noise, someone adds `continue-on-error: true`, and
within a fortnight nobody reads CI.

**The drift canary inverts pinning.** Exact pins protect the build *and hide*
upstream breakage until an upgrade months later. The canary installs unpinned on
a schedule and opens one rolling issue — the pinned build stays green, so it
never blocks anyone, but you learn while the change is still small enough to
bisect. This repo already documents the exact failure it watches for.

---

## Bugs found while writing this

Each is fixed and pinned by a regression test. They are listed because *how* they
were found is the transferable part — every one came from a test written to
assert a property, not from reading the code.

| Bug | Symptom |
|---|---|
| `except BaseException` in retry / map / breaker | Ctrl-C could not stop a sweep |
| Cached vectors reused across a `chunk_size` change | matrix and chunk list silently disagreed |
| Refusal-spike alert required a non-zero baseline | 0% → 50% — the most dramatic spike — never fired |
| Uncompared metric rendered as `stable` | a new metric read as verified-unchanged |
| Unguarded `state.values` | a clean budget stop became an `AttributeError` |
| PII counts ignored retrieved contexts | under-reported the case most worth alerting on |
| Exit code 2 documented but never returned | "judge broke" was indistinguishable from "quality dropped" |
