# 07 — Prompt chaining

**You build:** sequential, parallel, branching and map-reduce chains with LCEL,
each one traced and contract-checked.
**You learn:** why chaining makes a system *harder* to test, and the three
techniques that make it testable again.

```bash
pytest 07_prompt_chaining -v      # 19 tests, no model, under a second
```

---

## The number that should change your design

```
0.95 ** 4 == 0.8145
```

Four links that each work 95% of the time give you an **81%** chain. People
reason additively about this ("95%, minus a bit") and are wrong by 14
percentage points at four links, more at eight.

Two consequences, both tested:

- **Reliability is bounded above by your worst link.** No amount of polishing
  the other three beats the 60% one. "Improve every prompt" is the wrong
  response to a flaky chain — find the worst link.
- **To get 99% out of 5 links, each link needs 99.8%.** If you cannot build
  that link, the answer is not a better prompt. It is **fewer links**.

```python
required_link_reliability(0.99, 5)   # 0.99799
```

That is the argument for collapsing a chain, and it is an argument you can
make with a number rather than a feeling.

---

## Why chaining is a testing problem

One prompt has one failure mode: the answer is wrong. A four-link chain has
five — each link, plus the composition. And only the last link's output is
visible, so every one of those failures presents identically.

Three techniques, in the order they pay off:

### 1. Trace every link

`ChainTrace` records each link's input, output, duration and error. A failure
then has an **address** instead of a symptom.

It also distinguishes two things a chain-level assertion cannot tell apart:

| `trace.of("answer")` | means |
|---|---|
| a `LinkTrace` with `error` set | the link ran and failed |
| `None` | **the link never ran** |

In a branching chain the second is the more common cause of a surprising
answer, and it looks exactly like the first from outside.

### 2. Contract-check each boundary

```python
IS_LABEL.enforce("Category: FACTUAL")
# ContractViolation: classification: the router only knows three labels,
#                    so anything else routes nowhere (got 'Category: FACTUAL')
```

Without that contract the chain returns a plausible-looking refusal and you
spend an hour debugging the retriever. This is the ordinary SDET move — contract
tests at integration boundaries — applied to the `|`s.

The nastiest one to catch is `NON_EMPTY`. An empty link output does not raise;
it gets formatted into the next prompt as nothing at all, and the downstream
model confidently answers a question it was never asked.

### 3. Measure per-link success

`weakest_link(trace)` tells you where the time goes. Per-link success rates tell
you where the failures come from. Both are invisible if you only score the final
answer.

---

## The four shapes

| Shape | Function | Use when | Failure behaviour |
|---|---|---|---|
| Sequential | `sequential_chain` | each step needs the last one's output | a bad step poisons everything after it |
| Parallel | `parallel_chain` | independent views of one input | branches are isolated — one failure does not corrupt the others |
| Branch | `branching_chain` | different prompts for different inputs | **always have a default arm** |
| Map-reduce | `map_reduce_chain` | more documents than fit in context | the reduce step cannot recover what map dropped |

Two things worth internalising:

**Prefer parallel to sequential where you can.** Not for speed — for blast
radius. Independent branches cannot corrupt each other, so your failure modes
get cheaper. That is a design lever.

**Routing costs no model call.** `RunnableBranch` predicates are ordinary
Python. Paying a model to decide which prompt to use, when a regex would do, is
latency and variance bought for nothing. `test_branch_routing_costs_no_model_call`
asserts exactly one model call on a route-and-answer chain.

---

## The map-reduce trap that no metric catches

The reduce step sees only the summaries. Any fact the map step dropped is
unrecoverable, and the final answer is confidently *incomplete* rather than
visibly truncated.

Faithfulness will not catch it — the summary **is** faithful to the text it saw.
Catching it needs recall against the originals, which is why `map_reduce_chain`
returns the intermediate summaries instead of swallowing them.

---

## A flaky test I wrote, and the fix

The first version of `test_parallel_chain_returns_a_dict_keyed_by_branch` used
`FakeListChatModel`, which replays responses **in call order**. `RunnableParallel`
runs branches in a thread pool, so the replies got assigned to branches by
thread scheduling. The test passed or failed at random.

`keyed_model.py` fixes it by keying the fake on the **prompt** rather than on
call order. Same branch, same reply, regardless of when it runs.

Generalise it: **when faking a dependency that will be called concurrently, key
the fake on the request, never on call order.** That one applies far outside
LangChain.

---

## Interview answers this lesson gives you

> *"How would you test a multi-step LLM chain?"*

Not "score the final output". Trace each link so a failure has an address,
contract-check each boundary so the failure is caught where it happened, and
measure per-link success so you know which link to fix. Then point out that four
95% links are an 81% chain, and that the fix is usually fewer links rather than
better prompts.

**Next:** [08_mcp](../08_mcp/) — giving an agent tools from servers you do not own.
