# 08 — MCP: one agent, several tool servers

**You build:** three real MCP servers and one agent that uses all of them.
**You learn:** the five traps in multi-server MCP, and how to test an agent
whose tools live in someone else's repository.

```bash
make mcp-tools                    # what the servers actually expose
make threshold                    # the measurement in "no threshold works"
pytest 08_mcp -v                  # 26 tests, real subprocesses, no model
```

---

## The uncomfortable bit

**Your agent's tool surface is now decided by a process you do not control.**

A server upgrade can rename a tool, tighten a schema, or reword a description —
and your agent's behaviour changes with **no diff in your repository**. You
cannot unit-test the servers; they are not yours. What you can test is the seam,
and that is what this lesson is.

---

## Five traps, all measured against `langchain-mcp-adapters 0.3.2`

Every one was found by running the code here, not recalled from documentation.
Each has a test that fails if the behaviour changes.

### Trap 1 — one dead server kills them all

`MultiServerMCPClient.get_tools()` gathers every server in a single asyncio
`TaskGroup`. One server that fails to start raises an `ExceptionGroup` out of the
whole call and you get **no tools at all**, not even from the healthy servers.

Three MCP servers therefore means three single points of failure. The fix is six
lines — `load_tools_resiliently` loads each server independently — and the
version most people write is shorter and has the availability of the worst
server in the set.

Unwrap the `ExceptionGroup` when you log it. Its `str()` is
`"unhandled errors in a TaskGroup (1 sub-exception)"`, which tells an operator
nothing.

### Trap 2 — colliding tool names are silent

```
prefix=False: ['search', 'search']       # no error, no warning
prefix=True:  ['corpus_search', 'web_search']
```

Two servers exposing `search` produce two tools called `search`. The model emits
`{"name": "search"}` and which server runs is resolved by **list order** — that
is, by your config file.

This is the most likely way a working multi-MCP agent breaks when someone adds a
third server, and it is invisible in code review. `build_mcp_agent` **refuses**
to build on a colliding registry, because a 50/50 bug is cheaper at startup than
in production where it reproduces one time in two.

### Trap 3 — a sessionless tool call spawns a new server process

Measured with a stateful server:

```
no session:  pid=1665 count=1    pid=1669 count=1     <- state LOST
session:     pid=1673 count=1    pid=1673 count=2     <- state KEPT
```

Tools from `get_tools()` open a fresh connection **per invocation**. Anything the
server held in memory between calls — a cursor, a login, a cache, a transaction
— silently vanishes, and nothing in any log says so. Use `client.session(name)`
for a stateful server.

It costs latency too, and the number is not small. Measured against the real
gateway in lesson 09, one question with one tool call:

```json
{"tools_used": ["corpus_search"], "latency_ms": 1322.5}
```

A second and a third of that is Python interpreter startup plus an MCP
handshake, for a retrieval that takes milliseconds. Session reuse is a
correctness fix that happens to be a large performance fix.

### Trap 4 — an MCP tool does not return what a native tool returns

```python
native_tool.invoke({"a": 2, "b": 3})   # 5
mcp_tool.ainvoke({"a": 2, "b": 3})     # [{'type': 'text', 'text': '5', 'id': 'lc_...'}]
```

Swap a native tool for its MCP twin and every assertion you wrote breaks — as a
confusing type error far from the tool call. `text_of()` exists so you write that
flattening once.

### Trap 5 — MCP tools are async-only, and `.invoke()` raises

```
NotImplementedError: StructuredTool does not support sync invocation.
```

The adapter builds a `StructuredTool` with `coroutine` set and `func` left
`None`. There is no sync path and **no warning at bind time** — the agent
constructs, binds and plans perfectly, then dies on the first tool call.

The overwhelming majority of LangChain examples, including ones you will copy,
use `.invoke()`. So this passes every unit test and fails in integration. Async
all the way down.

---

## The metric that matters: tool-selection accuracy

An agent with three servers has a failure mode no RAG metric can see: **it asks
the wrong server.** The answer comes back fluent, cited, and faithful — because
it *is* faithful; the passages were just from the wrong place. The error is
upstream of everything faithfulness can measure.

Tool-selection accuracy needs **no judge**. Label which server should serve each
question, and count.

```
Tool-selection accuracy: 75% (6/8)
  MISROUTED 'is it raining in London right now?'   expected web, chose corpus
  MISROUTED 'did I cite anything that does not exist?'  expected evaluator, chose corpus
```

That is `KeywordRouter`, the zero-cost baseline. **An LLM router has to beat 75%
to be worth its latency and its variance.** Measuring against a cheap baseline is
ordinary engineering that eval work routinely skips.

### The routing set was saturated, and that was the interesting part

The first five cases scored the keyword baseline **5/5**. A set your cheapest
baseline aces cannot rank two routers — the same saturation problem lesson 01
found in recall@k, recurring in a completely unrelated metric.

The fix was three paraphrases carrying no keyword. `test_the_routing_set_has_discriminating_power`
asserts `0 < accuracy < 1` so nobody quietly deletes them.

---

## Snapshot the contract

```python
tool_contract(tools)["corpus_search"]
# {'description': 'Search the internal knowledge base for passages matching a query.',
#  'required': ['query'],
#  'parameters': {'k': 'integer', 'query': 'string'}}
```

`contract_diff` ranks changes by how silently they break you:

| Change | Why it is ranked there |
|---|---|
| `REMOVED tool x` | fails loudly on the next call |
| `BREAKING x: newly required [...]` | fails on the first call with the old argument shape — days later |
| `REWORDED x description` | **not cosmetic** — the description is the prompt the model routes on |

---

## A measurement that kills a common answer

"Add a relevance threshold so the system refuses when retrieval is weak" is
advice you will read everywhere and give in an interview. `make threshold` checks
whether it is true here:

```
real questions  n=42   min 0.208  median 0.313  max 0.610
gibberish       n=200  min 0.029  median 0.190  max 0.433

gibberish max 0.433 > real min 0.208  ->  distributions OVERLAP

  threshold   real rejected   gibberish accepted
     0.25         16.7%           10.0%
     0.30         42.9%            3.0%
```

**No threshold separates them.** At 0.25 you refuse one genuine question in six
and still answer one nonsense query in ten.

The cause is specific: hashed lexical embeddings collide unknown tokens into
slots real tokens occupy, so nothing ever scores zero. A neural embedder
separates far better — but you must **measure that on your own corpus** before
trusting it. The method transfers; the numbers do not.

So `corpus_search` returns a `LOW_CONFIDENCE` banner rather than suppressing
results, and the refusal decision stays where a deterministic check can back it
up. A test pins the overlap so nobody "fixes" the hint into a hard gate.

---

## `create_agent` or `StateGraph`?

| Use | When |
|---|---|
| `langchain.agents.create_agent` | the loop is standard (model → tools → model); you want middleware, structured output and checkpointing for free |
| LangGraph `StateGraph` (lesson 03) | you need a node the loop does not have: a validation gate, a fan-out, a human approval step mid-loop, a custom router |

`create_agent` returns a compiled LangGraph graph, so everything lesson 03 taught
still applies. Start there; drop to `StateGraph` the moment you need a step that
is not "call a tool". Reaching for `StateGraph` first writes 200 lines the
prebuilt already has; reaching for `create_agent` when you need a custom node
ends in middleware abused as control flow.

---

## Why these tests spawn real subprocesses

They cost about a second each, and the fast tier is slower for it. It is worth
paying: **every trap above is invisible to a mock**, because a mock is written
from the same wrong mental model that produced the bug. Still no model, no GPU,
no network and no API key.

**Next:** [09_serving](../09_serving/) — putting this agent on the public internet.
