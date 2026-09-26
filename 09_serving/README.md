# 09 — Exposing the agent publicly

**You build:** a FastAPI gateway over the lesson 08 agent, with auth, a per-key
tool allowlist, rate limiting, spend limits, timeouts and output guards.
**You learn:** what a public agent endpoint actually exposes, and why an API key
is nowhere near enough.

```bash
make serve-agent          # http://localhost:8001/docs  and  /live
pytest 09_serving -v      # 29 tests, no model
```

```bash
curl -s localhost:8001/v1/ask \
  -H 'x-api-key: sk_eval_demo_readonly' \
  -H 'content-type: application/json' \
  -d '{"question":"what is chunk overlap?"}'
```

---

## The uncomfortable answer

**"Expose the agent publicly" is not a deployment task.** A public agent endpoint
is a *remote tool execution service*, driven by untrusted text, billed to you.

A stranger with your URL gets:

- **arbitrary execution of every tool you bound**, with arguments they influence
  — and with MCP those tools come from a process you may not own
- **your inference spend**, with no natural ceiling. One request that loops
  twenty tool calls costs twenty times as much and looks like one request
- **a prompt-injection surface that includes your own retrieved documents.**
  Anything the agent reads can carry instructions — the user's message is not
  the only untrusted input
- **an oracle** for extracting your system prompt, your corpus and your tool list

None of that is fixed by adding an API key. A key tells you *who* is doing it,
which is necessary and nowhere near sufficient.

What is here is the **minimum**. A genuinely public endpoint belongs behind a
real gateway — WAF, mTLS or OAuth, egress control. Saying "I added an API key and
rate limiting" and stopping is the answer that fails an interview.

---

## The guards, and why each exists

| Guard | Code | The failure it prevents |
|---|---|---|
| Constant-time key compare | `verify` | `==` on a secret leaks length and prefix through timing |
| **Per-key tool allowlist** | `ApiKey.allowed_tools` | a leaked read-only key is a leak; a leaked key with `web_fetch_url` is an SSRF proxy on your egress IP |
| Token bucket | `TokenBucket` | a fixed window permits a double burst across the boundary |
| **Spend budget** | `SpendLimiter` | rate is not cost — one request can fan out into a dozen model calls |
| Input length cap | `validate_question` | the largest input you accept is the largest single bill you can be handed |
| Hard timeout | `asyncio.wait_for` | an agent loop with no deadline is an unbounded bill and a held socket |
| PII redaction | `guard_output` | personal data in the answer leaving your process |
| Leak canaries | `LEAK_CANARIES` | an answer reciting your system prompt |
| Request id everywhere | `RequestContext` | "it broke this afternoon" is not a supportable bug report |

### The one people leave out

`allowed_tools`. Two demo keys, deliberately unequal:

```python
readonly:  corpus_*, evaluator_*                    # cannot reach the network
power:     corpus_*, evaluator_*, web_search, web_fetch_url
```

Enforced twice on purpose — **preventively**, by binding only permitted tools so
the model never sees the others, and **detectively**, by `tool_calls_allowed`
checking afterwards. If the detective control ever fires, the preventive one has
a hole, and you want to learn that from your own alert rather than from a bill.

---

## The caveat that invalidates most tutorial rate limiters

The limiter here is **in-process**. Run `uvicorn --workers 4` and you have four
independent limiters, so your "60 per minute" is 240. Autoscale to eight pods and
it is 1920.

```python
def test_in_process_limiting_is_per_worker():
    workers = [TokenBucket(capacity=10, refill_per_second=0.0) for _ in range(4)]
    total = sum(sum(w.allow(now) for _ in range(10)) for w in workers)
    assert total == 40, "stated limit 10; four workers served 40"
```

Nearly every FastAPI rate-limiting example has this bug and none of them mention
it. The fix is shared state (Redis) or limiting at the gateway. `TokenBucket`
carries `shared_state_warning` on every instance so a deployment cannot pretend
otherwise.

---

## Status codes are an interface

| Code | Means | What the client should do |
|---|---|---|
| 401 | unauthorised — **and nothing more specific** | fix the key. "unknown key" vs "bad key" is a free enumeration oracle |
| 402 | budget exhausted | **stop.** Do not retry |
| 422 | input rejected before the model saw it | fix the request |
| 429 | rate limited, with `retry-after` | retry after that many seconds |
| 504 | the agent exceeded its deadline | retry once, then escalate |

429 and 402 are different on purpose: one means retry later, the other means stop
and talk to someone. A bare 429 with no `retry-after` invites a tight retry loop,
which is how a rate limit becomes a self-inflicted DDoS.

---

## Three design decisions worth defending

### 1. Every handler is async, all the way down

Not style. MCP tools have no synchronous implementation (lesson 08, trap 5), so a
`def` handler raises `NotImplementedError` inside the tools node. And a sync
handler in FastAPI runs in a threadpool — under load you would pay for threads to
sit blocked on network I/O.

### 2. `/healthz` and `/readyz` are different endpoints

- **Liveness** — "is this process wedged? restart it". Checks nothing downstream.
- **Readiness** — "can it serve? send it traffic". Reports degraded MCP servers.

Conflating them is a real outage pattern: a readiness probe that checks a
downstream dependency, wired to liveness, restarts every pod when that dependency
blips — turning a degraded dependency into a total outage.

And degraded ≠ unready. With `web` down the service still answers from `corpus`,
so `/readyz` returns 200 and *reports* the degradation. Refusing all traffic
because one optional tool source is down is worse than degrading.

### 3. `/v1/metrics` publishes no quality score

A gateway has no reference answers, so it cannot compute faithfulness.
Publishing one anyway is the exact failure this repository is built around — a
number that looks like a measurement and is not.

```python
def test_metrics_publish_no_quality_score(client):
    for forbidden in ("faithfulness", "accuracy", "relevancy", "quality"):
        assert forbidden not in client.get("/v1/metrics").json()
```

`p95_ms` is `null` before any traffic, never `0.0`. A zero draws a beautiful flat
line on a dashboard and means nothing happened.

---

## Streaming costs you a guard

`/v1/ask/stream` emits **tool steps**, not raw tokens — the user sees "searching
the knowledge base" and understands the ten-second wait.

The trade-off is real and worth stating: the output guards run on the **complete**
answer. Stream tokens and you have already shipped the first half of a response
you might have blocked. That is why the events here are structured steps.

---

## Deploying it

```dockerfile
# See Dockerfile in this directory.
```

The parts that matter more than the Dockerfile:

- **Keys come from a secret store, hashed.** `DEMO_KEYS` is hard-coded so the
  lesson runs with no setup; it is not a pattern.
- **Limit at the gateway, not in the process** — see the worker-multiplication
  measurement above.
- **Egress control.** `web_fetch_url` turns user input into an outbound request.
  An allowlist, not a blocklist.
- **One replica per MCP stdio server is a process fan-out.** Every replica spawns
  its own server subprocesses. At scale, move to HTTP-transport MCP servers.
- **Sessionless MCP tools cost a process spawn per call.** Measured on this
  gateway: `{"tools_used": ["corpus_search"], "latency_ms": 1322.5}` for a
  retrieval that takes milliseconds. Before tuning anything else, reuse the
  session (lesson 08, trap 3).

---

## A vacuous assertion I wrote, and how it was caught

`test_a_readonly_key_is_never_shown_the_dangerous_tool` originally did
`captured.append(model.bound_tools)` — capturing the list *before* `bind_tools`
reassigned it. The negative assertion passed against an empty list, proving
nothing. The positive assertion on the next line is what failed and exposed it.

The fix: capture the models and read `bound_tools` afterwards, plus
`assert readonly_tools and power_tools` so an empty list can never satisfy the
test again. **Every negative assertion needs a positive one beside it.**

**Next:** [10_live_monitoring](../10_live_monitoring/) — watching it run.
