# 10 — Live monitoring

**You build:** error-budget burn-rate alerting, drift detection, and the loop
that turns live traffic into new golden items.
**You learn:** the hard limit on what production monitoring can tell you, and
what to do inside that limit.

```bash
make serve-agent          # then open http://localhost:8001/live
pytest 10_live_monitoring -v      # 30 tests, no model
```

---

## The uncomfortable answer

**You cannot measure quality live.** There are no reference answers in
production.

Every live number is a proxy. Refusal rate, citation validity, latency, tool
mix, answer length — not one of them says an answer was *correct*. They say
something **changed**. That is genuinely valuable and it is not the same thing,
and conflating them is how a team ends up with a green dashboard above a system
that is quietly wrong.

The honest framing, and the one to give in an interview:

| | measures | against |
|---|---|---|
| offline evaluation (04, 05) | **quality** | reference answers |
| live monitoring (this) | **change** | recent history |
| the loop between them | turns traffic into new references | — |

The third row is the payoff. The dashboard says this in a banner, on the page,
so nobody has to infer it.

---

## Burn rate, not thresholds

> *"Alert when the error rate exceeds 1%."*

Fires at 3am for a two-minute blip, and stays silent through a week at 0.9% that
eats your entire quarterly budget. Both failures, one rule.

With a 99% SLO the budget is 1% of requests. **Burn rate** is how fast you spend
it:

```
burn = observed_error_rate / (1 - slo)
```

Burn 1 spends the budget exactly over the window. Burn 14.4 spends a 30-day
budget in two days.

| Window | Burn | Severity | Because |
|---|---|---|---|
| fast (1h) | ≥ 14.4× | **page** | consumes 2% of a 30-day budget in an hour |
| slow (6h) | ≥ 6× | ticket | consumes 5% — not an outage, but the budget will not last |

Fast catches the outage. Slow catches the bleed. **You need both**, and almost
every home-grown alert has only the first:

```python
def test_a_threshold_alert_would_miss_the_slow_bleed():
    assert not (0.08 > 0.10)                                   # naive rule: silent
    assert ErrorBudget().evaluate(fast_rate=0.0, slow_rate=0.08) is not None
```

Two more details that matter:

- **One incident, one alert.** During a real outage both windows breach. Fast is
  checked first and returns; emitting two alerts for one incident is how alert
  fatigue starts.
- **A 100% SLO is a configuration error.** It has no budget, so any failure is
  infinite burn. Better to surface that than divide by zero in the alerting path
  at the worst possible moment.

---

## The four live checks

### 1. Error-budget burn — pages

### 2. Refusal rate, **both directions**

The spike means retrieval or corpus coverage broke. The **collapse** is the
dangerous one:

> refusals went from 40% to 0% — the system stopped abstaining and started
> answering from parametric memory.

Latency improves. Errors are unchanged. The refusal graph goes **down**, which on
most dashboards is drawn green. A one-sided alert never fires on it. This one
pages.

### 3. Tool-mix drift — unique to a multi-server agent

Traffic silently moving from the corpus server to the web server is a behaviour
change with no code change and no error. Nothing else catches it:

```python
def test_tool_mix_drift_fires_with_nothing_else_wrong():
    alerts = {a.name for a in monitor.check()}
    assert "tool_mix_drift" in alerts
    assert "error_budget_burn" not in alerts
```

Measured with **total variation distance**, not KL divergence or PSI. KL is
undefined when a category is missing from one side — which is exactly the
interesting case, a server that stopped being used at all. TVD handles it with no
epsilon fudge that someone later forgets they added. Read it as "the share of
traffic that would have to move to make the two match".

### 4. Latency — p95, never the mean

The mean hides the tail users actually experience and that trips client timeouts.

---

## Every comparison is recent-vs-reference

No fixed thresholds. `reference` is the half of the window *before* the recent
half — deliberately not all history, because a reference that includes the
anomaly dilutes it, and a long-running incident slowly becomes the new normal.
That is alert blindness with extra steps.

`min_samples` stops a quiet service alerting on three requests. And
`test_a_steady_service_produces_no_alerts` is the test that keeps the monitor
usable: a monitor that alerts on healthy traffic gets muted, and a muted monitor
is worse than none.

---

## Closing the loop — the most valuable thing here

`promote_candidates` picks live requests worth a human label:

| Signal | Priority | Why it teaches you something |
|---|---|---|
| output guard blocked | 4 | a real attack, or a false positive — you cannot tell without looking |
| errored | 3 | a failure you have no test for |
| refused | 2 | correct abstention, or a coverage gap — indistinguishable from outside |
| unusually long answer | 1 | correlates with rambling and with hedging, both of which score badly offline |

Selection is biased towards the **informative**, not the representative.
Sampling traffic uniformly mostly returns more of what you already handle: cheap
to label, near-zero information.

Three rules the code enforces:

- **Deduplicate on the question**, not the request id, or one popular failing
  question floods the queue forty times.
- **The long-tail rule is off below 20 samples.** A percentile over one sample
  *is* that sample — without the floor a quiet service promotes every answer as
  "unusually long". Found by the integration test failing.
- **No auto-promotion.** `Candidate` has a `reason` and deliberately **no**
  `expected_answer` field to fill in. Auto-promoting a model's own output into
  the reference set teaches the system to grade its own homework, and it fails
  silently and permanently.

An eval dataset that never grows is describing a system that no longer exists.

---

## Monitoring must never break the thing it watches

Two places this is enforced:

```python
def test_a_failing_monitor_cannot_break_a_request(registry):
    state.monitor.record_response = explode
    assert response.status_code == 200      # the user still gets their answer
```

and `record_response` tolerates a malformed body rather than raising. A
monitoring call in the hot path is a dependency of every response; if it can
raise, it can cause the outage it exists to detect.

---

## The dashboard

`/live` — traffic, error rate, refusal rate, latency percentiles, server mix,
active alerts, and the label queue.

One piece of design worth copying: **every panel shows the sample count next to
the number.** "Refusal rate 0%" over three requests and over three thousand are
different facts that render identically, and the first gets acted on at 3am by
someone who did not check. Putting `n` beside the value costs nothing.

---

## Interview answers this lesson gives you

> *"How do you know your RAG system is working in production?"*

Start with what you **cannot** know: there are no references, so nothing live
measures correctness. Then: burn-rate alerting on two windows rather than a
threshold, refusal rate watched in both directions because the collapse is the
dangerous one, tool-mix drift for a multi-server agent, and a promotion loop that
feeds anomalous traffic back into the offline golden set — which is the only
place quality is actually measured.

**Back to:** [the root README](../README.md) · [INTERVIEW.md](../INTERVIEW.md)
