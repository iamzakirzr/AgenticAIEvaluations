# 03 — LangGraph: Agents, and Why They Need Different Metrics

**Goal:** build a tool-using agent as an explicit `StateGraph`, and understand
why evaluating one answer is no longer enough.

---

## Run it

```bash
pytest 03_langgraph/ -v          # 20 fast tests, no model needed
pytest -m ollama 03_langgraph/   # a real local model choosing its own actions
```

---

## Files

| File | What it is |
|---|---|
| `tools.py` | Four tools, each chosen to make a specific failure observable |
| `agent.py` | The `StateGraph`, plus `AgentTrace` — the unit of agent evaluation |
| `scripted_model.py` | A deterministic model that emits tool calls |
| `test_langgraph.py` | Every agent pathology, reproduced without an LLM |

---

## An agent is a while-loop with a model as its exit condition

```
    START -> [agent] --(has tool calls?)--> [tools] --+
                ^                                     |
                +-------------------------------------+
                |
                +--(no tool calls)--> END
```

That is the entire mechanism. `_should_continue` is five lines:

```python
if state["steps"] >= self.max_steps:      return "end"    # circuit breaker
if last_message.tool_calls:               return "tools"  # loop again
return "end"
```

There is no planner and no supervisor. `create_react_agent(model, tools)` builds
this exact graph in one line — use it in production, but not to learn, because
the loop is the thing you need to see.

**This is why agents fail in ways pipelines cannot.** A pipeline runs a fixed
number of steps. An agent runs until a model decides to stop, so it can loop
forever, call tools in a nonsensical order, or stop too early.

---

## Outcome metrics are not enough

`core/corpus/agent_metrics.md` puts it directly: an agent that reached the right
answer after nine wasted calls and one wrong turn is indistinguishable, by
outcome, from one that went straight there.

So `run()` returns an **`AgentTrace`**, not a string — every tool call, its
arguments, its result, and the order. Three properties come free from that:

```python
trace.tool_names       # ['search_knowledge_base', 'refuse']  -> Tool Correctness
trace.refused          # True/False, no judge needed
trace.repeated_calls() # loop detection, pure comparison
trace.hit_step_limit   # did the circuit breaker fire?
```

### Design for deterministic observability

`refuse` is a **tool**, not free text. That single decision turns "did the agent
refuse?" from an LLM-judged question into a boolean. Look for this
transformation constantly — the same trick numbered the passages in lesson 02 so
citation hallucination became an integer comparison.

---

## Every agent pathology, tested without a model

| Failure | Test | How it's caught |
|---|---|---|
| Infinite loop | `test_step_limit_stops_an_agent_that_would_loop_forever` | step circuit breaker |
| Repeated identical calls | `test_loop_detection_finds_repeated_identical_calls` | signature comparison |
| Hallucinated tool name | `test_agent_survives_a_hallucinated_tool_name` | error fed back as `ToolMessage` |
| Wrong arguments | `test_agent_can_recover_from_a_bad_argument` | tool returns an error, agent retries |
| Legitimate refinement misread as a loop | `test_different_arguments_are_not_flagged_as_a_loop` | args are part of the signature |

All deterministic, all milliseconds. These are the failures that actually take
agents down, and an LLM judge is the *worst* way to catch them: slow, expensive,
and non-deterministic about something perfectly deterministic.

---

## Two gotchas worth the debugging session they cost

**1. `add_messages` de-duplicates by message id.**

If a model returns the same `AIMessage` object twice, the second does not append
— it **replaces** the first in its original position. The newest message is then
no longer last, the conditional edge reads a stale `ToolMessage`, and the loop
exits after two steps looking perfectly healthy. The symptom is an agent that
mysteriously stops early with no error.

`scripted_model.py` assigns a fresh id on every emission, and
`test_every_emitted_message_gets_a_unique_id` pins it.

**2. Tools should return errors, not raise them.**

A raised exception kills the graph. A returned error string becomes a
`ToolMessage` the agent can read and recover from — *and* it shows up in the
trajectory instead of as a stack trace. Whether the agent actually recovers is
then a behaviour you can measure.

---

## The docstring is a prompt, not documentation

What the model sees for each tool is a JSON schema built from its **name**,
**type annotations** and **docstring**. If an agent keeps picking the wrong
tool, rewriting the docstring is usually a bigger lever than changing the system
prompt — and almost nobody tries it first.

`list_documents` says *"It does NOT return document contents, so it cannot
answer a factual question on its own."* That sentence exists because agents
genuinely get stuck calling a cheap listing tool repeatedly instead of
committing to a search.

---

Next: **[04_deepeval](../04_deepeval/)** — scoring all of this with a judge, and
finding out whether the judge can be trusted.
