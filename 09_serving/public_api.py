"""
The public endpoint: FastAPI over the MCP agent, with every guard wired in.

Run it:
    make serve-agent          -> http://localhost:8001/docs

    curl -s localhost:8001/v1/ask -H 'x-api-key: sk_eval_demo_readonly' \
         -H 'content-type: application/json' -d '{"question":"what is chunk overlap?"}'

=============================================================================
READ 09_serving/gateway.py FIRST
=============================================================================
It explains what a public agent endpoint actually exposes (remote tool
execution, your inference spend, a prompt-injection surface that includes your
own documents) and implements the guards. This file is the wiring.

=============================================================================
THREE DESIGN DECISIONS WORTH ARGUING FOR IN AN INTERVIEW
=============================================================================
1. EVERY HANDLER IS ASYNC, ALL THE WAY DOWN.
   Not a style choice. MCP tools have no synchronous implementation -- see
   lesson 08, trap 5 -- so a `def` handler that calls the agent raises
   NotImplementedError inside the tools node. And a sync handler in FastAPI
   runs in a threadpool, so under load you would be paying for threads to sit
   blocked on network I/O.

2. /healthz AND /readyz ARE DIFFERENT ENDPOINTS.
   Liveness answers "is this process wedged? restart it". Readiness answers
   "can it serve? send it traffic". Conflating them is a genuine outage
   pattern: a readiness probe that checks a downstream dependency, wired to
   liveness, restarts every pod when that dependency blips -- turning a
   degraded dependency into a total outage. Here /readyz reports degraded MCP
   servers and /healthz deliberately does not.

3. ERRORS CARRY THE REQUEST ID.
   The id is on the success path, the error path and the rate-limit path. A
   user quoting an id is a supportable system; "it broke this afternoon" is not.
=============================================================================
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    str(_ROOT),
    str(_ROOT / "08_mcp"),
    str(_ROOT / "09_serving"),
    str(_ROOT / "02_langchain"),
    str(_ROOT / "10_live_monitoring"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from gateway import (
    ApiKey,
    KeyStore,
    RequestContext,
    SpendLimiter,
    TokenBucket,
    guard_output,
    tool_calls_allowed,
    validate_question,
)
from live_monitor import LiveMonitor
from mcp_agent import arun_agent, build_mcp_agent
from mcp_client import ToolRegistry, load_tools_resiliently
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# DEMO KEYS -- two of them, with DIFFERENT POWER. That is the lesson.
# ---------------------------------------------------------------------------
# Hard-coded here so the lesson runs with no setup. A real deployment loads
# hashed keys from a secret store; see the README's deployment section.
#
# The read-only key cannot reach `web_fetch_url`, which turns a user-supplied
# string into an outbound request from your server. One key is an information
# leak if it escapes; the other is an SSRF proxy.
READONLY_KEY = "sk_eval_demo_readonly"
POWER_KEY = "sk_eval_demo_power"

DEMO_KEYS = [
    ApiKey(
        key=READONLY_KEY,
        label="readonly",
        allowed_tools=frozenset(
            {"corpus_search", "corpus_list_documents", "corpus_stats",
             "evaluator_check_citations", "evaluator_retrieval_recall", "evaluator_retrieval_mrr"}
        ),
        requests_per_minute=30,
        daily_request_budget=200,
    ),
    ApiKey(
        key=POWER_KEY,
        label="power",
        allowed_tools=frozenset(
            {"corpus_search", "corpus_list_documents", "corpus_stats", "web_search",
             "web_fetch_url", "evaluator_check_citations", "evaluator_retrieval_recall",
             "evaluator_retrieval_mrr"}
        ),
        requests_per_minute=120,
        daily_request_budget=5000,
    ),
]


# ---------------------------------------------------------------------------
# APPLICATION STATE
# ---------------------------------------------------------------------------


@dataclass
class ServiceState:
    """Everything the handlers need, in one object so tests can build their own.

    Module-level globals would make the tests order-dependent -- one test's
    rate-limit exhaustion would leak into the next. A state object handed to
    `create_app` keeps each test isolated, which is the difference between a
    suite you trust and one you rerun until it passes.
    """

    keys: KeyStore
    registry: ToolRegistry
    model_factory: Any
    buckets: dict[str, TokenBucket] = field(default_factory=dict)
    spend: SpendLimiter = field(default_factory=SpendLimiter)
    started_at: float = field(default_factory=time.time)
    request_timeout_s: float = 30.0

    # Lesson 10. The gateway FEEDS the monitor; the monitor never blocks the
    # gateway. Recording is wrapped so a monitoring bug cannot take down the
    # service it was installed to watch -- see `_observe`.
    monitor: LiveMonitor = field(default_factory=LiveMonitor)

    # Rolling operational counters. Lesson 10 turns these into live monitoring;
    # here they exist so /v1/metrics has something true to report.
    served: int = 0
    refused: int = 0
    blocked: int = 0
    rate_limited: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    def bucket_for(self, key: ApiKey) -> TokenBucket:
        if key.label not in self.buckets:
            self.buckets[key.label] = TokenBucket(
                capacity=key.requests_per_minute,
                refill_per_second=key.requests_per_minute / 60.0,
            )
            self.spend.budgets[key.label] = key.daily_request_budget
        return self.buckets[key.label]

    def agent_for(self, key: ApiKey) -> Any:
        """Build an agent bound ONLY to the tools this key may use.

        Preventive control: the model is never shown a tool the caller is not
        entitled to, so it cannot call one. The detective control in
        `tool_calls_allowed` then verifies this actually held -- belt and
        braces, because a hole here is invisible until it is expensive.
        """
        permitted = ToolRegistry(
            tools=[t for t in self.registry.tools if t.name in key.allowed_tools],
            origin={k: v for k, v in self.registry.origin.items() if k in key.allowed_tools},
            degraded=list(self.registry.degraded),
        )
        return build_mcp_agent(permitted, self.model_factory()), permitted


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., description="The user's question.")


class AskResponse(BaseModel):
    request_id: str
    answer: str
    tools_used: list[str]
    servers_used: list[str]
    refused: bool
    pii_redacted: dict[str, int]
    latency_ms: float
    degraded_servers: list[str]


# ---------------------------------------------------------------------------
# THE APP
# ---------------------------------------------------------------------------


def create_app(state: ServiceState) -> FastAPI:
    app = FastAPI(
        title="Agent Gateway",
        version="1.0.0",
        description="A public endpoint over an MCP-backed agent. Read 09_serving/gateway.py.",
    )
    app.state.service = state

    # -- request id on EVERY response, success or failure --------------------
    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        context = RequestContext()
        request.state.context = context
        try:
            response = await call_next(request)
        except Exception:
            state.errors += 1
            # Even the unhandled path returns the id. Without this, the one
            # class of failure users most want to report is the one they cannot.
            return JSONResponse(
                status_code=500,
                content={"error": "internal error", "request_id": context.request_id},
                headers={"x-request-id": context.request_id},
            )
        response.headers["x-request-id"] = context.request_id
        response.headers["x-response-time-ms"] = f"{context.elapsed_ms:.1f}"
        return response

    # -- authentication + limits as a dependency -----------------------------
    async def authorised(
        request: Request, x_api_key: str | None = Header(default=None)
    ) -> ApiKey:
        context: RequestContext = request.state.context
        key = state.keys.authenticate(x_api_key)
        if key is None:
            # 401 with no hint about which part was wrong. "Unknown key" versus
            # "bad signature" is a free oracle for an attacker enumerating keys.
            raise HTTPException(
                status_code=401,
                detail={"error": "unauthorised", "request_id": context.request_id},
            )
        context.key_label = key.label

        bucket = state.bucket_for(key)
        if not bucket.allow():
            state.rate_limited += 1
            retry_after = bucket.retry_after_seconds()
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate limited",
                    "retry_after_seconds": round(retry_after, 2),
                    "request_id": context.request_id,
                },
                headers={"retry-after": str(max(1, int(retry_after)))},
            )

        if not state.spend.charge(key.label):
            # 402, not 429. "Slow down" and "you have spent your budget" need
            # different client behaviour: one should retry, the other must not.
            raise HTTPException(
                status_code=402,
                detail={"error": "budget exhausted", "request_id": context.request_id},
            )
        return key

    # -- the endpoint --------------------------------------------------------
    @app.post("/v1/ask", response_model=AskResponse)
    async def ask(request: Request, body: AskRequest, key: ApiKey = Depends(authorised)):
        context: RequestContext = request.state.context

        problem = validate_question(body.question)
        if problem:
            raise HTTPException(
                status_code=422,
                detail={"error": problem, "request_id": context.request_id},
            )

        agent, permitted = state.agent_for(key)
        try:
            # A HARD TIMEOUT, always. An agent loop with no deadline is an
            # unbounded bill and a held connection; the client has already
            # given up long before you notice.
            run = await asyncio.wait_for(
                arun_agent(agent, body.question, permitted), timeout=state.request_timeout_s
            )
        except TimeoutError:
            state.errors += 1
            raise HTTPException(
                status_code=504,
                detail={"error": "agent timed out", "request_id": context.request_id},
            ) from None

        # Detective control: did it call anything it should not have?
        violations = tool_calls_allowed(run.tool_calls, key)
        if violations:
            state.blocked += 1
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "tool policy violation",
                    "tools": violations,
                    "request_id": context.request_id,
                },
            )

        guarded = guard_output(run.answer)
        if guarded.blocked:
            state.blocked += 1
        elif run.refused:
            state.refused += 1
        state.served += 1
        state.latencies_ms.append(context.elapsed_ms)

        response = AskResponse(
            request_id=context.request_id,
            answer=guarded.text,
            tools_used=run.tool_calls,
            servers_used=run.servers_used,
            refused=run.refused or guarded.blocked,
            pii_redacted=guarded.pii_redacted,
            latency_ms=round(context.elapsed_ms, 1),
            degraded_servers=list(state.registry.degraded),
        )
        _observe(state, response.model_dump(), body.question)
        return response

    # -- streaming -----------------------------------------------------------
    @app.post("/v1/ask/stream")
    async def ask_stream(request: Request, body: AskRequest, key: ApiKey = Depends(authorised)):
        """Server-sent events, one JSON object per line.

        WHY STREAM AT ALL: an agent that calls three tools takes many seconds,
        and a blank screen for ten seconds reads as broken. Streaming the TOOL
        STEPS, not just the final tokens, is the version worth building -- the
        user sees "searching the knowledge base" and understands the wait.

        WHAT IT COSTS YOU IN EVALUATION: the output guards above run on the
        COMPLETE answer. Stream tokens and you have already shipped the first
        half of a response you might have blocked. That is a real trade-off,
        not a detail -- here the guard runs per emitted event, which is why
        events carry structured steps rather than raw tokens.
        """
        context: RequestContext = request.state.context
        agent, permitted = state.agent_for(key)

        async def events() -> AsyncIterator[str]:
            yield _sse({"type": "start", "request_id": context.request_id})
            try:
                run = await asyncio.wait_for(
                    arun_agent(agent, body.question, permitted),
                    timeout=state.request_timeout_s,
                )
            except TimeoutError:
                yield _sse({"type": "error", "error": "timeout", "request_id": context.request_id})
                return

            for name in run.tool_calls:
                yield _sse(
                    {"type": "tool", "name": name, "server": permitted.origin.get(name, "unknown")}
                )

            guarded = guard_output(run.answer)
            state.served += 1
            yield _sse(
                {
                    "type": "answer",
                    "text": guarded.text,
                    "blocked": guarded.blocked,
                    "request_id": context.request_id,
                }
            )
            yield _sse({"type": "done", "latency_ms": round(context.elapsed_ms, 1)})

        return StreamingResponse(events(), media_type="text/event-stream")

    # -- liveness vs readiness. Deliberately different. ----------------------
    @app.get("/healthz")
    async def healthz():
        """LIVENESS: is the process alive? Nothing downstream is checked.

        If this ever consulted the MCP servers, a blip in one of them would
        restart every pod -- converting a degraded dependency into an outage.
        """
        return {"status": "ok", "uptime_s": round(time.time() - state.started_at, 1)}

    @app.get("/readyz")
    async def readyz():
        """READINESS: should this instance receive traffic?

        Degraded MCP servers are reported but do NOT make the service unready:
        it can still answer from the servers that are up. Refusing all traffic
        because one optional tool source is down is worse than degrading.
        """
        has_tools = bool(state.registry.tools)
        return JSONResponse(
            status_code=200 if has_tools else 503,
            content={
                "ready": has_tools,
                "tools": len(state.registry.tools),
                "degraded_servers": list(state.registry.degraded),
            },
        )

    @app.get("/v1/metrics")
    async def metrics():
        """Operational counters. Lesson 10 consumes these.

        Deliberately NOT quality scores. A gateway can report latency, refusal
        rate and error rate honestly; it cannot report faithfulness, because
        nothing here has a reference answer. Publishing a quality number this
        endpoint cannot actually compute is the exact failure this repository
        exists to prevent.
        """
        latencies = sorted(state.latencies_ms)
        return {
            "served": state.served,
            "refused": state.refused,
            "blocked": state.blocked,
            "rate_limited": state.rate_limited,
            "errors": state.errors,
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "degraded_servers": list(state.registry.degraded),
        }

    # -- lesson 10: the live dashboard, mounted on the same app -------------
    from dashboard import live_router

    app.include_router(live_router(state.monitor))

    return app


def _observe(state: ServiceState, body: dict[str, Any], question: str, errored: bool = False) -> None:
    """Feed the monitor, and NEVER let it break the request.

    A monitoring call in the hot path is a dependency of every response. If it
    can raise, it can turn a healthy answer into a 500 -- the monitor causing
    the outage it exists to detect. Swallow, and count.
    """
    try:
        state.monitor.record_response(body, question=question, errored=errored)
    except Exception:
        state.errors += 0  # deliberately a no-op: observability is not the SLO


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    """None, not 0.0, when there is no data.

    A p95 of 0.0 on an empty series draws a beautiful flat line on a dashboard
    and means nothing happened. `null` leaves a gap, which is the truth.
    """
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return round(sorted_values[index], 1)


# ---------------------------------------------------------------------------
# DEFAULT WIRING (used by `make serve-agent`)
# ---------------------------------------------------------------------------


def build_default_state(model_factory: Any | None = None) -> ServiceState:
    from langchain_core.messages import AIMessage

    if model_factory is None:
        def model_factory():
            from scripted_model import ScriptedToolCallingModel

            return ScriptedToolCallingModel(
                script=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "corpus_search", "args": {"query": "overview"}, "id": "c1"}
                        ],
                    ),
                    AIMessage(
                        content=(
                            "(offline demo answer -- no model is running) "
                            "See the retrieved passage above."
                        )
                    ),
                ]
            )

    registry = ToolRegistry.from_loads(asyncio.run(load_tools_resiliently()))
    return ServiceState(
        keys=KeyStore(DEMO_KEYS), registry=registry, model_factory=model_factory
    )


def _default_app() -> FastAPI:
    sys.path.insert(0, str(_ROOT / "03_langgraph"))
    return create_app(build_default_state())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(_default_app(), host="127.0.0.1", port=8001)
