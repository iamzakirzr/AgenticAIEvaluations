"""
Watching a deployed agent in production -- and the limit of what that can tell you.

=============================================================================
THE UNCOMFORTABLE ANSWER, FIRST
=============================================================================
YOU CANNOT MEASURE QUALITY LIVE. There are no reference answers in production.

Every live number is a proxy. Refusal rate, citation validity, latency, tool
mix, answer length -- not one of them says an answer was correct. They say
something CHANGED. That is genuinely valuable and it is not the same thing, and
conflating the two is how a team ends up with a green dashboard above a system
that is quietly wrong.

So the honest framing, and the one to give in an interview:

    offline evaluation (04, 05)  measures QUALITY against references
    live monitoring   (this)     detects CHANGE against recent history
    the loop between them        turns live traffic into new references

The third line is the payoff. `promote_candidates` below closes it: live
traffic that looks anomalous becomes labelled golden items, which makes the
offline suite -- the only thing that can measure quality -- better every week.
An eval dataset that never grows is describing a system that no longer exists.

=============================================================================
WHY BURN RATE INSTEAD OF A THRESHOLD ALERT
=============================================================================
"Alert when the error rate exceeds 1%" fires at 3am for a two-minute blip and
stays silent through a week at 0.9% that eats your entire quarterly budget.

Error-budget burn rate fixes both. With a 99% SLO, the budget is 1% of
requests. Burn rate is how fast you are consuming it:

    burn = observed_error_rate / (1 - slo)

Burn rate 1 spends the budget exactly over the window. Burn rate 14.4 spends
a 30-day budget in two days -- that is a page. Burn rate 6 over six hours is a
ticket, not a page. Two windows, two severities: the standard multi-window
multi-burn-rate scheme, and it is the single most transferable operational
idea in this file.

Fast window catches the outage. Slow window catches the slow bleed. You need
both, and almost every home-grown alert has only the first.
=============================================================================
"""

from __future__ import annotations

import re
import statistics
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# WHAT A LIVE EVENT IS
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveEvent:
    """One served request, reduced to what can be monitored without a reference.

    Note what is ABSENT: any correctness field. There is nowhere for one to come
    from in production, and leaving a `correct: bool | None` here would
    eventually be filled in by something that guessed.
    """

    request_id: str
    timestamp: float
    latency_ms: float
    tools_used: tuple[str, ...] = ()
    server: str = "unknown"
    refused: bool = False
    blocked: bool = False
    errored: bool = False
    answer_chars: int = 0
    invalid_citations: int = 0
    top_score: float | None = None
    question: str = ""


# ---------------------------------------------------------------------------
# ERROR BUDGET AND BURN RATE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BurnAlert:
    severity: str  # "page" or "ticket"
    window_label: str
    burn_rate: float
    observed_rate: float
    detail: str


@dataclass
class ErrorBudget:
    """Multi-window, multi-burn-rate alerting on a service-level objective.

    Defaults are Google's SRE workbook numbers and they are defaults for a
    reason: 14.4x over one hour consumes 2% of a 30-day budget, which is worth
    waking someone for; 6x over six hours consumes 5%, which is worth a ticket.

    Tune them to your budget, but tune them deliberately -- most people copy
    "alert above 1%" and then wonder why the pager is useless.
    """

    slo: float = 0.99
    page_burn_rate: float = 14.4
    ticket_burn_rate: float = 6.0

    @property
    def budget(self) -> float:
        """The fraction of requests allowed to fail. 99% SLO -> 1% budget."""
        return 1.0 - self.slo

    def burn_rate(self, observed_error_rate: float) -> float:
        if self.budget <= 0:
            # A 100% SLO has no budget, so any failure is infinite burn. This
            # is a real configuration mistake, not a hypothetical: "we want
            # 100% uptime" sets an objective that can only ever be violated.
            return float("inf") if observed_error_rate > 0 else 0.0
        return observed_error_rate / self.budget

    def evaluate(self, fast_rate: float, slow_rate: float) -> BurnAlert | None:
        """Fast window pages; slow window tickets. Fast is checked first.

        The ordering matters: during a real outage BOTH windows breach, and
        paging is the correct single response. Emitting two alerts for one
        incident is how alert fatigue starts.
        """
        fast_burn = self.burn_rate(fast_rate)
        if fast_burn >= self.page_burn_rate:
            return BurnAlert(
                severity="page",
                window_label="fast",
                burn_rate=fast_burn,
                observed_rate=fast_rate,
                detail=(
                    f"burning {fast_burn:.1f}x budget: {fast_rate:.1%} errors against a "
                    f"{self.budget:.1%} budget. At this rate the window's budget is gone in "
                    f"{1 / fast_burn:.2f} of the period."
                ),
            )
        slow_burn = self.burn_rate(slow_rate)
        if slow_burn >= self.ticket_burn_rate:
            return BurnAlert(
                severity="ticket",
                window_label="slow",
                burn_rate=slow_burn,
                observed_rate=slow_rate,
                detail=(
                    f"sustained {slow_burn:.1f}x burn at {slow_rate:.1%} errors -- not an "
                    f"outage, but the budget will not last the period."
                ),
            )
        return None


# ---------------------------------------------------------------------------
# DISTRIBUTION DRIFT
# ---------------------------------------------------------------------------


def total_variation_distance(a: Counter[str], b: Counter[str]) -> float:
    """How far apart two categorical distributions are, in [0, 1].

    TVD rather than KL divergence or PSI, on purpose: KL is undefined when a
    category is missing from one side, which is EXACTLY the interesting case
    (a server stopped being used at all). TVD handles it and needs no epsilon
    fudge, which is a fudge people then forget they added.

    0.0 identical, 1.0 disjoint. Read it as "the share of traffic that would
    have to move to make the two match".
    """
    total_a, total_b = sum(a.values()), sum(b.values())
    if not total_a or not total_b:
        return 0.0
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / total_a - b[k] / total_b) for k in keys)


# ---------------------------------------------------------------------------
# THE MONITOR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveAlert:
    name: str
    severity: str
    detail: str


@dataclass
class LiveMonitor:
    """Rolling live monitoring over the deployed agent's traffic.

    Every check compares RECENT against REFERENCE within the same stream, so
    there is no fixed threshold to get wrong on day one and no re-tuning when
    the traffic mix changes. Lesson 06's `OnlineMonitor` introduced that idea
    for a RAG pipeline; this applies it to a deployed multi-tool agent, where
    the tool mix is itself a signal.

    `min_samples` exists so a quiet service does not alert on three requests.
    Firing an alert off a sample of two is how a monitor loses its audience.
    """

    window: int = 400
    min_samples: int = 40
    fast_window: int = 60
    budget: ErrorBudget = field(default_factory=ErrorBudget)
    drift_threshold: float = 0.35
    latency_factor: float = 2.0

    events: deque[LiveEvent] = field(default_factory=lambda: deque(maxlen=4000), init=False)

    # -- ingest ------------------------------------------------------------

    def record(self, event: LiveEvent) -> None:
        self.events.append(event)

    def record_response(
        self, body: dict[str, Any], question: str = "", errored: bool = False
    ) -> LiveEvent:
        """Adapt a `/v1/ask` response body (lesson 09) into a LiveEvent.

        Deliberately tolerant of missing keys: a monitor that raises on an
        unexpected payload takes down the thing it was meant to watch. A
        monitor must never be able to break the service.
        """
        answer = str(body.get("answer", ""))
        tools = tuple(body.get("tools_used", []) or ())
        servers = body.get("servers_used", []) or []
        return self._record_built(
            LiveEvent(
                request_id=str(body.get("request_id", "")),
                timestamp=time.time(),
                latency_ms=float(body.get("latency_ms", 0.0) or 0.0),
                tools_used=tools,
                server=servers[0] if servers else "none",
                refused=bool(body.get("refused", False)),
                blocked=bool(body.get("answer") == "The request could not be completed."),
                errored=errored,
                answer_chars=len(answer),
                invalid_citations=len(re.findall(r"INVALID_CITATIONS", answer)),
                question=question,
            )
        )

    def _record_built(self, event: LiveEvent) -> LiveEvent:
        self.record(event)
        return event

    # -- windows -----------------------------------------------------------

    def _recent(self, n: int) -> list[LiveEvent]:
        return list(self.events)[-n:]

    def _reference(self) -> list[LiveEvent]:
        """The half of the window BEFORE the recent half.

        Not "all history": a reference that includes the anomaly dilutes it,
        and a long-running incident slowly becomes the new normal. That is
        alert blindness with extra steps.
        """
        values = list(self.events)
        half = self.window // 2
        return values[-self.window : -half]

    # -- checks ------------------------------------------------------------

    def check(self) -> list[LiveAlert]:
        alerts: list[LiveAlert] = []
        if len(self.events) < self.min_samples:
            return alerts

        recent = self._recent(self.window // 2)
        reference = self._reference()
        if not recent or not reference:
            return alerts

        # 1. ERROR BUDGET BURN -- the one that should page.
        fast = self._recent(self.fast_window)
        fast_rate = sum(e.errored for e in fast) / len(fast)
        slow_rate = sum(e.errored for e in recent) / len(recent)
        burn = self.budget.evaluate(fast_rate, slow_rate)
        if burn:
            alerts.append(
                LiveAlert(name="error_budget_burn", severity=burn.severity, detail=burn.detail)
            )

        # 2. REFUSAL RATE, BOTH DIRECTIONS. The collapse is the dangerous one
        #    and a one-sided alert never fires on it: the system has stopped
        #    refusing and started confabulating, and every other metric looks
        #    healthier than before.
        now = sum(e.refused for e in recent) / len(recent)
        before = sum(e.refused for e in reference) / len(reference)
        if now > max(before * 2.0, 0.2):
            alerts.append(
                LiveAlert(
                    "refusal_spike",
                    "ticket",
                    f"refusals {before:.1%} -> {now:.1%}: retrieval or corpus coverage broke",
                )
            )
        elif before >= 0.1 and now < before / 3.0:
            alerts.append(
                LiveAlert(
                    "refusal_collapse",
                    "page",
                    f"refusals {before:.1%} -> {now:.1%}: the system stopped abstaining. "
                    f"Answers that used to be refusals are now confabulations, and no "
                    f"other live metric will look worse.",
                )
            )

        # 3. TOOL-MIX DRIFT. Unique to a multi-server agent: traffic silently
        #    moving from the corpus server to the web server is a behaviour
        #    change with no code change and no error.
        recent_mix = Counter(e.server for e in recent)
        reference_mix = Counter(e.server for e in reference)
        distance = total_variation_distance(recent_mix, reference_mix)
        if distance >= self.drift_threshold:
            alerts.append(
                LiveAlert(
                    "tool_mix_drift",
                    "ticket",
                    f"routing shifted (TVD {distance:.2f}): {dict(reference_mix)} -> "
                    f"{dict(recent_mix)}. Nothing errored; the agent is answering from "
                    f"somewhere else.",
                )
            )

        # 4. LATENCY. p95, not the mean -- the mean hides the tail that users
        #    actually experience and that trips client timeouts.
        recent_p95 = _percentile([e.latency_ms for e in recent], 0.95)
        reference_p95 = _percentile([e.latency_ms for e in reference], 0.95)
        if reference_p95 and recent_p95 > reference_p95 * self.latency_factor:
            alerts.append(
                LiveAlert(
                    "latency_regression",
                    "ticket",
                    f"p95 {reference_p95:.0f}ms -> {recent_p95:.0f}ms",
                )
            )

        return alerts

    # -- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """What a dashboard should show. Every field is honestly computable.

        `None` where there is no data, never 0.0. A zero on a dashboard is
        indistinguishable from "healthy" and is read that way at 3am.
        """
        events = list(self.events)
        if not events:
            return {"samples": 0, "note": "no traffic yet"}
        latencies = [e.latency_ms for e in events]
        return {
            "samples": len(events),
            "error_rate": sum(e.errored for e in events) / len(events),
            "refusal_rate": sum(e.refused for e in events) / len(events),
            "blocked_rate": sum(e.blocked for e in events) / len(events),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "p99_ms": _percentile(latencies, 0.99),
            "server_mix": dict(Counter(e.server for e in events)),
            "tool_calls_per_request": round(
                statistics.mean(len(e.tools_used) for e in events), 2
            ),
            # Named to make the limit unmissable on the dashboard itself.
            "quality_note": "no reference answers in production -- these detect CHANGE, not quality",
        }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


# ---------------------------------------------------------------------------
# CLOSING THE LOOP: live traffic -> golden items
# ---------------------------------------------------------------------------


# Below this many samples there is no distribution, so the long-tail rule is
# off entirely. See `promote_candidates`.
MIN_TAIL_SAMPLES = 20


@dataclass(frozen=True)
class Candidate:
    """A live request worth labelling by a human and adding to the golden set."""

    request_id: str
    question: str
    reason: str
    priority: int  # higher first


def promote_candidates(monitor: LiveMonitor, limit: int = 20) -> list[Candidate]:
    """Pick live requests worth a human label. THE most valuable loop you can build.

    Selection is deliberately biased towards the INFORMATIVE, not the
    representative. Sampling traffic uniformly mostly gives you more of what you
    already handle: cheap to label, near-zero information. These four signals
    pick the requests that would change a score:

      blocked     the guard fired -- either a real attack or a false positive,
                  and you cannot tell which without looking
      refused     either correct abstention or a coverage gap, and the two are
                  indistinguishable from the outside
      errored     a failure you have no test for
      long tail   unusually long answers correlate with rambling and with
                  hedging, both of which score badly offline

    Every candidate still needs a HUMAN label. Auto-promoting a model's own
    output into the reference set is how a system learns to grade its own
    homework, and it fails silently and permanently.
    """
    events = list(monitor.events)
    if not events:
        return []

    # MIN_TAIL_SAMPLES exists because a percentile over a tiny sample is the
    # sample. With one request, that request IS the p95, so every answer on a
    # quiet service gets promoted as "unusually long" -- found by
    # test_promotion_reads_the_monitor_the_gateway_filled failing. Below the
    # floor there is no tail to speak of, so the rule is simply off.
    lengths = [e.answer_chars for e in events if e.answer_chars]
    long_cut = (
        _percentile(lengths, 0.95)
        if len(lengths) >= MIN_TAIL_SAMPLES
        else float("inf")
    )

    candidates: list[Candidate] = []
    for event in events:
        if not event.question:
            continue
        if event.blocked:
            candidates.append(Candidate(event.request_id, event.question, "output guard blocked", 4))
        elif event.errored:
            candidates.append(Candidate(event.request_id, event.question, "request errored", 3))
        elif event.refused:
            candidates.append(
                Candidate(event.request_id, event.question, "refused -- coverage gap?", 2)
            )
        # STRICTLY greater than the p95, not >=. With a flat distribution the
        # p95 equals the common value, and >= then promotes the entire
        # population as "unusually long". "Unusual" has to mean longer than the
        # cut, never equal to it.
        elif event.answer_chars > long_cut:
            candidates.append(
                Candidate(event.request_id, event.question, "unusually long answer", 1)
            )

    # Deduplicate on the QUESTION, not the request id. Popular questions
    # otherwise flood the queue and you label the same thing forty times.
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: -c.priority):
        key = candidate.question.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique[:limit]


__all__ = [
    "MIN_TAIL_SAMPLES",
    "BurnAlert",
    "Candidate",
    "ErrorBudget",
    "LiveAlert",
    "LiveEvent",
    "LiveMonitor",
    "promote_candidates",
    "total_variation_distance",
]
