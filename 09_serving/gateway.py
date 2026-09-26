"""
Exposing an agent to the public -- the guards, and why each one exists.

=============================================================================
THE UNCOMFORTABLE ANSWER, FIRST
=============================================================================
"Expose the agent publicly" is not a deployment task. A public agent endpoint
is a REMOTE TOOL EXECUTION SERVICE, driven by untrusted text, billed to you.

Concretely, a stranger with your URL gets:

  * arbitrary execution of every tool you bound, with arguments they influence
    -- and with MCP (lesson 08) those tools come from a process you may not own
  * your inference spend, with no natural ceiling; a loop that calls a tool
    twenty times costs twenty times as much and looks like one request
  * a prompt-injection surface that includes YOUR OWN RETRIEVED DOCUMENTS, not
    just the user's message. Anything the agent reads can carry instructions
  * an oracle for extracting your system prompt, your corpus and your tool list

None of that is fixed by adding an API key. An API key tells you WHO is doing
it, which is necessary and nowhere near sufficient.

The guards in this file are the minimum. They are not a security review, and
this endpoint should sit behind a real gateway (WAF, mTLS or OAuth, egress
control) before it is genuinely public. Say that out loud in an interview; the
candidates who say "I added an API key and rate limiting" and stop have not
thought about it.

=============================================================================
WHAT THIS FILE ACTUALLY IMPLEMENTS
=============================================================================
    authentication      per-key identity, constant-time comparison
    authorisation       a TOOL ALLOWLIST per key -- keys differ in power
    rate limiting       token bucket per key, with the honest caveat below
    spend limiting      a per-key request budget, because rate != cost
    request limits      body size and a hard timeout
    output guards       PII redaction and system-prompt leak detection
    observability       a request id on every response, including errors

=============================================================================
THE CAVEAT THAT INVALIDATES MOST TUTORIAL RATE LIMITERS
=============================================================================
The limiter here is IN-PROCESS. Run uvicorn with `--workers 4` and you have
four independent limiters, so your "60 per minute" is actually 240 per minute.
Autoscale to eight pods and it is 1920. Nearly every FastAPI rate-limiting
example has this bug and none of them mention it.

`TokenBucket.shared_state_warning` exists so the deployment cannot pretend
otherwise, and `test_in_process_limiting_is_per_worker` measures the
multiplication rather than describing it. The real fix is shared state (Redis)
or limiting at the gateway. Knowing WHY is the point.
=============================================================================
"""

from __future__ import annotations

import hmac
import secrets
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_ROOT), str(_ROOT / "02_langchain")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from production_pipeline import redact

# ---------------------------------------------------------------------------
# IDENTITY AND AUTHORISATION
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApiKey:
    """One caller. Note that a key carries POWER, not just identity.

    `allowed_tools` is the field people leave out, and it is the one that
    contains a compromise. A leaked read-only key is an incident; a leaked key
    that can call `web_fetch_url` is an SSRF proxy with your egress IP.
    """

    key: str
    label: str
    allowed_tools: frozenset[str]
    requests_per_minute: int = 60
    daily_request_budget: int = 1000


def verify(presented: str, expected: str) -> bool:
    """Constant-time comparison.

    `==` on secrets leaks length and prefix through timing. The attack is
    finicky over the internet and trivial over a LAN or from a co-tenant, and
    the fix costs one function call, so there is no argument for `==` here.
    """
    return hmac.compare_digest(presented.encode(), expected.encode())


class KeyStore:
    """In-memory key registry. A real one reads hashed keys from a database.

    Keys are never logged. `redact_key` is what goes in a log line: enough to
    correlate requests from one caller, not enough to replay them.
    """

    def __init__(self, keys: list[ApiKey] | None = None) -> None:
        self._keys = {k.key: k for k in (keys or [])}

    def add(self, key: ApiKey) -> None:
        self._keys[key.key] = key

    def authenticate(self, presented: str | None) -> ApiKey | None:
        if not presented:
            return None
        # Iterate ALL keys with constant-time compares rather than a dict
        # lookup: a dict hit/miss is itself a timing signal, and the key set is
        # small. At thousands of keys, store a hash and look that up instead.
        for key in self._keys.values():
            if verify(presented, key.key):
                return key
        return None

    @staticmethod
    def redact_key(key: str) -> str:
        return f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "***"


def generate_key(prefix: str = "sk_eval") -> str:
    """`secrets`, never `random`. `random` is seeded and predictable."""
    return f"{prefix}_{secrets.token_urlsafe(24)}"


# ---------------------------------------------------------------------------
# RATE LIMITING
# ---------------------------------------------------------------------------


@dataclass
class TokenBucket:
    """Classic token bucket: `capacity` tokens, refilled at `rate` per second.

    Chosen over a fixed window because a fixed window permits a double burst
    across the boundary: 60 requests at 11:59:59 and 60 more at 12:00:00 is 120
    in one second while never breaching "60 per minute". A bucket cannot do
    that -- the tokens are simply not there.
    """

    capacity: int
    refill_per_second: float
    tokens: float = field(init=False)
    updated_at: float = field(default_factory=time.monotonic)

    # Set on every instance so a deployment cannot forget. See the module
    # docstring: in-process limiting multiplies by your worker count.
    shared_state_warning: str = (
        "in-process limiter: effective limit multiplies by worker and replica count"
    )

    def __post_init__(self) -> None:
        self.tokens = float(self.capacity)

    def allow(self, now: float | None = None, cost: float = 1.0) -> bool:
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.updated_at)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_per_second)
        self.updated_at = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def retry_after_seconds(self, cost: float = 1.0) -> float:
        """Tell the caller WHEN to come back. A bare 429 invites a tight retry
        loop, which is how a rate limit turns into a self-inflicted DDoS."""
        if self.tokens >= cost or self.refill_per_second <= 0:
            return 0.0
        return (cost - self.tokens) / self.refill_per_second


@dataclass
class SpendLimiter:
    """A per-key request budget, separate from the rate limit.

    Rate is not cost. A caller at one request per minute, all day, is inside
    every rate limit and can still spend more than a burst of fifty. Agents
    make this worse: one request can fan out into a dozen model calls, so the
    request count understates the bill.

    Budget is the control that actually caps the invoice.
    """

    budgets: dict[str, int] = field(default_factory=dict)
    spent: dict[str, int] = field(default_factory=dict)

    def charge(self, key_label: str, units: int = 1) -> bool:
        budget = self.budgets.get(key_label)
        if budget is None:
            return True
        used = self.spent.get(key_label, 0)
        if used + units > budget:
            return False
        self.spent[key_label] = used + units
        return True

    def remaining(self, key_label: str) -> int | None:
        budget = self.budgets.get(key_label)
        return None if budget is None else budget - self.spent.get(key_label, 0)


# ---------------------------------------------------------------------------
# OUTPUT GUARDS
# ---------------------------------------------------------------------------

# Distinctive phrases from the system prompt. If they come back out, the model
# has been talked into reciting its instructions.
LEAK_CANARIES: tuple[str, ...] = (
    "Tool choice rules",
    "internal knowledge base",
    "Do not answer from your own knowledge",
)


@dataclass
class GuardResult:
    text: str
    pii_redacted: dict[str, int] = field(default_factory=dict)
    leaked: list[str] = field(default_factory=list)
    blocked: bool = False
    reason: str = ""


def guard_output(text: str, canaries: tuple[str, ...] = LEAK_CANARIES) -> GuardResult:
    """Redact PII, and block an answer that is reciting the system prompt.

    ORDER MATTERS AND IS EASY TO GET WRONG: redact first, then leak-check.
    Redaction rewrites the text, and a canary check on the pre-redaction string
    would pass a response whose redacted form still leaks -- and vice versa. We
    check the text we are actually going to send.

    Blocking a leak rather than redacting it is deliberate. A response that
    recites the system prompt is evidence the model has been successfully
    steered, and the rest of that response is not trustworthy either.
    """
    redaction = redact(text)
    cleaned = redaction.text
    found = [c for c in canaries if c.lower() in cleaned.lower()]
    if found:
        return GuardResult(
            text="The request could not be completed.",
            pii_redacted=dict(redaction.found),
            leaked=found,
            blocked=True,
            reason="system prompt leak",
        )
    return GuardResult(text=cleaned, pii_redacted=dict(redaction.found))


def tool_calls_allowed(called: list[str], key: ApiKey) -> list[str]:
    """Which of the tools the agent actually used were NOT permitted?

    This is a DETECTIVE control, run after the fact, and it is worth having in
    addition to the preventive one (binding only allowed tools). If it ever
    returns non-empty, the preventive control has a hole, and you want to know
    that from your own alert rather than from a bill.
    """
    return [name for name in called if name not in key.allowed_tools]


# ---------------------------------------------------------------------------
# REQUEST CONTEXT
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 2000


@dataclass
class RequestContext:
    """Everything one request needs to be traced, billed and explained.

    `request_id` goes on EVERY response including errors. Without it a user
    reporting "it failed at about three o'clock" is unactionable; with it they
    quote an id and you have the trace. It is the cheapest operability feature
    there is and it is routinely missing.
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    key_label: str = "anonymous"
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.started_at) * 1000


def validate_question(question: str) -> str | None:
    """Return an error string, or None if acceptable.

    The length cap is not politeness. An unbounded prompt is an unbounded bill,
    and the largest input you accept sets the largest single request you can be
    charged for. Enforce it BEFORE the model sees the text.
    """
    if not question or not question.strip():
        return "question must not be empty"
    if len(question) > MAX_QUESTION_CHARS:
        return f"question exceeds {MAX_QUESTION_CHARS} characters"
    return None


__all__ = [
    "LEAK_CANARIES",
    "MAX_QUESTION_CHARS",
    "ApiKey",
    "GuardResult",
    "KeyStore",
    "RequestContext",
    "SpendLimiter",
    "TokenBucket",
    "generate_key",
    "guard_output",
    "tool_calls_allowed",
    "validate_question",
    "verify",
]
