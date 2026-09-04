"""
core.resilience -- the failure handling every production LLM system needs.

=============================================================================
WHY A LEARNING REPO HAS A RESILIENCE MODULE
=============================================================================
The lessons so far assumed calls succeed. Production does not. An LLM call is
a network call to a service that is slow, occasionally overloaded, and priced
per token, which produces four failure modes a plain `llm.invoke()` handles
none of:

  1. TRANSIENT FAILURE      the server returns 503 or the connection resets.
                            Retrying works. Not retrying loses a request.

  2. HANGING CALL           the server accepts the request and never answers.
                            No exception is raised -- your worker just stops.
                            A retry without a timeout does not help, because
                            the first attempt never finishes.

  3. SUSTAINED OUTAGE       the model server is down. Retrying every request
                            turns one outage into a queue of stuck workers and
                            makes recovery slower. This is where a circuit
                            breaker earns its place.

  4. COST BLOWOUT           an eval loop over 10,000 items, or an agent that
                            loops, quietly spends real money. A budget is the
                            only mechanism that reliably stops it.

Everything here is dependency-free and synchronous, because the point is that
you can read it. In a real service you would likely use `tenacity` for retries
(ragas already does -- see RunConfig.max_retries) and your platform's circuit
breaker. Knowing what they do is what stops you misconfiguring them.

=============================================================================
THE SINGLE MOST IMPORTANT IDEA IN THIS FILE
=============================================================================
    RETRY ONLY WHAT IS RETRYABLE.

A 503 is worth retrying. A 401, a malformed prompt, or a schema-validation
failure is not -- retrying those burns time and money to fail identically three
more times, and it hides the real error behind a timeout.

`retry()` therefore takes an explicit `retry_on` predicate rather than blindly
catching Exception. Deciding what is retryable is a design decision, not a
default.
=============================================================================
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


# ===========================================================================
# EXCEPTIONS
# ===========================================================================


class BudgetExceeded(RuntimeError):
    """Raised when a run exceeds its token, cost, call or time allowance."""


class CircuitOpen(RuntimeError):
    """Raised when the circuit breaker is open and refusing calls fast."""


class OperationTimeout(TimeoutError):
    """Raised when a single attempt exceeded its deadline."""


# ===========================================================================
# RETRY
# ===========================================================================


def default_retryable(exc: BaseException) -> bool:
    """A conservative default: retry transport-ish failures only.

    Deliberately does NOT retry ValueError/TypeError/KeyError, which almost
    always mean a bug in your code or a malformed response -- retrying those
    just fails three times more slowly.
    """
    retryable_names = {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "ConnectionError",
        "TimeoutError",
        "OperationTimeout",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailable",
    }
    return type(exc).__name__ in retryable_names


@dataclass
class RetryPolicy:
    """Exponential backoff with jitter.

    WHY JITTER IS NOT OPTIONAL: without it, N workers that fail at the same
    moment all retry at the same moment, and keep colliding. That is the
    "thundering herd", and it converts a brief blip into a sustained outage.
    Randomising each delay spreads the retries out.

    WHY A CAP: unbounded exponential backoff eventually sleeps for minutes.
    `max_delay` keeps the worst case bounded.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter: float = 0.25  # +/- 25% randomisation
    retry_on: Callable[[BaseException], bool] = default_retryable

    def delay_for(self, attempt: int) -> float:
        """Delay before ``attempt`` (1-based: attempt 2 is the first retry)."""
        raw = min(self.base_delay * (2 ** (attempt - 2)), self.max_delay)
        if self.jitter <= 0:
            return raw
        spread = raw * self.jitter
        return max(0.0, raw + random.uniform(-spread, spread))


@dataclass
class RetryStats:
    attempts: int = 0
    retries: int = 0
    total_sleep: float = 0.0
    last_error: str = ""


def retry(
    fn: Callable[[], T],
    policy: RetryPolicy | None = None,
    stats: RetryStats | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` with retries, returning its result or re-raising.

    ``sleep`` is injectable so tests can run instantly instead of actually
    waiting -- otherwise a retry test takes seconds and everybody deletes it.
    """
    policy = policy or RetryPolicy()
    stats = stats if stats is not None else RetryStats()

    for attempt in range(1, policy.max_attempts + 1):
        stats.attempts = attempt
        try:
            return fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            stats.last_error = f"{type(exc).__name__}: {exc}"

            # Not retryable, or out of attempts -> give up immediately.
            if not policy.retry_on(exc) or attempt == policy.max_attempts:
                raise

            stats.retries += 1
            pause = policy.delay_for(attempt + 1)
            stats.total_sleep += pause
            sleep(pause)

    raise AssertionError("unreachable")  # pragma: no cover


# ===========================================================================
# TIMEOUT
# ===========================================================================


def call_with_timeout(
    fn: Callable[[], T],
    seconds: float,
    on_timeout: str = "operation exceeded its deadline",
) -> T:
    """Run ``fn`` in a worker thread and give up waiting after ``seconds``.

    IMPORTANT AND OFTEN MISUNDERSTOOD: this bounds how long YOU WAIT. It does
    not kill the underlying work -- Python cannot forcibly stop a thread. The
    HTTP request keeps running until the socket closes.

    That is still the right trade for an LLM call: the alternative is a worker
    blocked forever on a server that will never answer. But it means a timeout
    is not free, and setting it very low while the server keeps working means
    you pay for tokens you never read.

    The genuinely correct fix is to pass a timeout to the HTTP client as well
    (this repo does: `settings.request_timeout` reaches httpx and Ollama). This
    function is the outer guard for everything that does not accept one.
    """
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FuturesTimeout

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=seconds)
        except FuturesTimeout as exc:
            # cancel() cannot stop a running thread; we mark it and move on.
            future.cancel()
            raise OperationTimeout(f"{on_timeout} ({seconds}s)") from exc


# ===========================================================================
# CIRCUIT BREAKER
# ===========================================================================


@dataclass
class CircuitBreaker:
    """Stop hammering a service that is clearly down.

    THREE STATES:

        CLOSED     normal. Calls pass through. Consecutive failures are counted.
        OPEN       after `failure_threshold` consecutive failures, every call is
                   rejected IMMEDIATELY with CircuitOpen -- no waiting, no
                   timeout, no retry storm.
        HALF_OPEN  after `reset_after` seconds, ONE probe call is allowed. If it
                   succeeds the circuit closes; if it fails it re-opens.

    WHY THIS MATTERS MORE THAN IT LOOKS: during an outage, retrying every
    request means every worker is asleep in a backoff loop. Requests queue,
    memory grows, and when the service recovers it is immediately hit by the
    entire backlog -- so it falls over again. Failing fast keeps workers free
    and lets recovery actually happen.

    The cost is that you reject requests that might have succeeded. That is the
    trade, and it is usually worth it.
    """

    failure_threshold: int = 5
    reset_after: float = 30.0

    _failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _clock: Callable[[], float] = field(default=time.monotonic, init=False)

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at >= self.reset_after:
            return "half_open"
        return "open"

    def call(self, fn: Callable[[], T]) -> T:
        state = self.state
        if state == "open":
            raise CircuitOpen(
                f"circuit is open after {self._failures} consecutive failures; "
                f"retrying in {self.reset_after - (self._clock() - self._opened_at):.1f}s"
            )

        try:
            result = fn()
        except BaseException:
            self.record_failure()
            raise

        self.record_success()
        return result

    def record_success(self) -> None:
        """Reset. A single success in half-open closes the circuit."""
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.failure_threshold:
            self._opened_at = self._clock()


# ===========================================================================
# BUDGET
# ===========================================================================


@dataclass
class Budget:
    """A hard ceiling on what one run may consume.

    An eval sweep, an agent loop or a batch job can all spend far more than
    intended, and the failure is silent -- you find out from the bill or from a
    job that has been running for six hours.

    A budget converts that into a loud, early exception. Check it BEFORE each
    call (`check()`) and record after (`record()`).

    max_cost is in dollars. Set `input_cost_per_1k` / `output_cost_per_1k` to
    your provider's rates; leaving them at 0.0 (correct for local Ollama, where
    inference is free) makes cost accounting a no-op while token and call
    limits still apply.
    """

    max_calls: int | None = None
    max_tokens: int | None = None
    max_cost: float | None = None
    max_seconds: float | None = None

    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0

    calls: int = field(default=0, init=False)
    input_tokens: int = field(default=0, init=False)
    output_tokens: int = field(default=0, init=False)
    started: float = field(default_factory=time.monotonic, init=False)

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        return (
            self.input_tokens / 1000 * self.input_cost_per_1k
            + self.output_tokens / 1000 * self.output_cost_per_1k
        )

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        """Raise BudgetExceeded if any ceiling has been reached.

        Call this BEFORE issuing a request. Checking afterwards still spends
        the money you were trying not to spend.
        """
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"call budget exhausted ({self.calls}/{self.max_calls})")
        if self.max_tokens is not None and self.tokens >= self.max_tokens:
            raise BudgetExceeded(f"token budget exhausted ({self.tokens}/{self.max_tokens})")
        if self.max_cost is not None and self.cost >= self.max_cost:
            raise BudgetExceeded(f"cost budget exhausted (${self.cost:.4f}/${self.max_cost:.4f})")
        if self.max_seconds is not None and self.elapsed >= self.max_seconds:
            raise BudgetExceeded(
                f"time budget exhausted ({self.elapsed:.1f}s/{self.max_seconds:.1f}s)"
            )

    def record(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Account for one completed call."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    def report(self) -> str:
        parts = [f"{self.calls} calls", f"{self.tokens} tokens"]
        if self.input_cost_per_1k or self.output_cost_per_1k:
            parts.append(f"${self.cost:.4f}")
        parts.append(f"{self.elapsed:.1f}s")
        return " | ".join(parts)


# ===========================================================================
# BOUNDED PARALLELISM
# ===========================================================================


def map_bounded(
    fn: Callable[[T], object],
    items: Iterable[T],
    max_workers: int = 4,
    stop_on_error: bool = False,
) -> list[tuple[T, object | None, BaseException | None]]:
    """Run ``fn`` over ``items`` with at most ``max_workers`` in flight.

    Returns ``(item, result, error)`` triples IN INPUT ORDER, so a failure in
    one item does not lose the results of the others and you can still line
    results up against the dataset that produced them.

    WHY NOT JUST ThreadPoolExecutor.map: `.map()` re-raises the first exception
    and discards every other result. For an evaluation sweep that is the wrong
    behaviour -- one judge failure on item 7 should not destroy the other 47
    scores. Returning errors alongside results is what lets the caller report
    "45 scored, 3 judge failures" instead of nothing at all.

    WHY BOUND IT AT ALL: a local Ollama server processes requests roughly
    serially anyway; firing 200 concurrent requests just builds a queue,
    inflates latency and can exhaust file descriptors. Against a hosted API it
    gets you rate limited. RAGAS defaults to max_workers=16 for the same reason.
    """
    from concurrent.futures import ThreadPoolExecutor

    items = list(items)
    results: list[tuple[T, object | None, BaseException | None]] = [
        (item, None, None) for item in items
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        for future, index in futures.items():
            try:
                results[index] = (items[index], future.result(), None)
            except BaseException as exc:  # noqa: BLE001 - recorded, not swallowed
                results[index] = (items[index], None, exc)
                if stop_on_error:
                    # Best effort: cancel work that has not started yet.
                    for pending in futures:
                        pending.cancel()
                    break

    return results
