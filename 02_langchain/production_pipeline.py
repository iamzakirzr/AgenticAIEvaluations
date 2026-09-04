"""
PRODUCTION SCENARIOS for the RAG pipeline.

=============================================================================
FOUR THINGS THE LESSON-02 PIPELINE DOES NOT SURVIVE
=============================================================================

  1. NO RELEVANCE THRESHOLD.  Vector search always returns top_k. For a
     question the corpus cannot answer it returns four irrelevant chunks with
     a straight face, and the model writes something plausible from them.
     lesson 02 defends against this with the prompt alone. In production you
     want a second, deterministic line of defence: if the best chunk scores
     below a threshold, refuse WITHOUT calling the model at all. That is both
     safer and cheaper -- a refusal costs zero tokens.

  2. NO PII HANDLING.  Users paste email addresses, card numbers and national
     insurance numbers into chat boxes. Those then travel into your prompt,
     your logs, your traces and your eval datasets. Redaction has to happen at
     the boundary, before any of that.

  3. NO FALLBACK.  When the primary model is down, the whole feature is down.
     A fallback chain degrades instead: try the good model, fall back to a
     smaller/faster one, and record which was used so your metrics do not
     silently mix two systems.

  4. NO CONCURRENCY CONTROL.  Answering 500 evaluation questions serially takes
     hours; answering them with unbounded parallelism exhausts the model server
     and gets you rate limited.

=============================================================================
THE PRINCIPLE THESE SHARE
=============================================================================
Every one of them is a DETERMINISTIC guard around a non-deterministic core.
None of them requires an LLM to decide anything. Whenever you can move a
safety property out of the model's judgement and into code, do it -- the code
version is cheaper, faster, testable, and cannot be talked out of its decision
by a prompt injection.
=============================================================================
"""

from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline import RagPipeline
from prompts import format_context

from core.golden import REFUSAL
from core.resilience import (
    Budget,
    CircuitBreaker,
    RetryPolicy,
    map_bounded,
    retry,
)
from core.trace import RagTrace

# ===========================================================================
# 1. PII REDACTION
# ===========================================================================
# WHY REGEXES AND NOT A MODEL: a redaction step that itself calls an LLM is
# slow, costs money, can be prompt-injected, and fails open. Regexes are
# none of those. They also MISS things -- names, addresses, free-text
# identifiers -- so this is a first line of defence, not a compliance
# programme. For regulated data you want a dedicated PII detector (Presidio and
# similar) behind the same interface.
#
# The ordering below matters: card numbers are matched before generic long
# digit runs, or a card would be redacted as a phone number and the more
# specific label lost.
# ===========================================================================

PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # 13-16 digits, optionally separated by spaces or hyphens.
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    # UK National Insurance number.
    ("NINO", re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b", re.IGNORECASE)),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\d[ -]?){9,14}\d(?!\w)")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


@dataclass
class RedactionResult:
    text: str
    found: dict[str, int] = field(default_factory=dict)

    @property
    def had_pii(self) -> bool:
        return bool(self.found)


def redact(text: str) -> RedactionResult:
    """Replace recognisable personal data with typed placeholders.

    Placeholders keep the SHAPE of the sentence ("email me at [EMAIL]"), so the
    model can still understand the request. Deleting the text outright often
    makes the question unanswerable.

    Returns counts by type as well, because "we redacted 4 emails today" is an
    operational metric worth alerting on -- a sudden spike usually means a new
    integration started forwarding raw customer records.
    """
    found: dict[str, int] = {}
    redacted = text
    for label, pattern in PII_PATTERNS:
        redacted, count = pattern.subn(f"[{label}]", redacted)
        if count:
            found[label] = found.get(label, 0) + count
    return RedactionResult(text=redacted, found=found)


# ===========================================================================
# 2. RELEVANCE THRESHOLD
# ===========================================================================


@dataclass
class ThresholdPolicy:
    """Refuse before generating when retrieval is clearly not confident.

    core/corpus/hallucination.md lists this as a mitigation: "Apply a relevance
    threshold to retrieval scores and refuse when the best chunk falls below
    it, rather than always returning the top K regardless of quality."

    CHOOSING THE NUMBER IS AN EMPIRICAL QUESTION, NOT A GUESS. Score the
    unanswerable golden items, look at the distribution of top scores, and pick
    a value that separates them from the answerable ones. `suggest_threshold()`
    below does exactly that. A threshold picked by intuition either refuses
    everything or nothing.
    """

    min_top_score: float = 0.10
    # Require at least this many chunks above the bar, not just the best one.
    # Guards against a single lucky keyword match.
    min_supporting_chunks: int = 1

    def should_refuse(self, trace: RagTrace) -> bool:
        if not trace.retrieved:
            return True
        above = [c for c in trace.retrieved if c.score >= self.min_top_score]
        return len(above) < self.min_supporting_chunks


def suggest_threshold(
    answerable_top_scores: list[float],
    unanswerable_top_scores: list[float],
) -> tuple[float, float]:
    """Pick a threshold from data, and report what it would cost you.

    Returns ``(threshold, false_refusal_rate)``.

    The trade-off, stated in core/corpus/hallucination.md: raising the bar
    reduces hallucination but increases unhelpful refusals on questions the
    system could have answered. Both directions must be measured together, or
    you optimise your way into either a liar or a system that refuses
    everything.

    Strategy here: take the highest score any UNANSWERABLE question achieved,
    and sit just above it. That is the smallest threshold that refuses all of
    them, so the false-refusal rate it reports is the honest price.
    """
    if not unanswerable_top_scores:
        return 0.0, 0.0

    threshold = max(unanswerable_top_scores) + 1e-6
    if not answerable_top_scores:
        return threshold, 0.0

    wrongly_refused = sum(1 for s in answerable_top_scores if s < threshold)
    return threshold, wrongly_refused / len(answerable_top_scores)


# ===========================================================================
# 3. THE PRODUCTION PIPELINE
# ===========================================================================


@dataclass
class GuardStats:
    """What the guards did. These are the numbers you put on a dashboard."""

    requests: int = 0
    pii_redacted: int = 0
    refused_low_relevance: int = 0
    fallback_used: int = 0
    retries: int = 0
    failures: int = 0

    def report(self) -> str:
        return (
            f"{self.requests} requests | {self.pii_redacted} redacted | "
            f"{self.refused_low_relevance} refused (low relevance) | "
            f"{self.fallback_used} fallbacks | {self.retries} retries | "
            f"{self.failures} failures"
        )


class ProductionRagPipeline:
    """The lesson-02 pipeline wrapped in the guards it needs to be deployed.

    Composition rather than inheritance, deliberately: the teaching pipeline
    stays simple and readable, and everything here is visibly OPTIONAL. You can
    read `pipeline.py` to understand RAG and this file to understand operating
    it, without either one obscuring the other.
    """

    def __init__(
        self,
        pipeline: RagPipeline,
        fallback_llm=None,
        threshold: ThresholdPolicy | None = None,
        redact_pii: bool = True,
        retry_policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        budget: Budget | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.fallback_llm = fallback_llm
        self.threshold = threshold or ThresholdPolicy()
        self.redact_pii = redact_pii
        self.retry_policy = retry_policy or RetryPolicy(max_attempts=3)
        self.breaker = breaker or CircuitBreaker(failure_threshold=5, reset_after=30)
        self.budget = budget
        self.stats = GuardStats()

    def answer(self, question: str, sleep=time.sleep) -> RagTrace:
        """Answer with every guard applied, in the order that matters."""
        self.stats.requests += 1
        if self.budget is not None:
            self.budget.check()

        # --- GUARD 1: redact BEFORE anything sees the text -----------------
        # Before the prompt, before the logs, before the trace. Redacting later
        # means the raw value already reached somewhere it should not have.
        redaction = RedactionResult(text=question)
        if self.redact_pii:
            redaction = redact(question)
            if redaction.had_pii:
                self.stats.pii_redacted += 1

        clean_question = redaction.text

        # --- GUARD 2: retrieve, then check confidence ----------------------
        retrieval_started = time.perf_counter()
        chunks = self.pipeline.retrieve(clean_question)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

        probe = RagTrace(question=clean_question, answer="", retrieved=chunks)
        if self.threshold.should_refuse(probe):
            # Refuse WITHOUT calling the model: safer and free.
            self.stats.refused_low_relevance += 1
            trace = RagTrace(
                question=clean_question,
                answer=REFUSAL,
                retrieved=chunks,
                retrieval_ms=retrieval_ms,
                chunk_size=self.pipeline.chunk_size,
                top_k=self.pipeline.top_k,
            )
            trace.metadata.update(
                refused_reason="below relevance threshold",
                top_score=max((c.score for c in chunks), default=0.0),
                pii_found=redaction.found,
                model_used="none (refused before generation)",
            )
            return trace

        # --- GUARD 3: generate, with retry, breaker and fallback -----------
        context = format_context([c.text for c in chunks])
        messages = self.pipeline.prompt.format_messages(
            context=context, question=clean_question
        )

        generation_started = time.perf_counter()
        answer_text, model_used = self._generate_with_fallback(messages, sleep)
        generation_ms = (time.perf_counter() - generation_started) * 1000

        trace = RagTrace(
            question=clean_question,
            answer=answer_text,
            retrieved=chunks,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            chat_model=model_used,
            embed_model=type(self.pipeline.embeddings).__name__,
            chunk_size=self.pipeline.chunk_size,
            top_k=self.pipeline.top_k,
        )
        trace.metadata.update(
            pii_found=redaction.found,
            model_used=model_used,
            top_score=max((c.score for c in chunks), default=0.0),
        )

        if self.budget is not None:
            # Rough accounting. A real deployment reads usage_metadata off the
            # response; 4 chars/token is the standard back-of-envelope ratio.
            self.budget.record(
                input_tokens=sum(len(str(m.content)) for m in messages) // 4,
                output_tokens=len(answer_text) // 4,
            )
        return trace

    def _generate_with_fallback(self, messages, sleep) -> tuple[str, str]:
        """Primary model, then fallback. Records which one answered.

        RECORDING WHICH MODEL ANSWERED IS NOT OPTIONAL. If a fallback silently
        serves 30% of traffic, your quality metrics are the average of two
        different systems and every conclusion you draw from them is wrong.
        """
        stats_holder = {"retries": 0}

        def call(llm):
            def attempt():
                return self.breaker.call(lambda: llm.invoke(messages))

            from core.resilience import RetryStats

            retry_stats = RetryStats()
            try:
                response = retry(
                    attempt, self.retry_policy, stats=retry_stats, sleep=sleep
                )
            finally:
                stats_holder["retries"] += retry_stats.retries
            return response.content if hasattr(response, "content") else str(response)

        primary_name = _model_name(self.pipeline.llm)
        try:
            text = call(self.pipeline.llm)
            self.stats.retries += stats_holder["retries"]
            return text, primary_name
        except Exception:  # noqa: BLE001 - deliberately broad: any failure falls back
            self.stats.retries += stats_holder["retries"]
            if self.fallback_llm is None:
                self.stats.failures += 1
                raise

        # The fallback gets its own circuit breaker state via a fresh call, and
        # is deliberately NOT retried as aggressively -- if the primary is down
        # you want an answer quickly, not three more rounds of backoff.
        self.stats.fallback_used += 1
        try:
            response = self.fallback_llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            return text, f"{_model_name(self.fallback_llm)} (fallback)"
        except Exception:
            self.stats.failures += 1
            raise

    # ---- batch -------------------------------------------------------------

    def answer_many(
        self, questions: list[str], max_workers: int = 4
    ) -> list[tuple[str, RagTrace | None, BaseException | None]]:
        """Answer a batch with bounded parallelism, keeping partial results.

        Returns ``(question, trace, error)`` triples in input order. One failure
        does not lose the rest -- which is what lets an evaluation sweep report
        "45 scored, 3 failed" instead of nothing.
        """
        return map_bounded(self.answer, questions, max_workers=max_workers)


def _model_name(llm) -> str:
    return str(getattr(llm, "model", type(llm).__name__))
