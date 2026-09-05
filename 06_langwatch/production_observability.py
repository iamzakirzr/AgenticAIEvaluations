"""
PRODUCTION SCENARIO: traces you can legally keep, and that tell you something.

=============================================================================
FOUR THINGS THAT BITE ONCE TRACING IS REAL
=============================================================================

  1. TRACES LEAVE YOUR PROCESS. That is the entire point of a hosted
     observability platform -- and it means every prompt, every answer and
     every retrieved chunk is transmitted to and stored by a third party. If a
     user pasted a card number into the chat box, you have just exported it.
     Redaction must happen BEFORE the span is built, not before it is read.

  2. SAMPLING DECIDES WHAT YOU CAN INVESTIGATE. Judged metrics cost a model
     call each, so you sample. Sample the wrong way and the traces you kept are
     exactly the boring ones.

  3. ALERTS NEED A DIRECTION AND A BASELINE. "Faithfulness is 0.72" is not
     actionable. "Refusal rate has tripled in an hour" is.

  4. PRODUCTION TRAFFIC IS THE BEST SOURCE OF GOLDEN DATA. Your hand-written
     dataset tests what you imagined users would ask. The traces contain what
     they actually asked, and the gap between those two is where systems fail.

=============================================================================
THE ONE NON-NEGOTIABLE
=============================================================================
    REDACT BEFORE THE SPAN IS CREATED.

Not before it is exported, not in a processor, not in the dashboard. Once the
raw string is an attribute on a span, it is in a buffer that some other thread
is already shipping. Redaction at any later point is a race you will lose.
=============================================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
for path in (str(_ROOT), str(_ROOT / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)

from production_pipeline import redact

from core.golden import GoldenItem
from core.trace import RagTrace

# ===========================================================================
# 1. REDACTION AT THE BOUNDARY
# ===========================================================================


@dataclass
class ExportPolicy:
    """What may leave the process, and in what form.

    Defaults are deliberately conservative. Exporting less is recoverable --
    you turn a flag on. Exporting a customer's card number to a third party is
    not.
    """

    redact_pii: bool = True

    # Retrieved chunks are usually your own documents, so they are safe to
    # export and enormously useful for debugging. Set False if your corpus is
    # itself confidential.
    include_contexts: bool = True

    # Truncate long fields. Traces with 50KB of context are expensive to store,
    # slow to render, and rarely more informative than the first 2KB.
    max_field_chars: int = 2000

    # Hash rather than transmit. Lets you correlate "this user hit the bug
    # eleven times" without ever exporting who they are.
    hash_user_ids: bool = True


def hash_identifier(value: str, salt: str = "") -> str:
    """A stable pseudonym for a user id.

    Same input -> same hash, so you can count and correlate. The original is
    not recoverable from the hash, so it is not personal data in the trace.

    THE SALT MATTERS: without one, a hash of a short identifier space (an email,
    a numeric id) is trivially reversed with a rainbow table. Use a secret salt
    from your environment in production, not a literal.
    """
    return hashlib.sha256(f"{salt}{value}".encode()).hexdigest()[:16]


def prepare_for_export(
    trace: RagTrace,
    policy: ExportPolicy | None = None,
    user_id: str | None = None,
    salt: str = "",
) -> dict[str, Any]:
    """Build the span payload, with redaction applied BEFORE construction.

    Returns a plain dict rather than touching a span, so this is testable in
    isolation and so the ordering guarantee is obvious: nothing raw is ever
    handed to the tracer.
    """
    policy = policy or ExportPolicy()

    # Counts accumulate across EVERY field we clean, not just the question and
    # answer. Retrieved chunks are usually your own documents, but "usually" is
    # not a guarantee -- a support corpus can contain customer data, and an
    # exported count that ignores contexts under-reports exactly the case you
    # would most want alerted on.
    found: dict[str, int] = {}

    def clean(text: str) -> str:
        out = text
        if policy.redact_pii:
            # ONE redact() call per field: the earlier version ran it twice
            # (once to clean, once to count), which doubled the work and let
            # the two paths drift apart.
            result = redact(text)
            out = result.text
            for label, count in result.found.items():
                found[label] = found.get(label, 0) + count
        if len(out) > policy.max_field_chars:
            out = out[: policy.max_field_chars] + f"... [truncated {len(out)} chars]"
        return out

    payload: dict[str, Any] = {
        "question": clean(trace.question),
        "answer": clean(trace.answer),
        "retrieval_ms": round(trace.retrieval_ms, 2),
        "generation_ms": round(trace.generation_ms, 2),
        "chat_model": trace.chat_model,
        "embed_model": trace.embed_model,
        "chunk_size": trace.chunk_size,
        "top_k": trace.top_k,
        "doc_ids": trace.retrieved_doc_ids,
        "top_score": round(max((c.score for c in trace.retrieved), default=0.0), 4),
    }

    if policy.include_contexts:
        payload["contexts"] = [clean(c) for c in trace.contexts]

    # Set AFTER every field has been cleaned, because `found` accumulates as
    # clean() runs. Counts, not values: "we redacted 3 emails" is safe to
    # export and is itself a useful alerting signal.
    payload["pii_redacted"] = found

    if user_id is not None:
        payload["user"] = (
            hash_identifier(user_id, salt) if policy.hash_user_ids else user_id
        )

    return payload


# ===========================================================================
# 2. SAMPLING
# ===========================================================================


@dataclass
class SamplingPolicy:
    """Decide which traces get expensive judged evaluation.

    HEAD SAMPLING (`rate`) is the deterministic hash-based decision from
    lesson 06: cheap, stable, and made before you know how the request went.

    TAIL SAMPLING (`always_keep_*`) overrides it AFTER the fact for requests
    that are interesting regardless of the dice. This is the part people miss.
    Pure random sampling at 10% keeps a representative slice of BORING traffic
    and throws away nine out of ten of your actual incidents.

    The rule of thumb: sample the normal, keep all of the abnormal.
    """

    rate: float = 0.1
    always_keep_errors: bool = True
    always_keep_refusals: bool = True
    always_keep_slow_ms: float = 5000.0
    always_keep_low_confidence: float = 0.05

    def should_evaluate(self, trace: RagTrace, errored: bool = False) -> tuple[bool, str]:
        """Return ``(sample, reason)``. The reason is what makes it debuggable."""
        if self.always_keep_errors and errored:
            return True, "tail: request errored"

        if self.always_keep_refusals and "does not contain this information" in (
            trace.answer or ""
        ).lower():
            return True, "tail: refusal"

        if trace.total_ms >= self.always_keep_slow_ms:
            return True, f"tail: slow ({trace.total_ms:.0f}ms)"

        top = max((c.score for c in trace.retrieved), default=0.0)
        if top <= self.always_keep_low_confidence:
            return True, f"tail: low retrieval confidence ({top:.3f})"

        if _hash_bucket(trace.question) < self.rate:
            return True, f"head: sampled at {self.rate:.0%}"

        return False, "not sampled"


def _hash_bucket(text: str) -> float:
    """Deterministic bucket in [0, 1).

    Deterministic so that re-running a request samples identically -- otherwise
    you cannot reproduce a scored trace while investigating it, and two services
    handling the same request disagree about whether it was sampled.
    """
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


# ===========================================================================
# 3. ALERTING
# ===========================================================================


@dataclass
class Alert:
    metric: str
    message: str
    severity: str = "warning"
    current: float = 0.0
    reference: float = 0.0


@dataclass
class OnlineMonitor:
    """Rolling-window monitoring of the reference-free online metrics.

    WHY A ROLLING BASELINE RATHER THAN A FIXED THRESHOLD: a fixed threshold on
    refusal rate is wrong on day one, because the correct rate depends on the
    traffic mix, which changes. What is always meaningful is a SUDDEN CHANGE
    relative to recent history.

    Two directions matter, and people usually only watch one:

      refusal rate SPIKES  -> retrieval broke, or the corpus lost coverage
      refusal rate COLLAPSES -> the model stopped refusing and is now
                                confabulating. This is the dangerous one, and a
                                one-sided alert never fires on it.
    """

    window: int = 200
    min_samples: int = 30
    change_factor: float = 2.0

    # Absolute floor for a spike out of a zero baseline. A relative test alone
    # can never fire when the reference rate is 0, which is precisely the
    # "retrieval just broke completely" case.
    absolute_spike: float = 0.2

    _refusals: deque[int] = field(default_factory=lambda: deque(maxlen=1000), init=False)
    _errors: deque[int] = field(default_factory=lambda: deque(maxlen=1000), init=False)
    _latencies: deque[float] = field(default_factory=lambda: deque(maxlen=1000), init=False)
    _confidences: deque[float] = field(default_factory=lambda: deque(maxlen=1000), init=False)
    _pii: deque[int] = field(default_factory=lambda: deque(maxlen=1000), init=False)

    def record(self, trace: RagTrace, errored: bool = False) -> None:
        refused = "does not contain this information" in (trace.answer or "").lower()
        self._refusals.append(int(refused))
        self._errors.append(int(errored))
        self._latencies.append(trace.total_ms)
        self._confidences.append(max((c.score for c in trace.retrieved), default=0.0))
        self._pii.append(sum(trace.metadata.get("pii_found", {}).values()))

    def _recent_and_reference(self, series: deque) -> tuple[list[float], list[float]]:
        values = list(series)
        half = self.window // 2
        recent = values[-half:]
        reference = values[-self.window : -half]
        return recent, reference

    def check(self) -> list[Alert]:
        alerts: list[Alert] = []
        if len(self._refusals) < self.min_samples:
            return alerts

        recent, reference = self._recent_and_reference(self._refusals)
        if reference and recent:
            now = sum(recent) / len(recent)
            before = sum(reference) / len(reference)

            # TWO spike conditions, and the second one is easy to forget.
            #
            # A relative test alone (`now > before * factor`) cannot fire when
            # the baseline is ZERO -- and 0% -> 50% refusals is the most
            # dramatic spike there is. Multiplying zero by anything is still
            # zero, so the alert stays silent through exactly the incident it
            # was written for. An absolute floor covers that case.
            relative_spike = before > 0.01 and now > before * self.change_factor
            spike_from_nothing = before <= 0.01 and now >= self.absolute_spike
            if relative_spike or spike_from_nothing:
                alerts.append(
                    Alert(
                        "refusal_rate",
                        f"refusal rate rose from {before:.1%} to {now:.1%} -- "
                        f"retrieval may be broken or the corpus lost coverage",
                        "warning",
                        now,
                        before,
                    )
                )
            if before > 0.05 and now < before / self.change_factor:
                # The alert people forget to write.
                alerts.append(
                    Alert(
                        "refusal_rate",
                        f"refusal rate COLLAPSED from {before:.1%} to {now:.1%} -- "
                        f"the model may have stopped refusing and started "
                        f"confabulating. This is the more dangerous direction.",
                        "critical",
                        now,
                        before,
                    )
                )

        recent_err, _ = self._recent_and_reference(self._errors)
        if recent_err:
            rate = sum(recent_err) / len(recent_err)
            if rate > 0.05:
                alerts.append(
                    Alert("error_rate", f"error rate is {rate:.1%}", "critical", rate, 0.05)
                )

        recent_conf, reference_conf = self._recent_and_reference(self._confidences)
        if recent_conf and reference_conf:
            now = sum(recent_conf) / len(recent_conf)
            before = sum(reference_conf) / len(reference_conf)
            if before > 0 and now < before * 0.5:
                alerts.append(
                    Alert(
                        "retrieval_confidence",
                        f"mean top-chunk similarity halved ({before:.3f} -> {now:.3f}) "
                        f"-- the index or the query distribution changed",
                        "warning",
                        now,
                        before,
                    )
                )

        recent_pii, _ = self._recent_and_reference(self._pii)
        if recent_pii and sum(recent_pii) > 0:
            per_request = sum(recent_pii) / len(recent_pii)
            if per_request > 0.1:
                alerts.append(
                    Alert(
                        "pii_rate",
                        f"{per_request:.2f} PII items redacted per request -- a new "
                        f"integration may be forwarding raw customer records",
                        "warning",
                        per_request,
                        0.1,
                    )
                )
        return alerts


# ===========================================================================
# 4. TRACES BACK INTO THE GOLDEN DATASET
# ===========================================================================


@dataclass
class DatasetCandidate:
    """A production question proposed for the golden dataset.

    NOTE it has no reference answer. That is the honest state: promoting a
    trace gives you a real QUESTION, and a human still has to supply the right
    ANSWER. Anything that auto-fills the reference from the system's own output
    is building a dataset that certifies the system's current behaviour as
    correct -- which is circular and will hide every existing bug forever.
    """

    question: str
    suggested_category: str
    reason: str
    observed_answer: str = ""
    observed_doc_ids: list[str] = field(default_factory=list)
    occurrences: int = 1

    def to_golden_stub(self, item_id: str) -> dict[str, Any]:
        return {
            "id": item_id,
            "question": self.question,
            "reference_answer": "TODO: a human must write this",
            "reference_doc_ids": self.observed_doc_ids,
            "category": self.suggested_category,
            "difficulty": 2,
            "rationale": f"promoted from production: {self.reason}",
        }


def propose_dataset_candidates(
    traces: Iterable[RagTrace],
    existing: Iterable[GoldenItem] = (),
    min_occurrences: int = 2,
) -> list[DatasetCandidate]:
    """Mine traces for questions worth adding to the golden dataset.

    Prioritises the traces that reveal something:

      - REFUSALS: either a genuine coverage gap (add the document) or an
        unhelpful refusal (a bug). Both are worth a dataset item.
      - LOW-CONFIDENCE ANSWERS: the system answered anyway despite weak
        retrieval, which is where hallucinations live.
      - REPEATED questions: asked often enough to matter.

    Deliberately skips anything already resembling a dataset question, so the
    set does not fill up with near-duplicates.
    """
    seen_questions = {_normalise(item.question) for item in existing}
    counts: dict[str, int] = {}
    first: dict[str, RagTrace] = {}
    # (reason, category, priority). Higher priority wins when a question is
    # seen several times with different outcomes -- a question that was ever
    # refused is more interesting than one that was merely asked often.
    verdicts: dict[str, tuple[str, str, int]] = {}

    for trace in traces:
        key = _normalise(trace.question)
        if not key or key in seen_questions:
            continue

        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, trace)

        verdict = _classify(trace)
        if key not in verdicts or verdict[2] > verdicts[key][2]:
            verdicts[key] = verdict

    candidates: list[DatasetCandidate] = []
    for key, count in counts.items():
        reason, category, priority = verdicts[key]
        # Priority 0 means "nothing notable happened" -- only promote those if
        # they are asked often enough to be worth a human's labelling time.
        if priority == 0 and count < min_occurrences:
            continue
        trace = first[key]
        candidates.append(
            DatasetCandidate(
                question=trace.question,
                suggested_category=category,
                reason=reason,
                observed_answer=trace.answer,
                observed_doc_ids=trace.retrieved_doc_ids,
                occurrences=count,
            )
        )

    # Most-repeated first: those are the ones worth a human's labelling time.
    return sorted(candidates, key=lambda c: (-c.occurrences, c.question))


def _classify(trace: RagTrace) -> tuple[str, str, int]:
    """Return ``(reason, suggested_category, priority)`` for one trace.

    Priority orders how interesting an outcome is, so that a question seen
    several times is promoted for its most informative occurrence rather than
    its most recent one.
    """
    answer = (trace.answer or "").lower()
    top = max((c.score for c in trace.retrieved), default=0.0)

    if "does not contain this information" in answer:
        # Either a genuine coverage gap (add the document) or an unhelpful
        # refusal (a bug). Both deserve a dataset item.
        return "refused in production", "unanswerable", 2

    if top < 0.05:
        # Answered anyway despite weak retrieval. This is where hallucinations
        # live, which makes it the most valuable trace to label.
        return (
            f"answered despite low retrieval confidence ({top:.3f})",
            "adversarial",
            3,
        )

    return "frequently asked", "single_hop", 0


def _normalise(question: str) -> str:
    return " ".join((question or "").lower().split())


def write_candidates(candidates: list[DatasetCandidate], path: Path) -> Path:
    """Write proposals as JSONL stubs for a human to complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(c.to_golden_stub(f"prod-{i:03d}"), ensure_ascii=False)
        for i, c in enumerate(candidates, start=1)
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path
