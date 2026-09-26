"""
Prompt chaining with LCEL -- and the thing nobody tells you about it.

=============================================================================
THE UNCOMFORTABLE BIT, FIRST
=============================================================================
Chaining prompts makes your system WORSE at being tested, not better.

One prompt has one failure mode: the answer is wrong. A four-link chain has
five: each link can be wrong, and the composition can be wrong even when every
link is individually right. And because only the last link's output is visible,
every one of those failures presents identically -- "the answer is wrong".

Worse, reliability multiplies. Four links at 95% each is 81% end to end:

    0.95 ** 4 == 0.8145...

That is the single most useful number in this file. A chain of "pretty good"
steps is not pretty good. `test_compounding_reliability_is_multiplicative`
pins it, because engineers consistently reason additively about this and are
consistently surprised.

So the eval technique that matters for chains is NOT "score the final answer".
It is:

    1. trace every link (`ChainTrace`), so a failure has an address
    2. assert a CONTRACT on each link's output, so the failure is caught at
       the link that caused it rather than four steps downstream
    3. measure per-link success, because that is what tells you which link to
       fix

=============================================================================
WHAT LCEL ACTUALLY IS
=============================================================================
`|` is `Runnable.__or__`. That is the whole trick. Every LangChain component
implements the `Runnable` interface:

    invoke(input) -> output          ainvoke  for async
    batch([...])  -> [...]           abatch
    stream(input) -> Iterator        astream

and `a | b` returns a `RunnableSequence` that calls `b(a(x))`. Composition is
function composition with a shared interface -- there is no magic, no graph
compiler, no hidden state. Which means you can test any sub-chain in isolation
by simply... invoking it. That is the property this lesson exploits.

The four shapes you need, and all of them are in this file:

    SEQUENTIAL   a | b | c            each step feeds the next
    PARALLEL     RunnableParallel     fan out, fan in (dict in, dict out)
    BRANCH       RunnableBranch       route by a predicate
    MAP-REDUCE   .batch() then a      per-item work, then one summarising call
=============================================================================
"""

from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# ---------------------------------------------------------------------------
# 1. TRACING -- the part that makes a chain testable
# ---------------------------------------------------------------------------


@dataclass
class LinkTrace:
    """What one link did. The unit of blame."""

    name: str
    input: Any
    output: Any
    duration_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class ChainTrace:
    """Every link, in order.

    Compare with `core.trace.RagTrace`, which exists for the same reason one
    layer down: a pipeline that returns only a string cannot be debugged and
    cannot be evaluated beyond "was the string good".
    """

    links: list[LinkTrace] = field(default_factory=list)

    def add(self, link: LinkTrace) -> None:
        self.links.append(link)

    @property
    def names(self) -> list[str]:
        return [link.name for link in self.links]

    @property
    def failed(self) -> list[LinkTrace]:
        return [link for link in self.links if not link.ok]

    @property
    def total_ms(self) -> float:
        return sum(link.duration_ms for link in self.links)

    def of(self, name: str) -> LinkTrace | None:
        """The trace for one named link, or None if it never ran.

        `None` is itself a finding: in a branching chain, "the link never ran"
        is the most common cause of a surprising answer, and a chain-level
        assertion can never distinguish it from "the link ran and was wrong".
        """
        for link in self.links:
            if link.name == name:
                return link
        return None


def traced(name: str, trace: ChainTrace) -> Callable[[Runnable], Runnable]:
    """Wrap a runnable so that invoking it records a LinkTrace.

    Usage:
        trace = ChainTrace()
        chain = traced("classify", trace)(classifier) | traced("answer", trace)(answerer)

    Why not LangChain callbacks? You can, and in production you should -- see
    lesson 06 for the OpenTelemetry version. This explicit wrapper exists
    because it is READABLE: you can see, in the chain definition itself, what
    is being measured. For a teaching repo that beats a callback firing
    invisibly somewhere.
    """

    def wrap(runnable: Runnable) -> Runnable:
        def run(value: Any) -> Any:
            started = time.perf_counter()
            try:
                result = runnable.invoke(value)
            except Exception as exc:
                trace.add(
                    LinkTrace(
                        name=name,
                        input=value,
                        output=None,
                        duration_ms=(time.perf_counter() - started) * 1000,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
            trace.add(
                LinkTrace(
                    name=name,
                    input=value,
                    output=result,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
            return result

        return RunnableLambda(run)

    return wrap


# ---------------------------------------------------------------------------
# 2. CONTRACTS -- catch the failure at the link that caused it
# ---------------------------------------------------------------------------


class ContractViolation(ValueError):
    """A link produced output its successor cannot consume."""


@dataclass
class Contract:
    """A named, checkable promise about one link's output.

    This is the idea to steal from ordinary SDET work: the reason integration
    bugs are expensive is that the symptom appears far from the cause. Contract
    tests at each boundary move the symptom back to the cause. A prompt chain
    is an integration, and the boundaries are exactly the `|`s.

    Deliberately NOT pydantic: the check is arbitrary Python, because the
    interesting contracts are semantic ("mentions at least one citation",
    "is one of these three labels") rather than structural.
    """

    name: str
    check: Callable[[Any], bool]
    because: str

    def enforce(self, value: Any) -> Any:
        if not self.check(value):
            raise ContractViolation(f"{self.name}: {self.because} (got {value!r})")
        return value


def guard(contract: Contract) -> Runnable:
    """Turn a Contract into a runnable you can drop into a chain with `|`."""
    return RunnableLambda(contract.enforce)


# Three contracts used by the demo chain below. Each one is the difference
# between "the final answer was odd" and "link 1 returned 'Category: FACT'
# instead of 'fact'".
IS_LABEL = Contract(
    name="classification",
    check=lambda v: isinstance(v, str) and v.strip().lower() in {"factual", "opinion", "unknown"},
    because="the router only knows three labels, so anything else routes nowhere",
)

NON_EMPTY = Contract(
    name="non_empty",
    check=lambda v: isinstance(v, str) and bool(v.strip()),
    because="an empty string is silently concatenated into the next prompt and vanishes",
)

HAS_CITATION = Contract(
    name="has_citation",
    check=lambda v: bool(re.search(r"\[\d+\]", str(v))),
    because="an answer with no citation cannot be checked against its sources",
)


# ---------------------------------------------------------------------------
# 3. THE FOUR CHAIN SHAPES
# ---------------------------------------------------------------------------


def sequential_chain(model: Any) -> Runnable:
    """SHAPE 1 -- a | b | c.

    Draft, then critique, then revise. The classic "chain of prompts" that
    every tutorial shows, and the one whose reliability compounds worst,
    because a bad draft poisons both later links.
    """
    draft = (
        ChatPromptTemplate.from_template("Answer concisely: {question}")
        | model
        | StrOutputParser()
    )
    critique = (
        ChatPromptTemplate.from_template("List flaws in this answer, one per line:\n{draft}")
        | model
        | StrOutputParser()
    )
    revise = (
        ChatPromptTemplate.from_template(
            "Rewrite the answer, fixing every flaw.\n\nAnswer:\n{draft}\n\nFlaws:\n{critique}"
        )
        | model
        | StrOutputParser()
    )

    # RunnablePassthrough.assign ADDS a key to the dict flowing through, rather
    # than replacing it. That is what lets `revise` see BOTH draft and critique;
    # a plain `|` would have thrown the draft away.
    return (
        RunnablePassthrough.assign(draft=draft)
        | RunnablePassthrough.assign(critique=critique)
        | revise
    )


def parallel_chain(model: Any) -> Runnable:
    """SHAPE 2 -- fan out, fan in.

    Three independent views of the same input, computed CONCURRENTLY, then
    combined. `RunnableParallel` runs its branches in a thread pool, so the
    wall-clock cost is the slowest branch, not the sum.

    The eval-relevant property: the branches are independent, so a failure in
    one does not corrupt the others -- unlike a sequential chain. If you can
    express a step as parallel rather than sequential, your failure modes get
    cheaper. That is a design lever, not just a speed one.
    """
    summary = (
        ChatPromptTemplate.from_template("Summarise in one sentence: {text}")
        | model
        | StrOutputParser()
    )
    keywords = (
        ChatPromptTemplate.from_template("List 3 keywords, comma separated: {text}")
        | model
        | StrOutputParser()
    )
    sentiment = (
        ChatPromptTemplate.from_template("One word sentiment: {text}") | model | StrOutputParser()
    )
    return RunnableParallel(summary=summary, keywords=keywords, sentiment=sentiment)


def branching_chain(model: Any) -> Runnable:
    """SHAPE 3 -- route by a predicate.

    `RunnableBranch((predicate, runnable), ..., default)`. The predicate is
    ordinary Python over the input dict, so routing is free and deterministic
    -- you do NOT need a model to decide which prompt to use, and using one is
    a common and expensive mistake when a regex would do.

    TESTING NOTE: the routes you must test are the ones your golden set never
    triggers. Coverage here means "every branch taken at least once", exactly
    like branch coverage in ordinary testing, and it is the same bug when it is
    missing. `test_every_branch_is_reachable` asserts it.
    """
    factual = (
        ChatPromptTemplate.from_template("Answer using only the context.\n{question}")
        | model
        | StrOutputParser()
    )
    opinion = (
        ChatPromptTemplate.from_template("Explain the trade-offs, take no side.\n{question}")
        | model
        | StrOutputParser()
    )
    refuse = RunnableLambda(
        lambda _: "The provided context does not contain this information."
    )

    return RunnableBranch(
        (lambda x: x.get("label") == "factual", factual),
        (lambda x: x.get("label") == "opinion", opinion),
        refuse,  # the default arm -- ALWAYS have one; without it a novel label raises
    )


def map_reduce_chain(model: Any) -> Runnable:
    """SHAPE 4 -- per-item work, then one summarising call.

    The standard way to handle "more documents than fit in the context window".

    The trap it hides: the reduce step sees only the summaries, so any fact the
    map step dropped is unrecoverable, and the final answer will be confidently
    incomplete rather than visibly truncated. That failure is invisible to a
    faithfulness metric -- the summary IS faithful to the text it saw. Measuring
    it needs recall against the originals, which is why `map_reduce_chain`
    returns the intermediate summaries instead of swallowing them.
    """
    summarise_one = (
        ChatPromptTemplate.from_template("Summarise in one sentence:\n{text}")
        | model
        | StrOutputParser()
    )

    def run(payload: dict[str, Any]) -> dict[str, Any]:
        docs: Sequence[str] = payload["documents"]
        # .batch() is the parallel form of .invoke(). Same interface, N inputs.
        summaries = summarise_one.batch([{"text": d} for d in docs])
        combined = "\n".join(f"- {s}" for s in summaries)
        final = (
            ChatPromptTemplate.from_template("Combine these into one paragraph:\n{combined}")
            | model
            | StrOutputParser()
        ).invoke({"combined": combined})
        # Both, always. The intermediates are the only way to attribute a
        # missing fact to the map step rather than the reduce step.
        return {"summaries": summaries, "answer": final}

    return RunnableLambda(run)


# ---------------------------------------------------------------------------
# 4. RELIABILITY ARITHMETIC -- the number that should change your design
# ---------------------------------------------------------------------------


def end_to_end_reliability(link_reliabilities: Iterable[float]) -> float:
    """P(all links succeed) for independent links.

    Independence is an ASSUMPTION and usually an optimistic one: links that
    share a model share its bad days, so real chains do worse than this. Treat
    the result as a ceiling.

        >>> round(end_to_end_reliability([0.95] * 4), 4)
        0.8145
    """
    result = 1.0
    for value in link_reliabilities:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"reliability must be in [0, 1], got {value}")
        result *= value
    return result


def required_link_reliability(target: float, links: int) -> float:
    """Inverse: how good must each link be to hit an end-to-end target?

    Run it once and the design implication lands: to get 99% out of 5 links,
    every link needs 99.8%. If you cannot build that link, the answer is not a
    better prompt -- it is FEWER LINKS.
    """
    if links < 1:
        raise ValueError("a chain has at least one link")
    return target ** (1.0 / links)


def weakest_link(trace: ChainTrace) -> LinkTrace | None:
    """The slowest link that ran. Where to spend your optimisation budget."""
    return max(trace.links, key=lambda link: link.duration_ms, default=None)


# ---------------------------------------------------------------------------
# 5. A COMPLETE, TRACED, CONTRACT-CHECKED CHAIN
# ---------------------------------------------------------------------------


def build_guarded_chain(model: Any, trace: ChainTrace) -> Runnable:
    """classify -> [contract] -> route -> answer -> [contract].

    Every element of the lesson in one object: sequential composition, a
    branch, a contract at each boundary, and a trace entry per link.

    Invoke it with {"question": "..."} and read `trace.names` afterwards to see
    exactly which path ran.
    """
    classify = (
        ChatPromptTemplate.from_template(
            "Classify as exactly one of: factual, opinion, unknown.\n{question}"
        )
        | model
        | StrOutputParser()
        # Normalising BEFORE the contract is deliberate: models add whitespace
        # and capitalisation, and failing a contract on "  Factual\n" would be
        # pedantry rather than a finding.
        | RunnableLambda(lambda s: s.strip().lower())
    )

    def route(payload: dict[str, Any]) -> dict[str, Any]:
        return {"question": payload["question"], "label": payload["label"]}

    return (
        RunnablePassthrough.assign(label=traced("classify", trace)(classify | guard(IS_LABEL)))
        | traced("route", trace)(RunnableLambda(route))
        | traced("answer", trace)(branching_chain(model) | guard(NON_EMPTY))
    )


__all__ = [
    "HAS_CITATION",
    "IS_LABEL",
    "NON_EMPTY",
    "ChainTrace",
    "Contract",
    "ContractViolation",
    "LinkTrace",
    "branching_chain",
    "build_guarded_chain",
    "end_to_end_reliability",
    "guard",
    "map_reduce_chain",
    "parallel_chain",
    "required_link_reliability",
    "sequential_chain",
    "traced",
    "weakest_link",
]
