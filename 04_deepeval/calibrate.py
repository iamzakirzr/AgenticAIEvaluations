"""
JUDGE CALIBRATION: is your judge measuring anything at all?

Run it:   .venv/bin/python 04_deepeval/calibrate.py              # needs Ollama
          .venv/bin/python 04_deepeval/calibrate.py --demo       # no model, explains the maths

=============================================================================
WHY THIS IS THE MOST IMPORTANT FILE IN LESSON 04
=============================================================================
Every judged metric you have seen so far produces a confident-looking decimal.
None of them tell you whether the judge is any good.

core/corpus/llm_as_judge.md, stated plainly:

    "The central risk is that the judge is itself a fallible model. A metric
     produced by an uncalibrated judge is a number that looks rigorous and may
     measure nothing. Treating judge scores as ground truth without ever
     checking them against human labels is the most common serious mistake in
     applied evaluation."

Calibration is the fix, and it is not complicated:

    1. Hand-label a sample of outputs yourself (pass / fail).
    2. Run the judge on the same sample.
    3. Compute agreement, CORRECTED FOR CHANCE.

Almost nobody does step 3. Doing it -- and being able to say "llama3.1:8b
scores kappa 0.31 on faithfulness, so I do not gate on it" -- is a genuinely
differentiating thing to have on a CV.

=============================================================================
WHY COHEN'S KAPPA AND NOT PERCENTAGE AGREEMENT
=============================================================================
Raw agreement lies on imbalanced data. If 90% of answers are good, a judge that
says "good" every single time scores 90% agreement while having learned
absolutely nothing.

Kappa corrects for the agreement you would get by chance:

    kappa = (p_observed - p_chance) / (1 - p_chance)

The lazy judge above gets p_observed = p_chance, so kappa = 0 -- correctly
reporting that it is worthless. That is exactly the failure raw agreement hides.

Conventional interpretation (from the corpus):
    > 0.8   almost perfect
    > 0.6   substantial
    < 0.4   do not trust this judge to gate anything
=============================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for path in (str(_HERE), str(_HERE.parent), str(_HERE.parent / "02_langchain")):
    if path not in sys.path:
        sys.path.insert(0, path)


# ===========================================================================
# THE STATISTICS
# ===========================================================================


def cohens_kappa(human: list[int], judge: list[int]) -> float:
    """Chance-corrected agreement between two raters of binary labels.

    Args:
        human: your labels, 1 = pass, 0 = fail
        judge: the judge's labels for the same items, in the same order

    >>> cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])   # perfect
    1.0
    >>> cohens_kappa([1, 1, 1, 1], [1, 1, 1, 1])   # both always agree, but
    0.0                                            # there is no variance to
    ...                                            # agree ABOUT
    """
    if len(human) != len(judge):
        raise ValueError("label lists must be the same length")
    n = len(human)
    if n == 0:
        raise ValueError("cannot calibrate on an empty sample")

    observed = sum(1 for h, j in zip(human, judge) if h == j) / n

    # Chance agreement: the probability two independent raters with these
    # marginal rates would agree by luck alone.
    h_pos, j_pos = sum(human) / n, sum(judge) / n
    chance = (h_pos * j_pos) + ((1 - h_pos) * (1 - j_pos))

    if chance == 1.0:
        # Both raters gave a single constant label. There is no variance, so
        # kappa is undefined -- report 0.0, which is the honest answer: this
        # sample proves nothing about the judge.
        return 0.0

    return (observed - chance) / (1 - chance)


def interpret(kappa: float) -> str:
    if kappa > 0.8:
        return "almost perfect -- trustworthy"
    if kappa > 0.6:
        return "substantial -- usable as a gate"
    if kappa > 0.4:
        return "moderate -- usable for trends, not for gating"
    if kappa > 0.2:
        return "fair -- barely better than guessing"
    return "poor -- DO NOT gate anything on this judge"


@dataclass
class CalibrationResult:
    model: str
    metric: str
    n: int
    kappa: float
    raw_agreement: float
    judge_failures: int
    human_pass_rate: float
    judge_pass_rate: float

    def report(self) -> str:
        lines = [
            f"  judge model     : {self.model}",
            f"  metric          : {self.metric}",
            f"  sample size     : {self.n}",
            f"  raw agreement   : {self.raw_agreement:.1%}   <- the misleading number",
            f"  Cohen's kappa   : {self.kappa:.3f}   ({interpret(self.kappa)})",
            f"  human pass rate : {self.human_pass_rate:.1%}",
            f"  judge pass rate : {self.judge_pass_rate:.1%}",
        ]
        if self.judge_failures:
            lines.append(
                f"  JUDGE FAILURES  : {self.judge_failures} "
                f"(reported separately, NEVER scored as 0)"
            )
        if abs(self.human_pass_rate - self.judge_pass_rate) > 0.25:
            lines.append(
                "  NOTE: the pass rates diverge sharply -- the judge is "
                "systematically more lenient or stricter than you, which is a "
                "threshold problem, not necessarily a judgement problem."
            )
        return "\n".join(lines)


# ===========================================================================
# THE LABELLED SAMPLE
# ===========================================================================
# In real work YOU label these, by reading the outputs. They are hand-written
# here so the lesson is runnable, and deliberately include the hard cases:
# a faithfully-repeated falsehood, a partially-supported answer, and an answer
# that is correct but not derivable from the context.
# ===========================================================================

FAITHFULNESS_SAMPLE: list[dict] = [
    {
        "id": "f1",
        "context": ["nomic-embed-text produces 768 dimensions."],
        "answer": "nomic-embed-text produces 768 dimensions.",
        "human": 1,
        "note": "exactly supported",
    },
    {
        "id": "f2",
        "context": ["nomic-embed-text produces 768 dimensions."],
        "answer": "nomic-embed-text produces 768 dimensions and was released in 1997.",
        "human": 0,
        "note": "one supported claim, one fabricated",
    },
    {
        "id": "f3",
        "context": ["Chunk overlap is typically 10 to 20 percent of chunk size."],
        "answer": "Overlap is usually 10-20% of the chunk size.",
        "human": 1,
        "note": "paraphrase of a supported claim",
    },
    {
        "id": "f4",
        "context": ["Chunk overlap is typically 10 to 20 percent of chunk size."],
        "answer": "Overlap should always be exactly 50% of the chunk size.",
        "human": 0,
        "note": "contradicts the context (intrinsic hallucination)",
    },
    {
        "id": "f5",
        "context": ["HNSW is an approximate nearest neighbour index."],
        "answer": "HNSW is approximate, and its three parameters are M, ef_construction and ef_search.",
        "human": 0,
        "note": "TRUE in the world, but not derivable from THIS context (extrinsic)",
    },
    {
        "id": "f6",
        "context": ["Cohen's kappa corrects agreement for chance."],
        "answer": "Cohen's kappa corrects for chance agreement between two raters.",
        "human": 1,
        "note": "supported, mild elaboration that the context implies",
    },
    {
        "id": "f7",
        "context": ["Reciprocal rank is one divided by the rank of the first relevant result."],
        "answer": "The reciprocal rank at position 4 is 0.25.",
        "human": 1,
        "note": "arithmetic derived from a supported rule -- a genuinely hard call",
    },
    {
        "id": "f8",
        "context": ["Vector search always returns top_k results."],
        "answer": "Vector search returns nothing when no document is relevant.",
        "human": 0,
        "note": "directly contradicts the context",
    },
]


def run_calibration(judge_model: str | None = None, threshold: float = 0.8) -> CalibrationResult:
    """Score the labelled sample with a real judge and compare against the labels."""
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
    from ollama_judge import JudgeFailure, OllamaJudge


    judge = OllamaJudge(model=judge_model)
    human_labels: list[int] = []
    judge_labels: list[int] = []
    failures = 0

    print(f"Scoring {len(FAITHFULNESS_SAMPLE)} hand-labelled examples "
          f"with {judge.get_model_name()}...\n")

    for row in FAITHFULNESS_SAMPLE:
        case = LLMTestCase(
            input="(calibration item)",
            actual_output=row["answer"],
            retrieval_context=row["context"],
        )
        metric = FaithfulnessMetric(model=judge, threshold=threshold)
        try:
            metric.measure(case)
            # Binarise the judge's continuous score at the threshold, so it is
            # comparable with a human pass/fail label.
            label = 1 if metric.score >= threshold else 0
        except JudgeFailure as exc:
            # NEVER record this as 0. It is an infrastructure failure and
            # scoring it would fabricate a quality regression.
            failures += 1
            print(f"  {row['id']}  JUDGE FAILED: {str(exc)[:80]}")
            continue

        human_labels.append(row["human"])
        judge_labels.append(label)

        agree = "  " if row["human"] == label else "<-- DISAGREES"
        print(
            f"  {row['id']}  human={row['human']}  judge={label} "
            f"(raw {metric.score:.2f})  {agree}  {row['note']}"
        )

    if not human_labels:
        raise RuntimeError("every judge call failed -- nothing to calibrate")

    n = len(human_labels)
    return CalibrationResult(
        model=judge.get_model_name(),
        metric=f"Faithfulness (threshold {threshold})",
        n=n,
        kappa=cohens_kappa(human_labels, judge_labels),
        raw_agreement=sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / n,
        judge_failures=failures,
        human_pass_rate=sum(human_labels) / n,
        judge_pass_rate=sum(judge_labels) / n,
    )


def demo() -> None:
    """Explain the maths with no model, so the lesson works offline."""
    print("=" * 76)
    print("WHY RAW AGREEMENT LIES  (no model needed for this part)")
    print("=" * 76)

    # 9 good answers, 1 bad -- the imbalance you get in any half-decent system.
    human = [1] * 9 + [0]

    # Both judges are wrong exactly once, so BOTH score 90% raw agreement.
    # The difference is what they are wrong about.
    lazy = [1] * 10                  # always says "pass"; misses the one failure
    useful = [0] + [1] * 8 + [0]     # catches the failure, one false alarm

    for name, judge in (("lazy judge (always says pass)", lazy),
                        ("useful judge (caught the failure)", useful)):
        agreement = sum(1 for h, j in zip(human, judge) if h == j) / len(human)
        kappa = cohens_kappa(human, judge)
        print(f"\n{name}")
        print(f"  raw agreement : {agreement:.0%}   <- looks great either way")
        print(f"  Cohen's kappa : {kappa:.3f}   ({interpret(kappa)})")

    print(
        "\nBoth judges score 90% agreement. Only kappa distinguishes the one that\n"
        "learned something from the one that learned nothing. THIS is why you\n"
        "report kappa, and why 'my judge agrees with me 90% of the time' is not\n"
        "evidence of anything."
    )
    print("\nRun without --demo (and with Ollama up) to calibrate a real judge.")


def main() -> None:
    if "--demo" in sys.argv:
        demo()
        return

    from core.providers import ollama_available

    if not ollama_available():
        print("Ollama is not running. Showing the offline explanation instead.\n")
        demo()
        return

    print("=" * 76)
    print("JUDGE CALIBRATION -- Faithfulness")
    print("=" * 76)
    result = run_calibration()
    print("\n" + "=" * 76)
    print(result.report())
    print("=" * 76)
    print(
        "\nWHAT TO DO WITH THIS NUMBER:\n"
        "  kappa > 0.6  -> you may gate CI on this metric\n"
        "  kappa 0.4-0.6 -> track the trend, do not fail builds on it\n"
        "  kappa < 0.4  -> the metric is noise. Try a larger judge model\n"
        "                  (JUDGE_MODEL=qwen2.5:14b), simplify the criterion,\n"
        "                  or fall back to the deterministic metrics.\n\n"
        "Reporting 'my local judge scored kappa 0.31, so I did not gate on it'\n"
        "is a stronger result than reporting a green dashboard you never checked."
    )


if __name__ == "__main__":
    main()
