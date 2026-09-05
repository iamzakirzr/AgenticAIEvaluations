"""
Tests for the calibration maths. FAST TIER -- pure arithmetic, no model.

Run:  pytest 04_deepeval/test_calibration.py -v

If you only read one test file in this repo, read this one. It encodes the
reason to distrust a judge you have not measured.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from calibrate import FAITHFULNESS_SAMPLE, cohens_kappa, interpret


def test_perfect_agreement_is_one():
    assert cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_total_disagreement_is_negative():
    """Kappa below 0 means the judge is worse than a coin flip -- it is
    systematically inverted, which usually indicates a wired-up-backwards bug
    rather than a bad model."""
    assert cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1]) < 0


def test_a_lazy_judge_scores_zero_despite_high_raw_agreement():
    """THE point of using kappa at all.

    90% of answers are good. A judge that says "pass" unconditionally agrees
    with the human 90% of the time and has learned nothing whatsoever. Raw
    agreement calls that a success; kappa correctly calls it zero.
    """
    human = [1] * 9 + [0]
    lazy = [1] * 10

    raw_agreement = sum(1 for h, j in zip(human, lazy) if h == j) / len(human)
    assert raw_agreement == 0.9, "the lazy judge does look good on raw agreement"
    assert cohens_kappa(human, lazy) == pytest.approx(0.0), (
        "kappa failed to expose a judge with no discriminating power"
    )


def test_kappa_separates_two_judges_with_identical_raw_agreement():
    """Both judges are wrong exactly once, so both score 90%.

    One missed the only real failure; the other caught it and raised one false
    alarm. Those are very different judges, and only kappa can tell them apart.
    """
    human = [1] * 9 + [0]
    lazy = [1] * 10
    useful = [0] + [1] * 8 + [0]

    def agreement(judge):
        return sum(1 for h, j in zip(human, judge) if h == j) / len(human)

    assert agreement(lazy) == agreement(useful) == 0.9
    assert cohens_kappa(human, useful) > cohens_kappa(human, lazy) + 0.5


def test_no_variance_returns_zero_rather_than_nan():
    """When both raters give one constant label, kappa is mathematically
    undefined (division by zero). Returning 0.0 is the honest answer: this
    sample proves nothing about the judge. Returning NaN would propagate
    silently into a report."""
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 0.0


def test_mismatched_lengths_fail_loudly():
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([1, 0], [1])


def test_empty_sample_fails_loudly():
    with pytest.raises(ValueError, match="empty sample"):
        cohens_kappa([], [])


def test_interpretation_thresholds_match_the_corpus():
    """core/corpus/llm_as_judge.md: >0.8 almost perfect, >0.6 substantial,
    <0.4 do not trust. Keep the code and the documentation in agreement."""
    assert "almost perfect" in interpret(0.85)
    assert "substantial" in interpret(0.7)
    assert "DO NOT gate" in interpret(0.15)


# ===========================================================================
# THE LABELLED SAMPLE ITSELF
# ===========================================================================


def test_calibration_sample_is_balanced_enough_to_be_informative():
    """A sample where every item has the same label cannot measure a judge.

    Kappa would be undefined, and you would learn nothing. Enforcing a mix is
    the cheapest guard against a useless calibration run.
    """
    labels = [row["human"] for row in FAITHFULNESS_SAMPLE]
    assert 0 in labels and 1 in labels
    pass_rate = sum(labels) / len(labels)
    assert 0.25 <= pass_rate <= 0.75, (
        f"pass rate {pass_rate:.0%} is too skewed for a meaningful kappa"
    )


def test_calibration_sample_includes_the_hard_cases():
    """A calibration set of easy cases flatters the judge.

    These three are the ones that actually discriminate:
      - an extrinsic hallucination that is TRUE in the world
      - a contradiction of the context
      - a claim derived by arithmetic from a supported rule
    """
    notes = " ".join(row["note"] for row in FAITHFULNESS_SAMPLE)
    assert "extrinsic" in notes
    assert "contradicts" in notes
    assert "hard call" in notes


def test_every_sample_row_is_well_formed():
    for row in FAITHFULNESS_SAMPLE:
        assert row["human"] in (0, 1)
        assert row["context"] and row["answer"]
        assert row["note"], f"{row['id']} has no note explaining the label"
