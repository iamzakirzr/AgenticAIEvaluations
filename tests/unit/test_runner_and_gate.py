import json

import pytest
from deepeval.test_case import LLMTestCase

from evalkit.metrics import CitationMetric, DeterministicMetric
from evalkit.reporting import Recorder, aggregate, compare, load_outcomes
from evalkit.reporting.gate import main as gate_main
from evalkit.runner import MetricOutcome, assert_passed, run_metrics


class Exploding(DeterministicMetric):
    name = "Exploding"

    def score_case(self, test_case):
        raise RuntimeError("judge returned invalid JSON")


def tc(output):
    return LLMTestCase(input="q", actual_output=output, retrieval_context=["[a.md] S: t"])


def test_run_metrics_records_scores_and_turns_errors_into_failures():
    outcomes = run_metrics("rag", "c1", tc("x [a.md]"), [CitationMetric(), Exploding()])
    ok, boom = outcomes
    assert (ok.metric, ok.score, ok.success) == ("Citation", 1.0, True)
    assert (boom.score, boom.success) == (None, False)
    assert "invalid JSON" in boom.error


def test_assert_passed_explains_each_failure():
    outcomes = run_metrics("rag", "c1", tc("no citation"), [CitationMetric()])
    with pytest.raises(AssertionError, match=r"c1: 1/1 metric\(s\) failed\n  Citation: 0.000"):
        assert_passed(outcomes)


def o(metric, score, success=True, suite="rag", lower=False):
    return MetricOutcome(suite, "c", metric, score, 0.5, success, "r", lower_is_better=lower)


def test_aggregate():
    summary = aggregate([o("F", 1.0), o("F", 0.5, False), o("F", None, False)])
    assert summary["rag"]["F"] == {
        "mean": 0.75,
        "pass_rate": 0.3333,
        "n": 3,
        "errors": 0,
        "lower_is_better": False,
    }


def base(mean, pass_rate=1.0, lower=False):
    return {"mean": mean, "pass_rate": pass_rate, "n": 5, "errors": 0, "lower_is_better": lower}


@pytest.mark.parametrize(
    ("baseline", "current", "regressed"),
    [
        (base(0.80), base(0.78), False),  # within tolerance
        (base(0.80), base(0.70), True),  # mean dropped
        (base(0.80), base(0.90), False),  # improved
        (base(0.10, lower=True), base(0.30, lower=True), True),  # bias went UP
        (base(0.30, lower=True), base(0.10, lower=True), False),  # bias went down
        (base(0.80, 1.0), base(0.80, 0.6), True),  # same mean, more failures
        (base(0.80), base(None), True),  # every case errored
    ],
)
def test_gate_compare(baseline, current, regressed):
    result = compare({"rag": {"M": current}}, {"rag": {"M": baseline}}, tolerance=0.05)
    assert result.passed is not regressed


def test_gate_reports_but_does_not_fail_on_metrics_not_run():
    result = compare({}, {"rag": {"M": base(0.8)}})
    assert result.passed
    assert result.missing == ["rag/M"]


def test_gate_end_to_end(tmp_path, capsys):
    reports = tmp_path / "reports"
    rec = Recorder()
    rec.add([o("Faithfulness", 0.9), o("Faithfulness", 0.8)])
    rec.write(reports)
    assert len(load_outcomes(reports)) == 2
    baseline = tmp_path / "baselines" / "baseline.json"

    assert gate_main(["--reports-dir", str(reports), "--baseline", str(baseline), "--record"]) == 0
    assert json.loads(baseline.read_text())["rag"]["Faithfulness"]["mean"] == 0.85
    assert (reports / "summary.md").exists()

    rec2 = Recorder()
    rec2.add([o("Faithfulness", 0.5, False), o("Faithfulness", 0.6)])
    rec2.write(reports)
    assert gate_main(["--reports-dir", str(reports), "--baseline", str(baseline)]) == 1
    assert "REGRESSION rag/Faithfulness" in capsys.readouterr().out


def test_gate_with_no_outcomes_exits_2(tmp_path):
    assert gate_main(["--reports-dir", str(tmp_path)]) == 2
