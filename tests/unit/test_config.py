import pytest

from evalkit.config import Settings, Thresholds, get_thresholds


def test_shipped_thresholds_cover_every_metric_the_suites_use():
    t = get_thresholds()
    for suite, metric in [
        ("rag", "faithfulness"),
        ("rag", "citation"),
        ("agent", "tool_correctness"),
        ("agent", "step_efficiency"),
        ("chat", "knowledge_retention"),
        ("safety", "bias"),
    ]:
        assert 0 <= t.get(suite, metric) <= 1


def test_missing_threshold_names_the_file_to_edit():
    with pytest.raises(KeyError, match=r"config/thresholds\.yaml"):
        Thresholds({"rag": {}}).get("rag", "faithfulness")


@pytest.mark.parametrize("value", [1.5, -0.1, "high"])
def test_out_of_range_threshold_is_rejected(tmp_path, value):
    path = tmp_path / "t.yaml"
    path.write_text(f"rag:\n  faithfulness: {value!r}\n")
    with pytest.raises(ValueError, match="must be a number"):
        Thresholds.load(path)


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("JUDGE_MODEL", "claude-sonnet-5-5")
    s = Settings(_env_file=None)
    assert (s.judge_provider, s.judge_model) == ("anthropic", "claude-sonnet-5-5")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("JUDGE_PROVIDER", "gpt-banana")
    with pytest.raises(ValueError):
        Settings(_env_file=None)
