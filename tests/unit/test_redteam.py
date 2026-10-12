import asyncio
from types import SimpleNamespace

import pytest
from deepteam.attacks.single_turn import PromptInjection, Roleplay
from deepteam.test_case import RTTurn
from deepteam.vulnerabilities import Bias, PromptLeakage

from evalkit.config import get_settings
from evalkit.redteam import RedTeamConfig, build_attacks, load_config, summarize
from evalkit.redteam.runner import chat_callback
from evalkit.sut.base import ChatMessage


def test_shipped_config_resolves_to_deepteam_classes():
    config = load_config(get_settings().redteam_config_path)
    vulns = __import__("evalkit.redteam", fromlist=["x"]).build_vulnerabilities(config)
    assert any(isinstance(v, PromptLeakage) for v in vulns)
    assert any(isinstance(v, Bias) for v in vulns)
    attacks = build_attacks(config)
    assert any(isinstance(a, PromptInjection) for a in attacks)
    assert any(isinstance(a, Roleplay) for a in attacks)
    assert 0 < config.min_pass_rate <= 1


def test_typo_in_config_fails_before_any_model_call(tmp_path):
    path = tmp_path / "r.yaml"
    path.write_text(
        "target_purpose: x\nvulnerabilities: [{name: PromtLeakage}]\n"
        "attacks: [{name: PromptInjection}]\n"
    )
    with pytest.raises(ValueError, match="Unknown deepteam vulnerability: PromtLeakage"):
        load_config(path)


def test_unknown_attack_is_rejected():
    config = RedTeamConfig("x", [{"name": "Bias"}], [{"name": "MagicJailbreak"}])
    with pytest.raises(ValueError, match="Unknown deepteam attack"):
        build_attacks(config)


def test_callback_replays_attack_history_into_the_app():
    seen = {}

    class App:
        role_description = "r"

        def reply(self, history, message):
            seen["history"], seen["message"] = history, message
            return "no"

    turns = [RTTurn(role="user", content="a"), RTTurn(role="assistant", content="b")]
    out = asyncio.run(chat_callback(App())("c", turns))
    assert (out.role, out.content) == ("assistant", "no")
    assert seen["history"] == [ChatMessage("user", "a"), ChatMessage("assistant", "b")]
    assert seen["message"] == "c"


def _assessment(results, cases):
    return SimpleNamespace(
        overview=SimpleNamespace(vulnerability_type_results=results), test_cases=cases
    )


def _vt(name, vtype, passing, failing, errored=0):
    return SimpleNamespace(
        vulnerability=name,
        vulnerability_type=SimpleNamespace(value=vtype),
        pass_rate=0,
        passing=passing,
        failing=failing,
        errored=errored,
    )


def _case(score, error=None):
    return SimpleNamespace(
        vulnerability="Bias",
        vulnerability_type=SimpleNamespace(value="race"),
        attack_method="Roleplay",
        input="i",
        actual_output="o",
        reason="r",
        score=score,
        error=error,
    )


def test_summarize_counts_and_lists_failures():
    s = summarize(
        _assessment(
            [_vt("Bias", "race", 3, 1), _vt("PromptLeakage", "instructions", 4, 0, errored=2)],
            [_case(1), _case(0), _case(None, error="timeout")],
        )
    )
    assert (s.passing, s.failing, s.errored, s.scored) == (7, 1, 2, 8)
    assert s.pass_rate == 7 / 8
    assert len(s.failures) == 1
    assert s.by_vulnerability["Bias/race"]["failing"] == 1


def test_all_errored_run_has_no_pass_rate():
    s = summarize(_assessment([_vt("Bias", "race", 0, 0, errored=4)], []))
    assert s.pass_rate is None
    assert s.scored == 0
