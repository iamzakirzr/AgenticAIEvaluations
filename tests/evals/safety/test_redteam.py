"""DeepTeam red teaming of the support chatbot, configured by config/redteam.yaml."""

import pytest

from evalkit.redteam import load_config, run_red_team

pytestmark = pytest.mark.redteam


def test_red_team(chat_app, judge, settings):
    config = load_config(settings.redteam_config_path)
    summary = run_red_team(chat_app, config, judge, settings.reports_dir)

    assert summary.scored > 0, (
        f"No attack was scored ({summary.errored} errored). A run that proves nothing "
        "must not pass; check the judge model can follow DeepTeam's JSON schemas."
    )
    failures = "\n".join(
        f"  {f['vulnerability']}/{f['type']} via {f['attack']}: {f['reason']}"
        for f in summary.failures
    )
    assert summary.pass_rate >= config.min_pass_rate, (
        f"Defended {summary.passing}/{summary.scored} attacks "
        f"({summary.pass_rate:.0%} < {config.min_pass_rate:.0%}).\n{failures}"
    )
