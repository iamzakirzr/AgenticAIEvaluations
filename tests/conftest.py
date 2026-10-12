"""Shared fixtures and the tier logic.

- The default run (`pytest`) is the unit tier: `llm` and `redteam` tests are
  deselected by addopts in pyproject.toml.
- When `llm`/`redteam` tests ARE selected but no judge or SUT model is
  reachable, they are skipped with the exact reason, instead of erroring once
  per test case on a connection refused.
"""

from __future__ import annotations

import os

import pytest

import evalkit  # noqa: F401  (sets telemetry opt-out before deepeval loads)
from evalkit.config import Settings, Thresholds, get_settings, get_thresholds
from evalkit.judges import get_judge, judge_status
from evalkit.judges.factory import Status, ollama_status
from evalkit.reporting.results import Recorder
from evalkit.runner import MetricOutcome, assert_passed

_RECORDER = Recorder()
LIVE_MARKERS = ("llm", "redteam")


def sut_status(s: Settings) -> Status:
    if s.sut_provider == "scripted":
        return Status(True, "scripted")
    if s.sut_provider == "ollama":
        return ollama_status(s.ollama_base_url, s.sut_model)
    if not s.openai_api_key:
        return Status(False, "SUT_PROVIDER=openai needs OPENAI_API_KEY")
    return Status(True, f"openai:{s.sut_model}")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    live = [i for i in items if any(i.get_closest_marker(m) for m in LIVE_MARKERS)]
    if not live:
        return
    s = get_settings()
    for status, what in ((judge_status(s), "judge"), (sut_status(s), "system under test")):
        if not status.ok:
            skip = pytest.mark.skip(reason=f"{what} unavailable: {status.reason}")
            for item in live:
                item.add_marker(skip)
            return


def pytest_sessionstart(session):
    # Clear the previous run's outcomes (controller process only under xdist).
    if os.environ.get("PYTEST_XDIST_WORKER") is None:
        reports = get_settings().reports_dir
        for stale in reports.glob("outcomes-*.json"):
            stale.unlink()


def pytest_sessionfinish(session, exitstatus):
    path = _RECORDER.write(get_settings().reports_dir)
    if path and os.environ.get("PYTEST_XDIST_WORKER") is None:
        print(f"\nMetric outcomes written to {path}. Run `make gate` to aggregate.")


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def thresholds() -> Thresholds:
    return get_thresholds()


@pytest.fixture(scope="session")
def judge(settings):
    return get_judge(settings)


@pytest.fixture(scope="session")
def record():
    """Record outcomes for the report, then assert they all passed."""

    def _record(outcomes: list[MetricOutcome]) -> None:
        _RECORDER.add(outcomes)
        assert_passed(outcomes)

    return _record


# --- Systems under test ------------------------------------------------------
# To evaluate your own application, return your adapter from these fixtures
# (see evalkit.sut.base for the Protocols). Nothing else needs to change.


@pytest.fixture(scope="session")
def rag_app(settings):
    from evalkit.sut.factory import build_rag_app

    return build_rag_app(settings)


@pytest.fixture(scope="session")
def agent_app(settings):
    from evalkit.sut.factory import build_agent_app

    return build_agent_app(settings)


@pytest.fixture(scope="session")
def chat_app(settings):
    from evalkit.sut.factory import build_chat_app

    return build_chat_app(settings)
