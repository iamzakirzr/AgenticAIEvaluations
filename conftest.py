"""
Root pytest configuration -- shared fixtures and skip logic for every lesson.

=============================================================================
THE TWO-TIER TEST STRATEGY (this is the design idea worth stealing)
=============================================================================
Tests in this repo fall into tiers, separated by pytest markers:

  FAST TIER (no marker)
      No model, no network, fully deterministic, runs in about a second.
      Covers: retrieval metrics, chunking, dataset integrity, prompt
      assembly, graph routing, metric plumbing, report generation.
      Runs on every push in CI.

  JUDGED TIER (@pytest.mark.judge)
      Calls a real LLM to score outputs. Slow, non-deterministic, needs a
      running Ollama server. Run on demand and nightly, never as a PR gate.

  OLLAMA TIER (@pytest.mark.ollama)
      Needs a real model but not as a judge -- real embeddings, real
      generation. Deterministic-ish at temperature 0 but still slow.

  SAAS TIER (@pytest.mark.saas)
      Needs a third-party account (LangWatch). Never runs in CI.

Why this matters: an evaluation suite that takes 20 minutes and fails randomly
gets disabled within a fortnight. One that runs in one second and never lies
gets trusted, and the slow judged tier stays credible because it is not being
asked to do a job it is bad at.

Run them:
    pytest                    # fast tier only (the default)
    pytest -m ollama          # real models, no judging
    pytest -m judge           # the expensive judged metrics
    pytest -m "not saas"      # everything you can run locally
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make `core`, `01_embeddings`, etc. importable no matter where pytest is run
# from. Directories starting with digits are not valid Python identifiers, so
# lessons import shared code via `core.*` and keep their own modules local.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import settings
from core.golden import load_golden
from core.providers import (
    LexicalEmbeddings,
    installed_ollama_models,
    ollama_available,
)

# ---------------------------------------------------------------------------
# Skip helpers -- give a PRECISE reason, never a generic failure.
# ---------------------------------------------------------------------------
# A test that fails with "ConnectionRefused" deep inside httpx teaches nothing.
# A test that skips with "Ollama is not running at http://localhost:11434"
# tells you exactly what to do next.
# ---------------------------------------------------------------------------

requires_ollama = pytest.mark.skipif(
    not ollama_available(),
    reason=(
        f"Ollama is not reachable at {settings.ollama_base_url}. "
        "Start it with `ollama serve`, then pull the models: "
        f"`ollama pull {settings.chat_model} && ollama pull {settings.embed_model}`"
    ),
)


def requires_model(model_name: str):
    """Skip unless a specific model tag has been pulled locally."""
    installed = installed_ollama_models()
    return pytest.mark.skipif(
        not ollama_available() or model_name not in installed,
        reason=(
            f"Model {model_name!r} is not available. "
            f"Run `ollama pull {model_name}`. Currently installed: {installed or 'none'}"
        ),
    )


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden():
    """The full labelled dataset, loaded once per test session."""
    return load_golden()


@pytest.fixture(scope="session")
def answerable(golden):
    """Only the items the corpus can actually answer."""
    return [item for item in golden if item.is_answerable]


@pytest.fixture(scope="session")
def unanswerable(golden):
    """Only the items where the correct behaviour is refusal."""
    return [item for item in golden if not item.is_answerable]


@pytest.fixture
def lexical_embeddings():
    """Fresh, offline, deterministic embeddings for the fast tier."""
    return LexicalEmbeddings(dim=512)


def pytest_report_header(config):
    """Print the active configuration at the top of every test run.

    Small thing, large payoff: eval results are meaningless without knowing
    which models and which chunk size produced them, and printing it here means
    you can never accidentally compare two runs with different settings.
    """
    return [
        f"chat model : {settings.chat_model}",
        f"judge model: {settings.judge_model}",
        f"embed model: {settings.embed_model}",
        f"chunking   : size={settings.chunk_size} overlap={settings.chunk_overlap} top_k={settings.top_k}",
        f"ollama     : {'UP' if ollama_available() else 'DOWN'} at {settings.ollama_base_url}",
    ]
