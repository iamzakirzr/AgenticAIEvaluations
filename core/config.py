"""
core.config -- one place where every knob in the curriculum lives.

Every lesson reads its model names, URLs and retrieval parameters from here
rather than hard-coding them. That is not just tidiness: in lesson 04 and 05
you will deliberately change ``CHUNK_SIZE`` and ``TOP_K`` and watch evaluation
metrics move. If those numbers were scattered across ten files you could not
run that experiment.

Everything is overridable with an environment variable, so you can do:

    CHUNK_SIZE=200 pytest -m judge 05_ragas/

...and compare the scores against the default run without editing any code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Project root = the directory containing this package's parent.
ROOT = Path(__file__).resolve().parent.parent

# Where the knowledge base the RAG pipeline indexes lives.
CORPUS_DIR = ROOT / "core" / "corpus"

# Where the labelled evaluation questions live.
GOLDEN_PATH = ROOT / "core" / "golden.jsonl"

# Scratch space for vector indexes, eval reports, cached runs. Git-ignored.
ARTIFACTS_DIR = ROOT / ".artifacts"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:  # fail loudly rather than silently using default
        raise ValueError(f"Environment variable {name}={raw!r} is not an integer") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name}={raw!r} is not a float") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of configuration, built from the environment."""

    # ---------------- Ollama connection -----------------------------------
    # Ollama exposes TWO HTTP APIs on the same port:
    #   - its native API at /api/*        -> used by langchain-ollama
    #   - an OpenAI-compatible API at /v1 -> used by ragas and by anything
    #                                        that speaks the OpenAI protocol
    # We keep both, because different libraries in this curriculum need
    # different ones. This is a genuinely important detail: RAGAS 0.4 cannot
    # talk to Ollama's native API at all, only the /v1 shim.
    ollama_base_url: str = field(default_factory=lambda: _env_str("OLLAMA_BASE_URL", "http://localhost:11434"))

    # ---------------- Models ----------------------------------------------
    # The model that ANSWERS questions (the system under test).
    chat_model: str = field(default_factory=lambda: _env_str("CHAT_MODEL", "llama3.1:8b"))

    # The model that SCORES answers (the judge). Deliberately configurable and
    # deliberately separate: lesson 04 shows why judging with the same model
    # that generated the answer inflates your scores (self-preference bias).
    judge_model: str = field(default_factory=lambda: _env_str("JUDGE_MODEL", "llama3.1:8b"))

    # The model that turns text into vectors.
    embed_model: str = field(default_factory=lambda: _env_str("EMBED_MODEL", "nomic-embed-text"))

    # ---------------- Retrieval parameters (the experiment knobs) ---------
    # Characters per chunk. Too large -> retrieved context is mostly noise,
    # which tanks context precision. Too small -> facts get split across
    # chunks, which tanks context recall. Lesson 01 demonstrates both.
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 700))

    # Characters of overlap between neighbouring chunks, so a sentence that
    # straddles a boundary still appears whole in at least one chunk.
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))

    # How many chunks to retrieve per question.
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 4))

    # ---------------- Generation ------------------------------------------
    # Temperature 0 for BOTH the answerer and the judge. Non-zero temperature
    # on a judge is a common and expensive mistake: it makes your metric noisy
    # for no benefit, so you can no longer tell a real regression from sampling
    # variance.
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.0))

    # Seconds before we give up on a model call.
    request_timeout: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT", 120.0))

    @property
    def ollama_openai_base_url(self) -> str:
        """Ollama's OpenAI-compatible endpoint.

        RAGAS 0.4 builds its judge through `instructor`, which wraps an OpenAI
        client. Pointing that client here is how RAGAS talks to a local model.
        """
        return f"{self.ollama_base_url.rstrip('/')}/v1"


# A module-level instance is fine because Settings is frozen and env-driven.
settings = Settings()

# Make sure the scratch directory exists so lessons can write reports.
ARTIFACTS_DIR.mkdir(exist_ok=True)
