"""Build the judge LLM for the configured provider, and say whether it is usable.

`judge_status` is what lets the llm tier skip with a clear reason instead of
erroring 40 times on a connection refused.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from deepeval.models import (
    AnthropicModel,
    AzureOpenAIModel,
    DeepEvalBaseLLM,
    OllamaModel,
    OpenAIModel,
)

from evalkit.config import Settings, get_settings


@dataclass(frozen=True)
class Status:
    ok: bool
    reason: str


def get_judge(settings: Settings | None = None) -> DeepEvalBaseLLM:
    s = settings or get_settings()
    # temperature=0 everywhere: judge variance is the main source of flaky evals.
    if s.judge_provider == "ollama":
        return OllamaModel(model=s.judge_model, base_url=s.ollama_base_url, temperature=0)
    if s.judge_provider == "openai":
        return OpenAIModel(model=s.judge_model, api_key=s.openai_api_key, temperature=0)
    if s.judge_provider == "anthropic":
        return AnthropicModel(model=s.judge_model, api_key=s.anthropic_api_key, temperature=0)
    if s.judge_provider == "azure":
        return AzureOpenAIModel(
            model=s.judge_model,
            api_key=s.azure_openai_api_key,
            base_url=s.azure_openai_endpoint,
            deployment_name=s.azure_openai_deployment,
            api_version=s.azure_openai_api_version,
            temperature=0,
        )
    raise ValueError(f"Unknown judge provider: {s.judge_provider!r}")


def ollama_status(base_url: str, model: str) -> Status:
    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
        resp.raise_for_status()
    except Exception as exc:
        return Status(False, f"Ollama not reachable at {base_url} ({type(exc).__name__})")
    names = {m.get("name", "") for m in resp.json().get("models", [])}
    # `ollama pull llama3.1:8b` is listed as "llama3.1:8b"; a bare "llama3.1"
    # is listed as "llama3.1:latest".
    wanted = {model, f"{model}:latest"}
    if not names & wanted:
        return Status(False, f"Ollama model '{model}' not pulled (run: ollama pull {model})")
    return Status(True, f"ollama:{model}")


def judge_status(settings: Settings | None = None) -> Status:
    s = settings or get_settings()
    if s.judge_provider == "ollama":
        return ollama_status(s.ollama_base_url, s.judge_model)
    required = {
        "openai": [("OPENAI_API_KEY", s.openai_api_key)],
        "anthropic": [("ANTHROPIC_API_KEY", s.anthropic_api_key)],
        "azure": [
            ("AZURE_OPENAI_API_KEY", s.azure_openai_api_key),
            ("AZURE_OPENAI_ENDPOINT", s.azure_openai_endpoint),
            ("AZURE_OPENAI_DEPLOYMENT", s.azure_openai_deployment),
        ],
    }[s.judge_provider]
    missing = [name for name, value in required if not value]
    if missing:
        return Status(False, f"judge provider '{s.judge_provider}' needs {', '.join(missing)}")
    return Status(True, f"{s.judge_provider}:{s.judge_model}")
