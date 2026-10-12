"""Settings from the environment (or `.env`) and thresholds from YAML.

Everything that varies between a laptop, CI and production lives here, so
tests never hard-code a model name, a URL or a pass mark.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]

Provider = Literal["ollama", "openai", "anthropic", "azure"]
SutProvider = Literal["ollama", "openai", "scripted"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # The judge scores outputs. It should be at least as capable as the system
    # under test; a weak judge produces confident-looking noise.
    judge_provider: Provider = "ollama"
    judge_model: str = "llama3.1:8b"

    # The system under test. "scripted" runs the bundled apps with a
    # deterministic rule-based model, which is how the unit tier works.
    sut_provider: SutProvider = "ollama"
    sut_model: str = "llama3.1:8b"

    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_api_key: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment: str | None = None
    azure_openai_api_version: str = "2024-10-21"

    request_timeout_s: float = 120.0

    thresholds_path: Path = ROOT / "config" / "thresholds.yaml"
    redteam_config_path: Path = ROOT / "config" / "redteam.yaml"
    goldens_dir: Path = ROOT / "data" / "goldens"
    reports_dir: Path = ROOT / "reports"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class Thresholds:
    """Read-only view over config/thresholds.yaml: `suite -> metric -> float`."""

    def __init__(self, data: dict[str, dict[str, float]]):
        self._data = data

    @classmethod
    def load(cls, path: Path) -> Thresholds:
        with path.open() as fh:
            data = yaml.safe_load(fh) or {}
        for suite, metrics in data.items():
            if not isinstance(metrics, dict):
                raise ValueError(f"{path}: suite '{suite}' must map metric names to numbers")
            for name, value in metrics.items():
                if not isinstance(value, int | float) or not 0 <= value <= 1:
                    raise ValueError(f"{path}: {suite}.{name}={value!r} must be a number in [0, 1]")
        return cls(data)

    def get(self, suite: str, metric: str) -> float:
        try:
            return float(self._data[suite][metric])
        except KeyError:
            raise KeyError(
                f"No threshold for '{suite}.{metric}'. Add it to config/thresholds.yaml."
            ) from None


@lru_cache(maxsize=1)
def get_thresholds() -> Thresholds:
    return Thresholds.load(get_settings().thresholds_path)
