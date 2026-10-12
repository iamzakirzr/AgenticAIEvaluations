"""DeepTeam red teaming driven by config/redteam.yaml.

The attacker (simulator) and the grader (evaluator) are both the configured
judge model. A small local model makes a weak attacker, so a clean result from
an 8B model is weak evidence; run this against a stronger provider before
trusting a pass.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from deepeval.models import DeepEvalBaseLLM
from deepteam import red_team
from deepteam import vulnerabilities as vuln_module
from deepteam.attacks import multi_turn, single_turn
from deepteam.test_case import RTTurn

from evalkit.sut.base import ChatApp, ChatMessage


@dataclass
class RedTeamConfig:
    target_purpose: str
    vulnerabilities: list[dict[str, Any]]
    attacks: list[dict[str, Any]]
    attacks_per_vulnerability_type: int = 1
    min_pass_rate: float = 0.8


def load_config(path: Path) -> RedTeamConfig:
    data = yaml.safe_load(path.read_text())
    config = RedTeamConfig(**data)
    if not config.vulnerabilities or not config.attacks:
        raise ValueError(f"{path}: needs at least one vulnerability and one attack")
    # Resolve names now so a typo fails before any model is called.
    build_vulnerabilities(config)
    build_attacks(config)
    return config


def build_vulnerabilities(config: RedTeamConfig) -> list:
    out = []
    for spec in config.vulnerabilities:
        cls = getattr(vuln_module, spec["name"], None)
        if cls is None:
            raise ValueError(f"Unknown deepteam vulnerability: {spec['name']}")
        out.append(cls(types=spec["types"]) if spec.get("types") else cls())
    return out


def build_attacks(config: RedTeamConfig) -> list:
    out = []
    for spec in config.attacks:
        cls = getattr(single_turn, spec["name"], None) or getattr(multi_turn, spec["name"], None)
        if cls is None:
            raise ValueError(f"Unknown deepteam attack: {spec['name']}")
        out.append(cls(weight=spec.get("weight", 1), **spec.get("params", {})))
    return out


def chat_callback(app: ChatApp) -> Callable:
    """Adapt a ChatApp to DeepTeam's (input, turns) -> RTTurn callback.

    Multi-turn attacks pass the conversation so far, which is replayed as
    history so the target sees the same context it would in production.
    """

    async def callback(prompt: str, turns: list[RTTurn] | None = None) -> RTTurn:
        history = [
            ChatMessage(t.role, t.content) for t in turns or [] if t.role in ("user", "assistant")
        ]
        return RTTurn(role="assistant", content=app.reply(history, prompt))

    return callback


@dataclass
class RedTeamSummary:
    passing: int
    failing: int
    errored: int
    by_vulnerability: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def scored(self) -> int:
        return self.passing + self.failing

    @property
    def pass_rate(self) -> float | None:
        # Errored attacks are excluded from the rate but reported: a run where
        # every attack errored has proven nothing and must not read as 100%.
        return self.passing / self.scored if self.scored else None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "pass_rate": self.pass_rate}


def summarize(risk_assessment: Any) -> RedTeamSummary:
    by_vuln: dict[str, dict[str, Any]] = {}
    for r in risk_assessment.overview.vulnerability_type_results:
        key = f"{r.vulnerability}/{getattr(r.vulnerability_type, 'value', r.vulnerability_type)}"
        by_vuln[key] = {
            "pass_rate": r.pass_rate,
            "passing": r.passing,
            "failing": r.failing,
            "errored": r.errored,
        }
    failures = [
        {
            "vulnerability": tc.vulnerability,
            "type": str(getattr(tc.vulnerability_type, "value", tc.vulnerability_type)),
            "attack": tc.attack_method,
            "input": tc.input,
            "output": tc.actual_output,
            "reason": tc.reason,
        }
        for tc in risk_assessment.test_cases
        if not tc.error and tc.score is not None and tc.score <= 0
    ]
    return RedTeamSummary(
        passing=sum(v["passing"] for v in by_vuln.values()),
        failing=sum(v["failing"] for v in by_vuln.values()),
        errored=sum(v["errored"] for v in by_vuln.values()),
        by_vulnerability=by_vuln,
        failures=failures,
    )


def run_red_team(
    app: ChatApp, config: RedTeamConfig, judge: DeepEvalBaseLLM, reports_dir: Path
) -> RedTeamSummary:
    assessment = red_team(
        model_callback=chat_callback(app),
        vulnerabilities=build_vulnerabilities(config),
        attacks=build_attacks(config),
        simulator_model=judge,
        evaluation_model=judge,
        attacks_per_vulnerability_type=config.attacks_per_vulnerability_type,
        target_purpose=config.target_purpose,
        ignore_errors=True,
        max_concurrent=4,
    )
    summary = summarize(assessment)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "redteam.json").write_text(json.dumps(summary.to_dict(), indent=2))
    return summary
