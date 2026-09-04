"""
A DeepEval judge backed by a local Ollama model.

=============================================================================
WHAT DEEPEVAL REQUIRES OF A CUSTOM MODEL (verified against 4.2.1, not guessed)
=============================================================================
`DeepEvalBaseLLM` declares four abstract methods: `load_model`, `generate`,
`a_generate`, `get_model_name`.

The part that is NOT obvious from the docs: metrics do not call `generate`
directly. They call `generate_with_schema(prompt, schema=SomePydanticModel)`,
whose default implementation is:

    def generate_with_schema(self, *args, schema=None, **kwargs):
        if schema is not None:
            try:
                return self.generate(*args, schema=schema, **kwargs)
            except TypeError:
                pass          # provider doesn't accept a schema kwarg
        return self.generate(*args, **kwargs)

and the caller then does:

    if isinstance(result, schema_cls): use it directly
    else:                              parse the string as JSON

So there are two ways to satisfy DeepEval: return a validated pydantic object
(reliable), or return a JSON string it will parse for you (fragile). WE RETURN
THE PYDANTIC OBJECT, because the fragile path is precisely where small local
models fail.

=============================================================================
WHY STRUCTURED OUTPUT IS THE WHOLE BALLGAME FOR A LOCAL JUDGE
=============================================================================
core/corpus/llm_as_judge.md:

    "Open-weight models in the 7 to 8 billion parameter range ... are
     unreliable on tasks requiring structured output over several steps, such
     as claim-by-claim faithfulness. Two failure modes dominate: malformed
     JSON that cannot be parsed, and inconsistent verdicts across identical
     repeated calls."

A judge that is asked politely for JSON will produce prose, markdown fences and
trailing commentary. So we do not ask politely: Ollama's native `/api/chat`
accepts a `format` parameter containing a JSON SCHEMA, and constrains decoding
so the output must satisfy it. That converts "usually valid JSON" into "always
valid JSON", which is the single highest-leverage thing you can do to make a
small local judge usable.

(This is why the judge talks to Ollama's NATIVE api rather than its
OpenAI-compatible /v1 shim: `format` lives on the native endpoint. Lesson 05
uses the /v1 shim instead, because RAGAS goes through `instructor`, which
speaks the OpenAI protocol. Two libraries, two different correct answers --
that is normal, and pretending one wrapper fits both would break something.)

=============================================================================
THE RULE ABOUT FAILURES
=============================================================================
    "A judge failure must never be silently recorded as a score of zero. That
     converts an infrastructure problem into a fake quality regression and
     corrupts the baseline you compare against."

This module therefore RAISES on unrecoverable judge failure and counts every
failure on `JudgeStats`. It never invents a score. Suites report judge failures
separately from quality scores.
=============================================================================
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# DeepEval phones home and prompts for a cloud login unless told not to. Set
# BEFORE importing deepeval, or the import-time telemetry client is already up.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_UPDATE_WARNING_OPT_OUT", "YES")

import httpx  # noqa: E402
from deepeval.models import DeepEvalBaseLLM  # noqa: E402
from pydantic import BaseModel, ValidationError  # noqa: E402

from core.config import settings  # noqa: E402


class JudgeFailure(RuntimeError):
    """Raised when the judge could not produce a usable verdict.

    Deliberately an exception rather than a sentinel score. A score of 0.0 for
    "the judge broke" is indistinguishable from 0.0 for "the answer was
    terrible", and mixing the two silently poisons every baseline you compare
    against afterwards.
    """


@dataclass
class JudgeStats:
    """Operational counters, reported alongside (never mixed into) scores."""

    calls: int = 0
    schema_retries: int = 0
    failures: int = 0
    total_prompt_chars: int = 0
    failure_examples: list[str] = field(default_factory=list)

    def report(self) -> str:
        rate = (self.failures / self.calls * 100) if self.calls else 0.0
        lines = [
            f"judge calls        : {self.calls}",
            f"schema retries     : {self.schema_retries}",
            f"unrecovered fails  : {self.failures}  ({rate:.1f}%)",
        ]
        for example in self.failure_examples[:3]:
            lines.append(f"  failure: {example[:120]}")
        return "\n".join(lines)


class OllamaJudge(DeepEvalBaseLLM):
    """Scores DeepEval metrics using a local model, with schema enforcement.

    Usage:
        judge = OllamaJudge()
        metric = FaithfulnessMetric(model=judge, threshold=0.7)
    """

    def __init__(self, model: str | None = None, temperature: float | None = None) -> None:
        self.model_name = model or settings.judge_model
        # Temperature 0 for a judge is not a preference, it is a requirement.
        # core/corpus/llm_as_judge.md: sampling noise in a judge makes your
        # metric noisy, which destroys your ability to detect small regressions.
        self.temperature = settings.temperature if temperature is None else temperature
        self.stats = JudgeStats()
        self._client = httpx.Client(timeout=settings.request_timeout)
        super().__init__(model=self.model_name)

    # ---- DeepEvalBaseLLM interface ----------------------------------------

    def load_model(self) -> "OllamaJudge":
        """Required by the ABC. Ollama is a server, so there is nothing to load."""
        return self

    def get_model_name(self) -> str:
        """Appears in DeepEval's output. Include the settings that affect scores.

        Two eval runs judged by different models are not comparable, so the
        model identity must travel with the results.
        """
        return f"ollama/{self.model_name}@t{self.temperature}"

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **_: Any):
        """Return a validated pydantic object when a schema is given, else text."""
        self.stats.calls += 1
        self.stats.total_prompt_chars += len(prompt)

        if schema is None:
            return self._chat(prompt)

        # ---- attempt 1: constrained decoding against the JSON schema -------
        raw = self._chat(prompt, json_schema=schema.model_json_schema())
        parsed = self._try_parse(raw, schema)
        if parsed is not None:
            return parsed

        # ---- attempt 2: repair -------------------------------------------
        # Constrained decoding almost always succeeds, but a model can still
        # emit a structurally valid object with a semantically wrong field
        # (an empty required list, a string where a float belongs). One repair
        # attempt is worth it; more is throwing good compute after bad.
        self.stats.schema_retries += 1
        repair = (
            f"{prompt}\n\n"
            f"Your previous reply could not be parsed:\n{raw[:600]}\n\n"
            f"Reply with ONLY a JSON object matching this schema, nothing else:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        raw2 = self._chat(repair, json_schema=schema.model_json_schema())
        parsed = self._try_parse(raw2, schema)
        if parsed is not None:
            return parsed

        # ---- give up, LOUDLY ----------------------------------------------
        self.stats.failures += 1
        self.stats.failure_examples.append(raw2)
        raise JudgeFailure(
            f"{self.model_name} could not satisfy schema {schema.__name__} after a "
            f"repair attempt. Last reply: {raw2[:300]!r}\n"
            f"This is an INFRASTRUCTURE failure, not a quality score of 0. "
            f"Try a larger judge model (JUDGE_MODEL=qwen2.5:14b) or a simpler metric."
        )

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any):
        """Async variant. DeepEval runs metrics concurrently by default.

        Delegating to the sync path keeps one implementation of the retry and
        failure-accounting logic. It costs concurrency, which for a local
        single-GPU Ollama server you did not have anyway -- requests queue on
        the server regardless.
        """
        return self.generate(prompt, schema=schema, **kwargs)

    # ---- capability hints DeepEval consults --------------------------------

    def supports_json_mode(self) -> bool:
        return True

    def supports_structured_outputs(self) -> bool:
        return True

    def supports_temperature(self) -> bool:
        return True

    # ---- transport ---------------------------------------------------------

    def _chat(self, prompt: str, json_schema: dict | None = None) -> str:
        """One call to Ollama's NATIVE chat endpoint.

        The `format` field is the important part: passing a JSON schema makes
        Ollama constrain token sampling so the output must validate against it.
        This is the difference between a local judge that mostly works and one
        that is genuinely usable.
        """
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_schema is not None:
            payload["format"] = json_schema

        try:
            response = self._client.post(
                f"{settings.ollama_base_url}/api/chat", json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            self.stats.failures += 1
            raise JudgeFailure(
                f"Ollama request failed: {exc}. Is `ollama serve` running at "
                f"{settings.ollama_base_url}, and is {self.model_name!r} pulled?"
            ) from exc

        return response.json().get("message", {}).get("content", "")

    @staticmethod
    def _try_parse(raw: str, schema: type[BaseModel]) -> BaseModel | None:
        """Validate a reply into the schema, tolerating common wrappers."""
        if not raw or not raw.strip():
            return None

        text = raw.strip()

        # Strip markdown fences, which models add even under constrained
        # decoding when the schema permits a leading string.
        if text.startswith("```"):
            text = text.split("```")[1] if "```" in text[3:] else text[3:]
            text = text.removeprefix("json").strip()

        try:
            return schema.model_validate_json(text)
        except ValidationError:
            pass

        # Last resort: pull out the outermost {...} and try again. Handles a
        # model that prepended "Here is the JSON:".
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return schema.model_validate_json(text[start : end + 1])
            except ValidationError:
                return None
        return None


class ExplodingJudge(DeepEvalBaseLLM):
    """A judge that fails if it is ever called. Used to PROVE a metric is non-LLM.

    DeepEval 4.2.1 has a trap: several metrics that perform a purely
    deterministic comparison -- ToolCorrectnessMetric among them -- still
    construct a default OpenAI model during __init__ and raise if
    OPENAI_API_KEY is unset, even though they never actually call it.

    Passing this class satisfies the constructor without an API key, and turns
    "I believe this metric is deterministic" into a test that fails loudly the
    day that stops being true.
    """

    def load_model(self) -> "ExplodingJudge":
        return self

    def get_model_name(self) -> str:
        return "exploding-judge (should never be called)"

    def generate(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError(
            "A metric documented as non-LLM called the judge. It is not "
            "deterministic, and it cannot run in the fast CI tier."
        )

    async def a_generate(self, *args: Any, **kwargs: Any) -> str:
        return self.generate(*args, **kwargs)
