"""
Wiring RAGAS 0.4 to a local Ollama model.

=============================================================================
READ THIS FIRST: THE TWO THINGS THAT WILL BREAK YOUR RAGAS CODE
=============================================================================

--- 1. `import ragas` fails outright on a modern LangChain stack -------------

    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

ragas 0.4.3 has a module-level import of a class that langchain-community 0.4.x
removed, and it declares langchain-community as an UNPINNED dependency. So a
clean install is broken out of the box.

`core.compat.bootstrap()` installs a stub so the import succeeds. It MUST run
before the first `import ragas` in the process, which is why every file in this
lesson calls it at the very top -- and why ruff's E402 (imports at top of file)
is disabled in pyproject.toml. See core/compat.py for the full explanation.

--- 2. Every tutorial you find uses the DEPRECATED metric API ---------------

The classic form, which is everywhere online:

    from ragas.metrics import faithfulness, context_precision   # instances
    from ragas import evaluate
    evaluate(dataset, metrics=[faithfulness])

still half-works in 0.4.3 but emits:

    DeprecationWarning: Importing X from 'ragas.metrics' is deprecated and
    will be removed in v1.0. Please use 'ragas.metrics.collections' instead.

The current API is `ragas.metrics.collections`, which exports 39 metric
CLASSES that you instantiate with an llm and call with explicit arguments:

    from ragas.metrics.collections import Faithfulness
    metric = Faithfulness(llm=judge)
    result = await metric.ascore(
        user_input=..., response=..., retrieved_contexts=[...]
    )

This is a better API -- the arguments are explicit and type-checked instead of
being pulled from a loosely-specified dataset dict -- but it means almost every
RAGAS example you find is out of date. This lesson uses the new one.

=============================================================================
WHY THIS TALKS TO OLLAMA DIFFERENTLY THAN LESSON 04 DID
=============================================================================
Lesson 04's DeepEval judge uses Ollama's NATIVE /api/chat, because it needs the
`format` parameter for schema-constrained decoding.

RAGAS instead builds judges through `instructor`, which speaks the OpenAI
protocol. So here we point an OpenAI client at Ollama's OpenAI-COMPATIBLE
endpoint at /v1.

Same model, same server, two different endpoints, because two libraries want
different things. Forcing both through one wrapper would have broken one of
them -- which is the concrete version of the argument against a "unified"
abstraction layer.

The upside: because RAGAS only needs an OpenAI-compatible endpoint, this exact
code works against vLLM, LM Studio, llama.cpp's server, or any hosted
OpenAI-compatible API. Change the base URL, nothing else.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# MUST run before any ragas import anywhere in the process.
from core.compat import bootstrap

bootstrap()

from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory

from core.config import settings
from core.providers import get_openai_compatible_client


def build_ragas_llm(model: str | None = None):
    """A RAGAS judge backed by a local Ollama model.

    `llm_factory` returns an `InstructorLLM`: a wrapper that uses the
    `instructor` library to coerce model output into pydantic schemas. That
    matters for the same reason it mattered in lesson 04 -- RAGAS metrics like
    faithfulness decompose an answer into claims and need structured output at
    every step, and unstructured local models fail exactly there.

    provider="openai" does NOT mean OpenAI the company. It means "speaks the
    OpenAI protocol", and the injected client decides where the requests go.
    """
    return llm_factory(
        model=model or settings.judge_model,
        provider="openai",
        client=get_openai_compatible_client(),
        # Temperature 0: a judge with sampling noise makes your metric noisy,
        # which destroys your ability to detect small regressions.
        temperature=settings.temperature,
    )


def build_ragas_embeddings(model: str | None = None):
    """Embeddings for the metrics that need them.

    AnswerRelevancy is the main one: it asks the model to generate questions
    the answer would suit, then measures their embedding similarity to the real
    question. So it needs BOTH an llm and an embedder, and forgetting the
    second is a common setup error.
    """
    return embedding_factory(
        provider="openai",
        model=model or settings.embed_model,
        client=get_openai_compatible_client(),
    )


def ragas_ready() -> tuple[bool, str]:
    """Can we actually run judged RAGAS metrics right now?

    Returns (ready, reason). Used by tests to skip with a precise message
    rather than failing deep inside instructor with a connection error.
    """
    from core.providers import installed_ollama_models, ollama_available

    if not ollama_available():
        return False, f"Ollama is not reachable at {settings.ollama_base_url}"

    installed = installed_ollama_models()
    missing = [
        name
        for name in (settings.judge_model, settings.embed_model)
        if name not in installed
    ]
    if missing:
        return False, f"missing Ollama models: {missing}. Run: ollama pull {' '.join(missing)}"
    return True, "ready"
