"""
core.compat -- dependency compatibility shims.

=============================================================================
WHY THIS FILE EXISTS (read this, it is a real lesson, not boilerplate)
=============================================================================

When you install ragas 0.4.3 alongside a current LangChain stack, this happens:

    >>> import ragas
    ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'

Here is the actual chain of causes:

  1. ragas/llms/base.py has a MODULE-LEVEL import:
         from langchain_community.chat_models.vertexai import ChatVertexAI
     It only uses ChatVertexAI to populate a list of LLM classes that support
     n>1 completions. It is not needed for anything we do.

  2. `langchain-community` has been sunset (it prints a DeprecationWarning on
     import telling you so). In 0.4.x its Vertex AI chat model was removed --
     that integration now lives in the standalone `langchain-google-vertexai`
     package.

  3. ragas declares `langchain-community` as an UNPINNED dependency. So pip
     happily installs 0.4.2, and ragas then explodes on import.

This is worth internalising because it is the single most common way an
evaluation stack breaks in practice: **an eval library pins loosely against a
fast-moving orchestration library, and a transitive upgrade breaks you.** In a
real job, you will hit exactly this and be expected to diagnose it in minutes.

-----------------------------------------------------------------------------
THE THREE WAYS TO FIX IT, AND WHY WE CHOSE #3
-----------------------------------------------------------------------------

  (1) Downgrade langchain-community until the module exists again.
      Rejected: it drags langchain-core back below 1.0, which breaks
      langchain 1.4 and langgraph 1.2 -- the very things lesson 02 and 03
      teach. You cannot have both.

  (2) Downgrade ragas to a version that predates the bad import.
      Rejected: older ragas has a different (deprecated) metric API, so the
      lesson would teach you an API that is on its way out.

  (3) Register a stub module before ragas is imported.  <-- WHAT WE DO
      We insert a fake `langchain_community.chat_models.vertexai` into
      sys.modules containing a placeholder ChatVertexAI class. ragas's
      `isinstance(llm, ChatVertexAI)` check then simply always returns False,
      which is correct for us -- we never use Vertex AI. If anyone ever *does*
      try to construct it, our placeholder raises a loud, explanatory error
      rather than failing mysteriously.

The cost of (3) is that it is a monkeypatch and monkeypatches rot. So it is
isolated in this one file, it is loud about what it does, and
`test_compat.py` asserts it is still necessary -- when a future ragas removes
the bad import, that test fails and tells you to delete this shim.
=============================================================================
"""

from __future__ import annotations

import sys
import types

# Module path that ragas 0.4.3 imports but langchain-community 0.4.x no longer
# provides. Keep this as a constant so the test can reference the same string.
_MISSING_MODULE = "langchain_community.chat_models.vertexai"


class _UnavailableChatVertexAI:
    """Placeholder standing in for langchain_community's removed ChatVertexAI.

    ragas only uses the real class in an ``isinstance()`` check against a list
    of LLM types that support requesting multiple completions in one call.
    Since no LLM in this repo is a Vertex AI model, that check must return
    False -- and it does, because nothing we build is an instance of this
    class.

    Constructing it is always a mistake, so we fail loudly with a message that
    explains the situation rather than letting a subtle bug through.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "ChatVertexAI is not available in this project.\n"
            "This is a compatibility placeholder installed by core.compat so "
            "that `import ragas` succeeds. This repo evaluates local Ollama "
            "models, not Google Vertex AI. If you genuinely need Vertex AI, "
            "install `langchain-google-vertexai` and import ChatVertexAI from "
            "there instead."
        )


# Marker attribute so we can recognise our own stub in sys.modules and never
# mistake it for the real module.
_STUB_MARKER = "__core_compat_stub__"


def is_shim_needed() -> bool:
    """Return True if the REAL langchain_community Vertex AI module is absent.

    Used by the test suite: when this starts returning False, the upstream
    problem is fixed and `install_ragas_langchain_shim()` can be deleted.

    SUBTLETY worth understanding, because it is a bug we actually shipped and
    then caught: the obvious implementation is

        try: __import__(_MISSING_MODULE); return False
        except ImportError: return True

    ...which is WRONG once the shim is installed. Our stub lives in
    sys.modules, so `__import__` finds it and happily reports success -- the
    function then claims the shim is unnecessary precisely because the shim is
    working. The result is a test that passes or fails depending on whether
    anything imported ragas earlier in the session.

    So we check for our marker first and only fall back to a real import.
    """
    existing = sys.modules.get(_MISSING_MODULE)
    if existing is not None:
        # If it is our stub, the real module is still missing.
        return getattr(existing, _STUB_MARKER, False)

    try:
        __import__(_MISSING_MODULE)
    except ImportError:
        return True
    return False


def install_ragas_langchain_shim() -> bool:
    """Make ``import ragas`` work on a modern LangChain stack.

    Registers a stub module at ``langchain_community.chat_models.vertexai`` if
    and only if the real one is missing.

    IMPORTANT: this must run BEFORE the first ``import ragas`` anywhere in the
    process. That is why lesson files in 05_ragas/ call ``bootstrap()`` at the
    very top, before their ragas imports -- and why ruff's "imports must be at
    the top of the file" rule (E402) is disabled in pyproject.toml.

    Returns:
        True if a stub was installed, False if the real module was importable
        and nothing needed to be done.
    """
    if not is_shim_needed():
        return False

    # Already installed by a previous call in this process -- stay idempotent.
    if _MISSING_MODULE in sys.modules:
        return True

    stub = types.ModuleType(_MISSING_MODULE)
    stub.ChatVertexAI = _UnavailableChatVertexAI
    stub.__doc__ = (
        "Compatibility stub installed by core.compat. Not a real integration."
    )
    # Marker so is_shim_needed() can tell our stub from the genuine article.
    setattr(stub, _STUB_MARKER, True)
    sys.modules[_MISSING_MODULE] = stub
    return True


def bootstrap() -> None:
    """Single entry point every lesson calls before touching third-party libs.

    Today it only installs the ragas shim. It exists as a named function so
    that when the next incompatibility appears (and one will), there is an
    obvious place to put the fix and every lesson picks it up for free.
    """
    install_ragas_langchain_shim()
