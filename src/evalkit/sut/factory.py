"""Build the bundled apps for the configured SUT provider.

SUT_PROVIDER=scripted wires each app to its deterministic policy. That is how
the unit tier runs, and it is also useful in the llm tier to check the judge
against outputs you already know are right or wrong.
"""

from __future__ import annotations

from evalkit.config import Settings, get_settings
from evalkit.sut.agent_app import ToolAgent, keyword_policy
from evalkit.sut.chat_app import SupportChatbot, memory_policy
from evalkit.sut.llm_client import ChatModel, ScriptedChat, build_chat_model
from evalkit.sut.rag_app import RAGPipeline, extractive_policy


def _model(settings: Settings, scripted_policy) -> ChatModel:
    if settings.sut_provider == "scripted":
        return ScriptedChat(scripted_policy)
    return build_chat_model(settings)


def build_rag_app(settings: Settings | None = None) -> RAGPipeline:
    return RAGPipeline(_model(settings or get_settings(), extractive_policy))


def build_agent_app(settings: Settings | None = None) -> ToolAgent:
    return ToolAgent(_model(settings or get_settings(), keyword_policy))


def build_chat_app(settings: Settings | None = None) -> SupportChatbot:
    return SupportChatbot(_model(settings or get_settings(), memory_policy))
