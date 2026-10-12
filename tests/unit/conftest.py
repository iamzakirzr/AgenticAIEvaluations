import pytest

from evalkit.config import get_settings
from evalkit.sut.agent_app import ToolAgent, keyword_policy
from evalkit.sut.chat_app import SupportChatbot, memory_policy
from evalkit.sut.llm_client import ScriptedChat
from evalkit.sut.rag_app import RAGPipeline, TfidfRetriever, extractive_policy, load_corpus


@pytest.fixture(scope="session")
def retriever():
    return TfidfRetriever(load_corpus())


@pytest.fixture
def scripted_rag(retriever):
    return RAGPipeline(ScriptedChat(extractive_policy), retriever)


@pytest.fixture
def scripted_agent(retriever):
    from evalkit.sut.agent_app import default_tools

    return ToolAgent(ScriptedChat(keyword_policy), default_tools(retriever))


@pytest.fixture
def scripted_chat(retriever):
    return SupportChatbot(ScriptedChat(memory_policy), retriever)


@pytest.fixture(scope="session")
def goldens_dir():
    return get_settings().goldens_dir
