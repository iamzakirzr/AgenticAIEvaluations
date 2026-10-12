from evalkit.sut.base import ChatMessage
from evalkit.sut.chat_app import ROLE_DESCRIPTION


def test_history_is_sent_to_the_model(scripted_chat):
    history = [ChatMessage("user", "hi"), ChatMessage("assistant", "hello")]
    scripted_chat.reply(history, "How much is Pro?")
    sent = scripted_chat.model.calls[-1]
    assert [m["role"] for m in sent] == ["system", "user", "assistant", "user"]
    assert ROLE_DESCRIPTION in sent[0]["content"]
    assert sent[-1]["content"].endswith("Question: How much is Pro?")


def test_scripted_bot_recalls_facts_from_earlier_turns(scripted_chat):
    history = [
        ChatMessage("user", "Hi, my name is Priya."),
        ChatMessage("assistant", "Thanks, noted."),
    ]
    assert scripted_chat.reply(history, "What is my name?") == "Your name is Priya."


def test_scripted_bot_answers_from_docs(scripted_chat):
    answer = scripted_chat.reply([], "Where is customer data stored by default?")
    assert "Frankfurt" in answer
    assert "[security.md]" in answer
