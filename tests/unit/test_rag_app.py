from evalkit.sut.rag_app import REFUSAL


def test_answer_quotes_retrieved_text_and_cites_it(scripted_rag):
    r = scripted_rag.query("How much does the Pro plan cost?")
    assert "8 USD per user per month" in r.answer
    assert r.answer.endswith("[plans.md]")
    assert "plans.md" in r.sources
    assert all(c.startswith("[") for c in r.contexts)


def test_refuses_when_nothing_relevant_is_retrieved(scripted_rag):
    r = scripted_rag.query("Who is the CEO of Nimbus Notes?")
    assert r.answer == REFUSAL


def test_prompt_carries_instructions_context_and_question(scripted_rag):
    scripted_rag.query("Is there a minimum number of users on the Team plan?")
    system, user = scripted_rag.model.calls[-1]
    assert "cite its source" in system["content"]
    assert "[plans.md]" in user["content"]
    assert user["content"].rstrip().endswith("Team plan?")


def test_latency_is_recorded(scripted_rag):
    assert scripted_rag.query("Pro plan price").latency_s > 0
