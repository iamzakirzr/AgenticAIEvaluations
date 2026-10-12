import pytest

from evalkit.sut.rag_app import load_corpus


def test_corpus_is_chunked_by_section():
    chunks = load_corpus()
    assert len(chunks) == 15
    assert {c.source for c in chunks} == {
        "plans.md",
        "refunds.md",
        "security.md",
        "support.md",
        "accounts.md",
    }
    assert all(c.text and not c.text.startswith("#") for c in chunks)


@pytest.mark.parametrize(
    ("query", "source", "section"),
    [
        ("How much does the Pro plan cost?", "plans.md", "Plans"),
        ("annual subscription refund window", "refunds.md", "Eligibility"),
        (
            "Can I use SMS for two-factor authentication?",
            "security.md",
            "Two-factor authentication",
        ),
        ("service credit when uptime is low", "support.md", "Uptime commitment"),
        ("How do I restore my account after deleting it?", "accounts.md", "Deleting an account"),
    ],
)
def test_top_hit_is_the_right_section(retriever, query, source, section):
    top, _ = retriever.search(query, k=1)[0]
    assert (top.source, top.section) == (source, section)


def test_lexical_retrieval_misses_synonyms(retriever):
    """Known limitation, kept visible: 'price' never appears in the corpus ('costs' does).

    This is the failure mode embeddings fix. If you swap in a semantic
    retriever, flip this assertion.
    """
    assert retriever.search("price", k=1) == []


def test_unrelated_query_retrieves_nothing(retriever):
    assert retriever.search("recipe for banana bread", k=3) == []


def test_render_prefixes_the_source_for_citation(retriever):
    chunk, _ = retriever.search("Pro plan cost", k=1)[0]
    assert chunk.render().startswith("[plans.md] Plans: ")
