"""
ADVANCED RAGAS: generating a test set instead of hand-writing one.

=============================================================================
THE PROBLEM
=============================================================================
A golden dataset is the most valuable artefact in an eval setup and the most
tedious to produce. Hand-writing 200 questions with reference answers is days
of work, and the result reflects what YOU imagined users would ask.

RAGAS's `TestsetGenerator` reads your documents and generates questions,
including multi-hop ones, by building a knowledge graph over the corpus and
walking it.

=============================================================================
WHAT IT IS GOOD AT, AND WHAT IT IS NOT
=============================================================================
GOOD FOR BULK. Getting from 20 questions to 200 makes your means less noisy
and your category slices non-trivial. That is real value.

BAD FOR DISCOVERY. A model reading your corpus generates questions phrased in
the corpus's own vocabulary. That is exactly the bias that makes this repo's
recall metric saturate: the questions are easy because they are written from
the answers.

Concretely, a generator will not produce:
  - the question a confused user asks with the wrong terminology
  - the false-premise question that catches sycophancy
  - the question your corpus genuinely cannot answer

Those three categories are where the bugs are, and all three have to be
written by a human who knows the domain.

SO: generate for volume, hand-write for the failures. And a human confirms
every reference answer, or the dataset simply certifies current behaviour as
correct.
=============================================================================
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.compat import bootstrap

bootstrap()

from core.golden import GoldenItem, iter_corpus


def build_testset_generator(llm=None, embeddings=None):
    """A RAGAS TestsetGenerator wired to the local Ollama model.

    Signature verified against ragas 0.4.3:

        TestsetGenerator(llm, embedding_model, knowledge_graph=..., persona_list=...)

    `persona_list` is worth knowing about: it generates questions as different
    user types ("a new customer", "an engineer debugging an outage"), which is
    the closest the generator gets to producing genuine variety rather than
    rephrasings of the same lookup.
    """
    from ragas.testset import TestsetGenerator
    from ragas_setup import build_ragas_embeddings, build_ragas_llm

    return TestsetGenerator(
        llm=llm or build_ragas_llm(),
        embedding_model=embeddings or build_ragas_embeddings(),
    )


def corpus_as_langchain_documents(limit: int | None = None):
    """The corpus in the Document shape TestsetGenerator expects."""
    from langchain_core.documents import Document

    docs = []
    for doc_id, text in iter_corpus():
        docs.append(Document(page_content=text, metadata={"doc_id": doc_id}))
        if limit and len(docs) >= limit:
            break
    return docs


def generate_testset(size: int = 10, docs=None):
    """Generate `size` questions from the corpus. Requires a live model."""
    generator = build_testset_generator()
    return generator.generate_with_langchain_docs(
        docs if docs is not None else corpus_as_langchain_documents(),
        testset_size=size,
    )


@dataclass
class ReviewQueue:
    """Generated items, staged for human review rather than merged.

    THE RULE THIS ENFORCES: a generated question may enter the dataset, but a
    generated ANSWER may not become ground truth. The generator wrote it by
    reading the same corpus the system retrieves from, so accepting it makes
    the dataset circular -- it would certify the system's current behaviour as
    correct and hide every existing bug permanently.
    """

    items: list[GoldenItem]
    source: str = "ragas.TestsetGenerator"

    @classmethod
    def from_testset(cls, testset, category: str = "single_hop") -> ReviewQueue:
        items: list[GoldenItem] = []
        rows = testset.to_pandas().to_dict(orient="records") if hasattr(testset, "to_pandas") else []

        for index, row in enumerate(rows, start=1):
            question = row.get("user_input") or row.get("question") or ""
            if not question:
                continue
            items.append(
                GoldenItem(
                    id=f"gen-{index:03d}",
                    question=str(question),
                    # NOT row["reference"], deliberately. See the class docstring.
                    reference_answer="TODO: a human must confirm this",
                    reference_doc_ids=[],
                    category=category,
                    difficulty=2,
                    rationale="generated by RAGAS; needs human review",
                )
            )
        return cls(items=items)

    def to_jsonl(self, path: Path) -> Path:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {
                    "id": item.id,
                    "question": item.question,
                    "reference_answer": item.reference_answer,
                    "reference_doc_ids": item.reference_doc_ids,
                    "category": item.category,
                    "difficulty": item.difficulty,
                    "rationale": item.rationale,
                },
                ensure_ascii=False,
            )
            for item in self.items
        ]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def coverage_gaps(self) -> list[str]:
        """Which categories the generator did NOT produce.

        Always run this before congratulating yourself on 200 generated
        questions. A generator reading your corpus produces answerable,
        in-vocabulary lookups -- so the categories that catch real bugs will be
        missing, and their absence is invisible unless you check.
        """
        present = {item.category for item in self.items}
        required = {"single_hop", "multi_hop", "unanswerable", "adversarial"}
        return sorted(required - present)
