"""Section-level chunking and a TF-IDF retriever, in plain Python.

A real deployment would use embeddings and a vector store. For a reference
SUT, exact lexical retrieval over a tiny corpus is easier to reason about when
a contextual-precision score drops: you can recompute it by hand.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from evalkit.sut.text import tokens

CORPUS_DIR = Path(__file__).parent / "corpus"


@dataclass(frozen=True)
class Chunk:
    source: str  # file name, used for citations
    section: str
    text: str

    def render(self) -> str:
        return f"[{self.source}] {self.section}: {self.text}"


def load_corpus(directory: Path = CORPUS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        section, lines = None, []
        for line in path.read_text().splitlines():
            if line.startswith("## "):
                if section and lines:
                    chunks.append(Chunk(path.name, section, " ".join(lines)))
                section, lines = line[3:].strip(), []
            elif line.strip() and not line.startswith("# "):
                lines.append(line.strip())
        if section and lines:
            chunks.append(Chunk(path.name, section, " ".join(lines)))
    if not chunks:
        raise ValueError(f"No '## ' sections found in {directory}")
    return chunks


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        docs = [tokens(f"{c.section} {c.text}") for c in chunks]
        df = Counter(t for doc in docs for t in set(doc))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = [self._vector(doc) for doc in docs]

    def _vector(self, toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        vec = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        q = self._vector(tokens(query))
        scored = [
            (chunk, sum(w * vec.get(t, 0.0) for t, w in q.items()))
            for chunk, vec in zip(self.chunks, self.vectors, strict=True)
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s >= min_score]
