#!/usr/bin/env python
"""
Does a similarity threshold separate real questions from nonsense? MEASURE IT.

    python 08_mcp/measure_threshold.py

=============================================================================
WHY THIS SCRIPT EXISTS
=============================================================================
"Add a relevance threshold so the system refuses when retrieval is weak" is
advice you will read everywhere, give in an interview, and probably implement.
It is only sound if the two score distributions actually separate, and almost
nobody checks.

On this corpus with hashed lexical embeddings they do not separate at all. The
output below is the evidence. Run it against YOUR embedder and corpus before
you ship a threshold -- the method transfers even though the numbers will not.

Being able to say "I measured the overlap before choosing the threshold, and on
that corpus there wasn't one" is a materially stronger interview answer than
naming a number.
=============================================================================
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.golden import load_golden
from core.providers import LexicalEmbeddings, cosine_similarity

GIBBERISH_ALPHABET = "zqxwkjv"  # letters that are rare in the corpus


def gibberish(rng: random.Random) -> str:
    words = [
        "".join(rng.choice(GIBBERISH_ALPHABET) for _ in range(rng.randint(4, 9)))
        for _ in range(rng.randint(2, 6))
    ]
    return " ".join(words)


def top_scores(seed: int = 0, samples: int = 200) -> tuple[list[float], list[float]]:
    docs: list[str] = []
    for path in sorted((_ROOT / "core" / "corpus").glob("*.md")):
        docs.extend(p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if len(p.strip()) > 80)

    embedder = LexicalEmbeddings()
    matrix = embedder.embed_documents(docs)

    def best(query: str) -> float:
        vector = embedder.embed_query(query)
        return max(cosine_similarity(vector, row) for row in matrix)

    real = [best(item.question) for item in load_golden() if item.is_answerable]
    rng = random.Random(seed)
    noise = [best(gibberish(rng)) for _ in range(samples)]
    return real, noise


def main() -> int:
    real, noise = top_scores()
    real_sorted, noise_sorted = sorted(real), sorted(noise)

    print("Top-1 similarity score, per query")
    print(f"  real questions  n={len(real):<4} min {min(real):.3f}  median {statistics.median(real):.3f}  max {max(real):.3f}")
    print(f"  gibberish       n={len(noise):<4} min {min(noise):.3f}  median {statistics.median(noise):.3f}  max {max(noise):.3f}")
    print()

    overlap = max(noise) > min(real)
    print(f"  gibberish max {max(noise):.3f} {'>' if overlap else '<='} real min {min(real):.3f}"
          f"  ->  distributions {'OVERLAP' if overlap else 'separate'}")
    print()
    print("  threshold   real rejected   gibberish accepted")
    for threshold in (0.20, 0.25, 0.30, 0.35, 0.40):
        rejected = sum(1 for value in real_sorted if value < threshold) / len(real_sorted)
        accepted = sum(1 for value in noise_sorted if value >= threshold) / len(noise_sorted)
        print(f"     {threshold:.2f}       {rejected:>7.1%}        {accepted:>8.1%}")

    print()
    if overlap:
        print("VERDICT: no threshold separates the two. Every choice trades")
        print("genuine questions refused against nonsense answered. Use the score")
        print("as a hint, and put the refusal decision somewhere deterministic.")
    else:
        print("VERDICT: the distributions separate -- a threshold is defensible here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
