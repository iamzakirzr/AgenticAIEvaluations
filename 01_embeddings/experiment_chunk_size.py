"""
EXPERIMENT: prove that a configuration change moves an evaluation metric.

Run it:   make experiment-chunking
          (or:  .venv/bin/python 01_embeddings/experiment_chunk_size.py)

=============================================================================
WHY THIS FILE IS THE POINT OF THE WHOLE REPOSITORY
=============================================================================
Anyone can install a metric library and print a number. The skill that gets
you hired is being able to say:

    "I changed X. Metric Y moved by Z. Here is the mechanism, and here is why
     that trade is or is not worth taking."

That sentence is what this script produces. It sweeps chunk size across a
range, re-indexes the corpus at each setting, scores retrieval against the
golden dataset, and prints the curve.

WHAT YOU SHOULD SEE, AND WHY:

  Very small chunks (150 chars)
      A single fact gets split across a boundary, so no one chunk contains a
      whole answer. Some questions become unanswerable no matter how good the
      retriever is. -> RECALL falls.
      Meanwhile each retrieved chunk is tiny and tightly on-topic, so the
      fraction of retrieved text that is relevant goes UP.

  Very large chunks (2500 chars)
      Almost every chunk contains the answer to something, so recall is easy.
      But each retrieved chunk drags in paragraphs about unrelated topics.
      -> PRECISION falls, and the generator's context fills with noise.

  Somewhere in between is the setting that suits YOUR corpus. There is no
  universal answer, which is exactly why you measure instead of guessing.

A NOTE ON WHAT THIS EXPERIMENT CANNOT SEE:
These are DOCUMENT-level metrics. Recall stays high as long as the right
*document* is retrieved, even if the specific *sentence* was cut in half. So
document-level recall UNDERSTATES the damage small chunks do. The judged
metrics in 04_deepeval and 05_ragas (faithfulness, context recall) operate on
chunk text and will show the rest of the story. Knowing what a metric is blind
to is as important as knowing what it measures.
=============================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mini_rag import MiniRAG

from core.golden import load_golden
from core.metrics import evaluate_retrieval
from core.providers import LexicalEmbeddings

# Swept values. Overlap is held at ~17% of size so that we vary ONE thing at a
# time -- changing two variables at once makes the result uninterpretable, which
# is the most common flaw in amateur eval experiments.
CHUNK_SIZES = [150, 300, 500, 700, 1000, 1500, 2500]
TOP_K = 4


def run_one(chunk_size: int, top_k: int = TOP_K) -> dict[str, float]:
    """Index at one chunk size, score every answerable question, return metrics."""
    overlap = max(0, int(chunk_size * 0.17))

    rag = MiniRAG(
        embeddings=LexicalEmbeddings(dim=2048),
        chunk_size=chunk_size,
        overlap=overlap,
    ).index()

    golden = [item for item in load_golden() if item.is_answerable]
    results = {
        item.id: (rag.trace(item.question, top_k=top_k).retrieved_doc_ids, item.reference_doc_ids)
        for item in golden
    }
    report = evaluate_retrieval(results, k=top_k)

    return {
        "chunks": float(len(rag.chunks)),
        "mean_chars": sum(len(c.text) for c in rag.chunks) / len(rag.chunks),
        "recall": report.recall,
        "precision": report.precision,
        "mrr": report.mrr,
        "ndcg": report.ndcg,
        "n_failures": float(len(report.failures())),
    }


def main() -> None:
    print("=" * 84)
    print("CHUNK SIZE SWEEP -- retrieval quality against the golden dataset")
    print("=" * 84)
    print(f"top_k = {TOP_K}, overlap held at 17% of chunk size, lexical embeddings\n")

    header = (
        f"{'size':>6}{'chunks':>8}{'avg chars':>11}"
        f"{'recall':>9}{'precision':>11}{'MRR':>8}{'nDCG':>8}{'misses':>8}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for size in CHUNK_SIZES:
        m = run_one(size)
        rows.append((size, m))
        print(
            f"{size:>6}{int(m['chunks']):>8}{m['mean_chars']:>11.0f}"
            f"{m['recall']:>9.3f}{m['precision']:>11.3f}"
            f"{m['mrr']:>8.3f}{m['ndcg']:>8.3f}{int(m['n_failures']):>8}"
        )

    # ---- Interpretation, generated from the actual numbers ----------------
    print("\n" + "=" * 84)
    print("READING THE RESULT")
    print("=" * 84)

    best_recall = max(rows, key=lambda r: r[1]["recall"])
    best_precision = max(rows, key=lambda r: r[1]["precision"])
    smallest, largest = rows[0], rows[-1]

    print(f"  best recall    : size={best_recall[0]} at {best_recall[1]['recall']:.3f}")
    print(f"  best precision : size={best_precision[0]} at {best_precision[1]['precision']:.3f}")
    print()
    print(
        f"  Going from size={smallest[0]} to size={largest[0]}:\n"
        f"    recall    {smallest[1]['recall']:.3f} -> {largest[1]['recall']:.3f}  "
        f"({largest[1]['recall'] - smallest[1]['recall']:+.3f})\n"
        f"    precision {smallest[1]['precision']:.3f} -> {largest[1]['precision']:.3f}  "
        f"({largest[1]['precision'] - smallest[1]['precision']:+.3f})\n"
        f"    chunks    {int(smallest[1]['chunks'])} -> {int(largest[1]['chunks'])}"
    )

    if best_recall[0] != best_precision[0]:
        print(
            "\n  The two metrics peak at DIFFERENT settings. That is the trade-off,\n"
            "  visible in real numbers rather than described in the abstract. Which\n"
            "  one you optimise depends on your downstream generator: a model with a\n"
            "  long context and good noise resistance can afford lower precision;\n"
            "  a small local model cannot."
        )
    else:
        print(
            "\n  Both metrics peak at the same setting here -- convenient, but do not\n"
            "  generalise it. On a corpus with longer documents or more topic mixing\n"
            "  within a document, they usually diverge."
        )

    recall_spread = max(r[1]["recall"] for r in rows) - min(r[1]["recall"] for r in rows)
    if recall_spread < 0.05:
        print(
            f"\n  HONEST CAVEAT: recall barely moves across the whole sweep (spread"
            f" {recall_spread:.3f}).\n"
            "  It is effectively saturated. That is a property of the DATASET, not\n"
            "  evidence that chunk size does not matter. Two causes:\n"
            "    1. The corpus has 8 topically distinct documents, so picking the right\n"
            "       DOCUMENT is easy even when the right SENTENCE was cut in half.\n"
            "    2. The golden questions reuse vocabulary from the source documents,\n"
            "       which flatters lexical retrieval.\n"
            "  A saturated metric cannot detect regressions. Fixes, in order of value:\n"
            "    - watch precision and MRR instead, which are NOT saturated here\n"
            "    - add paraphrased questions that avoid the source's wording\n"
            "    - measure at chunk granularity, which the judged metrics in\n"
            "      04_deepeval and 05_ragas do\n"
            "  Noticing that your own metric has no headroom, and saying so, is the\n"
            "  difference between running an eval and understanding one."
        )

    print("\n  Reproduce any row directly:")
    print("    CHUNK_SIZE=300 .venv/bin/python 01_embeddings/walkthrough.py")


if __name__ == "__main__":
    main()
