"""
STEP-BY-STEP: what actually happens when text becomes a vector and gets found.

Run it:   make lesson-embeddings
          (or:  .venv/bin/python 01_embeddings/walkthrough.py)

This prints every intermediate state between "here is a sentence" and "here are
the retrieved chunks". Nothing is hidden and nothing is mocked -- the numbers
you see are the numbers the retriever uses.

Read the code alongside the output. Each STEP below maps to one printed block.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chunking import (  # noqa: E402
    chunk_stats,
    fixed_size_chunks,
    markdown_section_chunks,
    measure_overlap,
    recursive_chunks,
)
from mini_rag import MiniRAG  # noqa: E402

from core.golden import iter_corpus  # noqa: E402
from core.metrics import evaluate_retrieval  # noqa: E402
from core.providers import LexicalEmbeddings, cosine_similarity, tokenize  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sub(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 72 - len(title)))


# ===========================================================================
def step_1_tokenization() -> None:
    rule("STEP 1  TOKENIZATION -- text becomes a list of discrete symbols")
    print(
        "A model cannot do arithmetic on characters. The first move is always to\n"
        "cut text into TOKENS: the atomic units the rest of the pipeline counts.\n"
    )

    sentence = "Chunk overlap protects facts that straddle a boundary!"
    tokens = tokenize(sentence)

    print(f"raw text  : {sentence!r}")
    print(f"tokens    : {tokens}")
    print(f"vocabulary: {sorted(set(tokens))}  ({len(set(tokens))} unique)")

    sub("What we threw away, and why it matters")
    print(
        "Lowercasing merged 'Chunk' and 'chunk' into one token. That helps --\n"
        "you want them to match. It also merged 'US' (the country) with 'us'.\n"
        "Every tokenizer decision is a trade like this.\n\n"
        "Punctuation was dropped entirely. Fine for retrieval, disastrous if you\n"
        "were parsing code, where '!' and '!=' are meaningful.\n\n"
        "REAL MODELS USE SUBWORD TOKENIZATION (BPE) instead: 'tokenization' might\n"
        "become ['token', 'ization']. That gives a fixed vocabulary that can still\n"
        "represent words it has never seen, which whole-word tokenizers cannot."
    )


# ===========================================================================
def step_2_hashing() -> None:
    rule("STEP 2  FROM TOKENS TO VECTOR SLOTS -- the hashing trick")
    print(
        "We need a FIXED-LENGTH vector, but the vocabulary is open-ended. Two ways:\n\n"
        "  (a) Build a vocabulary dict word->index. Exact, but the dimension\n"
        "      depends on the corpus and grows as you add documents.\n"
        "  (b) HASH each token to a slot in [0, dim). Fixed dimension forever,\n"
        "      no dictionary to store or keep in sync. Cost: two words can\n"
        "      COLLIDE onto the same slot.\n\n"
        "We use (b). Below, watch distinct words land in distinct slots."
    )

    embedder = LexicalEmbeddings(dim=64)  # tiny dim so collisions are visible
    text = "chunk overlap protects facts at a boundary chunk chunk"
    tokens = tokenize(text)

    sub(f"Hashing into dim=64 (deliberately small to provoke collisions)")
    slots: dict[int, list[str]] = {}
    for token in tokens:
        from core.providers import _hash_token  # internal, shown on purpose

        slot = _hash_token(token, 64)
        slots.setdefault(slot, [])
        if token not in slots[slot]:
            slots[slot].append(token)

    for slot in sorted(slots):
        marker = "  <-- COLLISION" if len(slots[slot]) > 1 else ""
        print(f"  slot {slot:>3} <- {slots[slot]}{marker}")

    collisions = sum(1 for words in slots.values() if len(words) > 1)
    print(f"\n{collisions} collision(s) at dim=64. Real embedders use 384-1536 dimensions,")
    print("where collisions become rare enough to ignore.")

    sub("Term frequency: just counting")
    counts = Counter(tokens)
    print(f"  {dict(counts)}")
    print(
        "\nNote 'chunk' appears 3 times. We do NOT store 3 -- we store\n"
        "1 + log(3) = 2.10. That is SUBLINEAR term frequency: a word appearing\n"
        "100 times is not 100x as important as one appearing once."
    )


# ===========================================================================
def step_3_idf() -> None:
    rule("STEP 3  IDF -- why 'the' must not count as much as 'HNSW'")
    print(
        "Raw counts make every document look similar, because every document is\n"
        "full of 'the', 'is' and 'a'. Inverse Document Frequency fixes that by\n"
        "weighting each slot by how RARE it is across the corpus:\n\n"
        "    idf = log((1 + N) / (1 + df)) + 1\n\n"
        "  N  = number of documents\n"
        "  df = number of documents containing this term\n\n"
        "A term in every document gets a weight near 1.0 (no discriminating\n"
        "power). A term in one document out of eight gets a much higher weight."
    )

    docs = dict(iter_corpus())
    embedder = LexicalEmbeddings(dim=4096)  # large dim -> near-zero collisions
    embedder.embed_documents(list(docs.values()))

    from core.providers import _hash_token

    idf = embedder._idf()
    sub(f"IDF weights across the real {len(docs)}-document corpus")
    probes = ["the", "is", "a", "chunk", "hnsw", "kappa", "sycophancy"]
    print(f"  {'term':<14}{'df':>5}{'idf':>9}   interpretation")
    for term in probes:
        slot = _hash_token(term, 4096)
        df = embedder._doc_freq[slot]
        weight = idf[slot]
        note = "common, near-useless" if df >= len(docs) - 1 else (
            "rare, highly discriminating" if df <= 1 else "moderately specific"
        )
        print(f"  {term:<14}{int(df):>5}{weight:>9.3f}   {note}")

    print(
        "\nThis is why searching for 'the chunking document' works: 'the' and\n"
        "'document' contribute almost nothing, and 'chunking' does all the work."
    )


# ===========================================================================
def step_4_vector_and_cosine() -> None:
    rule("STEP 4  THE VECTOR, AND MEASURING THE ANGLE BETWEEN TWO OF THEM")

    docs = {
        "overlap": "Chunk overlap means consecutive chunks share text at their boundary.",
        "hnsw": "HNSW is a graph based approximate nearest neighbour index.",
        "cosine": "Cosine similarity measures the angle between two vectors.",
    }
    embedder = LexicalEmbeddings(dim=512)
    vectors = embedder.embed_documents(list(docs.values()))

    sub("One document's vector (512 dims, almost all zero)")
    vec = np.asarray(vectors[0])
    nonzero = np.nonzero(vec)[0]
    print(f"  text        : {docs['overlap']}")
    print(f"  dimension   : {len(vec)}")
    print(f"  non-zero    : {len(nonzero)} slots  ({len(nonzero)/len(vec):.1%} dense)")
    print(f"  L2 norm     : {np.linalg.norm(vec):.6f}   <- normalised to exactly 1")
    print(f"  first 5 hits: {[(int(i), round(float(vec[i]), 3)) for i in nonzero[:5]]}")
    print(
        "\nThis is a SPARSE vector -- mostly zeros, one slot per distinct word.\n"
        "Neural embeddings are DENSE: every one of the 768 dimensions carries a\n"
        "non-zero value, and no single dimension corresponds to a single word."
    )

    sub("Cosine similarity: query against each document")
    query = "what does chunk overlap do?"
    qvec = embedder.embed_query(query)
    print(f"  query: {query!r}\n")
    for (name, text), dvec in zip(docs.items(), vectors):
        score = cosine_similarity(qvec, dvec)
        bar = "#" * int(score * 50)
        print(f"  {score:>6.3f} {bar:<50} {name}")

    print(
        "\nThe overlap document wins because it shares the rare tokens 'chunk' and\n"
        "'overlap' with the query. That is the ENTIRE retrieval mechanism."
    )


# ===========================================================================
def step_5_where_lexical_breaks() -> None:
    rule("STEP 5  WHERE THIS BREAKS -- and why neural embeddings exist")

    docs = [
        "I drove my car to the office this morning.",
        "Prime factorisation underpins modern cryptography.",
    ]
    embedder = LexicalEmbeddings(dim=512)
    vectors = embedder.embed_documents(docs)

    print("Corpus:")
    for i, doc in enumerate(docs):
        print(f"  [{i}] {doc}")

    sub("Query A -- shares literal words")
    q1 = embedder.embed_query("I drove my car to the office")
    for i, dvec in enumerate(vectors):
        print(f"  doc[{i}]  {cosine_similarity(q1, dvec):.3f}")
    print("  -> correct: doc[0] wins.")

    sub("Query B -- SAME MEANING, different words")
    q2 = embedder.embed_query("I commuted by automobile to my workplace")
    for i, dvec in enumerate(vectors):
        print(f"  doc[{i}]  {cosine_similarity(q2, dvec):.3f}")
    print(
        "  -> 'automobile' hashes to a different slot than 'car'. 'commuted' to a\n"
        "     different slot than 'drove'. Lexically these sentences barely overlap,\n"
        "     even though a human reads them as the same statement."
    )

    print(
        "\nTHIS IS THE FAILURE THAT CREATED NEURAL EMBEDDINGS. A model trained on\n"
        "large text corpora learns to place 'car' and 'automobile' near each other\n"
        "in vector space, so paraphrase retrieval works.\n\n"
        "It is ALSO why hybrid search exists: lexical matching is still better at\n"
        "exact rare tokens -- error codes, SKUs, surnames -- where neural models\n"
        "blur distinctions. Production systems run both and fuse the rankings."
    )


# ===========================================================================
def step_6_chunking() -> None:
    rule("STEP 6  CHUNKING -- the same document, split four ways")

    doc_id, text = next(iter(iter_corpus()))
    print(f"Document: {doc_id}.md  ({len(text)} characters)\n")

    strategies = {
        "fixed, no overlap": fixed_size_chunks(text, doc_id, 700, 0),
        "fixed, 120 overlap": fixed_size_chunks(text, doc_id, 700, 120),
        "recursive, 120 overlap": recursive_chunks(text, doc_id, 700, 120),
        "markdown sections": markdown_section_chunks(text, doc_id, max_size=1200),
    }

    print(f"  {'strategy':<26}{'count':>6}{'mean':>8}{'min':>7}{'max':>7}{'stdev':>8}")
    for name, chunks in strategies.items():
        s = chunk_stats(chunks)
        print(
            f"  {name:<26}{int(s['count']):>6}{s['mean']:>8.0f}"
            f"{int(s['min']):>7}{int(s['max']):>7}{s['stdev']:>8.0f}"
        )

    sub("Overlap, verified rather than assumed")
    no_ov = strategies["fixed, no overlap"]
    with_ov = strategies["fixed, 120 overlap"]
    print(f"  fixed/no overlap   : chunk0->chunk1 shares {measure_overlap(no_ov[0], no_ov[1])} chars")
    print(f"  fixed/120 overlap  : chunk0->chunk1 shares {measure_overlap(with_ov[0], with_ov[1])} chars")
    print(
        "\n  Always MEASURE overlap rather than trusting the config value. A merge\n"
        "  step that silently drops overlap is a real bug, and nothing warns you --\n"
        "  your metrics just quietly get worse."
    )

    sub("What a hard cut destroys")
    print(f"  fixed-size chunk 0 ENDS   : ...{no_ov[0].text[-60:]!r}")
    print(f"  fixed-size chunk 1 BEGINS : {no_ov[1].text[:60]!r}...")
    print("\n  A sentence was guillotined. Neither chunk can answer a question about it.")
    print(f"\n  recursive chunk 0 ENDS    : ...{strategies['recursive, 120 overlap'][0].text[-60:]!r}")
    print("  -> ends on a real boundary, because recursive splitting tries")
    print("     paragraph breaks before it resorts to cutting mid-word.")


# ===========================================================================
def step_7_end_to_end() -> None:
    rule("STEP 7  END TO END -- retrieval scored against the golden dataset")

    from core.golden import load_golden

    rag = MiniRAG(embeddings=LexicalEmbeddings(dim=2048)).index()
    print(f"Indexed {len(rag.chunks)} chunks from the corpus.")
    print(f"Vector matrix shape: {rag.matrix.shape}  (chunks x dimensions)\n")

    golden = [item for item in load_golden() if item.is_answerable]

    sub("Three example queries")
    for item in golden[:3]:
        trace = rag.trace(item.question)
        hit = "HIT " if set(trace.retrieved_doc_ids) & set(item.reference_doc_ids) else "MISS"
        print(f"  [{hit}] {item.id}  {item.question[:58]}")
        print(f"         expected {item.reference_doc_ids} | got {trace.retrieved_doc_ids}")

    sub("Scored across every answerable question")
    results = {
        item.id: (rag.trace(item.question).retrieved_doc_ids, item.reference_doc_ids)
        for item in golden
    }
    report = evaluate_retrieval(results)
    print(report.format_table())

    failures = report.failures()
    if failures:
        print(f"\n  Questions where retrieval missed a required document ({len(failures)}):")
        lookup = {item.id: item for item in golden}
        for item_id in failures[:6]:
            print(f"    {item_id}  {lookup[item_id].question[:62]}")

    print(
        "\nTHIS IS THE NUMBER THAT MATTERS. Not a vibe check, not eyeballing three\n"
        "answers -- a reproducible score over a labelled dataset, computed with no\n"
        "LLM in the loop. It runs in milliseconds and costs nothing, which is why\n"
        "it can gate every pull request.\n\n"
        "These are lexical embeddings. Run `pytest -m ollama 01_embeddings/` to\n"
        "score the same questions with real neural embeddings and compare."
    )


def main() -> None:
    step_1_tokenization()
    step_2_hashing()
    step_3_idf()
    step_4_vector_and_cosine()
    step_5_where_lexical_breaks()
    step_6_chunking()
    step_7_end_to_end()
    print(f"\n{'=' * 78}\nNext: 01_embeddings/experiment_chunk_size.py, then 02_langchain/\n{'=' * 78}")


if __name__ == "__main__":
    main()
