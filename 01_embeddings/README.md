# 01 — Embeddings, Chunking and Retrieval, From Scratch

**Goal:** understand the mechanism before any framework hides it. By the end you
will have built a working retrieval system in ~80 lines of numpy and scored it
against a labelled dataset, with no LLM involved at all.

Nothing in this lesson needs Ollama except the three tests marked `@ollama`.

---

## Run it

```bash
make lesson-embeddings          # the step-by-step walkthrough (7 stages)
make experiment-chunking        # sweep chunk size, watch metrics move
pytest 01_embeddings/ -v        # fast tier, ~1s
pytest -m ollama 01_embeddings/ # re-run key claims on real neural embeddings
```

---

## Files

| File | What it is |
|---|---|
| `walkthrough.py` | Prints every intermediate state from raw text to ranked results. **Start here.** |
| `chunking.py` | Four splitting strategies implemented by hand |
| `mini_rag.py` | A complete retriever in ~80 lines, no framework |
| `experiment_chunk_size.py` | Sweeps chunk size and prints the metric curve |
| `test_embeddings.py` | Tests that demonstrate each claim the lesson makes |

---

## The seven steps

**1. Tokenization.** Text becomes discrete symbols. Every tokenizer decision is
a trade — lowercasing merges `Chunk`/`chunk` (good) and `US`/`us` (bad).

**2. The hashing trick.** Vocabulary is open-ended but vectors must be
fixed-length. Hash each token to a slot. Cost: collisions. The walkthrough runs
at `dim=64` so you can watch collisions actually happen.

**3. IDF.** Raw word counts make every document look alike because they are all
full of `the`. Inverse document frequency weights rare, discriminating terms
higher. The walkthrough prints real IDF weights over the corpus — `the` scores
1.000, `hnsw` scores 2.504.

**4. The vector and cosine similarity.** The vector is sparse (~2% non-zero) and
L2-normalised, so cosine similarity reduces to a dot product. This is the entire
retrieval mechanism.

**5. Where lexical breaks.** `car` and `automobile` hash to different slots, so a
lexical model scores identical meanings as unrelated. **This failure is why
neural embedding models exist.** `test_lexical_embeddings_fail_on_synonyms`
(fast tier) asserts the failure; `test_neural_embeddings_handle_synonyms_where_lexical_fails`
(`@ollama`) asserts a real model fixes it. Run both.

**6. Chunking.** Same document, four strategies, side by side with statistics.
Includes `measure_overlap`, so overlap is *verified* rather than assumed — a
merge step that silently drops overlap produces no error, just quietly worse
metrics.

**7. End to end.** Score retrieval over the golden dataset. Deterministic, free,
milliseconds.

---

## The trade-off, in real numbers

`make experiment-chunking` produces this (lexical embeddings, top_k=4):

```
  size  chunks  avg chars   recall  precision     MRR    nDCG  misses
   150     317        110    0.988      0.544   0.952   0.961       1
   300     138        246    1.000      0.542   0.964   0.975       0
   500      84        403    1.000      0.548   0.976   0.981       0
   700      62        546    1.000      0.534   0.988   0.991       0
  1000      41        807    1.000      0.462   0.988   0.991       0
  1500      27       1199    1.000      0.405   0.988   0.991       0
  2500      16       1934    1.000      0.361   0.972   0.979       0
```

Precision falls 0.548 → 0.361 as chunks grow: bigger chunks drag in more
irrelevant text. MRR peaks at 700. Recall dips only at 150, where facts start
getting cut across boundaries.

### The honest caveat, which matters more than the table

**Recall is saturated.** It moves by 0.012 across a 16× change in chunk size. A
metric with no headroom cannot detect a regression, so gating CI on recall here
would be theatre.

Two causes: the corpus has 8 topically distinct documents, so picking the right
*document* is easy even when the right *sentence* was cut in half; and the golden
questions reuse vocabulary from the source documents, which flatters lexical
matching.

That is why `test_retrieval_mrr_gate` gates on **MRR**, not recall. Choosing the
metric that can actually move is a real skill, and noticing your own metric has
no headroom is the difference between running an eval and understanding one.

The fix at the right granularity is chunk-level judged metrics — which is
lesson 04 and 05.

---

## Things worth internalising

- **Query and documents must use the same embedding model.** Different models
  put text in unrelated vector spaces. This breaks silently when an index is
  rebuilt with an upgraded model but the query path is not.
- **Exact search is correct below ~100k chunks.** `mini_rag.py` compares against
  every chunk. That is not a toy shortcut — it removes an entire class of recall
  bugs that approximate indexes introduce.
- **Document-level recall understates chunking damage.** It stays high as long
  as the right *document* is found, even if the answer sentence was bisected.
  Know what your metric is blind to.
- **Random fake embeddings make retrieval tests meaningless.** `core.providers`
  ships a real TF-IDF embedder for the fast tier precisely so recall@k is not
  measuring chance.

---

Next: **[02_langchain](../02_langchain/)** — the same pipeline, built with
LangChain 1.4, plus a chatbot and a real generator.
