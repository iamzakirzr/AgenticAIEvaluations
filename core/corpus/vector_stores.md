# Vector Stores

## What a vector store does

A vector store holds embeddings alongside their source text and metadata, and
answers nearest-neighbour queries: given a query vector, return the stored
vectors closest to it under some distance metric.

## Exact versus approximate search

Exact search, also called flat or brute-force search, compares the query against
every stored vector. It always returns the true nearest neighbours, and its cost
grows linearly with the number of vectors. For corpora up to roughly one hundred
thousand chunks, exact search is fast enough and is the correct default because
it removes an entire class of recall bugs.

Approximate nearest neighbour search, abbreviated ANN, trades a small amount of
accuracy for a large speedup by not examining every vector. The fraction of true
neighbours that an approximate index actually returns is called its recall, and
it is a tunable property, not a fixed one.

## HNSW

Hierarchical Navigable Small World, or HNSW, is the most widely deployed ANN
index. It builds a multi-layer graph where each vector is a node connected to
its neighbours. Search starts at a sparse top layer, greedily walks toward the
query, then descends to denser layers to refine.

HNSW has three parameters that matter:

- `M` controls how many neighbours each node keeps. Higher M gives better recall
  and uses more memory.
- `ef_construction` controls how hard the index works while being built. Higher
  values give a better graph and slower indexing.
- `ef_search` controls how hard the index works at query time. Raising it
  improves recall at the cost of latency, and it can be changed after the index
  is built, unlike the other two.

Chroma uses HNSW under the hood and defaults to cosine distance.

## Metadata filtering

Production systems almost always need to restrict search to a subset, such as
documents a particular user is allowed to see, or documents from the last
quarter. This is metadata filtering.

Filtering interacts badly with approximate indexes. If the filter is very
selective, the graph walk spends its time in nodes that the filter excludes and
recall collapses. This is called the pre-filtering versus post-filtering problem.
Post-filtering retrieves K results then discards those failing the filter, which
can return fewer than K. Pre-filtering restricts the candidate set first, which
is correct but harder to do efficiently in a graph index.

## Choosing a store

Chroma is embedded, runs in-process, requires no server, and is ideal for
learning and for corpora up to a few hundred thousand chunks. FAISS is a library
rather than a database, is extremely fast, and offers the widest range of index
types, but has no built-in persistence of metadata. Qdrant, Weaviate and Milvus
are dedicated servers that support sharding, replication and efficient filtered
search at scale. pgvector adds vector search to PostgreSQL and is often the
right answer when the data already lives in Postgres, because it avoids running
a second database.
