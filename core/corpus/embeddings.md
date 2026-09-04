# Text Embeddings

## What an embedding is

An embedding is a fixed-length list of floating point numbers that represents a
piece of text. The defining property is that texts with similar meaning produce
vectors that point in similar directions. Meaning is encoded in *direction*, not
in magnitude.

## Dimensionality

The length of the vector is called its dimensionality. Common sizes:

- `nomic-embed-text` produces 768 dimensions.
- `all-MiniLM-L6-v2` produces 384 dimensions.
- `text-embedding-3-small` produces 1536 dimensions.

Higher dimensionality can capture more nuance but costs more memory and makes
search slower. Storage cost is straightforward: one million chunks at 768
dimensions in float32 requires roughly 3 gigabytes of raw vector data, because
each vector is 768 times 4 bytes, which is 3072 bytes.

## Cosine similarity

Similarity between two embeddings is almost always measured with cosine
similarity, defined as the dot product of the two vectors divided by the product
of their magnitudes. The result ranges from -1 to 1, where 1 means the vectors
point in exactly the same direction.

In practice embedding vectors are usually L2-normalised first, meaning they are
scaled to have a magnitude of exactly 1. Once normalised, cosine similarity is
simply the dot product, which is much faster to compute.

Euclidean distance is an alternative but is sensitive to vector magnitude, which
usually reflects document length rather than meaning. This is why cosine is the
default in nearly every vector database.

## Lexical versus neural embeddings

The oldest form of embedding is lexical: count the words in a document and use
those counts as the vector. TF-IDF weights each word by term frequency times
inverse document frequency, so that rare, discriminating words count more than
common words like "the".

Lexical embeddings have one fatal weakness: they cannot recognise synonyms. The
words "car" and "automobile" occupy different dimensions, so a lexical model
scores them as completely unrelated. Neural embedding models are trained on
large text corpora specifically to place synonyms and paraphrases near each
other, which is why they replaced lexical methods for semantic search.

Lexical methods remain useful because they handle rare exact terms well, such as
product codes, error numbers and proper nouns, where neural models often blur
distinctions. Combining both is called hybrid search.

## Symmetric versus asymmetric search

Some embedding models are trained for symmetric search, where the query and the
document are similar in length and style, such as finding duplicate questions.
Others are trained for asymmetric search, where a short query must match a long
passage. Retrieval-augmented generation is an asymmetric task.

Models like `nomic-embed-text` support instruction prefixes to distinguish the
two roles: prefixing a document with `search_document:` and a query with
`search_query:` measurably improves retrieval quality. Forgetting these prefixes
is a common and silent cause of poor retrieval.
