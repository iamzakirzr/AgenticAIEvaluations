# Retrieval

## The basic loop

Retrieval in a RAG system has three steps. The query is embedded using the same
model that embedded the documents. The vector store finds the chunks whose
vectors are most similar to the query vector. The top K chunks are inserted into
the prompt given to the generator.

Using a different embedding model for queries than for documents produces
meaningless results, because the two models place text in unrelated vector
spaces. This is a frequent bug when an index is rebuilt with an upgraded model
but the query path is not updated.

## Choosing top K

Top K is how many chunks to retrieve. Typical values are 3 to 10.

Raising K increases the chance that the answer is present somewhere in the
context, which improves context recall. It also drags in more irrelevant text,
which lowers context precision and can actively harm answer quality, because
models are measurably worse at using information buried in the middle of a long
context. This effect is known as "lost in the middle".

## Maximum marginal relevance

Plain top-K similarity search often returns several near-identical chunks,
because a document that mentions a topic repeatedly produces many similar
vectors. Maximum Marginal Relevance, usually abbreviated MMR, addresses this by
selecting chunks that are similar to the query but dissimilar to the chunks
already selected. It trades a little relevance for diversity, controlled by a
lambda parameter where 1.0 is pure relevance and 0.0 is pure diversity.

## Hybrid search

Hybrid search runs both a lexical search, typically BM25, and a vector search,
then merges the two ranked lists. It is more robust than either alone because
the two methods fail in different ways: vector search misses exact rare tokens
such as error codes, and lexical search misses paraphrases.

The standard merge algorithm is Reciprocal Rank Fusion, which scores each
document by the sum over result lists of one divided by the quantity k plus the
document's rank in that list. The constant k is conventionally set to 60. RRF
needs only the ranks, not the raw scores, which is what makes it safe to combine
two retrievers whose scores are on incomparable scales.

## Reranking

A reranker is a second-stage model that rescores the retrieved candidates. The
usual design retrieves 25 to 50 candidates cheaply with vector search, then
applies a cross-encoder that reads the query and each candidate together and
outputs a relevance score, keeping only the best 3 to 5.

Cross-encoders are far more accurate than embedding similarity because they let
the query and document attend to each other directly, rather than compressing
each into a vector independently. They are also far slower, which is precisely
why they are used only on a shortlist rather than the whole corpus.

## Query transformation

The user's raw question is often a poor search query. Common transformations:

**Query rewriting** turns a conversational follow-up such as "what about the
second one" into a standalone question using the conversation history. Without
it, multi-turn RAG retrieves nonsense on every follow-up.

**Multi-query expansion** generates several paraphrases of the question,
retrieves for each, and takes the union, which improves recall when the user's
phrasing does not match the document's.

**HyDE**, short for Hypothetical Document Embeddings, asks the model to write a
fake answer to the question and embeds that instead of the question. The fake
answer is in the same style and vocabulary as real documents, so it often
matches better than the question does.
