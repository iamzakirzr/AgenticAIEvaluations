# Chunking

## Why documents must be split

Embedding models have a maximum input length, and more importantly, a single
vector cannot faithfully represent a long, topically diverse document. If an
entire 50-page manual is embedded as one vector, that vector sits at the average
of every topic in the manual and is close to nothing in particular. Splitting
documents into chunks gives each topic its own vector.

## Chunk size

Chunk size is the most consequential retrieval parameter. It controls a direct
trade-off:

- Chunks that are too large retrieve a lot of irrelevant text alongside the
  relevant sentence. This lowers context precision and wastes the generator's
  context window.
- Chunks that are too small split a single fact across a boundary, so no single
  chunk contains the whole answer. This lowers context recall.

A common starting range for prose is 500 to 1000 characters. There is no
universally correct value; it depends on how information is distributed in your
documents, which is why it should be tuned by measurement rather than guessed.

## Chunk overlap

Overlap means consecutive chunks share some text at their boundary. If chunk one
ends at character 700 and overlap is 120, chunk two begins at character 580.

The purpose is to protect facts that straddle a boundary. Without overlap, a
sentence split down the middle appears incompletely in both neighbouring chunks
and is retrievable from neither. A typical overlap is 10 to 20 percent of the
chunk size.

Overlap costs storage and introduces near-duplicate chunks in results, which can
crowd out genuinely different sources. Excessive overlap is a common cause of
retrieved contexts that all say the same thing.

## Splitting strategies

**Fixed-size splitting** cuts every N characters. It is simple and fast but
routinely cuts sentences in half.

**Recursive character splitting** is the usual default. It tries a prioritised
list of separators, splitting on paragraph breaks first, then single newlines,
then sentences, then words, only falling back to a hard character cut when no
separator produces a small enough piece. LangChain implements this as
`RecursiveCharacterTextSplitter`.

**Document-structure splitting** uses the document's own markup, splitting
markdown on headings or HTML on section tags. This keeps semantically coherent
units together and attaches useful metadata such as the section title.

**Semantic splitting** embeds each sentence and starts a new chunk where the
similarity between consecutive sentences drops below a threshold, placing
boundaries at genuine topic shifts. It is the most expensive option because it
requires embedding the corpus twice.

## Contextual retrieval

A refinement is to prepend a short, model-generated summary of the parent
document to each chunk before embedding it. This restores context that chunking
destroyed, such as which product a paragraph of specifications refers to. It
raises indexing cost substantially but reliably improves retrieval on documents
where chunks are ambiguous in isolation.
