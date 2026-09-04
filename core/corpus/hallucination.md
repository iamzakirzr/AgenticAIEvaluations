# Hallucination and Grounding

## Definition

A hallucination is generated content that is presented as fact but is not
supported by the model's source of truth. In a RAG system the source of truth is
the retrieved context, which makes hallucination measurable: a claim is
hallucinated if it does not follow from the retrieved chunks.

## Types

**Intrinsic hallucination** contradicts the provided context. The context says
the timeout is 30 seconds and the answer says 60.

**Extrinsic hallucination** adds information absent from the context. It may
even be true in the world, but it did not come from the sources, so the system
cannot justify it and the user cannot verify it.

**Citation hallucination** attributes a claim to a source that does not contain
it. This is especially damaging because the citation makes the claim look
verified.

## Why RAG reduces but does not eliminate hallucination

Retrieval supplies facts the model would otherwise invent, which removes a large
class of errors. Three causes remain.

First, the model may ignore the context and answer from its pretraining weights,
particularly when the context contradicts something it learned during training.

Second, retrieval may return nothing relevant, and a model that has not been
instructed to refuse will produce a plausible answer anyway.

Third, the model may combine two retrieved facts into an unsupported conclusion,
which is an inference error rather than a recall error.

## Mitigations

Instruct the model explicitly to answer only from the provided context and to
say that it does not know when the context is insufficient. This is the single
highest-value prompt change in a RAG system, and its effect is measurable as a
rise in faithfulness on unanswerable questions.

Require inline citations to specific chunk identifiers, then verify
programmatically that every cited identifier was actually retrieved. This turns
citation hallucination into a deterministic, checkable error.

Apply a relevance threshold to retrieval scores and refuse when the best chunk
falls below it, rather than always returning the top K regardless of quality.

Run a separate grounding check after generation that verifies each claim against
the context, and either flag or regenerate answers that fail.

## Evaluating refusal behaviour

A golden dataset that contains only answerable questions cannot detect the worst
failure mode, which is confident fabrication on questions the corpus does not
cover. Unanswerable questions must be included deliberately, with the expected
behaviour being a refusal.

There is a trade-off to manage. Pushing a system to refuse more often reduces
hallucination but increases unhelpful refusals on questions it could have
answered. Both directions must be measured together, because optimising one in
isolation produces either a liar or a system that refuses everything.
