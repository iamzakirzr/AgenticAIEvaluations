"""Shared spine for the curriculum.

Deliberately small. Everything here is used by two or more lessons; anything
used by only one lesson lives in that lesson's directory instead.

  config.py     every tunable knob, env-overridable
  compat.py     dependency shims (read it, it explains a real breakage)
  providers.py  the embedding/chat model seam -- swap models here
  trace.py      RagTrace, the object every evaluator consumes
  golden.py     the labelled dataset loader
  metrics.py    retrieval metrics that need no LLM (the free CI tier)
  corpus/       the markdown knowledge base the RAG pipeline indexes
  golden.jsonl  the labelled dataset itself
"""
