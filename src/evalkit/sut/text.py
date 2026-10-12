"""Tokenisation shared by the retriever and the scripted policies."""

from __future__ import annotations

import re

_STOPWORDS = """a an and are as at be by can do does for from has have how i if in is it its
    of on or
    that the their there this to was what when where which who will with you your my me we our
    after before than then any all not no more much many"""
STOPWORDS = frozenset(_STOPWORDS.split())

_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokens(text: str) -> list[str]:
    out = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS:
            continue
        # Crude plural folding: "notebooks" -> "notebook". Good enough for a
        # 5-document corpus and keeps the retriever dependency-free.
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.append(word)
    return out


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
