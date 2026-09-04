"""
Chunking strategies, implemented from scratch so you can see the mechanism.

=============================================================================
WHY IMPLEMENT THESE BY HAND WHEN LANGCHAIN HAS THEM?
=============================================================================
Because `RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)` is
one line that hides four decisions, and every one of those decisions shows up
later as a number in an evaluation report. When context recall drops after you
change chunk size, you need to know exactly what changed about the text -- not
what changed about a config value.

Lesson 02 uses the real LangChain splitter. This file exists so that when you
get there, nothing about it is mysterious.

=============================================================================
THE CENTRAL TRADE-OFF, STATED ONCE
=============================================================================
    chunks too LARGE  -> retrieved context is mostly irrelevant text
                      -> CONTEXT PRECISION falls
                      -> the generator is distracted, and pays for wasted tokens

    chunks too SMALL  -> a single fact is split across a boundary, so no one
                         chunk contains the whole answer
                      -> CONTEXT RECALL falls
                      -> the generator cannot answer even with perfect retrieval

There is no universally correct size. There is only the size that measures best
on YOUR corpus, which is why `experiment_chunk_size.py` exists next door.
=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    """A slice of a document, with enough provenance to evaluate retrieval."""

    text: str
    doc_id: str
    index: int          # position within the document, 0-based
    start_char: int     # offset into the original text -- lets you verify overlap
    end_char: int

    def __len__(self) -> int:
        return len(self.text)


# ===========================================================================
# STRATEGY 1 -- FIXED SIZE
# ===========================================================================


def fixed_size_chunks(text: str, doc_id: str, size: int, overlap: int = 0) -> list[Chunk]:
    """Cut every ``size`` characters, stepping forward by ``size - overlap``.

    The simplest possible splitter, and the one that most clearly demonstrates
    why overlap exists.

    THE STEP CALCULATION IS THE WHOLE IDEA:

        size=100, overlap=0   -> starts at 0, 100, 200, ...   (step 100)
        size=100, overlap=20  -> starts at 0,  80, 160, ...   (step  80)

    With overlap, characters 80-100 appear in BOTH chunk 0 and chunk 1. A
    sentence spanning that boundary is therefore complete in chunk 1, where
    with no overlap it would be truncated in chunk 0 and headless in chunk 1 --
    retrievable, in useful form, from neither.

    Its flaw: it has no idea what a sentence is, so it cuts words in half.
    """
    if overlap >= size:
        # Guard against an infinite loop. This is a real bug people ship:
        # step <= 0 means the loop never advances.
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    index = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end]
        if piece.strip():
            chunks.append(Chunk(piece, doc_id, index, start, end))
            index += 1
        if end == len(text):
            break
        start += step
    return chunks


# ===========================================================================
# STRATEGY 2 -- RECURSIVE CHARACTER SPLITTING (the practical default)
# ===========================================================================

# Ordered from "most semantically meaningful boundary" to "least". The splitter
# walks this list and uses the FIRST separator that yields pieces small enough.
# This is exactly the priority list LangChain's RecursiveCharacterTextSplitter
# uses by default.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_chunks(
    text: str,
    doc_id: str,
    size: int,
    overlap: int = 0,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split on the most meaningful boundary that produces small enough pieces.

    THE ALGORITHM, IN WORDS:
      1. Try to split on paragraph breaks. If every resulting piece fits within
         ``size``, done -- and every chunk is a whole paragraph.
      2. A piece is still too big? Re-split just that piece on single newlines.
      3. Still too big? On sentence ends. Then on spaces. Then, as a last
         resort, mid-word.

    The recursion means only the oversized parts get chopped aggressively;
    well-behaved paragraphs are left intact. That is why this beats fixed-size
    splitting on almost every real corpus: most chunks end up on a natural
    boundary, and only the pathological ones get butchered.

    NOTE ON OVERLAP: here overlap is applied by carrying trailing text from the
    previous chunk into the next when merging pieces, rather than by rewinding
    a character offset. Overlap is a property of the MERGE step, which is a
    detail the one-line library call hides completely.
    """
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _recursive_split(text, size, seps)
    return _merge_pieces(pieces, doc_id, size, overlap, text)


def _recursive_split(text: str, size: int, separators: list[str]) -> list[str]:
    """Break text into pieces each no larger than ``size`` where possible."""
    if len(text) <= size:
        return [text] if text.strip() else []

    if not separators:
        # No separators left: hard-cut. This is the fallback that guarantees
        # termination.
        return [text[i : i + size] for i in range(0, len(text), size)]

    separator, *rest = separators

    if separator == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(separator)
    out: list[str] = []
    for part in parts:
        # Put the separator back, so rejoined chunks read naturally.
        candidate = part + separator if separator != " " else part + " "
        if len(candidate) <= size:
            if candidate.strip():
                out.append(candidate)
        else:
            # Too big even after this split -- recurse with the next separator.
            out.extend(_recursive_split(part, size, rest))
    return out


def _merge_pieces(
    pieces: list[str], doc_id: str, size: int, overlap: int, original: str
) -> list[Chunk]:
    """Greedily pack small pieces into chunks up to ``size``, carrying overlap.

    Without this step, splitting on "\\n\\n" would emit one chunk per paragraph,
    including one-line paragraphs. Tiny chunks are bad: they carry too little
    context to be interpretable on their own, and they inflate the index.
    """
    chunks: list[Chunk] = []
    buffer = ""
    index = 0

    def flush(buf: str) -> None:
        nonlocal index
        if not buf.strip():
            return
        # Locate the chunk in the original text so start/end offsets are real.
        start = original.find(buf[:40]) if len(buf) >= 40 else original.find(buf)
        start = max(start, 0)
        chunks.append(Chunk(buf.strip(), doc_id, index, start, start + len(buf)))
        index += 1

    for piece in pieces:
        if len(buffer) + len(piece) <= size:
            buffer += piece
        else:
            flush(buffer)
            # Carry the tail of the previous chunk forward as overlap.
            tail = buffer[-overlap:] if overlap > 0 else ""
            buffer = tail + piece
    flush(buffer)
    return chunks


# ===========================================================================
# STRATEGY 3 -- MARKDOWN STRUCTURE
# ===========================================================================

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def markdown_section_chunks(text: str, doc_id: str, max_size: int | None = None) -> list[Chunk]:
    """Split on markdown headings, keeping each section whole.

    WHY THIS OFTEN WINS ON DOCUMENTATION: the author already decided where the
    topic boundaries are, and encoded that decision as headings. Splitting
    anywhere else throws away information you were given for free.

    It also enables a powerful trick: prefix each chunk with its heading path
    ("Retrieval > Reranking"). A chunk that begins "A reranker is a second-stage
    model" is ambiguous alone; prefixed with its section title it is not. This
    is a cheap approximation of contextual retrieval.

    The catch: section sizes are wildly uneven. A 4000-character section still
    needs sub-splitting, which is what ``max_size`` does.
    """
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return recursive_chunks(text, doc_id, max_size or len(text))

    chunks: list[Chunk] = []
    index = 0
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section = text[start:end].strip()
        if not section:
            continue

        if max_size is None or len(section) <= max_size:
            chunks.append(Chunk(section, doc_id, index, start, end))
            index += 1
        else:
            # Oversized section: sub-split it, but PREPEND THE HEADING to every
            # piece so the fragments stay interpretable in isolation.
            heading = match.group(0).strip()
            for sub in recursive_chunks(section, doc_id, max_size):
                chunks.append(
                    Chunk(f"{heading}\n\n{sub.text}", doc_id, index, start, end)
                )
                index += 1
    return chunks


# ===========================================================================
# ANALYSIS HELPERS -- for the walkthrough and the experiments
# ===========================================================================


def chunk_stats(chunks: list[Chunk]) -> dict[str, float]:
    """Descriptive statistics you should look at before trusting a splitter.

    High variance in chunk size is a warning sign: it means some chunks are
    starved of context while others are bloated, and a single top_k value
    cannot serve both.
    """
    if not chunks:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "stdev": 0.0}

    lengths = [len(c) for c in chunks]
    mean = sum(lengths) / len(lengths)
    variance = sum((length - mean) ** 2 for length in lengths) / len(lengths)
    return {
        "count": float(len(chunks)),
        "mean": mean,
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "stdev": variance**0.5,
    }


def measure_overlap(a: Chunk, b: Chunk) -> int:
    """Number of characters the end of ``a`` shares with the start of ``b``.

    Use this to VERIFY overlap is doing what you configured. A surprisingly
    common bug is overlap silently being 0 because the splitter's merge step
    dropped it -- and nothing tells you, the metrics just quietly get worse.

    CAVEAT worth knowing: this returns the LONGEST suffix-of-a matching a
    prefix-of-b, which over-reports on PERIODIC text. Given "abcabcabc...",
    two chunks with zero configured overlap still share a long suffix/prefix
    purely by coincidence. Real prose is not periodic, so this is reliable in
    practice -- but it makes synthetic test data like ``"abcdefghij" * 30``
    misleading, which is a trap worth having stepped in once.
    """
    max_possible = min(len(a.text), len(b.text))
    for length in range(max_possible, 0, -1):
        if a.text[-length:] == b.text[:length]:
            return length
    return 0
