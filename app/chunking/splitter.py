"""
app/chunking/splitter.py

Phase 2 deliverable: split page-level text (from app.ingestion.loader)
into smaller, overlapping chunks suitable for embedding + retrieval.

Implemented from scratch first (no LangChain) so you see exactly what
"recursive character splitting" means mechanically. The LangChain
equivalent (RecursiveCharacterTextSplitter) is shown separately below
this module for comparison - don't import it here.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import uuid

logger = logging.getLogger(__name__)

# Try splitting on these, in order, before falling back to a hard cut.
# Paragraph breaks first (most semantically meaningful), then lines,
# then sentences, then words, then - worst case - just cut mid-word.
DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    """
    One retrieval-sized unit of text, with enough metadata to trace
    it back to its exact source page - this is what gets embedded
    and stored in the vector DB in Phase 3/4, and what citations
    (Phase 8) are built from.
    """
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int  # position of this chunk within its page (0, 1, 2...)
    text: str
    char_count: int


def _split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """
    Recursively split `text` using the first separator that actually
    produces pieces small enough to work with. This is the core
    mechanic of "recursive character splitting."

    Input:  a block of text, a target chunk_size, and an ordered list
            of separators to try
    Output: a list of text pieces, each <= chunk_size where possible
            (the final fallback - splitting on "" - guarantees this,
            since it just hard-cuts by character count)
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Should not happen since "" is always the last separator,
        # but guard against it explicitly rather than crashing.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    remaining_separators = separators[1:]

    if sep == "":
        # Hard fallback: just cut every chunk_size characters.
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = text.split(sep)

    # Re-merge small pieces back together up to chunk_size, so we're
    # not producing a chunk per sentence when several sentences would
    # comfortably fit together under chunk_size.
    results: list[str] = []
    current = ""
    for piece in pieces:
        candidate = current + sep + piece if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                results.append(current)
            if len(piece) > chunk_size:
                # This single piece is still too big - recurse with
                # the next separator down the list.
                results.extend(_split_text(piece, chunk_size, remaining_separators))
                current = ""
            else:
                current = piece
    if current:
        results.append(current)

    return [r for r in results if r.strip()]


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """
    Prepend the tail of each piece onto the start of the next one,
    so information near a chunk boundary isn't isolated with zero
    surrounding context in either chunk.

    Input:  list of non-overlapping text pieces
    Output: list of the same length, each (after the first) prefixed
            with up to `overlap` characters from the end of the
            previous piece.
    """
    if overlap <= 0 or len(pieces) <= 1:
        return pieces

    overlapped = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        overlapped.append(prev_tail + pieces[i])
    return overlapped


def chunk_page_text(
    text: str,
    doc_id: str,
    filename: str,
    page_number: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    """
    Chunk a single page's text into Chunk objects.

    Input:  raw page text + its source metadata + chunking parameters
    Output: list of Chunk, each carrying enough metadata to trace back
            to (filename, page_number) for citations later.
    """
    if not text or not text.strip():
        return []

    seps = separators if separators is not None else DEFAULT_SEPARATORS
    pieces = _split_text(text, chunk_size, seps)
    pieces = _add_overlap(pieces, chunk_overlap)

    chunks = []
    for idx, piece in enumerate(pieces):
        chunks.append(Chunk(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            filename=filename,
            page_number=page_number,
            chunk_index=idx,
            text=piece,
            char_count=len(piece),
        ))
    return chunks


def chunk_page_records(
    page_records: list,  # list[PageRecord] from app.ingestion.loader
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """
    Chunk an entire list of PageRecord objects (e.g. everything that
    came out of load_pdfs_from_directory in Phase 1).

    Skips pages flagged is_low_content, since chunking word-list noise
    (like the attention-visualization pages) just pollutes the vector
    store with useless entries - this is exactly the low-content flag
    from Phase 1 paying off.
    """
    all_chunks: list[Chunk] = []
    skipped_pages = 0

    for record in page_records:
        if record.is_low_content:
            skipped_pages += 1
            continue

        chunks = chunk_page_text(
            text=record.text,
            doc_id=record.doc_id,
            filename=record.filename,
            page_number=record.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(chunks)

    logger.info(
        f"Produced {len(all_chunks)} chunks from {len(page_records)} pages "
        f"({skipped_pages} low-content pages skipped)"
    )
    return all_chunks


if __name__ == "__main__":
    from app.ingestion.loader import load_pdfs_from_directory

    records = load_pdfs_from_directory("data/raw")
    chunks = chunk_page_records(records, chunk_size=1000, chunk_overlap=200)

    print(f"\n{len(records)} pages -> {len(chunks)} chunks\n")

    # Show the first 3 chunks in full so you can eyeball chunk quality
    for c in chunks[:3]:
        print(f"--- Chunk {c.chunk_index} | {c.filename} p.{c.page_number} | {c.char_count} chars ---")
        print(c.text)
        print()

    # Distribution check: are chunk sizes reasonable, or are we
    # producing a lot of tiny fragments (a sign chunk_size or
    # separators need tuning)?
    sizes = [c.char_count for c in chunks]
    if sizes:
        print(f"Chunk size stats: min={min(sizes)}, max={max(sizes)}, "
              f"avg={sum(sizes) / len(sizes):.0f}")