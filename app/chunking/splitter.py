"""
app/chunking/splitter.py

Splits page-level text into overlapping, retrieval-sized chunks.
Key enhancements:
  - Word-boundary safe overlap (prevents mid-word sliced artifacts like 'ansformer').
  - Section title and document metadata preservation.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import uuid

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    char_count: int
    section_title: str = "General"


def _split_text(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split text using semantic separators down to chunk_size."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = separators[0]
    remaining_separators = separators[1:]

    if sep == "":
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    pieces = text.split(sep)
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
                results.extend(_split_text(piece, chunk_size, remaining_separators))
                current = ""
            else:
                current = piece

    if current:
        results.append(current)

    return [r for r in results if r.strip()]


def _find_word_boundary_tail(text: str, max_chars: int) -> str:
    """Extract up to max_chars from end of text without cutting words in half."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    first_space = tail.find(" ")
    if first_space != -1 and first_space < len(tail) - 1:
        return tail[first_space + 1:]
    return tail


def _add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the tail of each chunk onto the next, aligned to word boundaries."""
    if overlap <= 0 or len(pieces) <= 1:
        return pieces

    overlapped = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = _find_word_boundary_tail(pieces[i - 1], overlap)
        if prev_tail:
            overlapped.append(prev_tail + " " + pieces[i])
        else:
            overlapped.append(pieces[i])
    return overlapped


def chunk_page_text(
    text: str,
    doc_id: str,
    filename: str,
    page_number: int,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    section_title: str = "General",
    separators: Optional[list[str]] = None,
) -> list[Chunk]:
    """Chunk a single page's text into Chunk instances."""
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
            text=piece.strip(),
            char_count=len(piece.strip()),
            section_title=section_title,
        ))
    return chunks


def chunk_page_records(
    page_records: list,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Chunk]:
    """Chunk a list of PageRecords skipping low-content noise."""
    all_chunks: list[Chunk] = []
    skipped_pages = 0

    for record in page_records:
        if record.is_low_content:
            skipped_pages += 1
            continue

        section_title = getattr(record, "section_title", "General")
        chunks = chunk_page_text(
            text=record.text,
            doc_id=record.doc_id,
            filename=record.filename,
            page_number=record.page_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            section_title=section_title,
        )
        all_chunks.extend(chunks)

    logger.info(
        f"Produced {len(all_chunks)} chunks from {len(page_records)} pages "
        f"({skipped_pages} low-content pages skipped)"
    )
    return all_chunks