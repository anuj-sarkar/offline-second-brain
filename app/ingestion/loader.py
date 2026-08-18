"""
app/ingestion/loader.py

Phase 1 deliverable: load PDFs from a directory, extract text per page,
attach metadata, and flag pages that are unlikely to be useful for
retrieval (near-empty, or figure-heavy pages like the ones we found
in the "Attention Is All You Need" appendix).

This module does NOT chunk or embed anything - it only produces a
clean, structured list of page-level records. Chunking is Phase 2's job.
Keeping these responsibilities separate is deliberate: you should be
able to test "did ingestion work" independently of "did chunking work".
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.ingestion.cleaning import clean_page_text

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Below this character count, we flag a page as "low content" rather
# than silently trusting it. Tune this empirically once you see more
# documents - it is NOT a universal constant, just a starting heuristic.
LOW_CONTENT_CHAR_THRESHOLD = 100

# A second, structural signal: pages extracted from figures/diagrams
# (like attention-visualization heatmaps) often have real text volume
# (well above LOW_CONTENT_CHAR_THRESHOLD) but are formatted as one
# word per line - axis labels, not prose. Average line length is a
# cheap way to detect this pattern: normal prose wraps at maybe
# 40-80 chars/line after extraction; a one-word-per-line dump averages
# under ~12 chars/line. Found this heuristic AFTER seeing it cause a
# real downstream bug in BM25 search (Phase 9) - not designed upfront.
MIN_AVG_LINE_LENGTH = 12
MIN_LINE_COUNT_FOR_CHECK = 30  # only apply this check on pages with
                                 # enough lines that the average is meaningful


@dataclass
class PageRecord:
    """
    One page of extracted text plus everything needed to trace it
    back to its source later (this is what makes citations possible
    in Phase 8 - without this metadata, you cannot cite anything).
    """
    doc_id: str
    filename: str
    source_path: str
    page_number: int
    text: str
    char_count: int
    is_low_content: bool
    extraction_error: Optional[str] = field(default=None)


def _is_low_content(text: str, char_count: int) -> bool:
    """
    Flag a page as low-content using TWO signals, not just raw length:

    1. Simple volume: near-empty pages (broken extraction, blank pages).
    2. Structural: pages with real text volume but formatted as one
       word/label per line (figure/diagram text dumps) - these have
       plenty of characters but almost no actual prose, and can cause
       real downstream problems (e.g. inflating BM25 scores via
       document-length normalization on very short "documents") if
       treated as normal indexable text.
    """
    if char_count < LOW_CONTENT_CHAR_THRESHOLD:
        return True

    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= MIN_LINE_COUNT_FOR_CHECK:
        avg_line_length = char_count / len(lines)
        if avg_line_length < MIN_AVG_LINE_LENGTH:
            return True

    return False


def _make_doc_id(filename: str) -> str:
    """
    Stable-ish ID for a document. Using a UUID here for simplicity;
    in a later phase you may want a content hash instead, so that
    re-ingesting the *same* file twice doesn't create duplicate IDs.
    """
    return str(uuid.uuid4())


def load_single_pdf(pdf_path: Path) -> list[PageRecord]:
    """
    Load one PDF and return a list of PageRecord, one per page.

    Handles two distinct failure modes separately, because they need
    different responses downstream:
      1. The whole file fails to open (corrupted/not a real PDF) ->
         return an empty list, log an error, ingestion continues
         with other files.
      2. A single page fails to extract, but the file itself is fine ->
         record that page with extraction_error set, keep going with
         the rest of the pages.
    """
    filename = pdf_path.name
    doc_id = _make_doc_id(filename)
    records: list[PageRecord] = []

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError) as e:
        logger.error(f"Failed to open '{filename}': {e}")
        return records  # empty - caller just sees zero pages for this file

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            text = clean_page_text(text)  # strip known boilerplate/footnote noise (Phase 10)
            error = None
        except Exception as e:
            # A single bad page should not kill the whole document.
            logger.warning(f"'{filename}' page {page_num}: extraction failed ({e})")
            text = ""
            error = str(e)

        char_count = len(text)
        is_low_content = _is_low_content(text, char_count)

        if is_low_content:
            logger.info(
                f"'{filename}' page {page_num}: flagged low-content "
                f"({char_count} chars)"
            )

        records.append(PageRecord(
            doc_id=doc_id,
            filename=filename,
            source_path=str(pdf_path),
            page_number=page_num,
            text=text,
            char_count=char_count,
            is_low_content=is_low_content,
            extraction_error=error,
        ))

    logger.info(f"'{filename}': extracted {len(records)} pages")
    return records


def load_pdfs_from_directory(directory: str) -> list[PageRecord]:
    """
    Load every .pdf file in a directory (non-recursive).

    Input:  path to a directory, e.g. "data/raw"
    Output: flat list of PageRecord across ALL documents in that
            directory - this is what Phase 2 (chunking) will consume.

    A directory with zero PDFs, or a directory that doesn't exist,
    should not crash the pipeline - it should just produce zero
    records and log clearly why.
    """
    dir_path = Path(directory)

    if not dir_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    pdf_files = sorted(dir_path.glob("*.pdf"))

    if not pdf_files:
        logger.warning(f"No PDF files found in: {directory}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {directory}")

    all_records: list[PageRecord] = []
    for pdf_path in pdf_files:
        records = load_single_pdf(pdf_path)
        all_records.extend(records)

    return all_records


def summarize(records: list[PageRecord]) -> None:
    """Print a quick human-readable summary - useful for manual verification."""
    if not records:
        print("No records to summarize.")
        return

    by_doc: dict[str, list[PageRecord]] = {}
    for r in records:
        by_doc.setdefault(r.filename, []).append(r)

    print(f"\n{'=' * 60}")
    print(f"INGESTION SUMMARY: {len(by_doc)} document(s), {len(records)} page(s) total")
    print(f"{'=' * 60}")

    for filename, pages in by_doc.items():
        low_content_count = sum(1 for p in pages if p.is_low_content)
        error_count = sum(1 for p in pages if p.extraction_error)
        print(f"\n{filename}")
        print(f"  Pages: {len(pages)}")
        print(f"  Low-content pages: {low_content_count}")
        print(f"  Pages with extraction errors: {error_count}")
        if low_content_count:
            flagged = [p.page_number for p in pages if p.is_low_content]
            print(f"  Flagged page numbers: {flagged}")


if __name__ == "__main__":
    records = load_pdfs_from_directory("data/raw")
    summarize(records)