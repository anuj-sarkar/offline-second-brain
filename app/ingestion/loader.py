"""
app/ingestion/loader.py

Layout-aware document ingestion for Offline Second Brain.
Supports:
  - PDF (.pdf) via PyMuPDF (fitz) with two-column reading-order sorting and section detection.
  - Markdown (.md) and Plain Text (.txt) notes.

Produces PageRecord instances ready for chunking and indexing.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid

import pymupdf

from app.ingestion.cleaning import clean_page_text
from app.ingestion.markdown_loader import parse_markdown_sections

logger = logging.getLogger(__name__)

LOW_CONTENT_CHAR_THRESHOLD = 100
MIN_AVG_LINE_LENGTH = 12
MIN_LINE_COUNT_FOR_CHECK = 30


@dataclass
class PageRecord:
    doc_id: str
    filename: str
    source_path: str
    page_number: int
    text: str
    char_count: int
    is_low_content: bool
    section_title: str = "General"
    extraction_error: Optional[str] = field(default=None)


def _is_low_content(text: str, char_count: int) -> bool:
    """Flag a page as low-content if near-empty or structurally garbled."""
    if char_count < LOW_CONTENT_CHAR_THRESHOLD:
        return True

    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) >= MIN_LINE_COUNT_FOR_CHECK:
        avg_line_length = char_count / len(lines)
        if avg_line_length < MIN_AVG_LINE_LENGTH:
            return True

    return False


def _make_doc_id(filename: str) -> str:
    return str(uuid.uuid4())


def extract_pdf_page_layout_aware(page: pymupdf.Page) -> tuple[str, str]:
    """
    Extract text from a PyMuPDF page preserving 2-column reading order.
    Returns (extracted_text, detected_section_title).
    """
    page_width = page.rect.width
    mid_point = page_width / 2.0

    # Extract text blocks: (x0, y0, x1, y1, text, block_no, block_type)
    blocks = page.get_text("blocks")
    if not blocks:
        return "", "General"

    # Filter only text blocks (block_type == 0)
    text_blocks = [b for b in blocks if len(b) >= 5 and b[4].strip() and (len(b) < 7 or b[6] == 0)]

    # Check if page is two-column: blocks on left (x0 < mid_point - 20) and right (x0 > mid_point - 20)
    has_left = any(b[0] < mid_point - 30 for b in text_blocks)
    has_right = any(b[0] > mid_point - 30 for b in text_blocks)
    is_two_column = has_left and has_right

    if is_two_column:
        # Sort left column first (column 0), then right column (column 1), ordered top-to-bottom
        def block_sort_key(b):
            column_idx = 1 if b[0] > (mid_point - 20) else 0
            return (column_idx, b[1])  # (col, y0)

        sorted_blocks = sorted(text_blocks, key=block_sort_key)
    else:
        # Single column: top-to-bottom
        sorted_blocks = sorted(text_blocks, key=lambda b: b[1])

    text_parts = [b[4].strip() for b in sorted_blocks if b[4].strip()]
    full_text = "\n\n".join(text_parts)

    # Detect prominent heading if present
    section_title = "General"
    for b in sorted_blocks:
        first_line = b[4].strip().split("\n")[0]
        if len(first_line) < 60 and (
            first_line.isupper()
            or any(first_line.startswith(prefix) for prefix in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "I.", "II.", "III.", "IV.", "V."])
        ):
            section_title = first_line
            break

    return full_text, section_title


def load_single_pdf(pdf_path: Path) -> list[PageRecord]:
    """Load one PDF using layout-aware PyMuPDF extraction."""
    filename = pdf_path.name
    doc_id = _make_doc_id(filename)
    records: list[PageRecord] = []

    try:
        doc = pymupdf.open(str(pdf_path))
    except Exception as e:
        logger.error(f"Failed to open '{filename}': {e}")
        return records

    for page_num in range(1, len(doc) + 1):
        page = doc[page_num - 1]
        try:
            raw_text, detected_section = extract_pdf_page_layout_aware(page)
            text = clean_page_text(raw_text)
            error = None
        except Exception as e:
            logger.warning(f"'{filename}' page {page_num}: extraction failed ({e})")
            text = ""
            detected_section = "General"
            error = str(e)

        char_count = len(text)
        is_low_content = _is_low_content(text, char_count)

        if is_low_content:
            logger.debug(f"'{filename}' page {page_num}: flagged low-content ({char_count} chars)")

        records.append(PageRecord(
            doc_id=doc_id,
            filename=filename,
            source_path=str(pdf_path),
            page_number=page_num,
            text=text,
            char_count=char_count,
            is_low_content=is_low_content,
            section_title=detected_section,
            extraction_error=error,
        ))

    doc.close()
    logger.info(f"'{filename}': extracted {len(records)} pages via PyMuPDF")
    return records


def load_single_markdown_or_text(file_path: Path) -> list[PageRecord]:
    """Load one .md or .txt file as structured PageRecords."""
    filename = file_path.name
    doc_id = _make_doc_id(filename)
    records: list[PageRecord] = []

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_content = f.read()
    except Exception as e:
        logger.error(f"Failed to read '{filename}': {e}")
        return records

    sections = parse_markdown_sections(raw_content)
    for idx, sec in enumerate(sections, start=1):
        cleaned_text = clean_page_text(sec["text"])
        char_count = len(cleaned_text)
        is_low_content = char_count < 30

        records.append(PageRecord(
            doc_id=doc_id,
            filename=filename,
            source_path=str(file_path),
            page_number=idx,  # section index acts as page number for citations
            text=cleaned_text,
            char_count=char_count,
            is_low_content=is_low_content,
            section_title=sec["section_title"],
        ))

    logger.info(f"'{filename}': extracted {len(records)} section records")
    return records


def load_documents_from_directory(directory: str) -> list[PageRecord]:
    """
    Load all supported documents (.pdf, .md, .txt) from directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    supported_extensions = {".pdf", ".md", ".txt"}
    files = sorted([f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() in supported_extensions])

    if not files:
        logger.warning(f"No supported document files found in: {directory}")
        return []

    logger.info(f"Found {len(files)} document(s) in {directory}")

    all_records: list[PageRecord] = []
    for file_path in files:
        if file_path.suffix.lower() == ".pdf":
            records = load_single_pdf(file_path)
        else:
            records = load_single_markdown_or_text(file_path)
        all_records.extend(records)

    return all_records


# Alias for backward compatibility
def load_pdfs_from_directory(directory: str) -> list[PageRecord]:
    return load_documents_from_directory(directory)