"""
Phase 1 - Step 1: Minimal PDF text extraction (no LangChain, no abstractions).

Goal: see exactly what raw text extraction looks like, page by page,
before we do any cleaning or chunking. Run this and READ the output
carefully - garbled text, broken lines, and repeated headers/footers
you see here are the exact problems we solve in later steps.
"""

from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, one entry per page.

    Input:  path to a PDF file
    Output: list of dicts, one per page, like:
            {"page_number": 1, "text": "...", "source": "attention.pdf"}

    This is intentionally the simplest possible version - just enough
    to prove extraction works and to eyeball the output quality.
    """
    reader = PdfReader(pdf_path)
    filename = Path(pdf_path).name

    pages = []
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages.append({
            "page_number": page_num,
            "text": text,
            "source": filename,
        })
    return pages


def main():
    pdf_path = "data/raw/attention_is_all_you_need.pdf"  # change if needed

    pages = extract_pdf_text(pdf_path)

    print(f"Extracted {len(pages)} pages from {pdf_path}\n")
    print("=" * 70)

    # Print the first 3 pages in full so you can inspect extraction quality
    for page in pages[:3]:
        print(f"\n--- PAGE {page['page_number']} (source: {page['source']}) ---\n")
        print(page["text"])
        print("\n" + "=" * 70)

    # Print character counts for ALL pages - a quick way to spot pages
    # that extracted suspiciously empty (e.g. scanned images, broken pages)
    print("\nPage-by-page character counts:")
    for page in pages:
        char_count = len(page["text"]) if page["text"] else 0
        flag = "  <-- SUSPICIOUSLY SHORT" if char_count < 100 else ""
        print(f"  Page {page['page_number']:>3}: {char_count:>5} chars{flag}")


if __name__ == "__main__":
    main()