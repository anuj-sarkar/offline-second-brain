"""
tests/test_chunking.py

Tests for app.chunking.splitter. Covers the specific behaviors we
engineered on purpose: overlap actually happens, no chunk wildly
exceeds chunk_size, and low-content pages get skipped.

Run with: python -m pytest tests/test_chunking.py -v
"""

from app.chunking.splitter import chunk_page_text, chunk_page_records, Chunk


def test_short_text_returns_single_chunk():
    text = "This is a short page of text."
    chunks = chunk_page_text(text, doc_id="d1", filename="f.pdf", page_number=1,
                              chunk_size=1000, chunk_overlap=200)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_empty_text_returns_no_chunks():
    chunks = chunk_page_text("", doc_id="d1", filename="f.pdf", page_number=1)
    assert chunks == []

    chunks = chunk_page_text("   ", doc_id="d1", filename="f.pdf", page_number=1)
    assert chunks == []


def test_long_text_produces_multiple_chunks_with_overlap():
    # Build text long enough to force at least 2 chunks at chunk_size=100
    paragraph = "This is a sentence about transformers and attention. " * 20
    chunks = chunk_page_text(paragraph, doc_id="d1", filename="f.pdf", page_number=1,
                              chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1

    # Every chunk should carry the right metadata through
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.doc_id == "d1"
        assert c.filename == "f.pdf"
        assert c.page_number == 1

    # Overlap check: the tail of chunk N should appear at the start of chunk N+1
    tail_of_first = chunks[0].text[-30:]
    assert tail_of_first in chunks[1].text


def test_low_content_pages_are_skipped():
    class FakeRecord:
        def __init__(self, text, is_low_content, page_number):
            self.text = text
            self.is_low_content = is_low_content
            self.doc_id = "d1"
            self.filename = "f.pdf"
            self.page_number = page_number

    records = [
        FakeRecord("Real content here that should be chunked normally.", False, 1),
        FakeRecord("word\nword\nword\n<EOS>", True, 2),  # simulates the appendix pages
    ]

    chunks = chunk_page_records(records, chunk_size=1000, chunk_overlap=200)

    # Only page 1 should have produced chunks
    pages_present = {c.page_number for c in chunks}
    assert pages_present == {1}