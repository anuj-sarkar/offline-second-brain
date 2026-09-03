"""
tests/test_chunking.py

Tests for app.chunking.splitter.
"""

from app.chunking.splitter import Chunk, chunk_page_records, chunk_page_text


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
    paragraph = "This is a sentence about transformers and attention. " * 20
    chunks = chunk_page_text(paragraph, doc_id="d1", filename="f.pdf", page_number=1,
                              chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1

    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.doc_id == "d1"
        assert c.filename == "f.pdf"
        assert c.page_number == 1

    # Overlap check: words from chunk 0 tail should appear in chunk 1 start
    words_chunk0 = set(chunks[0].text.split()[-3:])
    words_chunk1 = set(chunks[1].text.split()[:5])
    assert bool(words_chunk0 & words_chunk1)


def test_low_content_pages_are_skipped():
    class FakeRecord:
        def __init__(self, text, is_low_content, page_number):
            self.text = text
            self.is_low_content = is_low_content
            self.doc_id = "d1"
            self.filename = "f.pdf"
            self.page_number = page_number
            self.section_title = "General"

    records = [
        FakeRecord("Real content here that should be chunked normally.", False, 1),
        FakeRecord("word\nword\nword\n<EOS>", True, 2),
    ]

    chunks = chunk_page_records(records, chunk_size=1000, chunk_overlap=200)
    pages_present = {c.page_number for c in chunks}
    assert pages_present == {1}


def test_math_block_delimiters_kept_together():
    text = (
        "Introductory explanation of the attention mechanism.\n\n"
        "$$ \\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V $$\n\n"
        "Concluding remarks regarding theoretical complexity."
    )
    chunks = chunk_page_text(text, doc_id="d1", filename="f.pdf", page_number=1, chunk_size=150, chunk_overlap=30)
    # Ensure math block $$ is not severed into single $ or split in middle
    math_chunk = next((c for c in chunks if "\\text{Attention}" in c.text), None)
    assert math_chunk is not None
    assert "$$ \\text{Attention}" in math_chunk.text
    assert "\\right)V $$" in math_chunk.text