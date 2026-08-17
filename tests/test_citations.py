"""
tests/test_citations.py

Tests for app.generation.citations. Uses hand-built fake answer text
and retrieved_chunks so we can precisely control what's "real" vs
"fabricated" - no need to call Ollama for this, since we're testing
parsing/verification logic, not generation quality.

Run with: python -m pytest tests/test_citations.py -v
"""

from app.generation.citations import extract_citations, verify_citations


def test_extract_citations_finds_all_matches():
    text = (
        "The Transformer uses self-attention [Source: attention.pdf, Page 2]. "
        "It also avoids recurrence [Source: attention.pdf, Page 1]."
    )
    citations = extract_citations(text)
    assert citations == [("attention.pdf", 2), ("attention.pdf", 1)]


def test_extract_citations_returns_empty_list_when_none_present():
    text = "This answer has no citations at all."
    assert extract_citations(text) == []


def test_verify_citations_marks_real_citation_as_verified():
    answer = "Self-attention is key. [Source: attention.pdf, Page 2]"
    retrieved_chunks = [
        {"filename": "attention.pdf", "page_number": 2, "text": "..."},
        {"filename": "attention.pdf", "page_number": 5, "text": "..."},
    ]
    report = verify_citations(answer, retrieved_chunks)

    assert report.verified_count == 1
    assert report.fabricated_count == 0
    assert report.all_verified


def test_verify_citations_flags_fabricated_page_number():
    # Correct filename, but page 99 was never actually retrieved
    answer = "Self-attention is key. [Source: attention.pdf, Page 99]"
    retrieved_chunks = [
        {"filename": "attention.pdf", "page_number": 2, "text": "..."},
    ]
    report = verify_citations(answer, retrieved_chunks)

    assert report.verified_count == 0
    assert report.fabricated_count == 1
    assert not report.all_verified


def test_verify_citations_flags_fabricated_filename():
    answer = "Self-attention is key. [Source: nonexistent_paper.pdf, Page 2]"
    retrieved_chunks = [
        {"filename": "attention.pdf", "page_number": 2, "text": "..."},
    ]
    report = verify_citations(answer, retrieved_chunks)

    assert report.fabricated_count == 1


def test_verify_citations_handles_no_citations_in_answer():
    answer = "The provided documents don't contain enough information to answer this."
    report = verify_citations(answer, retrieved_chunks=[])

    assert report.has_any_citations is False
    assert report.verified_count == 0
    assert report.fabricated_count == 0