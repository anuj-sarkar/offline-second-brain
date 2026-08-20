"""
tests/test_markdown_loader.py

Tests for app.ingestion.markdown_loader.
"""

from app.ingestion.markdown_loader import parse_markdown_sections


def test_parse_markdown_sections_extracts_headers():
    md = """# Introduction
This is the intro paragraph.

## Methodology
Here is the method.

### Results
Here are the results.
"""
    sections = parse_markdown_sections(md)
    assert len(sections) == 3
    assert sections[0]["section_title"] == "Introduction"
    assert "This is the intro" in sections[0]["text"]
    assert sections[1]["section_title"] == "Methodology"
    assert sections[2]["section_title"] == "Results"


def test_parse_plain_text():
    text = "Just a raw string with no markdown headers."
    sections = parse_markdown_sections(text)
    assert len(sections) == 1
    assert sections[0]["section_title"] == "Document"
    assert sections[0]["text"] == text
