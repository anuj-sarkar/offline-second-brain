"""
tests/test_ingestion.py

Minimal tests for the loader. Not exhaustive - just enough to catch
regressions on the behaviors we specifically engineered:
metadata presence, low-content flagging, graceful handling of a
missing directory.

Run with: pytest tests/test_ingestion.py -v
(pip install pytest if you don't have it yet)
"""

from app.ingestion.loader import load_pdfs_from_directory, PageRecord


def test_missing_directory_returns_empty_list():
    records = load_pdfs_from_directory("data/this_directory_does_not_exist")
    assert records == []


def test_records_have_required_metadata():
    records = load_pdfs_from_directory("data/raw")
    assert len(records) > 0, "Expected at least one PDF in data/raw for this test"

    first = records[0]
    assert isinstance(first, PageRecord)
    assert first.doc_id
    assert first.filename
    assert first.source_path
    assert first.page_number >= 1
    assert isinstance(first.char_count, int)


def test_structural_heuristic_catches_word_list_pages():
    from app.ingestion.loader import _is_low_content

    # Shaped like the real attention-visualization appendix page:
    # plenty of characters, but one word per line - not prose.
    word_list_text = "\n".join(["It", "is", "in", "this", "spirit"] * 30)
    assert len(word_list_text) > 100  # passes the raw char-count check...
    assert _is_low_content(word_list_text, len(word_list_text)) is True  # ...but structural check still catches it


def test_structural_heuristic_does_not_flag_normal_prose():
    from app.ingestion.loader import _is_low_content

    prose = (
        "This is a normal paragraph of extracted PDF text. It wraps across "
        "several lines the way real body text does, with each line holding "
        "a full sentence fragment rather than a single isolated word.\n"
    ) * 10
    assert _is_low_content(prose, len(prose)) is False


def test_low_content_pages_are_flagged():
    records = load_pdfs_from_directory("data/raw")
    flagged = [r for r in records if r.is_low_content]
    not_flagged = [r for r in records if not r.is_low_content]

    # We expect at least some normal-content pages in a real paper
    assert len(not_flagged) > 0

    # Every flagged page should be flagged for a legitimate reason:
    # either genuinely near-empty (< 100 chars), OR structurally
    # word-list-like (real char volume, but very short average line
    # length - e.g. figure/diagram label dumps). We no longer assert
    # ALL flagged pages are under 100 chars, since the structural
    # heuristic added in Phase 9 can legitimately flag longer pages.
    for r in flagged:
        lines = [line for line in r.text.split("\n") if line.strip()]
        avg_line_length = r.char_count / len(lines) if lines else 0
        is_near_empty = r.char_count < 100
        is_word_list_like = len(lines) >= 30 and avg_line_length < 12
        assert is_near_empty or is_word_list_like, (
            f"Page {r.page_number} of {r.filename} was flagged low-content "
            f"but matches neither known reason (char_count={r.char_count})"
        )