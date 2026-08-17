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


def test_low_content_pages_are_flagged():
    records = load_pdfs_from_directory("data/raw")
    flagged = [r for r in records if r.is_low_content]
    not_flagged = [r for r in records if not r.is_low_content]

    # We expect at least some normal-content pages in a real paper
    assert len(not_flagged) > 0

    # Every flagged page should actually be under the threshold
    for r in flagged:
        assert r.char_count < 100