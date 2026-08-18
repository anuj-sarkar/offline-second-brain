"""
tests/test_cleaning.py

Tests for app.ingestion.cleaning, built from the EXACT real patterns
we observed in earlier phases - not synthetic examples.

Run with: python -m pytest tests/test_cleaning.py -v
"""

from app.ingestion.cleaning import clean_page_text


def test_strips_ieee_watermark():
    text = (
        "Some real paper content here.\n"
        "Authorized licensed use limited to: Indian Institute of Technology "
        "(ISM) Dhanbad. Downloaded on July 23,2026 at 14:05:16 UTC from IEEE "
        "Xplore.  Restrictions apply.\n"
        "More content after."
    )
    cleaned = clean_page_text(text)
    assert "Authorized licensed use limited to" not in cleaned
    assert "Some real paper content here." in cleaned
    assert "More content after." in cleaned


def test_strips_arxiv_line():
    text = "Abstract text here.\narXiv:1706.03762v7  [cs.CL]  2 Aug 2023\nMore text."
    cleaned = clean_page_text(text)
    assert "arXiv:1706.03762" not in cleaned
    assert "Abstract text here." in cleaned


def test_strips_conference_line():
    text = (
        "Abstract text here.\n"
        "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.\n"
        "More text."
    )
    cleaned = clean_page_text(text)
    assert "Conference on Neural Information" not in cleaned
    assert "Abstract text here." in cleaned


def test_strips_footnote_marker_blocks():
    text = (
        "Abstract text describing the paper's contribution.\n"
        "∗Equal contribution. Listing order is random. Jakob proposed replacing "
        "RNNs with self-attention and started the effort to evaluate this idea.\n\n"
        "Section 1 starts here."
    )
    cleaned = clean_page_text(text)
    assert "Equal contribution" not in cleaned
    assert "Abstract text describing the paper's contribution." in cleaned
    assert "Section 1 starts here." in cleaned


def test_strips_ieee_manuscript_footnote():
    text = (
        "Abstract text about the algorithm.\n"
        "Manuscript received August 18, 2000; revised February 5, 2001 and "
        "September 7, 2001. The work of K. Deb was supported by the Ministry "
        "of Human Resources and Development, India. Publisher Item Identifier "
        "S 1089-778X(02)04101-2.\n"
        "I. INTRODUCTION starts here."
    )
    cleaned = clean_page_text(text)
    assert "Manuscript received" not in cleaned
    assert "Publisher Item Identifier" not in cleaned
    assert "Abstract text about the algorithm." in cleaned
    assert "I. INTRODUCTION starts here." in cleaned


def test_strips_ieee_running_header():
    text = "182 IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 6, NO. 2, APRIL 2002\nReal content follows."
    cleaned = clean_page_text(text)
    assert "IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION" not in cleaned
    assert "Real content follows." in cleaned


def test_strips_ieee_copyright_line():
    text = "Paper content here.\n1089-778X/02$17.00 © 2002 IEEE"
    cleaned = clean_page_text(text)
    assert "© 2002 IEEE" not in cleaned
    assert "Paper content here." in cleaned


def test_does_not_remove_normal_content():
    text = "This is a completely normal paragraph with no boilerplate patterns at all."
    cleaned = clean_page_text(text)
    assert cleaned == text