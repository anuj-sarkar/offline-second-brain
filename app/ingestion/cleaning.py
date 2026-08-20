"""
app/ingestion/cleaning.py

Text cleaning and normalization for extracted document text.
Applies:
  1. Unicode normalization (hyphens, asterisks, quotes, whitespace).
  2. De-hyphenation across line breaks (e.g. 'trans-\\nformer' -> 'transformer').
  3. Pattern-based removal of publisher watermarks, headers, and footnote blocks.
  4. Paragraph and blank-line consolidation.
"""

import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

# IEEE Xplore per-page watermark
_IEEE_WATERMARK_PATTERN = re.compile(
    r"Authorized licensed use limited to:.*?Restrictions apply\.",
    re.DOTALL,
)

# arXiv identifier line
_ARXIV_LINE_PATTERN = re.compile(r"arXiv:\S+\s+\[[\w.]+\]\s+\d+\s+\w+\s+\d{4}")

# Conference attribution line
_CONFERENCE_LINE_PATTERN = re.compile(r"^\d+(?:st|nd|rd|th)\s+Conference on.*$", re.MULTILINE)

# Footnote marker lines: author-contribution blocks
_FOOTNOTE_MARKER_PATTERN = re.compile(
    r"[*∗†‡]\s*(?:Equal contribution|Work performed).*?(?=\n\n|\Z)",
    re.DOTALL,
)

# IEEE journal-style manuscript footnote block
_IEEE_MANUSCRIPT_FOOTNOTE_PATTERN = re.compile(
    r"Manuscript received.*?Publisher Item Identifier[^\n]*\.",
    re.DOTALL,
)

# IEEE running header
_IEEE_RUNNING_HEADER_PATTERN = re.compile(
    r"^\d+\s+IEEE TRANSACTIONS ON [A-Z ]+,\s*VOL\.\s*\d+,\s*NO\.\s*\d+,.*$",
    re.MULTILINE,
)

# IEEE copyright line
_IEEE_COPYRIGHT_LINE_PATTERN = re.compile(r"\d{4}-\d{3,4}X/\d{2}\$[\d.]+\s*©\s*\d{4}\s*IEEE")

# Unicode character mappings for normalization
_UNICODE_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u00a0": " ",
    "\u200b": "",
}


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters and standardise quotes, dashes, and spaces."""
    for char, replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return unicodedata.normalize("NFKC", text)


def dehyphenate_linebreaks(text: str) -> str:
    """Join words broken across line wraps (e.g. 'multi-\\nobjective' -> 'multiobjective')."""
    return re.sub(r"([a-zA-Z]+)-\n\s*([a-zA-Z]+)", r"\1\2", text)


def clean_page_text(text: str) -> str:
    """
    Strip known boilerplate and clean up extracted document text.
    Applied before chunking.
    """
    if not text:
        return ""

    original_length = len(text)

    # 1. Apply known pattern removals
    cleaned = _IEEE_WATERMARK_PATTERN.sub("", text)
    cleaned = _ARXIV_LINE_PATTERN.sub("", cleaned)
    cleaned = _CONFERENCE_LINE_PATTERN.sub("", cleaned)
    cleaned = _FOOTNOTE_MARKER_PATTERN.sub("", cleaned)
    cleaned = _IEEE_MANUSCRIPT_FOOTNOTE_PATTERN.sub("", cleaned)
    cleaned = _IEEE_RUNNING_HEADER_PATTERN.sub("", cleaned)
    cleaned = _IEEE_COPYRIGHT_LINE_PATTERN.sub("", cleaned)

    # 2. De-hyphenate broken line wraps
    cleaned = dehyphenate_linebreaks(cleaned)

    # 3. Unicode normalize
    cleaned = normalize_unicode(cleaned)

    # 4. Collapse consecutive blank lines and trim
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    removed_chars = original_length - len(cleaned)
    if removed_chars > 0:
        logger.debug(f"Cleaned {removed_chars} chars of noise/boilerplate")

    return cleaned