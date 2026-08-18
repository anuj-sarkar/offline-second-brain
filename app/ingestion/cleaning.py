"""
app/ingestion/cleaning.py

Phase 10 deliverable: pattern-based cleaning of known boilerplate and
footnote noise, discovered from REAL text we extracted and inspected
in earlier phases - not designed speculatively upfront.

Three patterns found and confirmed across this project's real PDFs:
  1. IEEE's per-page download watermark, repeated on nearly every
     page of the NSGA-II/MOEA-D papers (seen in Phase 9's retrieved
     chunks).
  2. Author-contribution footnotes (*, dagger, double-dagger markers)
     glued directly after the abstract with no separator (seen in
     Phase 1's very first extraction of page 1).
  3. Conference/arXiv identifier lines (seen in that same page 1).

This is intentionally NOT a general-purpose PDF-structure parser -
it targets specific, observed noise patterns. A more general solution
(font-size-aware extraction via PyMuPDF) is a bigger change, noted
but not implemented here per the "don't add libraries just because"
principle - this cheaper fix is tested first.
"""

import logging
import re

logger = logging.getLogger(__name__)

# IEEE Xplore's per-page download watermark - confirmed present
# verbatim in real retrieved chunks (Phase 9 results).
_IEEE_WATERMARK_PATTERN = re.compile(
    r"Authorized licensed use limited to:.*?Restrictions apply\.",
    re.DOTALL,
)

# arXiv identifier line, e.g. "arXiv:1706.03762v7  [cs.CL]  2 Aug 2023"
_ARXIV_LINE_PATTERN = re.compile(r"arXiv:\S+\s+\[[\w.]+\]\s+\d+\s+\w+\s+\d{4}")

# Conference attribution line, e.g.
# "31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA."
_CONFERENCE_LINE_PATTERN = re.compile(r"^\d+(?:st|nd|rd|th)\s+Conference on.*$", re.MULTILINE)

# Footnote marker lines: a line starting with *, ∗ (Unicode asterisk
# operator - U+2217, what PDF fonts commonly use instead of plain
# ASCII "*"), †, or ‡, followed by author-contribution text. We strip
# from the FIRST such marker to the end of that paragraph block, since
# these footnotes consistently appear as a contiguous block glued
# after the abstract in our real documents (verified in Phase 1).
_FOOTNOTE_MARKER_PATTERN = re.compile(
    r"[*∗†‡]\s*(?:Equal contribution|Work performed).*?(?=\n\n|\Z)",
    re.DOTALL,
)


# IEEE journal-style manuscript footnote block, e.g.
# "Manuscript received August 18, 2000; revised... Publisher Item
# Identifier S 1089-778X(02)04101-2." - a completely different
# footnote convention than the NeurIPS-style asterisk footnotes above,
# confirmed present in the NSGA-II paper's real page 1 text.
_IEEE_MANUSCRIPT_FOOTNOTE_PATTERN = re.compile(
    r"Manuscript received.*?Publisher Item Identifier[^\n]*\.",
    re.DOTALL,
)

# IEEE running header repeated at the top of pages, e.g.
# "182 IEEE TRANSACTIONS ON EVOLUTIONARY COMPUTATION, VOL. 6, NO. 2, APRIL 2002"
_IEEE_RUNNING_HEADER_PATTERN = re.compile(
    r"^\d+\s+IEEE TRANSACTIONS ON [A-Z ]+,\s*VOL\.\s*\d+,\s*NO\.\s*\d+,.*$",
    re.MULTILINE,
)

# IEEE copyright/DOI line, e.g. "1089-778X/02$17.00 © 2002 IEEE"
_IEEE_COPYRIGHT_LINE_PATTERN = re.compile(r"\d{4}-\d{3,4}X/\d{2}\$[\d.]+\s*©\s*\d{4}\s*IEEE")


def clean_page_text(text: str) -> str:
    """
    Strip known boilerplate/footnote patterns from extracted page text.

    Input:  raw extracted page text (from app.ingestion.loader)
    Output: cleaned text, with confirmed-noise patterns removed

    Applied BEFORE chunking, so noise never enters a chunk in the
    first place - cheaper and more reliable than trying to detect
    and discount it after the fact during retrieval.
    """
    original_length = len(text)

    cleaned = _IEEE_WATERMARK_PATTERN.sub("", text)
    cleaned = _ARXIV_LINE_PATTERN.sub("", cleaned)
    cleaned = _CONFERENCE_LINE_PATTERN.sub("", cleaned)
    cleaned = _FOOTNOTE_MARKER_PATTERN.sub("", cleaned)
    cleaned = _IEEE_MANUSCRIPT_FOOTNOTE_PATTERN.sub("", cleaned)
    cleaned = _IEEE_RUNNING_HEADER_PATTERN.sub("", cleaned)
    cleaned = _IEEE_COPYRIGHT_LINE_PATTERN.sub("", cleaned)

    # Collapse the extra blank lines/whitespace left behind by removals
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()

    removed_chars = original_length - len(cleaned)
    if removed_chars > 0:
        logger.info(f"Cleaning removed {removed_chars} chars of boilerplate/footnote noise")

    return cleaned


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from app.ingestion.loader import load_pdfs_from_directory

    records = load_pdfs_from_directory("data/raw")

    # Test on the NSGA-II paper's page 1 - this is the actual page
    # relevant to Phase 9's finding: "What is NSGA-II?" never surfaced
    # the definitional abstract chunk, and we hypothesized footnote/
    # boilerplate dilution as a possible cause.
    page_1 = next(
        r for r in records
        if r.filename == "a_fast_and_elitist_multiobjective_genetic_algorithm_NSGA-II.pdf"
        and r.page_number == 1
    )

    print("=" * 70)
    print("BEFORE CLEANING:")
    print("=" * 70)
    print(page_1.text)
    print(f"\n[{len(page_1.text)} characters]")

    cleaned = clean_page_text(page_1.text)

    print("\n" + "=" * 70)
    print("AFTER CLEANING:")
    print("=" * 70)
    print(cleaned)
    print(f"\n[{len(cleaned)} characters]")