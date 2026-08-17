"""
app/generation/citations.py

Phase 8 deliverable: verify that citations the LLM produces in its
answer actually correspond to chunks that were retrieved - not
fabricated filenames or page numbers that merely sound plausible.

This matters because grounding instructions (Phase 7) reduce
hallucination of FACTS, but don't automatically guarantee the
CITATIONS themselves are accurate - a model can still write a
correct-sounding [Source: paper.pdf, Page 4] even if page 4 was
never actually retrieved, or even if that filename isn't in your
library at all. This module catches that specific failure mode.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches: [Source: filename.pdf, Page 7]  (case-insensitive on "Source"/"Page",
# tolerant of extra whitespace)
CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^,]+?),\s*Page\s*(\d+)\]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    filename: str
    page_number: int
    is_verified: bool  # True if this citation matches an actually-retrieved chunk


@dataclass
class CitationReport:
    citations_found: list[Citation]
    verified_count: int
    fabricated_count: int

    @property
    def all_verified(self) -> bool:
        return self.fabricated_count == 0

    @property
    def has_any_citations(self) -> bool:
        return len(self.citations_found) > 0


def extract_citations(answer_text: str) -> list[tuple[str, int]]:
    """
    Pull every [Source: filename, Page N] citation out of the answer text.

    Input:  the LLM's generated answer string
    Output: list of (filename, page_number) tuples, in the order they
            appear in the text (duplicates preserved - if the model
            cites the same source twice, that's worth knowing too)
    """
    matches = CITATION_PATTERN.findall(answer_text)
    return [(filename.strip(), int(page)) for filename, page in matches]


def verify_citations(answer_text: str, retrieved_chunks: list[dict]) -> CitationReport:
    """
    Check every citation the model produced against what was actually
    retrieved for this query.

    Input:  the generated answer text, and the list of chunk dicts
            that were actually passed into the prompt as context
    Output: a CitationReport - which citations are real, which are
            fabricated (filename/page combination never retrieved)

    A citation is considered "verified" only if BOTH filename and
    page_number exactly match a retrieved chunk - a correct filename
    with a wrong page number still counts as fabricated, since it
    would point the user to the wrong location in the source document.
    """
    retrieved_set = {
        (chunk["filename"], chunk["page_number"]) for chunk in retrieved_chunks
    }

    raw_citations = extract_citations(answer_text)
    citations: list[Citation] = []

    for filename, page_number in raw_citations:
        is_verified = (filename, page_number) in retrieved_set
        citations.append(Citation(
            filename=filename,
            page_number=page_number,
            is_verified=is_verified,
        ))
        if not is_verified:
            logger.warning(
                f"Fabricated citation detected: [Source: {filename}, Page {page_number}] "
                f"was not among the retrieved chunks for this query."
            )

    verified_count = sum(1 for c in citations if c.is_verified)
    fabricated_count = len(citations) - verified_count

    return CitationReport(
        citations_found=citations,
        verified_count=verified_count,
        fabricated_count=fabricated_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from app.generation.rag_pipeline import answer_question

    questions = [
        "What is the main innovation of the Transformer architecture?",
        "What is NSGA-II?",
    ]

    for q in questions:
        result = answer_question(q, top_k=5)
        report = verify_citations(result.answer, result.retrieved_chunks)

        print(f"\n{'=' * 70}")
        print(f"QUESTION: {q}")
        print(f"{'=' * 70}")
        print(f"\nANSWER:\n{result.answer}\n")
        print(f"Citations found: {len(report.citations_found)}")
        print(f"  Verified: {report.verified_count}")
        print(f"  Fabricated: {report.fabricated_count}")
        for c in report.citations_found:
            status = "OK" if c.is_verified else "FABRICATED"
            print(f"  [{status}] {c.filename}, Page {c.page_number}")