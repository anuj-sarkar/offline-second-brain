"""
app/generation/citations.py

Verifies that citations produced by the LLM match retrieved chunks AND are
lexically/semantically grounded in the cited source text rather than hallucinated.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Matches [Source: filename, Page X] or [Source: filename]
CITATION_PATTERN = re.compile(
    r"\[Source:\s*([^,]+?)(?:,\s*Page\s*(\d+))?\]",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with", "source", "page", "this", "these"
}


@dataclass
class Citation:
    filename: str
    page_number: int
    is_verified: bool       # True if (filename, page_number) exists in retrieved context
    is_grounded: bool       # True if the claim sentence shares semantic/lexical overlap with chunk
    claim_text: str = ""    # The sentence making the cited claim


@dataclass
class CitationReport:
    citations_found: list[Citation]
    verified_count: int
    fabricated_count: int
    ungrounded_count: int

    @property
    def all_verified(self) -> bool:
        return self.fabricated_count == 0

    @property
    def all_grounded(self) -> bool:
        return self.ungrounded_count == 0

    @property
    def has_any_citations(self) -> bool:
        return len(self.citations_found) > 0


def extract_claims_and_citations(answer_text: str) -> list[tuple[str, str, int]]:
    """
    Extract each sentence along with its citation (filename, page_number).
    Returns list of (claim_sentence, filename, page_number).
    """
    results = []
    # Split text into sentences
    sentences = re.split(r"(?<=[.!?])\s+", answer_text)

    for sentence in sentences:
        matches = CITATION_PATTERN.findall(sentence)
        for match in matches:
            filename = match[0].strip()
            page_str = match[1].strip() if len(match) > 1 and match[1] else "1"
            try:
                page_num = int(page_str)
            except ValueError:
                page_num = 1
            results.append((sentence.strip(), filename, page_num))

    return results


def extract_citations(answer_text: str) -> list[tuple[str, int]]:
    """Pull (filename, page_number) tuples from answer text."""
    matches = CITATION_PATTERN.findall(answer_text)
    results = []
    for match in matches:
        filename = match[0].strip()
        page_str = match[1].strip() if len(match) > 1 and match[1] else "1"
        try:
            page_num = int(page_str)
        except ValueError:
            page_num = 1
        results.append((filename, page_num))
    return results


def _check_claim_grounding(claim_text: str, chunk_text: str) -> bool:
    """
    Check if non-stopword tokens in the claim sentence are substantiated in the chunk text.
    """
    # Clean citations out of the sentence before checking
    clean_sentence = CITATION_PATTERN.sub("", claim_text).lower()
    words = re.findall(r"\b[a-z0-9-]+\b", clean_sentence)
    content_words = [w for w in words if w not in _STOPWORDS and len(w) > 2]

    if not content_words:
        return True  # Trivial sentence

    chunk_lower = chunk_text.lower()
    matched_words = sum(1 for w in content_words if w in chunk_lower)
    overlap_ratio = matched_words / len(content_words)

    # Require at least 40% keyword grounding in the cited source
    return overlap_ratio >= 0.4


def verify_citations(answer_text: str, retrieved_chunks: list[dict]) -> CitationReport:
    """
    Verify citations against retrieved chunks and check claim-support grounding.
    """
    chunk_map: dict[tuple[str, int], list[str]] = {}
    for chunk in retrieved_chunks:
        key = (chunk.get("filename", ""), chunk.get("page_number", 1))
        chunk_map.setdefault(key, []).append(chunk.get("text", ""))

    claims_and_citations = extract_claims_and_citations(answer_text)
    citations: list[Citation] = []

    for claim, filename, page_number in claims_and_citations:
        key = (filename, page_number)
        is_verified = key in chunk_map

        is_grounded = True
        if is_verified:
            matching_texts = chunk_map[key]
            is_grounded = any(_check_claim_grounding(claim, text) for text in matching_texts)

        citations.append(Citation(
            filename=filename,
            page_number=page_number,
            is_verified=is_verified,
            is_grounded=is_grounded,
            claim_text=claim,
        ))

    # If fallback extract without sentence split finds additional citations
    if not citations:
        for filename, page_number in extract_citations(answer_text):
            is_verified = (filename, page_number) in chunk_map
            citations.append(Citation(
                filename=filename,
                page_number=page_number,
                is_verified=is_verified,
                is_grounded=is_verified,
            ))

    verified_count = sum(1 for c in citations if c.is_verified)
    fabricated_count = len(citations) - verified_count
    ungrounded_count = sum(1 for c in citations if c.is_verified and not c.is_grounded)

    return CitationReport(
        citations_found=citations,
        verified_count=verified_count,
        fabricated_count=fabricated_count,
        ungrounded_count=ungrounded_count,
    )