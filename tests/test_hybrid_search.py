"""
tests/test_hybrid_search.py

Tests for app.retrieval.hybrid_search. RRF fusion logic is tested
with small hand-built fake ranked lists (no need to hit Ollama/Chroma
for this - it's pure ranking-math logic). bm25_search is tested with
a tiny real BM25Okapi index over toy documents.

Run with: python -m pytest tests/test_hybrid_search.py -v
"""

from rank_bm25 import BM25Okapi

from app.retrieval.hybrid_search import (
    reciprocal_rank_fusion, bm25_search, _tokenize,
)


def test_reciprocal_rank_fusion_favors_chunk_ranked_high_in_both_lists():
    dense_results = [
        {"chunk_id": "A", "text": "dense top"},
        {"chunk_id": "B", "text": "dense second"},
    ]
    bm25_results = [
        {"chunk_id": "B", "text": "bm25 top"},
        {"chunk_id": "C", "text": "bm25 second"},
    ]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, top_k=3)

    # B ranked well in BOTH lists - should come out on top despite
    # never being #1 in either individual list
    assert fused[0]["chunk_id"] == "B"


def test_reciprocal_rank_fusion_includes_chunks_from_either_list():
    dense_results = [{"chunk_id": "A", "text": "only in dense"}]
    bm25_results = [{"chunk_id": "Z", "text": "only in bm25"}]

    fused = reciprocal_rank_fusion(dense_results, bm25_results, top_k=5)
    fused_ids = {c["chunk_id"] for c in fused}

    assert "A" in fused_ids
    assert "Z" in fused_ids


def test_reciprocal_rank_fusion_respects_top_k():
    dense_results = [{"chunk_id": str(i), "text": "t"} for i in range(10)]
    fused = reciprocal_rank_fusion(dense_results, [], top_k=3)
    assert len(fused) == 3


def test_tokenize_strips_trailing_punctuation():
    # Regression test for a real bug: "NSGA-II?" must tokenize to the
    # same term as "NSGA-II," or "NSGA-II." elsewhere in a corpus,
    # or BM25 silently fails to match the single most important term
    # in a query - discovered by testing on a real query, not by
    # inspection alone.
    assert _tokenize("What is NSGA-II?") == ["what", "is", "nsga-ii"]
    assert _tokenize("NSGA-II, a genetic algorithm.") == ["nsga-ii", "a", "genetic", "algorithm"]


def test_bm25_search_ranks_exact_term_match_highest():
    chunk_records = [
        {"chunk_id": "1", "text": "This paper discusses genetic algorithms broadly.",
         "filename": "a.pdf", "page_number": 1, "doc_id": "d1"},
        {"chunk_id": "2", "text": "NSGA-II NSGA-II NSGA-II is a fast elitist algorithm.",
         "filename": "b.pdf", "page_number": 1, "doc_id": "d2"},
        {"chunk_id": "3", "text": "Stock markets fell sharply today amid concerns.",
         "filename": "c.pdf", "page_number": 1, "doc_id": "d3"},
    ]
    tokenized = [_tokenize(c["text"]) for c in chunk_records]
    bm25 = BM25Okapi(tokenized)

    results = bm25_search(bm25, chunk_records, "NSGA-II", top_k=3)

    assert results[0]["chunk_id"] == "2"  # the chunk repeating the exact term wins