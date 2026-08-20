"""
tests/test_reranker.py

Tests for app.retrieval.reranker using FlashRank.
"""

from app.retrieval.reranker import rerank_chunks


def test_reranker_returns_ranked_chunks_with_scores():
    query = "What is NSGA-II?"
    candidates = [
        {
            "chunk_id": "c1",
            "text": "Stock market trading strategies and quarterly financial earnings.",
            "filename": "finance.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": "c2",
            "text": "NSGA-II is a fast and elitist multiobjective genetic algorithm.",
            "filename": "nsga2.pdf",
            "page_number": 1,
        },
    ]

    results = rerank_chunks(query, candidates, top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "c2"  # Definitional match should be rank #1
    assert "rerank_score" in results[0]
    assert results[0]["rerank_score"] > results[1]["rerank_score"]


def test_reranker_handles_empty_candidates():
    assert rerank_chunks("query", [], top_k=5) == []


def test_reranker_handles_single_candidate():
    candidates = [{"chunk_id": "c1", "text": "Single passage", "filename": "doc.pdf", "page_number": 1}]
    results = rerank_chunks("query", candidates, top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
