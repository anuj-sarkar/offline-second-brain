"""
app/retrieval/reranker.py

Local cross-encoder reranking using FlashRank.
Runs 100% locally and offline on CPU using quantized ONNX models.
Cross-encoders jointly process (query, passage) pairs, capturing complex token-level
interactions that bi-encoder cosine similarity misses.
"""

import logging
from typing import Optional

from flashrank import Ranker, RerankRequest

logger = logging.getLogger(__name__)

# Default lightweight, high-accuracy cross-encoder model
DEFAULT_RERANKER_MODEL = "ms-marco-MiniLM-L-12-v2"

_RANKER_INSTANCE: Optional[Ranker] = None


def get_ranker(model_name: str = DEFAULT_RERANKER_MODEL) -> Ranker:
    """Lazy-load singleton Ranker instance."""
    global _RANKER_INSTANCE
    if _RANKER_INSTANCE is None:
        logger.info(f"Initializing FlashRank Ranker ({model_name})...")
        _RANKER_INSTANCE = Ranker(model_name=model_name)
    return _RANKER_INSTANCE


def rerank_chunks(
    query: str,
    candidate_chunks: list[dict],
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
) -> list[dict]:
    """
    Rerank candidate chunks using local cross-encoder.

    Input:
      query: User question or search query
      candidate_chunks: Pool of retrieved chunks (from dense or hybrid search)
      top_k: Number of highest-relevance chunks to return
    Output:
      top_k reranked chunks with 'rerank_score' attached, ordered best-first.
    """
    if not candidate_chunks:
        return []

    if len(candidate_chunks) == 1:
        chunk = dict(candidate_chunks[0])
        chunk.setdefault("rerank_score", 1.0)
        return [chunk]

    try:
        ranker = get_ranker(model_name=model_name)
        passages = [
            {
                "id": str(c.get("chunk_id", i)),
                "text": c.get("text", ""),
                "meta": c,
            }
            for i, c in enumerate(candidate_chunks)
        ]

        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = ranker.rerank(rerank_request)

        reranked = []
        for r in ranked_results[:top_k]:
            chunk_data = dict(r["meta"])
            chunk_data["rerank_score"] = float(r["score"])
            reranked.append(chunk_data)

        logger.info(f"Reranked {len(candidate_chunks)} candidates -> top {len(reranked)}")
        return reranked

    except Exception as e:
        logger.error(f"Reranking failed with error: {e}. Falling back to original candidates.")
        return candidate_chunks[:top_k]
