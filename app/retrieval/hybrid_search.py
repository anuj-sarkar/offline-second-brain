"""
app/retrieval/hybrid_search.py

Hybrid search combining dense semantic search (ChromaDB) and sparse keyword search (BM25).
Optionally reranks candidates using a local cross-encoder (FlashRank).
Caches the BM25 index on disk to avoid rebuilding from scratch on every query.
"""

import logging
from pathlib import Path
import pickle
import re

from rank_bm25 import BM25Okapi

from app.embeddings.embedder import embed_text
from app.retrieval.reranker import rerank_chunks
from app.retrieval.vectorstore import (
    create_collection,
    get_client,
    search as dense_search,
)

logger = logging.getLogger(__name__)

RRF_K = 60
DEFAULT_BM25_CACHE_PATH = "vectorstore/bm25_index.pkl"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """Tokenize text preserving internal hyphens while stripping punctuation."""
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(collection) -> tuple[BM25Okapi, list[dict]]:
    """Build an in-memory BM25 index over all chunks in ChromaDB."""
    all_items = collection.get()
    ids = all_items.get("ids", [])
    documents = all_items.get("documents", [])
    metadatas = all_items.get("metadatas", [])

    if not ids:
        return BM25Okapi([["empty"]]), []

    chunk_records = [
        {
            "chunk_id": ids[i],
            "text": documents[i],
            "filename": metadatas[i].get("filename", "unknown"),
            "page_number": metadatas[i].get("page_number", 1),
            "doc_id": metadatas[i].get("doc_id", ""),
            "section_title": metadatas[i].get("section_title", "General"),
        }
        for i in range(len(ids))
    ]

    tokenized_corpus = [_tokenize(c["text"]) for c in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"Built BM25 index over {len(chunk_records)} chunks")
    return bm25, chunk_records


def save_bm25_index(
    bm25: BM25Okapi,
    chunk_records: list[dict],
    cache_path: str = DEFAULT_BM25_CACHE_PATH,
) -> None:
    """Serialize BM25 index and chunk records to disk."""
    try:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"bm25": bm25, "records": chunk_records}, f)
        logger.info(f"Saved BM25 index cache to {cache_path}")
    except Exception as e:
        logger.warning(f"Could not save BM25 index cache: {e}")


def load_or_build_bm25_index(
    collection,
    cache_path: str = DEFAULT_BM25_CACHE_PATH,
    force_rebuild: bool = False,
) -> tuple[BM25Okapi, list[dict]]:
    """Load cached BM25 index if valid; otherwise build and save."""
    cache_file = Path(cache_path)
    collection_count = collection.count()

    if not force_rebuild and cache_file.exists():
        try:
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
                bm25 = data["bm25"]
                records = data["records"]
                if len(records) == collection_count:
                    return bm25, records
        except Exception as e:
            logger.warning(f"Failed to load BM25 cache: {e}. Rebuilding...")

    bm25, records = build_bm25_index(collection)
    save_bm25_index(bm25, records, cache_path=cache_path)
    return bm25, records


def bm25_search(
    bm25: BM25Okapi,
    chunk_records: list[dict],
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Search chunks using BM25 Okapi."""
    if not chunk_records:
        return []

    tokenized_query = _tokenize(query)
    if not tokenized_query:
        return []

    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in ranked_indices:
        chunk = chunk_records[idx]
        results.append({**chunk, "bm25_score": float(scores[idx])})
    return results


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Combine ranked lists using standard RRF."""
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup.setdefault(cid, chunk)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused = []
    for cid in ranked_ids:
        chunk = dict(chunk_lookup[cid])
        chunk["fused_score"] = scores[cid]
        fused.append(chunk)
    return fused


def hybrid_search(
    query: str,
    top_k: int = 5,
    candidate_pool: int = 15,
    use_reranker: bool = True,
) -> list[dict]:
    """
    Execute hybrid retrieval:
      1. Dense search (with 'search_query:' task prefix).
      2. BM25 keyword search (from cached index).
      3. If use_reranker=True: Rerank candidates with FlashRank cross-encoder.
         Else: Fuse using RRF.
    """
    client = get_client()
    collection = create_collection(client)

    if collection.count() == 0:
        return []

    # 1. Dense retrieval
    query_vector = embed_text(query, is_query=True)
    dense_results = dense_search(collection, query_vector, top_k=candidate_pool)

    # 2. BM25 retrieval
    bm25, chunk_records = load_or_build_bm25_index(collection)
    keyword_results = bm25_search(bm25, chunk_records, query, top_k=candidate_pool)

    # 3. Rerank or Fuse
    if use_reranker:
        # Merge unique candidates from both streams
        seen_ids = set()
        candidates = []
        for c in dense_results + keyword_results:
            cid = c["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                candidates.append(c)

        return rerank_chunks(query, candidates, top_k=top_k)
    else:
        return reciprocal_rank_fusion(dense_results, keyword_results, top_k=top_k)