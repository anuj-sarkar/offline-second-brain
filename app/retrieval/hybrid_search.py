"""
app/retrieval/hybrid_search.py

Phase 9 deliverable: hybrid search combining BM25 (keyword-based) and
dense vector search, fused via Reciprocal Rank Fusion (RRF).

This directly targets a real failure we reproduced twice: pure dense
search failed to surface the NSGA-II paper's own definitional
abstract chunk for the query "What is NSGA-II?", because chunks
densely packed with the term in results/parameter-tuning contexts
scored similarly on pure semantic similarity. BM25 fixes exactly this
kind of exact-terminology gap.

Note: this module re-embeds/re-scores over the FULL corpus for BM25,
since rank_bm25 doesn't persist to disk like ChromaDB does - it's an
in-memory index built fresh each run. Fine at our current scale (a
few hundred chunks); worth revisiting if your library grows large.
"""

import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from app.retrieval.vectorstore import get_client, create_collection, search as dense_search
from app.embeddings.embedder import embed_text

logger = logging.getLogger(__name__)

# RRF constant - dampens the influence of very high individual ranks,
# a standard default from the original RRF paper (Cormack et al.).
# Not something you need to tune unless you have a specific reason to.
RRF_K = 60


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    """
    Lowercase + strip punctuation before splitting into tokens.

    This matters more than it looks: a naive text.lower().split() would
    turn "NSGA-II?" into the token "nsga-ii?" (with the question mark
    glued on), which then NEVER exactly matches "nsga-ii" as it appears
    in document text (itself often followed by a comma or period).
    That single mismatch was enough to silently break BM25's exact-term
    matching for the whole query - discovered by testing on a real
    query and noticing garbage results, not by inspection alone.

    The regex keeps internal hyphens (so "nsga-ii" and "non-dominated"
    stay as single meaningful tokens) but strips surrounding punctuation
    like ?, ., ,, ! that would otherwise get glued onto a word.
    """
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(collection) -> tuple[BM25Okapi, list[dict]]:
    """
    Build an in-memory BM25 index over every chunk currently in the
    ChromaDB collection.

    Input:  a ChromaDB collection (from app.retrieval.vectorstore)
    Output: (bm25_index, chunk_records) - the index itself, plus the
            parallel list of chunk metadata/text it was built from
            (needed to map BM25's integer positions back to real
            filename/page_number/text)
    """
    all_items = collection.get()
    ids = all_items["ids"]
    documents = all_items["documents"]
    metadatas = all_items["metadatas"]

    chunk_records = [
        {
            "chunk_id": ids[i],
            "text": documents[i],
            "filename": metadatas[i]["filename"],
            "page_number": metadatas[i]["page_number"],
            "doc_id": metadatas[i]["doc_id"],
        }
        for i in range(len(ids))
    ]

    tokenized_corpus = [_tokenize(c["text"]) for c in chunk_records]
    bm25 = BM25Okapi(tokenized_corpus)

    logger.info(f"Built BM25 index over {len(chunk_records)} chunks")
    return bm25, chunk_records


def bm25_search(bm25: BM25Okapi, chunk_records: list[dict], query: str, top_k: int = 10) -> list[dict]:
    """
    Keyword search using BM25.

    Input:  the BM25 index, its parallel chunk_records, a query string
    Output: list of chunk dicts ranked by BM25 score (best first),
            same dict shape as dense search results for consistency
    """
    tokenized_query = _tokenize(query)
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
    """
    Combine two ranked lists into one, using Reciprocal Rank Fusion.

    Core idea: a chunk's fused score is the sum of 1/(rank + k) across
    every list it appears in. A chunk that ranks well in BOTH dense
    and keyword search rises to the top; a chunk that ranks well in
    only one still gets meaningful credit, rather than being drowned
    out by scale differences between BM25 scores and cosine similarity
    (which aren't on the same numeric scale at all - RRF sidesteps
    that problem entirely by using RANK POSITION, not raw scores).

    Input:  dense_results and bm25_results, each a ranked list of
            chunk dicts (best first), both containing 'chunk_id'
    Output: fused, deduplicated, re-ranked list of chunk dicts,
            length top_k, each with a fused_score added
    """
    scores: dict[str, float] = {}
    chunk_lookup: dict[str, dict] = {}

    for rank, chunk in enumerate(dense_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup[cid] = chunk

    for rank, chunk in enumerate(bm25_results):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
        chunk_lookup.setdefault(cid, chunk)  # keep dense version's fields if both present

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:top_k]

    fused = []
    for cid in ranked_ids:
        chunk = dict(chunk_lookup[cid])
        chunk["fused_score"] = scores[cid]
        fused.append(chunk)
    return fused


def hybrid_search(query: str, top_k: int = 5, candidate_pool: int = 15) -> list[dict]:
    """
    Full hybrid search: run dense + BM25 in parallel, fuse with RRF.

    Input:  a query string, final top_k to return, and how many
            candidates to pull from EACH method before fusing
            (wider than top_k, so fusion has real material to work
            with rather than just re-sorting an already-truncated list)
    Output: top_k fused results, ranked best-first
    """
    client = get_client()
    collection = create_collection(client)

    # Dense search
    query_vector = embed_text(query)
    dense_results = dense_search(collection, query_vector, top_k=candidate_pool)

    # BM25 search
    bm25, chunk_records = build_bm25_index(collection)
    keyword_results = bm25_search(bm25, chunk_records, query, top_k=candidate_pool)

    # Fuse
    fused = reciprocal_rank_fusion(dense_results, keyword_results, top_k=top_k)
    return fused


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    query = "What is NSGA-II?"

    print(f"Query: {query}\n")
    print("=" * 70)
    print("DENSE-ONLY RESULTS (for comparison):")
    print("=" * 70)
    client = get_client()
    collection = create_collection(client)
    query_vector = embed_text(query)
    dense_only = dense_search(collection, query_vector, top_k=5)
    for rank, r in enumerate(dense_only, start=1):
        print(f"Rank {rank}: {r['filename']} p.{r['page_number']} (similarity: {r['similarity']:.4f})")
        print(f"  {r['text'][:150]}...")

    print("\n" + "=" * 70)
    print("HYBRID (BM25 + DENSE, fused via RRF):")
    print("=" * 70)
    hybrid_results = hybrid_search(query, top_k=5)
    for rank, r in enumerate(hybrid_results, start=1):
        print(f"Rank {rank}: {r['filename']} p.{r['page_number']} (fused score: {r['fused_score']:.4f})")
        print(f"  {r['text'][:150]}...")