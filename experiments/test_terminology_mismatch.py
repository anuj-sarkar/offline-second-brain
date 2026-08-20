"""
experiments/test_terminology_mismatch.py

Compares Dense vs. Hybrid vs. FlashRank Rerank on terminology mismatch queries.
Tests how robust the retrieval pipeline is when the user's phrasing avoids
the document's exact vocabulary.
"""

import logging

from app.generation.rag_pipeline import retrieve_context

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EXACT_TERMINOLOGY_QUERY = "What is encoder-decoder attention?"
PARAPHRASED_QUERY = "How does information from the input side get combined with what the output side is generating?"

if __name__ == "__main__":
    for label, query in [
        ("EXACT TERMINOLOGY", EXACT_TERMINOLOGY_QUERY),
        ("PARAPHRASED (terminology mismatch)", PARAPHRASED_QUERY),
    ]:
        print(f"\n{'=' * 75}")
        print(f"{label}: '{query}'")
        print("=" * 75)

        for strat in ["dense", "hybrid_no_rerank", "hybrid"]:
            results = retrieve_context(query, top_k=3, retrieval_strategy=strat)
            print(f"\n--- Strategy: {strat} ---")
            for rank, r in enumerate(results, start=1):
                score = r.get("rerank_score", r.get("similarity", r.get("fused_score", 0)))
                print(f"Rank {rank}: {r['filename']} p.{r['page_number']} ({r.get('section_title', 'General')}) [score: {score:.4f}]")
                print(f"  {r['text'][:140]}...")