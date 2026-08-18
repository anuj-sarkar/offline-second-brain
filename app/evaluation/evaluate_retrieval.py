"""
app/evaluation/evaluate_retrieval.py

Phase 11 deliverable: runs the full evaluation dataset through a
given retrieval strategy, computes aggregate Recall@K / Precision@K /
MRR, and saves results to experiments/ so different configurations
(dense vs hybrid, different top_k) can be compared side by side
over time, not just eyeballed one query at a time like we've been
doing since Phase 5.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.dataset import EVAL_DATASET
from app.evaluation.metrics import recall_at_k, precision_at_k, mean_reciprocal_rank
from app.retrieval.vectorstore import get_client, create_collection, search as dense_search
from app.retrieval.hybrid_search import hybrid_search
from app.embeddings.embedder import embed_text

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = "experiments"


def evaluate_strategy(strategy_name: str, top_k: int = 5) -> dict:
    """
    Run the full EVAL_DATASET through one retrieval strategy, compute
    aggregate metrics.

    Input:  strategy_name - either "dense" or "hybrid"
            top_k - how many results to retrieve per query
    Output: a results dict with per-question and aggregate metrics,
            ready to be saved to disk or compared against another run
    """
    client = get_client()
    collection = create_collection(client)

    per_question_results = []

    for eq in EVAL_DATASET:
        if strategy_name == "dense":
            query_vector = embed_text(eq.question)
            results = dense_search(collection, query_vector, top_k=top_k)
        elif strategy_name == "hybrid":
            results = hybrid_search(eq.question, top_k=top_k)
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        r_at_k = recall_at_k(results, eq.relevant_pages, top_k)
        p_at_k = precision_at_k(results, eq.relevant_pages, top_k)
        rr = mean_reciprocal_rank(results, eq.relevant_pages)

        per_question_results.append({
            "question": eq.question,
            "recall_at_k": r_at_k,
            "precision_at_k": p_at_k,
            "reciprocal_rank": rr,
        })

        logger.info(
            f"[{strategy_name}] {eq.question!r} -> "
            f"Recall@{top_k}={r_at_k:.2f}, Precision@{top_k}={p_at_k:.2f}, RR={rr:.2f}"
        )

    n = len(per_question_results)
    aggregate = {
        "mean_recall_at_k": sum(r["recall_at_k"] for r in per_question_results) / n,
        "mean_precision_at_k": sum(r["precision_at_k"] for r in per_question_results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in per_question_results) / n,
    }

    return {
        "strategy": strategy_name,
        "top_k": top_k,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_questions": n,
        "aggregate": aggregate,
        "per_question": per_question_results,
    }


def save_results(results: dict) -> str:
    """Save one evaluation run's results to experiments/, timestamped so multiple runs accumulate rather than overwrite."""
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = results["timestamp"].replace(":", "-").split(".")[0]
    filename = f"{EXPERIMENTS_DIR}/eval_{results['strategy']}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {filename}")
    return filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print(f"Evaluating {len(EVAL_DATASET)} questions across two retrieval strategies...\n")

    dense_results = evaluate_strategy("dense", top_k=5)
    save_results(dense_results)

    hybrid_results = evaluate_strategy("hybrid", top_k=5)
    save_results(hybrid_results)

    print("\n" + "=" * 70)
    print("COMPARISON: dense vs. hybrid (aggregate, top_k=5)")
    print("=" * 70)
    print(f"{'Metric':<20} {'Dense':>10} {'Hybrid':>10}")
    for metric in ["mean_recall_at_k", "mean_precision_at_k", "mrr"]:
        d = dense_results["aggregate"][metric]
        h = hybrid_results["aggregate"][metric]
        print(f"{metric:<20} {d:>10.3f} {h:>10.3f}")