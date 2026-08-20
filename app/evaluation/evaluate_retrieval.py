"""
app/evaluation/evaluate_retrieval.py

Evaluation benchmark comparing:
  - Dense baseline (with task prefix)
  - Hybrid RRF baseline (Dense + BM25)
  - Hybrid + FlashRank Reranker (Cross-encoder)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.evaluation.dataset import EVAL_DATASET
from app.evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from app.generation.rag_pipeline import retrieve_context

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = "experiments"


def evaluate_strategy(strategy_name: str, top_k: int = 5) -> dict:
    """Run EVAL_DATASET questions through a strategy and compute IR metrics."""
    per_question_results = []

    for eq in EVAL_DATASET:
        results = retrieve_context(eq.question, top_k=top_k, retrieval_strategy=strategy_name)

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
            f"[{strategy_name}] {eq.question[:40]}... -> "
            f"Recall@{top_k}={r_at_k:.2f}, Precision@{top_k}={p_at_k:.2f}, RR={rr:.2f}"
        )

    n = len(per_question_results)
    aggregate = {
        "mean_recall_at_k": sum(r["recall_at_k"] for r in per_question_results) / n if n else 0.0,
        "mean_precision_at_k": sum(r["precision_at_k"] for r in per_question_results) / n if n else 0.0,
        "mrr": sum(r["reciprocal_rank"] for r in per_question_results) / n if n else 0.0,
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
    Path(EXPERIMENTS_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = results["timestamp"].replace(":", "-").split(".")[0]
    filename = f"{EXPERIMENTS_DIR}/eval_{results['strategy']}_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {filename}")
    return filename


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    strategies = ["dense", "hybrid_no_rerank", "hybrid"]
    all_results = {}

    print(f"\nEvaluating {len(EVAL_DATASET)} questions across {len(strategies)} strategies (top_k=5)...\n")

    for strat in strategies:
        res = evaluate_strategy(strat, top_k=5)
        save_results(res)
        all_results[strat] = res

    print("\n" + "=" * 75)
    print(f"{'Metric':<25} {'Dense':>12} {'Hybrid (RRF)':>16} {'Hybrid + Reranker':>18}")
    print("=" * 75)
    for metric in ["mean_recall_at_k", "mean_precision_at_k", "mrr"]:
        d = all_results["dense"]["aggregate"][metric]
        h_rrf = all_results["hybrid_no_rerank"]["aggregate"][metric]
        h_rerank = all_results["hybrid"]["aggregate"][metric]
        print(f"{metric:<25} {d:>12.3f} {h_rrf:>16.3f} {h_rerank:>18.3f}")
    print("=" * 75)