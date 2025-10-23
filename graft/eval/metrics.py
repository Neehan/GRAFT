"""Shared metrics for retrieval evaluation: Recall@K, MRR."""

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved, gold, k):
    """Compute Recall@K for a single query.

    Args:
        retrieved: List of retrieved document IDs (ranked)
        gold: List/set of gold document IDs
        k: Cutoff for top-k

    Returns:
        Recall@K score (fraction of gold docs in top-k)
    """
    retrieved_k = set(retrieved[:k])
    gold_set = set(gold)
    return len(retrieved_k & gold_set) / len(gold_set) if gold_set else 0.0


def compute_mrr(retrieved, gold):
    """Compute Mean Reciprocal Rank for a single query.

    Args:
        retrieved: List of retrieved document IDs (ranked)
        gold: List/set of gold document IDs

    Returns:
        MRR score (1/rank of first gold doc, 0 if none found)
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def aggregate_and_save_results(all_scores, output_path, method_name):
    """Aggregate per-query scores and save results to JSON.

    Args:
        all_scores: Dict with keys like 'recall@5', 'recall@10', etc.
                    Each value is a list of per-query scores
        output_path: Path to save results JSON
        method_name: Name of method (for logging)
    """
    results = {metric: np.mean(scores) for metric, scores in all_scores.items()}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"{method_name} results:")
    for metric, score in results.items():
        logger.info(f"  {metric}: {score:.4f}")

    return results
