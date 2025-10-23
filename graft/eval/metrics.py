"""Shared metrics for multi-hop retrieval evaluation: Joint-Recall, Recall@K, nDCG, MRR."""

import json
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved, gold, k):
    """Compute Recall@K (any) for a single query.

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


def compute_joint_recall_at_k(retrieved, gold, k):
    """Compute Joint-Recall@K for a single query (multi-hop retrieval).

    Args:
        retrieved: List of retrieved document IDs (ranked)
        gold: List/set of gold document IDs
        k: Cutoff for top-k

    Returns:
        1.0 if ALL gold docs are in top-k, 0.0 otherwise
    """
    retrieved_k = set(retrieved[:k])
    gold_set = set(gold)
    return 1.0 if gold_set.issubset(retrieved_k) else 0.0


def compute_mrr(retrieved, gold, k):
    """Compute Mean Reciprocal Rank@K for a single query.

    Args:
        retrieved: List of retrieved document IDs (ranked)
        gold: List/set of gold document IDs
        k: Cutoff for top-k

    Returns:
        MRR@K score (1/rank of first gold doc in top-k, 0 if none found)
    """
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg_at_k(retrieved, gold, k):
    """Compute nDCG@K for a single query.

    Args:
        retrieved: List of retrieved document IDs (ranked)
        gold: List/set of gold document IDs
        k: Cutoff for top-k

    Returns:
        nDCG@K score
    """
    retrieved_k = retrieved[:k]
    gold_set = set(gold)

    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, doc_id in enumerate(retrieved_k)
        if doc_id in gold_set
    )

    ideal_k = min(len(gold_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_k))

    return dcg / idcg if idcg > 0 else 0.0


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
