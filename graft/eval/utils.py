"""Common evaluation utilities shared across methods."""

import logging
from tqdm import tqdm

from graft.eval.metrics import (
    compute_recall_at_k,
    compute_mrr,
    aggregate_and_save_results,
)

logger = logging.getLogger(__name__)


def prepare_queries(dataset, title_to_id):
    """Extract queries with gold document IDs from dataset.

    Args:
        dataset: HuggingFace dataset with supporting_facts
        title_to_id: Mapping from title to document ID

    Returns:
        List of dicts with keys: id, question, gold_ids
    """
    queries = []
    for item in dataset:
        supporting_facts = item.get("supporting_facts", {})
        if supporting_facts and "title" in supporting_facts:
            gold_ids = [
                title_to_id[t] for t in supporting_facts["title"] if t in title_to_id
            ]
            if gold_ids:
                queries.append(
                    {
                        "id": item["id"],
                        "question": item["question"],
                        "gold_ids": gold_ids,
                    }
                )
    return queries


def evaluate_retrieval(queries, retrieval_fn, topk, method_name, output_path):
    """Generic evaluation loop for any retrieval method.

    Args:
        queries: List of query dicts (from prepare_queries)
        retrieval_fn: Function that takes query_text and returns list of doc IDs
        topk: Number of docs to retrieve
        method_name: Name for logging
        output_path: Path to save results JSON

    Returns:
        Results dict with metrics
    """
    all_scores = {
        "recall@5": [],
        "recall@10": [],
        "recall@50": [],
        "mrr": [],
    }

    for q in tqdm(queries, desc=f"Evaluating {method_name}"):
        query_text = q["question"]
        gold_ids = q["gold_ids"]

        retrieved = retrieval_fn(query_text, topk)

        all_scores["recall@5"].append(compute_recall_at_k(retrieved, gold_ids, 5))
        all_scores["recall@10"].append(compute_recall_at_k(retrieved, gold_ids, 10))
        all_scores["recall@50"].append(compute_recall_at_k(retrieved, gold_ids, 50))
        all_scores["mrr"].append(compute_mrr(retrieved, gold_ids))

    return aggregate_and_save_results(all_scores, output_path, method_name)
