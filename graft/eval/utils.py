"""Common evaluation utilities shared across methods."""

import logging
from tqdm import tqdm
from datasets import load_dataset

from graft.eval.metrics import (
    compute_recall_at_k,
    compute_joint_recall_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    aggregate_and_save_results,
)

logger = logging.getLogger(__name__)


def prepare_queries(split, doc_id_to_id):
    """Extract queries with gold document IDs from mteb/hotpotqa.

    Args:
        split: Dataset split (train/dev/test)
        doc_id_to_id: Mapping from document ID to node ID

    Returns:
        List of dicts with keys: id, question, gold_ids
    """
    queries_ds = load_dataset("mteb/hotpotqa", "queries", split="queries")
    qrels_ds = load_dataset("mteb/hotpotqa", "default", split=split)

    qid_to_query = {item["_id"]: item["text"] for item in queries_ds}

    query_gold_docs = {}
    for item in qrels_ds:
        qid = item["query-id"]
        doc_id = item["corpus-id"]
        score = item["score"]

        if score > 0 and doc_id in doc_id_to_id:
            if qid not in query_gold_docs:
                query_gold_docs[qid] = {"query": qid_to_query.get(qid), "gold_ids": []}
            query_gold_docs[qid]["gold_ids"].append(doc_id_to_id[doc_id])

    queries = []
    for qid, data in query_gold_docs.items():
        if data["query"] and data["gold_ids"]:
            queries.append(
                {
                    "id": qid,
                    "question": data["query"],
                    "gold_ids": data["gold_ids"],
                }
            )

    return queries


def evaluate_retrieval(
    queries, retrieval_fn, topk, method_name, output_path, eval_config, batch_size=1
):
    """Generic evaluation loop for any retrieval method.

    Args:
        queries: List of query dicts (from prepare_queries)
        retrieval_fn: Function that takes query_text(s) and k, returns list of doc IDs (or list of lists for batch)
        topk: Number of docs to retrieve
        method_name: Name for logging
        output_path: Path to save results JSON
        eval_config: Config dict with eval settings (recall_k, joint_recall_k, ndcg_k, mrr_k)
        batch_size: Number of queries to process at once (for multi-GPU optimization)

    Returns:
        Results dict with metrics
    """
    recall_k_values = eval_config["recall_k"]
    joint_recall_k = eval_config["joint_recall_k"]
    ndcg_k = eval_config["ndcg_k"]
    mrr_k = eval_config["mrr_k"]

    all_scores = {f"joint_recall@{joint_recall_k}": []}
    for k in recall_k_values:
        all_scores[f"recall@{k}"] = []
    all_scores[f"ndcg@{ndcg_k}"] = []
    all_scores[f"mrr@{mrr_k}"] = []

    for i in tqdm(range(0, len(queries), batch_size), desc=f"Evaluating {method_name}"):
        batch_queries = queries[i : i + batch_size]
        query_texts = [q["question"] for q in batch_queries]

        retrieved_batch = retrieval_fn(query_texts, topk)

        for q, retrieved in zip(batch_queries, retrieved_batch):
            gold_ids = q["gold_ids"]
            all_scores[f"joint_recall@{joint_recall_k}"].append(
                compute_joint_recall_at_k(retrieved, gold_ids, joint_recall_k)
            )
            for k in recall_k_values:
                all_scores[f"recall@{k}"].append(
                    compute_recall_at_k(retrieved, gold_ids, k)
                )
            all_scores[f"ndcg@{ndcg_k}"].append(
                compute_ndcg_at_k(retrieved, gold_ids, ndcg_k)
            )
            all_scores[f"mrr@{mrr_k}"].append(compute_mrr(retrieved, gold_ids, mrr_k))

    return aggregate_and_save_results(all_scores, output_path, method_name)
