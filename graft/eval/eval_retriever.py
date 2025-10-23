"""Evaluate retriever with Recall@K, nDCG@K, MRR on multi-hop QA datasets."""

import json
import logging
import faiss
import torch
import numpy as np
from tqdm import tqdm

from graft.models.encoder import Encoder

logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved, gold, k):
    retrieved_k = set(retrieved[:k])
    gold_set = set(gold)
    return len(retrieved_k & gold_set) / len(gold_set) if gold_set else 0.0


def compute_mrr(retrieved, gold):
    for i, doc_id in enumerate(retrieved):
        if doc_id in gold:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_retriever(encoder_path, index_path, config, output_path, dataset, graph):
    """Evaluate retriever on pre-loaded dataset.

    Args:
        encoder_path: Path to encoder checkpoint
        index_path: Path to FAISS index
        config: Config dict
        output_path: Path to save results JSON
        dataset: Pre-loaded HuggingFace dataset
        graph: Pre-loaded PyG graph
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(
        model_name=config["encoder"]["model_name"],
        max_len=config["encoder"]["max_len"],
        pool=config["encoder"]["pool"],
        proj_dim=config["encoder"]["proj_dim"],
        freeze_layers=0
    )

    encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    encoder.to(device)
    encoder.eval()

    index = faiss.read_index(index_path)

    title_to_id = graph.title_to_id

    queries = []
    for item in dataset:
        supporting_facts = item.get("supporting_facts", {})
        if supporting_facts and "title" in supporting_facts:
            gold_ids = [title_to_id[t] for t in supporting_facts["title"] if t in title_to_id]
            if gold_ids:
                queries.append({
                    "id": item["id"],
                    "question": item["question"],
                    "gold_ids": gold_ids
                })

    topk = config["index"]["topk"]
    recall_at_5 = []
    recall_at_10 = []
    recall_at_50 = []
    mrr_scores = []

    for q in tqdm(queries, desc="Evaluating retriever"):
        query_text = q["question"]
        gold_ids = q["gold_ids"]

        query_embed = encoder.encode([query_text], device).cpu().numpy()
        faiss.normalize_L2(query_embed)

        distances, indices = index.search(query_embed, topk)
        retrieved = indices[0].tolist()

        recall_at_5.append(compute_recall_at_k(retrieved, gold_ids, 5))
        recall_at_10.append(compute_recall_at_k(retrieved, gold_ids, 10))
        recall_at_50.append(compute_recall_at_k(retrieved, gold_ids, 50))
        mrr_scores.append(compute_mrr(retrieved, gold_ids))

    results = {
        "recall@5": np.mean(recall_at_5),
        "recall@10": np.mean(recall_at_10),
        "recall@50": np.mean(recall_at_50),
        "mrr@10": np.mean(mrr_scores)
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results:")
    for k, v in results.items():
        logger.info(f"  {k}: {v:.4f}")
