"""Load query-doc pairs from mteb/hotpotqa using HF datasets."""

import logging
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_query_pairs(split, graph_path, config, log=True):
    """Load query-doc pairs from HF dataset (chunked graph).

    Args:
        split: Dataset split (train/dev/test)
        graph_path: Path to graph .pt file
        config: Config dict with dataset name
        log: Whether to log the loading info
    """
    graph = torch.load(graph_path, weights_only=False)
    doc_id_to_node_ids = graph.doc_id_to_node_ids

    dataset = config["data"]["dataset"]
    queries_ds = load_dataset(dataset, "queries", split="queries")
    qrels_ds = load_dataset(dataset, "default", split=split)

    qid_to_query = {item["_id"]: item["text"] for item in queries_ds}

    pairs = []
    for item in qrels_ds:
        qid = item["query-id"]
        doc_id = item["corpus-id"]
        score = item["score"]

        if score > 0 and doc_id in doc_id_to_node_ids:
            query_text = qid_to_query.get(qid)
            if query_text:
                for node_id in doc_id_to_node_ids[doc_id]:
                    pairs.append({"query": query_text, "pos_node": node_id, "qid": qid})

    if log:
        logger.info(f"Loaded {len(pairs)} query-doc pairs from {split} split")
    return pairs
