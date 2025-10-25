"""Load query-doc pairs from mteb/hotpotqa using HF datasets."""

import logging
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_query_pairs(split, graph_path, config, log=True):
    """Load query-doc pairs from HF dataset (chunked graph).

    Returns list of dicts with:
        - query: query text
        - pos_nodes: list of positive node IDs (all chunks from all supporting docs)
        - qid: query ID

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

    qid_to_query = {item["_id"]: item["text"] for item in queries_ds}  # type: ignore

    qid_to_docs = {}
    for item in qrels_ds:
        qid = item["query-id"]  # type: ignore
        doc_id = item["corpus-id"]  # type: ignore
        score = item["score"]  # type: ignore

        if score > 0 and doc_id in doc_id_to_node_ids:
            if qid not in qid_to_docs:
                qid_to_docs[qid] = []
            qid_to_docs[qid].append(doc_id)

    pairs = []
    for qid, doc_ids in qid_to_docs.items():
        query_text = qid_to_query.get(qid)
        if not query_text:
            continue

        all_nodes = []
        for doc_id in doc_ids:
            all_nodes.extend(doc_id_to_node_ids[doc_id])

        if all_nodes:
            pairs.append({"query": query_text, "pos_nodes": all_nodes, "qid": qid})

    if log:
        logger.info(
            f"Loaded {len(pairs)} query pairs (with {sum(len(p['pos_nodes']) for p in pairs)} total positive nodes) from {split} split"
        )
    return pairs
