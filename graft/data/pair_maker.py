"""Load query-doc pairs from HotpotQA supporting facts using HF datasets."""

import logging
import torch

logger = logging.getLogger(__name__)


def load_query_pairs(dataset, graph_path):
    """Load query-doc pairs from HotpotQA HF dataset (chunked graph).

    Args:
        dataset: Pre-loaded HuggingFace dataset
        graph_path: Path to graph .pt file
    """
    graph = torch.load(graph_path, weights_only=False)
    title_to_node_ids = graph.title_to_node_ids

    pairs = []
    for item in dataset:
        qid = item["id"]
        question = item["question"]
        supporting_facts = item.get("supporting_facts", {})

        if supporting_facts and "title" in supporting_facts:
            for title in supporting_facts["title"]:
                if title in title_to_node_ids:
                    for node_id in title_to_node_ids[title]:
                        pairs.append(
                            {"query": question, "pos_node": node_id, "qid": qid}
                        )

    logger.info(f"Loaded {len(pairs)} query-doc pairs")
    return pairs
