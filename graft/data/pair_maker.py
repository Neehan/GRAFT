"""Load query-doc pairs and create positive node mappings from HotpotQA supporting facts using HF datasets."""

import json
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
                        pairs.append({
                            "query": question,
                            "pos_node": node_id,
                            "qid": qid
                        })

    logger.info(f"Loaded {len(pairs)} query-doc pairs")
    return pairs


def create_pos_map_from_hotpot(dataset, graph_path, output_path):
    """Create qid->positive node mapping from HotpotQA (chunked graph).

    Args:
        dataset: Pre-loaded HuggingFace dataset
        graph_path: Path to graph .pt file
        output_path: Path to save pos_map JSON
    """
    graph = torch.load(graph_path, weights_only=False)
    title_to_node_ids = graph.title_to_node_ids

    qid2pos = {}
    for item in dataset:
        qid = item["id"]
        supporting_facts = item.get("supporting_facts", {})

        if supporting_facts and "title" in supporting_facts:
            pos_ids = []
            for title in supporting_facts["title"]:
                if title in title_to_node_ids:
                    pos_ids.extend(title_to_node_ids[title])
            if pos_ids:
                qid2pos[qid] = pos_ids

    with open(output_path, "w") as f:
        json.dump(qid2pos, f)

    logger.info(f"Created pos_map: {len(qid2pos)} queries")
