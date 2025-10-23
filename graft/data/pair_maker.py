"""Load query-doc pairs and create positive node mappings from HotpotQA supporting facts using HF datasets."""

import json
import logging
import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_query_pairs(split, graph_path):
    """Load query-doc pairs from HotpotQA HF dataset."""
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    graph = torch.load(graph_path)
    title_to_id = graph.title_to_id

    pairs = []
    for item in dataset:
        qid = item["id"]
        question = item["question"]
        supporting_facts = item.get("supporting_facts", {})

        if supporting_facts and "title" in supporting_facts:
            for title in supporting_facts["title"]:
                if title in title_to_id:
                    pairs.append({
                        "query": question,
                        "pos_node": title_to_id[title],
                        "qid": qid
                    })

    logger.info(f"Loaded {len(pairs)} query-doc pairs from {split} split")
    return pairs


def create_pos_map_from_hotpot(split, graph_path, output_path):
    """Create qid->positive node mapping from HotpotQA."""
    dataset = load_dataset("hotpot_qa", "distractor", split=split)
    graph = torch.load(graph_path)
    title_to_id = graph.title_to_id

    qid2pos = {}
    for item in dataset:
        qid = item["id"]
        supporting_facts = item.get("supporting_facts", {})

        if supporting_facts and "title" in supporting_facts:
            pos_ids = []
            for title in supporting_facts["title"]:
                if title in title_to_id:
                    pos_ids.append(title_to_id[title])
            if pos_ids:
                qid2pos[qid] = pos_ids

    with open(output_path, "w") as f:
        json.dump(qid2pos, f)

    logger.info(f"Created pos_map: {len(qid2pos)} queries from {split} split")
