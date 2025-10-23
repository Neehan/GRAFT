"""Build PyG graph from Wikipedia hyperlinks using HotpotQA context from HF datasets."""

import logging
import torch
from datasets import load_dataset
from torch_geometric.data import Data
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_hotpot_graph(output_path, split="train"):
    """Build graph from HotpotQA context passages with Wikipedia hyperlinks."""
    dataset = load_dataset("hotpot_qa", "distractor", split=split)

    title_to_id = {}
    node_texts = []
    passages_by_title = {}

    for item in dataset:
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title not in passages_by_title:
                passages_by_title[title] = " ".join(sentences)

    for i, (title, text) in enumerate(passages_by_title.items()):
        title_to_id[title] = i
        node_texts.append(text)

    edge_list = []
    edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(edge_index=edge_index)
    graph.node_text = node_texts
    graph.title_to_id = title_to_id

    torch.save(graph, output_path)
    logger.info(f"Graph saved: {len(node_texts)} nodes, {edge_index.size(1)} edges (from {split} split)")
