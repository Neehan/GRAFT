"""Build PyG graph from Wikipedia hyperlinks using HotpotQA context from HF datasets."""

import logging
import re
import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def extract_hyperlinks(text, title_to_id):
    """Extract Wikipedia hyperlinks from text (markdown-style [[Title]] format)."""
    links = re.findall(r'\[\[(.*?)\]\]', text)
    valid_links = [link for link in links if link in title_to_id]
    return valid_links


def build_hotpot_graph(dataset, output_path):
    """Build graph from HotpotQA context passages with Wikipedia hyperlinks.

    Args:
        dataset: Pre-loaded HuggingFace dataset
        output_path: Path to save graph .pt file
    """
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
    for title, text in passages_by_title.items():
        src_id = title_to_id[title]
        links = extract_hyperlinks(text, title_to_id)
        for link in links:
            dst_id = title_to_id[link]
            edge_list.append((src_id, dst_id))

    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(edge_index=edge_index)
    graph.node_text = node_texts
    graph.title_to_id = title_to_id

    torch.save(graph, output_path)
    logger.info(f"Graph saved: {len(node_texts)} nodes, {edge_index.size(1)} edges")
