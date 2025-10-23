"""Build PyG graph from HotpotQA with chunking and supporting fact edges."""

import logging
import torch
from torch_geometric.data import Data
from collections import defaultdict

logger = logging.getLogger(__name__)


def chunk_text(text, chunk_size, overlap):
    """Split text into overlapping chunks by tokens."""
    tokens = text.split()
    chunks = []

    if len(tokens) <= chunk_size:
        return [text]

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = ' '.join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

        if end >= len(tokens):
            break

    return chunks


def build_hotpot_graph(dataset, output_path, chunk_size=200, chunk_overlap=50):
    """Build graph from HotpotQA with chunking and supporting fact co-occurrence edges.

    Args:
        dataset: Pre-loaded HuggingFace dataset
        output_path: Path to save graph .pt file
        chunk_size: Max tokens per chunk
        chunk_overlap: Overlapping tokens between chunks
    """
    passage_texts = {}

    for item in dataset:
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title not in passage_texts:
                passage_texts[title] = " ".join(sentences)

    node_id = 0
    title_to_node_ids = defaultdict(list)
    node_texts = []
    edge_list = []

    for title, text in passage_texts.items():
        chunks = chunk_text(text, chunk_size, chunk_overlap)
        chunk_ids = []

        for chunk in chunks:
            node_texts.append(chunk)
            chunk_ids.append(node_id)
            title_to_node_ids[title].append(node_id)
            node_id += 1

        for i in range(len(chunk_ids) - 1):
            edge_list.append((chunk_ids[i], chunk_ids[i + 1]))
            edge_list.append((chunk_ids[i + 1], chunk_ids[i]))

    for item in dataset:
        supporting_titles = item.get("supporting_facts", {}).get("title", [])
        if len(supporting_titles) < 2:
            continue

        for i, title1 in enumerate(supporting_titles):
            for title2 in supporting_titles[i+1:]:
                if title1 in title_to_node_ids and title2 in title_to_node_ids:
                    for node1 in title_to_node_ids[title1]:
                        for node2 in title_to_node_ids[title2]:
                            edge_list.append((node1, node2))
                            edge_list.append((node2, node1))

    if edge_list:
        edge_index = torch.tensor(list(set(edge_list)), dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(edge_index=edge_index)
    graph.node_text = node_texts
    graph.title_to_node_ids = dict(title_to_node_ids)

    torch.save(graph, output_path)
    logger.info(f"Graph saved: {len(node_texts)} nodes ({len(passage_texts)} passages), {edge_index.size(1)} edges")
