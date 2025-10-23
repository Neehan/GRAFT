"""Build PyG graph from HotpotQA with chunking and supporting fact edges."""

import logging
import torch
from torch_geometric.data import Data
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_hotpot_graph(
    corpus, qrels_train, output_path, chunk_size=200, chunk_overlap=50
):
    """Build graph from mteb/hotpotqa with chunking and co-occurrence edges.

    Args:
        corpus: Filtered corpus dataset (only docs in train split)
        qrels_train: Train split qrels for building graph edges
        output_path: Path to save graph .pt file
        chunk_size: Max tokens per chunk
        chunk_overlap: Overlapping tokens between chunks
    """
    passage_texts = {}
    doc_id_to_title = {}

    for item in corpus:
        doc_id = item["_id"]
        title = item["title"]
        text = item["text"]
        passage_texts[doc_id] = text
        doc_id_to_title[doc_id] = title

    node_id = 0
    doc_id_to_node_ids = defaultdict(list)
    node_texts = []
    edge_list = []

    for doc_id, text in passage_texts.items():
        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        chunk_ids = []

        for chunk in chunks:
            node_texts.append(chunk)
            chunk_ids.append(node_id)
            doc_id_to_node_ids[doc_id].append(node_id)
            node_id += 1

        for i in range(len(chunk_ids) - 1):
            edge_list.append((chunk_ids[i], chunk_ids[i + 1]))
            edge_list.append((chunk_ids[i + 1], chunk_ids[i]))

    query_to_docs = defaultdict(set)
    for item in qrels_train:
        qid = item["query-id"]
        doc_id = item["corpus-id"]
        score = item["score"]

        if score > 0:
            query_to_docs[qid].add(doc_id)

    for doc_ids in query_to_docs.values():
        doc_ids = list(doc_ids)
        if len(doc_ids) < 2:
            continue

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc_id1, doc_id2 = doc_ids[i], doc_ids[j]
                if doc_id1 in doc_id_to_node_ids and doc_id2 in doc_id_to_node_ids:
                    for node1 in doc_id_to_node_ids[doc_id1]:
                        for node2 in doc_id_to_node_ids[doc_id2]:
                            edge_list.append((node1, node2))
                            edge_list.append((node2, node1))

    if edge_list:
        edge_index = torch.tensor(list(set(edge_list)), dtype=torch.long).T
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = Data(edge_index=edge_index)
    graph.node_text = node_texts
    graph.doc_id_to_node_ids = dict(doc_id_to_node_ids)
    graph.doc_id_to_id = {
        doc_id: doc_id_to_node_ids[doc_id][0] for doc_id in doc_id_to_node_ids
    }

    num_nodes = len(node_texts)
    num_edges = edge_index.size(1)

    logger.info(
        f"Graph saved: {num_nodes} nodes ({len(passage_texts)} passages), {num_edges} edges"
    )
    _compute_graph_stats(edge_index, num_nodes)

    torch.save(graph, output_path)


def _chunk_text(text, chunk_size, overlap):
    """Split text into overlapping chunks by tokens."""
    tokens = text.split()
    chunks = []

    if len(tokens) <= chunk_size:
        return [text]

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

        if end >= len(tokens):
            break

    return chunks


def _compute_graph_stats(edge_index, num_nodes):
    """Compute and log graph degree statistics.

    Args:
        edge_index: Edge index tensor (2, E)
        num_nodes: Total number of nodes
    """
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    if edge_index.size(1) > 0:
        for node_id in edge_index[0]:
            degrees[node_id] += 1

    mean_degree = degrees.float().mean().item()
    std_degree = degrees.float().std().item()
    median_degree = degrees.float().median().item()
    max_degree = degrees.max().item()

    num_isolated = (degrees == 0).sum().item()
    pct_isolated = 100 * num_isolated / num_nodes if num_nodes > 0 else 0

    logger.info(
        f"Degree stats: mean={mean_degree:.2f}, std={std_degree:.2f}, "
        f"median={median_degree:.0f}, max={max_degree}"
    )
    logger.info(f"Isolated nodes: {num_isolated}/{num_nodes} ({pct_isolated:.1f}%)")
