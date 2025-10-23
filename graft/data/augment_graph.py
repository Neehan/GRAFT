"""Augment graph with kNN edges based on embedding similarity."""

import logging
import torch
import numpy as np
import faiss
from pathlib import Path
from tqdm import tqdm

from graft.models.encoder import Encoder

logger = logging.getLogger(__name__)


def embed_graph_nodes(graph, config, device=None):
    """Embed all graph nodes using pretrained encoder.

    Args:
        graph: PyG graph with node_text attribute
        config: Config dict with encoder settings
        device: torch device

    Returns:
        numpy array of embeddings (num_nodes, hidden_dim)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = config["encoder"]["model_name"]
    logger.info(f"Loading encoder: {model_name}")
    encoder = Encoder(
        model_name=model_name,
        max_len=config["encoder"]["max_len"],
        pool=config["encoder"]["pool"],
        freeze_layers=0,
    )
    encoder.to(device)
    encoder.eval()

    node_texts = graph.node_text
    embeddings = []
    batch_size = config["encoder"].get("batch_size", 128)

    logger.info(f"Embedding {len(node_texts)} nodes...")
    for i in tqdm(range(0, len(node_texts), batch_size), desc="Embedding"):
        batch = node_texts[i : i + batch_size]
        batch_embeds = encoder.encode(batch, device)
        embeddings.append(batch_embeds.cpu().numpy())

    return np.vstack(embeddings)


def build_knn_edges(embeddings, k):
    """Build undirected kNN edges using FAISS.

    Args:
        embeddings: numpy array (num_nodes, hidden_dim)
        k: Number of nearest neighbors

    Returns:
        List of (src, dst) tuples
    """
    num_nodes, dim = embeddings.shape

    logger.info(f"Building FAISS index for {num_nodes} nodes...")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    logger.info(f"Querying k={k} nearest neighbors...")
    distances, indices = index.search(embeddings, k + 1)

    edges = []
    for src_node in tqdm(range(num_nodes), desc="Building edges"):
        neighbors = indices[src_node][1:]
        for dst_node in neighbors:
            edges.append((src_node, int(dst_node)))
            edges.append((int(dst_node), src_node))

    return list(set(edges))


def augment_with_knn(
    input_graph_path,
    output_graph_path,
    config,
    k,
    knn_only=False,
    force_recompute=False,
):
    """Augment graph with kNN edges.

    Args:
        input_graph_path: Path to input graph .pt file
        output_graph_path: Path to save augmented graph
        config: Config dict with encoder settings
        k: Number of nearest neighbors
        knn_only: If True, use only kNN edges (ablation). If False, add to existing edges (hybrid)
        force_recompute: If True, ignore cached embeddings

    Returns:
        Path to output graph
    """
    logger.info(f"Loading graph from {input_graph_path}")
    graph = torch.load(input_graph_path, weights_only=False)

    model_name = config["encoder"]["model_name"]
    embeddings_cache = str(input_graph_path).replace(
        ".pt", f"_embeddings_{model_name.replace('/', '_')}.npy"
    )

    if Path(embeddings_cache).exists() and not force_recompute:
        logger.info(f"Loading cached embeddings from {embeddings_cache}")
        embeddings = np.load(embeddings_cache)
    else:
        embeddings = embed_graph_nodes(graph, config)
        logger.info(f"Caching embeddings to {embeddings_cache}")
        np.save(embeddings_cache, embeddings)

    old_num_edges = graph.edge_index.size(1)
    old_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
    if old_num_edges > 0:
        for node_id in graph.edge_index[0]:
            old_degrees[node_id] += 1
    old_mean_degree = old_degrees.float().mean().item()

    knn_edges = build_knn_edges(embeddings, k)

    if knn_only:
        logger.info(f"Mode: kNN-only (ablation) - using only semantic edges")
        new_edge_index = torch.tensor(knn_edges, dtype=torch.long).T
    else:
        logger.info(
            f"Mode: Hybrid - keeping structural edges + adding semantic kNN edges"
        )
        existing_edges = graph.edge_index.T.tolist()
        combined_edges = list(set(existing_edges + knn_edges))
        new_edge_index = torch.tensor(combined_edges, dtype=torch.long).T

    graph.edge_index = new_edge_index

    new_num_edges = new_edge_index.size(1)
    new_degrees = torch.zeros(len(graph.node_text), dtype=torch.long)
    if new_num_edges > 0:
        for node_id in new_edge_index[0]:
            new_degrees[node_id] += 1
    new_mean_degree = new_degrees.float().mean().item()

    logger.info(f"Graph augmentation complete:")
    logger.info(f"  Before: {old_num_edges} edges, mean degree={old_mean_degree:.2f}")
    logger.info(f"  After:  {new_num_edges} edges, mean degree={new_mean_degree:.2f}")
    if not knn_only:
        logger.info(f"  Added: {new_num_edges - old_num_edges} kNN edges")

    Path(output_graph_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_graph_path)
    logger.info(f"Augmented graph saved to {output_graph_path}")

    return output_graph_path


if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Augment graph with kNN edges based on semantic similarity"
    )
    parser.add_argument("input_graph", type=str, help="Path to input graph .pt file")
    parser.add_argument("output_graph", type=str, help="Path to save augmented graph")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument(
        "--k", type=int, default=5, help="Number of nearest neighbors (default: 5)"
    )
    parser.add_argument(
        "--knn-only",
        action="store_true",
        help="Use only kNN edges (ablation), discard structural edges",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recompute embeddings even if cached",
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    augment_with_knn(
        args.input_graph,
        args.output_graph,
        config,
        args.k,
        args.knn_only,
        args.force_recompute,
    )
